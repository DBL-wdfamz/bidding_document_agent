# -*- coding: utf-8 -*-
"""
初始化官方知识库脚本（独立运行）
- 扫描 static/kb_initialize/<KB名称> 下的文件（支持: pdf/txt/md/docx/csv/xlsx）
- 在 DB 中创建/更新 official KB（owner=0）
- 在 {KB_BASE_ROOT}/0/{knowledge_bases_id}/faiss_index 重建/更新向量索引
- 重置 kb_files 表记录（官方库用文件名作为唯一标识 hash）

依赖：
  pip install sqlalchemy aiosqlite  # 若用 sqlite
  或根据你的 DATABASE_URL 安装相应驱动（eg: asyncpg）

注意：
  读取/写入向量索引依赖你的 KBIndex 实现：
  from knowledge_base_package.knowledge_base import KnowledgeBase as KBIndex
"""

import os
import time
import argparse
import hashlib
import secrets
from pathlib import Path
from typing import List

import asyncio
from sqlalchemy import String, Integer, Text, UniqueConstraint, ForeignKey, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# === 与线上一致：允许的扩展名 ===
ALLOWED_SUFFIX = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx"}

# === 你的向量检索实现（与线上保持一致） ===
from knowledge_base_package.knowledge_base import KnowledgeBase as KBIndex  # noqa: E402


# ---------------- SQLAlchemy 最小模型（与线上表结构字段名保持一致） ----------------

class Base(DeclarativeBase):
    pass

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_bases_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    owner: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint("owner", "name", name="uq_kb_owner_name"),  # 防重复（可选）
    )

class KbFile(Base):
    __tablename__ = "kb_files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_id: Mapped[int] = mapped_column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)   # 逻辑名
    hash: Mapped[str] = mapped_column(String(255), nullable=False)   # 官方：文件名作为“标识”
    type: Mapped[str] = mapped_column(String(50), nullable=False)    # 扩展名


# ---------------- 工具函数 ----------------

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def find_seed_kbs(seed_dir: Path) -> List[Path]:
    """返回所有子目录（每个子目录即一个知识库）"""
    if not seed_dir.exists():
        return []
    return [p for p in sorted(seed_dir.iterdir()) if p.is_dir()]

def collect_files(kb_dir: Path) -> List[Path]:
    files = []
    for p in sorted(kb_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX:
            files.append(p)
    return files


# ---------------- 核心逻辑 ----------------

async def init_official_kb_for_dir(
    session: AsyncSession,
    kb_name: str,
    seed_files: List[Path],
    kb_base_root: Path,
    rebuild_index: bool,
) -> dict:
    """
    对单个 seed 目录执行初始化/重建：
    - DB 中查找 (owner=0, name=kb_name) 不存在则创建
    - 根据 kb_row.knowledge_bases_id 构造索引目录
    - 若 --rebuild-index 则删除整个 faiss_index 目录
    - 删除旧源（按 basename），添加新文档
    - 重置 kb_files 记录
    """
    # 1) 查/建 KB 行
    res = await session.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.owner == 0,
            KnowledgeBase.name == kb_name
        )
    )
    kb_row = res.scalar_one_or_none()
    created = False
    if not kb_row:
        kb_hash = sha256_hex(f"official:{kb_name}:{time.time()}:{secrets.token_hex(8)}")
        kb_row = KnowledgeBase(
            knowledge_bases_id=kb_hash,
            owner=0,
            name=kb_name
        )
        session.add(kb_row)
        await session.commit()
        await session.refresh(kb_row)
        created = True

    # 2) 索引目录
    index_dir = kb_base_root / "0" / kb_row.knowledge_bases_id / "faiss_index"
    if rebuild_index and index_dir.exists():
        # 全量重建：删掉旧索引目录
        import shutil
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    kb = KBIndex(db_path=str(index_dir))

    # 3) 清理并添加文档
    for p in seed_files:
        try:
            kb.delete_documents_by_source(p.name)  # 约定按 basename 作为 source
        except Exception:
            pass

    if seed_files:
        docs = KBIndex.load_documents_from_paths([str(p) for p in seed_files])
        kb.add_documents(docs)

    # 4) 重置 kb_files 列表
    await session.execute(delete(KbFile).where(KbFile.knowledge_id == kb_row.id))
    for p in seed_files:
        session.add(KbFile(
            knowledge_id=kb_row.id,
            name=p.name,
            hash=p.name,               # 官方：以文件名作为“hash”标识
            type=p.suffix.lower()
        ))
    await session.commit()

    return {
        "name": kb_name,
        "id": kb_row.id,
        "knowledge_bases_id": kb_row.knowledge_bases_id,
        "files": len(seed_files),
        "created": created,
        "index_dir": str(index_dir),
    }


async def main():
    parser = argparse.ArgumentParser(description="Initialize official knowledge bases (owner=0)")
    parser.add_argument("--seed-dir", default=os.getenv("OFFICIAL_KB_SEED_DIR", "static/kb_initialize"), help="Directory containing official KB folders")
    parser.add_argument("--kb-base-root", default=os.getenv("KB_BASE_ROOT", "user_knowledge_bases"), help="Base dir for FAISS indexes")
    parser.add_argument("--db", default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db"), help="DATABASE_URL")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index directory (clean then re-add)")
    parser.add_argument("--create-tables", action="store_true", help="Create tables if not exist")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir).resolve()
    kb_base_root = Path(args.kb_base_root).resolve()
    kb_base_root.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(args.db, echo=False, future=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        if args.create_tables:
            # 仅创建这两个表（若不存在）
            await conn.run_sync(Base.metadata.create_all)

    kb_dirs = find_seed_kbs(seed_dir)
    if not kb_dirs:
        print(f"[WARN] No KB folders found under: {seed_dir}")
        return

    summary = []
    async with SessionLocal() as session:
        for kb_dir in kb_dirs:
            kb_name = kb_dir.name
            files = collect_files(kb_dir)
            info = await init_official_kb_for_dir(
                session=session,
                kb_name=kb_name,
                seed_files=files,
                kb_base_root=kb_base_root,
                rebuild_index=args.rebuild_index,
            )
            print(f"[OK] KB='{info['name']}' id={info['id']} kb_hash={info['knowledge_bases_id']} "
                  f"files={info['files']} created={info['created']} index_dir={info['index_dir']}")
            summary.append(info)

    # 汇总
    print("\n=== Summary ===")
    for s in summary:
        print(f"- {s['name']}  (id={s['id']}, kb={s['knowledge_bases_id']})  files={s['files']}  "
              f"{'CREATED' if s['created'] else 'UPDATED'}  → {s['index_dir']}")

if __name__ == "__main__":
    asyncio.run(main())
