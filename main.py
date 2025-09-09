import os
import re
import time
import asyncio
import mimetypes
import tempfile
import secrets
import shutil
from datetime import timedelta, datetime
from typing import Annotated, Optional, Union, List, Literal

from fastapi import FastAPI, Depends, HTTPException, status, Body, UploadFile, File
from fastapi import Path as PathParam
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse, StreamingResponse
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from passlib.context import CryptContext
from sqlalchemy import String, DateTime, func, select, UniqueConstraint, Integer, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv
from dataclasses import dataclass

import hashlib
import json
from pathlib import Path

from json2doc import render_docx,docx_to_pdf
from bid_review_package.draft_generator import generate_draft_stream
from bid_review_package.review_agents import (
    review_fiscal,
    review_style,
    review_policy,
    review_document_number,
    review_completeness
)
from bid_review_package.content_extractor import extract_content as _ec
from knowledge_base_package.knowledge_base import KnowledgeBase as KBIndex
from template import generate_template_from_file
from docnumber.models import AsyncSessionLocal as DocNumSession, engine as DOC_ENGINE, create_tables as docnumber_create_tables
from docnumber.docnumber import (
    create_user_document_number,
    get_document_number_by_user_id,
    delete_document_number,
    get_and_increment_document_number,
)

# =========================
# 环境 & 常量
# =========================
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
DOCNUMBER_DEFAULT_PREFIX = os.getenv("DOCNUMBER_DEFAULT_PREFIX", "测试公司(2025)第")
DOCNUMBER_DEFAULT_START  = os.getenv("DOCNUMBER_DEFAULT_START", "100号")
EXPORT_ROOT = Path(os.getenv("EXPORT_ROOT", r"D:\project\document_agent\exports")).resolve()
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "document_agent/media")).resolve()
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
KB_BASE_ROOT = Path(os.getenv("KB_BASE_ROOT", "user_knowledge_bases")).resolve()
KB_BASE_ROOT.mkdir(parents=True, exist_ok=True)
FORMWORK_ROOT = Path(os.getenv("FORMWORK_ROOT", "static/formwork")).resolve()
FORMWORK_ROOT.mkdir(parents=True, exist_ok=True)
MAX_REF_CHARS = 2000

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# 可选的刷新 token 黑名单（简单演示：进程内内存，生产建议做持久化）
revoked_refresh_tokens: set[str] = set()

# =========================
# 数据库 & ORM
# =========================

class Base(DeclarativeBase):
    pass

class LoginIn(BaseModel):
    username: str
    password: str

class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_user_email"),
                      UniqueConstraint("username", name="uq_user_username"))
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Template(Base):
    __tablename__ = "templates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 模板文件路径（JSON）

class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA256
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    template_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    knowledge_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # 原始输入
    file: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # 预留
    document: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 最终内容（建议 Text，如需很大可改 Text）
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Review(Base):
    __tablename__ = "reviews"
    reviewsid: Mapped[str] = mapped_column(String(64), primary_key=True)  # 哈希
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)         # 审查类型
    original: Mapped[str] = mapped_column(Text, nullable=False)
    modified: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_bases_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    owner: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

class KbFile(Base):
    __tablename__ = "kb_files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 与 knowledge_bases(id) 外键一致，并开启级联
    knowledge_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    hash: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)



engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# =========================
# 工具函数：密码 & Token
# =========================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)

def create_token(subject: dict, expires_delta: timedelta, token_type: str) -> str:
    now = int(time.time())                    # 用同一时钟源
    payload = {
        "sub": subject.get("sub"),
        "uid": subject.get("uid"),
        "type": token_type,
        "iat": now,
        "exp": now + int(expires_delta.total_seconds()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_access_token(uid: int, username: str) -> str:
    return create_token({"sub": username, "uid": uid}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")

def create_refresh_token(uid: int, username: str) -> str:
    return create_token({"sub": username, "uid": uid}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh")

def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

# =========================
# Pydantic Schemas
# =========================
class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token 过期秒数")

class RegisterIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=64)

class PublicUser(BaseModel):
    id: int
    email: EmailStr
    username: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class MePatch(BaseModel):
    # 可以只更新其中一个或多个
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(default=None, min_length=3, max_length=32)

class ChangePasswordIn(BaseModel):
    old_password: str = Field(min_length=6, max_length=64)
    new_password: str = Field(min_length=6, max_length=64)


class JobOverview(BaseModel):
    """用于任务列表的摘要视图"""
    job_id: str
    project_name: str
    create_time: datetime
    model_config = ConfigDict(from_attributes=True)


class JobDetail(BaseModel):
    """用于单个任务的详细视图"""
    job_id: str
    project_name: str
    user_id: int
    template_id: Optional[int]
    knowledge_id: Optional[str]
    content: Optional[str]
    document: Optional[str]
    create_time: datetime

    # 添加 model_config 来解决兼容性问题
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class KnowledgeBaseOut(BaseModel):
    id: int
    knowledge_bases_id: str
    owner: int
    name: str
    model_config = ConfigDict(from_attributes=True)

class ReviewOut(BaseModel):
    reviewsid: str
    job_id: str
    type: str
    original: str
    modified: Optional[str]
    reason: Optional[str]
    model_config = ConfigDict(from_attributes=True)

class TemplateOut(BaseModel):
    id: int
    name: str
    owner: int
    content: str
    model_config = ConfigDict(from_attributes=True)

class GenerateIn(BaseModel):
    project_name: str
    information: str
    knowledge_base_id: Union[str, List[Union[str, int]]] = ""
    template_id: str
    references: list = []

class GenerateOut(BaseModel):
    job_id: str
    result: dict  # 最终合并后的 JSON

class JobDownloadIn(BaseModel):
    type: Literal["docx", "pdf"]

class ReviewDraftIn(BaseModel):
    jobid: str
    knowledge_base_id: Union[str, List[Union[str, int]]]

class KbFileOut(BaseModel):
    id: int
    knowledge_id: int
    name: str
    hash: str
    type: str
    model_config = ConfigDict(from_attributes=True)

class KBCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=255, description="知识库名称")

class KBUploadFileItem(BaseModel):
    name: str = Field(min_length=1, max_length=255, description="文件逻辑名称")
    file_id: str = Field(min_length=1, description="上传接口返回的哈希文件名（带后缀）")

class KBUploadIn(BaseModel):
    id: int
    files: List[KBUploadFileItem]

class JobSaveIn(BaseModel):
    content: str = Field(min_length=1, description="要写入 jobs.document 的内容（字符串，可为JSON文本）")

class TemplateUploadIn(BaseModel):
    name: str = Field(min_length=1, max_length=255, description="模板名称")
    file_id: str = Field(min_length=1, description="媒体库哈希文件名（含后缀），如 734d...c99657.docx")

class PublicUserWithWenhao(PublicUser):
    wenhao: str

class DocNumberUpdateIn(BaseModel):
    # 二选一：传 full，或传 prefix+number_part
    full: Optional[str] = Field(default=None, description="完整文号，如：测试公司(2025)第207号")
    prefix: Optional[str] = Field(default=None, description="文号前缀，如：测试公司(2025)第")
    number_part: Optional[str] = Field(default=None, description="文号尾部，如：207号")

# =========================
# 依赖：获取当前用户
# =========================
async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)],
                           db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise credentials_error
        uid = payload.get("uid")
        if uid is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = await db.get(User, uid)
    if not user:
        raise credentials_error
    return user

# 如果你未来有“角色/权限”，可以这样包装一个依赖
def require_auth():
    async def _inner(user: Annotated[User, Depends(get_current_user)]) -> User:
        return user
    return _inner

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def merge_template_and_content(draft: dict, tmpl: dict) -> dict:
    """
    将 generate_draft_json 产出的草稿(draft) 与 模板 JSON(tmpl) 合并：
    - 保留 draft.docInfo 与 draft.contentBlocks
    - 如果模板里有 pageSetup / styles，则覆盖或补充到最终结果
    """
    final_obj = dict(draft)  # 浅拷贝
    if "pageSetup" in tmpl:
        final_obj["pageSetup"] = tmpl["pageSetup"]
    if "styles" in tmpl:
        final_obj["styles"] = tmpl["styles"]
    return final_obj

def normalize_kb_ids(kb_ids: Union[str, List[Union[str, int]]]) -> List[str]:
    """
    将 knowledge_base_id 归一化为字符串 ID 列表。
    支持：["1","2"] / [1,2] / "1,2" / "" / "  " 等。
    """
    if isinstance(kb_ids, list):
        out = []
        for x in kb_ids:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    # 字符串
    s = (kb_ids or "").strip()
    if not s:
        return []
    # 逗号分隔
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def safe_slug(s: str, default: str = "document") -> str:
    """将任意字符串转成文件名安全的 slug。"""
    s = s.strip()
    if not s:
        return default
    # 只保留 中英文、数字、下划线、横杠；空白改为下划线
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_\-]", "", s)
    return s or default

def build_export_paths(job_id: str, project_name: str) -> dict:
    """
    返回该 job 的 docx 与 pdf 输出路径。
    文件名示例：{job_id}_{project}.docx / .pdf
    """
    proj = safe_slug(project_name, "document")
    base = f"{job_id}_{proj}"
    docx_path = EXPORT_ROOT / f"{base}.docx"
    pdf_path  = EXPORT_ROOT / f"{base}.pdf"
    return {"docx": docx_path, "pdf": pdf_path}

def sse_event(event: str, data: dict) -> bytes:
    # SSE: event: xxx\ndata: <json>\n\n
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

def try_extract_json(text: str) -> dict | None:
    """
    从不完整的文本缓冲中尽力抽取最外层 JSON 对象。
    找到第一个 '{' 与最后一个 '}'，尝试 json.loads。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i

def _find_key_start(s: str, key: str, from_idx: int = 0) -> int:
    # 粗糙但高效：匹配 "docInfo" 或 "contentBlocks"
    pat = re.compile(r'"' + re.escape(key) + r'"\s*:')
    m = pat.search(s, from_idx)
    return m.start() if m else -1


def _ndjson_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

async def stream_and_store_reviews(
    db: AsyncSession,
    job: Job,
    test_content: str,
):
    """
    并行调用五个流式 text：
    - 解析每个 text 产出的 JSON 对象（逐条建议）
    - 立刻写入 reviews 表
    - 同时以 NDJSON 行（event=data）流式返回给前端
    """
    # text 标签到审查类型名（存入 reviews.type）
    agents = [
        (review_fiscal,           "财政审查"),
        (review_style,            "风格审查"),
        (review_policy,           "政策审查"),
        (review_document_number,  "文号审查"),
        (review_completeness,     "完整性审查"),
    ]

    queue: asyncio.Queue = asyncio.Queue()

    async def producer(review_func, review_type: str):
        buffer = ""
        in_array = False
        try:
            async for chunk in review_func(test_content):
                buffer += chunk
                # 找到 JSON 数组开头
                if not in_array:
                    if "[" in buffer:
                        in_array = True
                        buffer = buffer[buffer.find("["):]
                    else:
                        continue
                # 尝试从 buffer 中切出一个个对象
                while "{" in buffer and "}" in buffer:
                    s = buffer.find("{")
                    e = buffer.find("}")
                    if s < e:
                        obj_str = buffer[s:e+1]
                        try:
                            obj = json.loads(obj_str)
                            await queue.put((review_type, obj))
                            buffer = buffer[e+1:]
                        except json.JSONDecodeError:
                            # 对象不完整，再等下一个 chunk
                            break
                    else:
                        # 错位，截断到下一个 '{'
                        buffer = buffer[s:]
                        break
        finally:
            await queue.put((review_type, None))  # 结束标记

    # 启动所有生产者
    prods = [asyncio.create_task(producer(fn, typ)) for fn, typ in agents]

    finished = {typ: False for _, typ in agents}
    total_finished = 0
    total_agents = len(agents)

    # 消费：直到所有生产者结束
    while total_finished < total_agents:
        review_type, item = await queue.get()
        if item is None:
            if not finished[review_type]:
                finished[review_type] = True
                total_finished += 1
                # 通知前端某类型结束
                yield _ndjson_line({"phase": "agent_done", "type": review_type})
        else:
            # 解析每条建议并入库 + 推给前端
            original = str(item.get("original", "")).strip()
            modified = item.get("modified")
            reason = item.get("reason")
            # 生成 reviewsid
            rid_src = f"{job.job_id}|{review_type}|{original}|{modified}|{reason}|{time.time()}"
            reviewsid = sha256_hex(rid_src)[:64]

            # 入库
            db.add(Review(
                reviewsid=reviewsid,
                job_id=job.job_id,
                type=review_type,
                original=original or "",
                modified=modified,
                reason=reason,
            ))
            # 尽量批量提交：这里按条提交也可（稳妥起见可每 N 条 flush/commit 一次）
            await db.commit()

            # 推流：每条建议一行
            yield _ndjson_line({
                "phase": "suggestion",
                "type": review_type,
                "reviewsid": reviewsid,
                "data": item
            })

    # 等待所有生产者结束
    await asyncio.gather(*prods, return_exceptions=True)
    # 最后一行：全部完成
    yield _ndjson_line({"phase": "done", "job_id": job.job_id})

async def ensure_user_docnumber(user_id: int) -> None:
    """若用户尚未初始化文号，则按默认前缀与起始号创建。"""
    async with DocNumSession() as ddb:
        rec = await get_document_number_by_user_id(ddb, user_id)
        if not rec:
            await create_user_document_number(ddb, user_id, DOCNUMBER_DEFAULT_PREFIX, DOCNUMBER_DEFAULT_START)

async def get_user_wenhao_text(user_id: int) -> str:
    """获取当前用户的完整文号文本（不自增）。若不存在则初始化并返回起始号。"""
    async with DocNumSession() as ddb:
        rec = await get_document_number_by_user_id(ddb, user_id)
        if rec:
            return f"{rec.prefix}{rec.number_part}"
    # 若尚未初始化则立即初始化并返回默认
    await ensure_user_docnumber(user_id)
    return f"{DOCNUMBER_DEFAULT_PREFIX}{DOCNUMBER_DEFAULT_START}"

async def increment_user_wenhao(user_id: int) -> str:
    """文号 +1，并返回自增后的完整文号文本。"""
    async with DocNumSession() as ddb:
        full = await get_and_increment_document_number(ddb, user_id)
        # full 可能已经是完整文本；若返回的是结构，请按实际返回拼接
        return str(full)


@dataclass
class ArrayStreamState:
    found_key: bool = False
    array_start_idx: int = -1
    scan_idx: int = 0
    in_str: bool = False
    esc: bool = False
    depth_obj: int = 0      # 对象/数组内部配平深度（数组内的单个元素）
    item_buf: list[str] = None
    items_emitted: int = 0
    array_done: bool = False

def _extract_json_substring(buf: str) -> str | None:
    """仿照你的测试程序做法：取第一个 '{' 到最后一个 '}' 之间的子串。"""
    start_idx = buf.find('{')
    end_idx = buf.rfind('}')
    if start_idx != -1 and end_idx > start_idx:
        return buf[start_idx:end_idx + 1]
    return None

def extract_text_blocks_from_document(doc: dict) -> list[str]:
    """
    从 document JSON 中提取所有文本（contentBlocks[].content[].text）
    返回每个段落的纯文本列表
    """
    out = []
    try:
        blocks = doc.get("contentBlocks", [])
        for blk in blocks:
            for run in blk.get("content", []):
                txt = str(run.get("text", "")).strip()
                if txt:
                    out.append(txt)
    except Exception:
        pass
    return out

def chunk_texts(texts: list[str], max_len: int = 500) -> list[str]:
    """
    将多段文本合并后按 max_len 粗略切块，尽量在中文标点/空格处断开
    """
    big = "。".join(texts)  # 合并；简单用句号衔接
    big = re.sub(r"\s+", " ", big).strip()
    if not big:
        return []
    chunks = []
    start = 0
    n = len(big)
    while start < n:
        end = min(n, start + max_len)
        # 向右扩展到最近的句号/分号/顿号/逗号/空格
        if end < n:
            m = re.search(r"[。；；、，,]\s*", big[end-50:end+50])  # 窗口
            if m:
                end = end - 50 + m.end()
        chunk = big[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def uniq_rules(rules: list[str], limit: int = 12) -> list[str]:
    """简单去重并限量"""
    seen = set()
    out = []
    for r in rules:
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= limit:
            break
    return out

async def query_kbs_and_collect_ctx(
    db: AsyncSession,
    owner_id: int,
    kb_ids: list[str],
    query_text: str,
    per_kb_top_k: int = 4,
    recall_k: int = 20,
) -> list[dict]:
    """
    1) 用 knowledge_bases.id 批量查库，拿 knowledge_bases_id / name
    2) 组成本地 FAISS 索引路径：
         document_agent\\knowledge_base_package\\user_knowledge_bases\\{owner_id}\\{knowledge_bases_id}\\faiss_index
       （注意：Windows 默认路径分隔符；用 Path 统一拼接更安全）
    3) 对每个 KB 执行 query_with_score(...)，汇总为一个 list[dict]
    4) 返回结构尽量自描述，便于上游模板/提示词直接使用
    """
    if not kb_ids:
        return []

    # 将传入的 "1" / 1 统一成 int 并过滤非法值
    kb_int_ids: list[int] = []
    for x in kb_ids:
        try:
            kb_int_ids.append(int(str(x).strip()))
        except Exception:
            continue
    if not kb_int_ids:
        return []

    # 只查属于当前用户的 KB，避免越权
    rows = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id.in_(kb_int_ids),
            KnowledgeBase.owner == owner_id,
        )
    )
    kb_rows = rows.scalars().all()
    if not kb_rows:
        return []

    merged: list[dict] = []
    for kb_row in kb_rows:
        # 统一用 Path 拼，避免分隔符问题（Windows/Linux 皆可）
        kb_path = Path("document_agent") / "knowledge_base_package" / "user_knowledge_bases" / str(owner_id) / kb_row.knowledge_bases_id / "faiss_index"
        print(kb_path)
        kb = KBIndex(db_path=str(kb_path))
        try:
            hits = kb.query_with_score(query_text=query_text, top_k=per_kb_top_k, recall_k=recall_k)
        except Exception as e:
            # 某个 KB 查询异常时，不阻断整体流程
            merged.append({
                "kb_id": kb_row.id,
                "knowledge_bases_id": kb_row.knowledge_bases_id,
                "kb_name": kb_row.name,
                "error": f"query failed: {e}",
            })
            continue

        for h in hits:
            doc = h.get("document")
            score = h.get("score")
            # 兼容 LangChain Document
            source = getattr(doc, "metadata", {}).get("source", "未知来源") if hasattr(doc, "metadata") else (doc.get("metadata", {}).get("source", "未知来源") if isinstance(doc, dict) else "未知来源")
            content = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else (doc.get("page_content", "") if isinstance(doc, dict) else "")
            snippet = (content or "").replace("\n", " ")[:500]
            if snippet == "init":
                continue

            merged.append({
                "kb_id": kb_row.id,
                "knowledge_bases_id": kb_row.knowledge_bases_id,
                "kb_name": kb_row.name,
                "score": float(score) if score is not None else None,
                "source": source,
                "text": snippet,
            })

    # 如果你们的分数是“分数越小越好”或“越大越好”，可以在这里排序/裁剪
    # 例如：从所有 KB 命中的合并结果中，取前 N 条作为最终 retrieved_ctx
    # 下面示例是“分数越大越相关”的做法（若相反，改成 reverse=False）
    try:
        merged.sort(key=lambda x: (x.get("score") is None, x.get("score")), reverse=True)
    except Exception:
        pass
    return merged

def build_review_payload(document_obj: dict, rules: list[str]) -> str:
    """
    构造传给 review agents 的 test_content（字符串，JSON 序列化）
    """
    docInfo = document_obj.get("docInfo", {})
    contentBlocks = document_obj.get("contentBlocks", [])
    payload = {
        "text_to_review": {
            "docInfo": {
                "docType": docInfo.get("docType", ""),
                "title": docInfo.get("title", ""),
                "docNumber": docInfo.get("docNumber", ""),
                "urgency": docInfo.get("urgency", ""),
                "security": docInfo.get("security", ""),
            },
            "contentBlocks": contentBlocks
        },
        "retrieved_rules": rules
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def _split_full_wenhao(full: str) -> tuple[str, str]:
    """
    将 '测试公司(2025)第207号' 切成 ('测试公司(2025)第', '207号')
    规则：尽量匹配末尾的“数字+号”
    """
    full = (full or "").strip()
    m = re.search(r"(\d+号)\s*$", full)
    if not m:
        raise ValueError("无法从 full 中解析出 number_part（需形如：……207号）")
    number_part = m.group(1)
    prefix = full[: m.start(1)]
    if not prefix:
        raise ValueError("解析后的 prefix 为空")
    return prefix, number_part

def _safe_extract_content(file_path: Path) -> str:
    """
    调用用户提供的 extract_content(path) -> str
    若函数不存在或报错，返回可读的占位信息，不阻断主流程。
    """
    # 1) 在全局命名空间里找（你可能已 from xxx import extract_content 过）
    extract_content = globals().get("extract_content")
    # 2) 尝试延迟导入（如果你有固定模块名，可在这里改成 from your_package import extract_content）
    if extract_content is None:
        try:
            extract_content = _ec
        except Exception:
            extract_content = None

    if extract_content is None:
        return "[未找到 extract_content(path) 函数，无法提取文件文本]"

    try:
        text = extract_content(str(file_path))  # 你要求的调用方式
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if len(text) > MAX_REF_CHARS:
            text = text[:MAX_REF_CHARS] + "……【内容过长，已截断】"
        return text or "[文件内容为空]"
    except Exception as e:
        return f"[提取失败: {e}]"


def build_references_hint_and_append_payload_info(
    information: str,
    references: list | None,
) -> tuple[str, str]:
    """
    使用 references 里的 name 作为展示名，file_id 仅用于定位文件。
    """
    references = references or []
    if not isinstance(references, list) or not references:
        return "", information

    lines = ["用户提供了以下文件："]
    any_ok = False

    for item in references:
        try:
            name = str(item.get("name", "")).strip() or "未命名文件"
            file_id = str(item.get("file_id", "")).strip()
        except Exception:
            continue
        if not file_id:
            continue

        path = (MEDIA_ROOT / file_id).resolve()
        ext = Path(file_id).suffix.lower()
        # 展示名 = references.name + 源文件的后缀（若 name 已经带后缀就不重复）
        display = f"{name}{ext}" if ext and not name.endswith(ext) else name

        if not path.exists():
            lines.append(f"一份{display}文件内容如下：\n[未找到文件：{path}]\n")
            any_ok = True
            continue

        extracted = _safe_extract_content(path)
        lines.append(f"一份{display}文件内容如下：\n{extracted}\n")
        any_ok = True

    if not any_ok:
        return "", information

    hint_text = "\n".join(lines).strip()
    combined = (information or "").strip()
    if combined:
        combined = combined + "\n\n" + hint_text
    else:
        combined = hint_text
    return hint_text, combined




# 你的真实实现可替换这个占位版本

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Auth Demo", version="1.0.0")

@app.on_event("startup")
async def on_startup():
    await init_db()
    await docnumber_create_tables()

# ========== Auth 路由 ==========
@app.post("/auth/register", response_model=PublicUser, summary="注册")
async def register(payload: RegisterIn, db: Annotated[AsyncSession, Depends(get_db)]):
    # 检查 email/username 是否已存在
    exists_email = await db.execute(select(User).where(User.email == payload.email))
    if exists_email.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email 已被注册")

    exists_name = await db.execute(select(User).where(User.username == payload.username))
    if exists_name.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="用户名已被占用")

    user = User(
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    try:
        await ensure_user_docnumber(user.id)
    except Exception:
        # 文号初始化失败不阻断注册
        pass

    return user

@app.post("/auth/login", response_model=TokenPair, summary="登录（JSON）")
async def login_json(payload: LoginIn, db: Annotated[AsyncSession, Depends(get_db)]):
    q = await db.execute(
        select(User).where((User.email == payload.username) | (User.username == payload.username))
    )
    user = q.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="用户名或密码错误")

    access = create_access_token(user.id, user.username)
    refresh = create_refresh_token(user.id, user.username)
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

@app.post("/auth/refresh", response_model=TokenPair, summary="用 refresh 刷新 access")
async def refresh_token(refresh_token: Annotated[str, Body(embed=True, description="刷新令牌")],
                        db: Annotated[AsyncSession, Depends(get_db)]):
    # 简单黑名单检查
    if refresh_token in revoked_refresh_tokens:
        raise HTTPException(status_code=401, detail="Refresh token 已失效")

    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Token 类型错误")
        uid = payload.get("uid")
        sub = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="无效或过期的 refresh token")

    user = await db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")

    access = create_access_token(user.id, user.username)
    # 可选择是否滚动刷新 refresh（这里演示：每次也给新的 refresh）
    new_refresh = create_refresh_token(user.id, user.username)
    return TokenPair(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

@app.post("/auth/logout", summary="登出（示例：使 refresh 失效）")
async def logout(refresh_token: Annotated[str, Body(embed=True)]):
    # 仅演示：将 refresh 加入黑名单（进程内）
    revoked_refresh_tokens.add(refresh_token)
    return {"ok": True}

# ========== 个人中心 ==========
@app.get("/me", response_model=PublicUserWithWenhao, summary="获取个人信息（含当前文号）")
async def get_me(
    current_user: Annotated[User, Depends(require_auth())],
):
    # 确保已初始化文号，并获取当前文号（不自增）
    try:
        await ensure_user_docnumber(current_user.id)
        wenhao_text = await get_user_wenhao_text(current_user.id)
    except Exception:
        wenhao_text = ""

    return PublicUserWithWenhao(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at,
        wenhao=wenhao_text,
    )


@app.patch("/me", response_model=PublicUserWithWenhao, summary="更新个人资料（邮箱/用户名）")
async def update_me(
    patch: MePatch,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    if patch.email and patch.email != current_user.email:
        q = await db.execute(select(User).where(User.email == patch.email))
        if q.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email 已被占用")
        current_user.email = patch.email

    if patch.username and patch.username != current_user.username:
        q = await db.execute(select(User).where(User.username == patch.username))
        if q.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="用户名已被占用")
        current_user.username = patch.username

    await db.commit()
    await db.refresh(current_user)

    # 返回时附带当前文号（不自增）
    try:
        await ensure_user_docnumber(current_user.id)
        wenhao_text = await get_user_wenhao_text(current_user.id)
    except Exception:
        wenhao_text = ""

    return PublicUserWithWenhao(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at,
        wenhao=wenhao_text,
    )


@app.post("/me/change-password", summary="修改密码")
async def change_password(payload: ChangePasswordIn,
                          db: Annotated[AsyncSession, Depends(get_db)],
                          current_user: Annotated[User, Depends(require_auth())]):
    if not verify_password(payload.old_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="原密码不正确")
    current_user.password_hash = hash_password(payload.new_password)
    await db.commit()
    return {"ok": True}

# ========== 示例：受保护业务接口 ==========
@app.get("/secure/ping", summary="示例：带 Token 的受保护接口")
async def secure_ping(current_user: Annotated[User, Depends(require_auth())]):
    return {"pong": True, "user": current_user.username, "time": int(time.time())}



# ========= 在 FastAPI 路由处追加 =========
@app.post("/api/v1/jobs/generate", summary="生成文档（NDJSON 流：先模板后增量）- 需JWT")
async def generate_job_ndjson(
    payload: GenerateIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    # 1) 读取模板
    try:
        tmpl_id_int = int(payload.template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="template_id 必须是整数 ID（字符串形式）")

    tmpl = await db.get(Template, tmpl_id_int)
    if not tmpl:
        raise HTTPException(status_code=404, detail="模板不存在")

    tmpl_path = Path(tmpl.content).expanduser().resolve()
    if not tmpl_path.exists():
        raise HTTPException(status_code=500, detail=f"模板文件不存在: {tmpl_path}")

    try:
        with tmpl_path.open("r", encoding="utf-8") as f:
            tmpl_json = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模板文件读取/解析失败: {e}")

    page_setup = dict(tmpl_json.get("pageSetup", {}))
    tmpl_styles = tmpl_json.get("styles", {})

    # ★ 注入文号（不自增）
    wenhao_text = await get_user_wenhao_text(current_user.id)
    page_setup["wenhao"] = {
        "style": "wenhao",
        "content": wenhao_text
    }

    # 同步回模板对象，确保后续 merge 和首行 template 推流都带上文号
    tmpl_json["pageSetup"] = page_setup

    # 2) 知识库检索
    kb_ids = normalize_kb_ids(payload.knowledge_base_id)
    retrieved_ctx = await query_kbs_and_collect_ctx(
        db = db,
        owner_id = current_user.id,
        kb_ids = kb_ids,
        query_text = payload.information,
        per_kb_top_k = 4,
        recall_k = 20,
    )

    refs_hint_text, combined_information = build_references_hint_and_append_payload_info(
        information=payload.information,
        references=payload.references,
    )

    # 3) 组织提示文本
    generation_data = {
        "user_input": {
            "project_name": payload.project_name,
            "core_requirements": combined_information,
        },
        "template_styles": tmpl_styles,
        "retrieved_context": retrieved_ctx,
    }
    prompt_text = json.dumps(generation_data, ensure_ascii=False)
    print(combined_information)
    # 4) 生成 job_id（用于最终入库）
    now_ts = datetime.utcnow().isoformat()
    job_id = sha256_hex(f"{current_user.id}:{payload.project_name}:{payload.template_id}:{now_ts}")

    # 5) 定义 NDJSON 流生成器
    async def gen_ndjson():
        # 5.1 先发模板（第一行）
        yield _ndjson_line({
            "phase": "template",
            "data": {"pageSetup": page_setup, "styles": tmpl_styles}
        })

        buffer = ""
        # 5.2 转发模型流（每个 chunk 一行 JSON）
        try:
            async for chunk in generate_draft_stream(prompt_text):
                # 原样流出 chunk（作为 JSON 的字符串值）
                yield _ndjson_line({
                    "phase": "draft_chunk",
                    "data": chunk
                })
                buffer += chunk
        except Exception as e:
            # 报错也以 JSON 行返回
            yield _ndjson_line({"phase": "error", "message": f"生成失败: {str(e)}"})

        # 5.3 收尾：尝试从缓冲里提取完整 JSON，合并模板，并入库
        try:
            json_str = _extract_json_substring(buffer)
            if json_str:
                parsed = json.loads(json_str)
                final_json = merge_template_and_content(draft=parsed, tmpl=tmpl_json)

                raw_content = {
                    "project_name": payload.project_name,
                    "information": payload.information,
                    "knowledge_base_id": kb_ids,
                    "template_id": payload.template_id,
                    "references": payload.references,
                }

                job = Job(
                    job_id=job_id,
                    project_name=payload.project_name,
                    user_id=current_user.id,
                    template_id=tmpl_id_int,
                    knowledge_id=",".join(kb_ids) if kb_ids else None,  # VARCHAR(512)
                    content=payload.information,
                    file=None,
                    document=json.dumps(final_json, ensure_ascii=False),
                )
                db.add(job)
                await db.commit()

                yield _ndjson_line({"phase": "finalized", "job_id": job_id})
            else:
                yield _ndjson_line({"phase": "warn", "message": "未能从生成流中提取完整 JSON，未入库"})
        except Exception as e:
            yield _ndjson_line({"phase": "error", "message": f"入库失败: {str(e)}"})

    return StreamingResponse(
        gen_ndjson(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反代禁用缓冲（如 Nginx）
        },
    )


@app.post("/api/v1/review-draft", summary="基于 job 文档的并行流式审查（落库 reviews）- 需JWT")
async def review_draft_api(
    payload: ReviewDraftIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    # 1) 查 job
    job = await db.get(Job, payload.jobid)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")

    # （可选）只允许本人
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问该任务")

    if not job.document:
        raise HTTPException(status_code=400, detail="该 job 的 document 为空")

    # 2) 解析 document
    try:
        doc_obj = json.loads(job.document)
        if not isinstance(doc_obj, dict):
            raise ValueError("document 必须为 JSON 对象")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"document JSON 解析失败: {e}")

    # 3) 提取文本并切块
    texts = extract_text_blocks_from_document(doc_obj)
    if not texts:
        raise HTTPException(status_code=400, detail="未能从 document 中提取任何文本")

    chunks = chunk_texts(texts, max_len=500)

    # 4) 归一化 KB IDs
    kb_ids = normalize_kb_ids(payload.knowledge_base_id)

    # 5) 针对每个 chunk 检索规则，汇总去重
    all_rules = []
    try:
        for c in chunks:
            rules = await query_kbs_and_collect_ctx(
                db=db,
                owner_id=current_user.id,
                kb_ids=kb_ids,
                query_text=c,
                per_kb_top_k=4,
                recall_k=20,
            )
            if rules:
                all_rules.extend(rules)
            print(rules)
    except Exception as e:
        # 检索异常时，继续用已有内容（也可以直接报错）
        all_rules.append(f"检索规则异常：{e}")

    rules_final = uniq_rules(all_rules, limit=20)

    # 6) 构造传给审查 agents 的 payload（字符串）
    review_input = build_review_payload(doc_obj, rules_final)

    # 7) 定义 NDJSON 生成器（边审查边落库边推流）
    async def gen():
        # 先把基础信息发一行
        yield _ndjson_line({
            "phase": "start",
            "job_id": job.job_id,
            "project_name": job.project_name,
            "kb_ids": kb_ids,
            "chunks": len(chunks),
            "rules_count": len(rules_final)
        })

        # 并行审查 + 落库 + 推流
        async for line in stream_and_store_reviews(db, job, review_input):
            yield line

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@app.post("/api/v1/jobs/{job_id}/download", summary="下载已生成的文档（docx/pdf）- 需JWT")
async def download_job_document(
    job_id: str,
    payload: JobDownloadIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    # 1) 查询 job
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")

    # （可选安全校验）当前用户只能下载自己的 job
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问该任务")

    if not job.document:
        raise HTTPException(status_code=400, detail="该任务不存在可导出的 document JSON")

    # 2) 解析 document JSON
    try:
        json_data = json.loads(job.document)
        if not isinstance(json_data, dict):
            raise ValueError("document 必须为 JSON 对象")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"document JSON 解析失败: {e}")

    # 3) 生成输出路径
    paths = build_export_paths(job_id=job.job_id, project_name=job.project_name)
    out_docx = paths["docx"]
    out_pdf = paths["pdf"]

    # 4) 渲染与转换
    file_to_return: Path
    media_type: str

    try:
        # 先确保目录存在
        out_docx.parent.mkdir(parents=True, exist_ok=True)

        if payload.type == "docx":
            # 直接渲染 docx
            # 注意：这里按你的函数签名调用 render_docx(json_data, output_path)
            render_docx(json_data, str(out_docx))
            file_to_return = out_docx
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            # 先渲染 docx，再转 pdf
            render_docx(json_data, str(out_docx))
            docx_to_pdf(str(out_docx), str(out_pdf))
            file_to_return = out_pdf
            media_type = "application/pdf"

    except HTTPException:
        # 透传上面的业务异常
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档生成失败: {e}")

    # 5) 校验文件存在并返回
    if not file_to_return.exists():
        raise HTTPException(status_code=500, detail="目标文件生成失败或未找到")

    # 指定下载文件名（Content-Disposition）
    download_name = file_to_return.name
    return FileResponse(
        path=str(file_to_return),
        media_type=media_type,
        filename=download_name
    )

@app.get("/api/v1/jobs/", response_model=List[JobOverview], summary="获取当前用户的任务概览列表 - 需JWT")
async def get_my_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    """
    通过JWT验证用户身份，检索并返回该用户创建的所有任务的概览列表。
    结果按创建时间降序排列。
    """
    q = select(Job).where(Job.user_id == current_user.id).order_by(Job.create_time.desc())
    result = await db.execute(q)
    jobs = result.scalars().all()
    return jobs


@app.get("/api/v1/jobs/{job_id}/detail", response_model=JobDetail, summary="获取单个任务的详细信息 - 需JWT")
async def get_job_detail(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    """
    通过 job_id 检索单个任务的详细信息。
    同时会验证该任务是否属于当前登录用户，防止数据越权访问。
    """
    # 通过主键高效查找
    job = await db.get(Job, job_id)

    # 关键安全校验：
    # 1. 检查任务是否存在
    # 2. 检查任务是否属于当前用户
    if not job or job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or access denied"
        )
    return job


@app.get("/api/v1/jobs/{job_id}/reviews", response_model=List[ReviewOut], summary="获取指定任务的审查结果 - 需JWT")
async def get_job_reviews(
        job_id: str,
        db: Annotated[AsyncSession, Depends(get_db)],
        current_user: Annotated[User, Depends(require_auth())],
):
    """
    通过 job_id 获取该任务的所有审查结果。
    """
    # 1. 验证 job_id 是否存在且属于当前用户
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this job")

    # 2. 查询 reviews 表中所有与该 job_id 关联的记录
    q = select(Review).where(Review.job_id == job_id)
    result = await db.execute(q)
    reviews = result.scalars().all()

    # 3. 返回结果列表
    return reviews


@app.get("/api/v1/knowledgebases/", response_model=List[KnowledgeBaseOut],
         summary="获取当前用户的所有知识库信息 - 需JWT")
async def get_my_knowledgebases(
        db: Annotated[AsyncSession, Depends(get_db)],
        current_user: Annotated[User, Depends(require_auth())],
):
    """
    通过JWT验证用户身份，检索并返回该用户创建的所有知识库的详细信息。
    """
    # 使用 select 语句查询 knowledge_bases 表中 owner 等于当前用户 ID 的所有记录
    q = select(KnowledgeBase).where(KnowledgeBase.owner == current_user.id)

    result = await db.execute(q)
    knowledge_bases = result.scalars().all()

    # 返回 KnowledgeBaseOut 列表，Pydantic 会自动从 SQLAlchemy 对象中转换数据
    return knowledge_bases


@app.get("/api/v1/templates/", response_model=List[TemplateOut], summary="获取当前用户的所有模板信息 - 需JWT")
async def get_my_templates(
        db: Annotated[AsyncSession, Depends(get_db)],
        current_user: Annotated[User, Depends(require_auth())],
):
    """
    通过JWT验证用户身份，检索并返回该用户创建的所有模板的详细信息。
    """
    # 使用 select 语句查询 templates 表中 owner 等于当前用户 ID 的所有记录
    q = select(Template).where(Template.owner == current_user.id)

    result = await db.execute(q)
    templates = result.scalars().all()

    # 返回 TemplateOut 列表，Pydantic 会自动从 SQLAlchemy 对象中转换数据
    return templates

@app.post("/api/v1/files/uploads/", summary="上传文件到媒体库（按哈希命名）- 需JWT")
async def upload_file_api(
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(require_auth())] = None,
):
    """
    接收 multipart/form-data 的文件字段 `file`，
    将文件以 SHA-256 哈希值命名存入 MEDIA_ROOT，保留扩展名（若有）。
    返回文件哈希、原始文件名、内容类型。
    """
    # 读取并增量计算哈希，同时写入临时文件，避免大文件占内存
    hasher = hashlib.sha256()
    suffix = Path(file.filename or "").suffix.lower() if file.filename else ""

    # 如果没有扩展名，尝试基于 MIME 猜测
    if not suffix and file.content_type:
        guessed = mimetypes.guess_extension(file.content_type) or ""
        # 某些情况会猜到 ".jpe" 之类的古怪后缀，这里原样保留；需要可再做映射
        suffix = guessed.lower()

    # 将上传内容流式写入临时文件并计算哈希
    with tempfile.NamedTemporaryFile(dir=str(MEDIA_ROOT), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB/块
            if not chunk:
                break
            hasher.update(chunk)
            tmp.write(chunk)

    file_hash = hasher.hexdigest()
    dest_path = MEDIA_ROOT / f"{file_hash}{suffix}"

    try:
        if dest_path.exists():
            # 已存在同哈希文件：删除临时文件，复用现有文件
            tmp_path.unlink(missing_ok=True)
        else:
            # 原子替换到目标文件名
            tmp_path.replace(dest_path)
    finally:
        # 防御性清理（若 replace 失败等）
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "file_hash": file_hash,
        "file_name": file.filename,
        "file_type": suffix or "",
    }

@app.get("/api/v1/knowledgebases/{knowledgebase_id}/files",
         response_model=List[KbFileOut],
         summary="获取指定知识库的文件列表 - 需JWT")
async def get_kb_files(
    knowledgebase_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    # 1) 校验知识库存在 & 归属当前用户
    kb = await db.get(KnowledgeBase, knowledgebase_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.owner != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this knowledge base")

    # 2) 查询 kb_files
    result = await db.execute(select(KbFile).where(KbFile.knowledge_id == knowledgebase_id))
    files = result.scalars().all()
    return files

@app.post("/api/v1/knowledgebases/create",
          response_model=KnowledgeBaseOut,
          summary="创建知识库（生成64位哈希ID并初始化索引目录）- 需JWT")
async def create_knowledge_base(
    payload: KBCreateIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    """
    1) 通过 JWT 获取 owner（当前用户）
    2) 生成 64 位哈希作为 knowledge_bases_id
    3) 入库 knowledge_bases
    4) 创建目录 user_knowledge_bases/{user_id}/{knowledge_bases_id}/faiss_index
    5) 使用索引类 KBIndex 初始化索引（如类构造会自动建库/加载）
    """
    owner_id = current_user.id

    # 生成 64 位哈希（避免碰撞：带随机盐）
    kb_hash = sha256_hex(f"{owner_id}:{payload.name}:{time.time()}:{secrets.token_hex(16)}")

    # 3) 入库
    kb_row = KnowledgeBase(
        knowledge_bases_id=kb_hash,
        owner=owner_id,
        name=payload.name.strip(),
    )
    db.add(kb_row)
    await db.commit()
    await db.refresh(kb_row)

    # 4) 目录：user_knowledge_bases/{user_id}/{knowledge_bases_id}/faiss_index
    kb_index_dir = KB_BASE_ROOT / str(owner_id) / kb_row.knowledge_bases_id / "faiss_index"
    kb_index_dir.mkdir(parents=True, exist_ok=True)

    # 5) 初始化索引（若你的索引类需要其他初始化方法，可在这里调用）
    try:
        _kb = KBIndex(db_path=str(kb_index_dir))
        # 如果需要显式 build/init，可在这里执行：例如 _kb.build() / _kb.init()
    except Exception as e:
        # 如果索引初始化失败，按需决定是否回滚数据库。这里仅记录警告不回滚。
        # 也可选择 raise HTTPException(500, "索引初始化失败") 并删除记录/目录。
        pass

    return kb_row

@app.delete("/api/v1/knowledgebases/{knowledgebase_id}",
            summary="删除知识库（级联删除 DB 记录与本地索引目录）- 需JWT")
async def delete_knowledge_base(
    knowledgebase_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    """
    1) 仅允许删除当前用户的 KB
    2) 删除 DB 记录（kb_files 通过外键级联清理）
    3) 删除本地目录 user_knowledge_bases/{owner}/{knowledge_bases_id}
    """
    kb = await db.get(KnowledgeBase, knowledgebase_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.owner != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this knowledge base")

    # 先缓存目录信息，避免提交后对象状态变化
    kb_owner = kb.owner
    kb_hash = kb.knowledge_bases_id
    kb_dir = KB_BASE_ROOT / str(kb_owner) / kb_hash

    # 先删 DB，确保外键约束/级联行为正确
    try:
        await db.delete(kb)
        await db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库删除失败: {e}")

    # 再删文件系统目录（即使失败，也不回滚 DB；返回告知前端）
    fs_removed = False
    fs_error = None
    try:
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
            fs_removed = True
    except Exception as e:
        fs_error = str(e)

    resp = {
        "ok": True,
        "id": knowledgebase_id,
        "knowledge_bases_id": kb_hash,
        "fs_removed": fs_removed,
    }
    if fs_error:
        resp["fs_error"] = fs_error
    return resp

@app.post("/api/v1/knowledgebases/upload", summary="向知识库添加文件并建立索引 - 需JWT")
async def upload_files_to_knowledgebase(
    payload: KBUploadIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
):
    # 1) 校验知识库存在且归属当前用户
    kb_row = await db.get(KnowledgeBase, payload.id)
    if not kb_row:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb_row.owner != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this knowledge base")

    # 2) 允许的文件类型
    allowed_suffix = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx"}

    # 3) 组装索引目录与 KB 索引实例
    kb_index_dir = KB_BASE_ROOT / str(current_user.id) / kb_row.knowledge_bases_id / "faiss_index"
    kb_index_dir.mkdir(parents=True, exist_ok=True)
    kb = KBIndex(db_path=str(kb_index_dir))

    # 4) 收集有效文件路径
    file_paths: List[tuple[Path, str, str]] = []  # (path, logical_name, file_hash)
    results = []

    for item in payload.files:
        p = (MEDIA_ROOT / item.file_id).resolve()
        suffix = p.suffix.lower()
        file_hash = Path(item.file_id)

        if suffix not in allowed_suffix:
            results.append({"file_id": item.file_id, "ok": False, "reason": f"unsupported suffix {suffix}"})
            continue


        if not p.exists():
            results.append({"file_id": item.file_id, "ok": False, "reason": "file not found in MEDIA_ROOT"})
            continue

        file_paths.append((p, item.name, file_hash))
        results.append({"file_id": item.file_id, "ok": True, "suffix": suffix, "name": item.name, "hash": file_hash})

    if not file_paths:
        # 没有任何可用文件，直接返回
        return {
            "knowledgebase_id": payload.id,
            "knowledge_bases_id": kb_row.knowledge_bases_id,
            "indexed": 0,
            "inserted": 0,
            "details": results,
        }

    # 5) 先删除同名源，避免重复（忽略不存在错误）
    for p, _, _ in file_paths:
        try:
            kb.delete_documents_by_source(p.name)
        except Exception:
            # 不阻断流程
            pass

    # 6) 加载与写入索引
    try:
        docs = KBIndex.load_documents_from_paths([str(p) for p in file_paths])
        kb.add_documents(docs)
        indexed_ok = len(file_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"indexing failed: {e}")

    # 7) 写入 kb_files（若已有同名条目，是否允许重复由你决定；这里简单允许重复）
    inserted = 0
    for p, logical_name, file_hash in file_paths:
        try:
            db.add(KbFile(
                knowledge_id=kb_row.id,
                name=logical_name,
                hash=file_hash,
                type=p.suffix.lower() or ""
            ))
            inserted += 1
        except Exception:
            # 单条失败不影响其他
            pass
    await db.commit()

    return {
        "knowledgebase_id": payload.id,
        "knowledge_bases_id": kb_row.knowledge_bases_id,
        "indexed": indexed_ok,
        "inserted": inserted,
        "details": results,
    }

@app.delete("/api/v1/knowledgebases/{knowledgebase_id}/files/{file_hash}",
            summary="从知识库删除文件（删索引+删记录）- 需JWT")
async def delete_kb_file(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
    knowledgebase_id: int = PathParam(..., description="knowledge_bases.id"),
    file_hash: str = PathParam(..., description="带后缀的哈希文件名，如 734d...57.pdf"),
):
    # 1) 校验 KB
    kb_row = await db.get(KnowledgeBase, knowledgebase_id)
    if not kb_row:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb_row.owner != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this knowledge base")

    # 2) 找到将要删除的 kb_files 记录（通常唯一）
    q = await db.execute(
        select(KbFile).where(
            KbFile.knowledge_id == knowledgebase_id,
            KbFile.hash == file_hash
        )
    )
    rows = q.scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="File record not found in this knowledge base")

    # 3) 组装索引目录并删除向量库中的对应文档
    # 路径仍按“用户ID + knowledge_bases_id”来定位索引目录（与你之前的创建/上传保持一致）
    kb_index_dir = KB_BASE_ROOT / str(kb_row.owner) / kb_row.knowledge_bases_id / "faiss_index"
    kb_index_dir.mkdir(parents=True, exist_ok=True)
    kb = KBIndex(db_path=str(kb_index_dir))

    # 注意：你给的删除方式要求传入“源文件名”（现在等于 file_hash，且“带后缀”）
    # 如：ok = kb.delete_documents_by_source("734d7e...c99657.pdf")
    try:
        ok = kb.delete_documents_by_source(file_hash)
    except Exception as e:
        # 索引层失败不阻断 DB 记录删除，但会返回提示
        ok = False
        idx_error = str(e)
    else:
        idx_error = None

    # 4) 删除 kb_files 记录
    deleted = 0
    for r in rows:
        try:
            await db.delete(r)
            deleted += 1
        except Exception:
            pass
    await db.commit()

    resp = {
        "ok": True,
        "knowledgebase_id": knowledgebase_id,
        "knowledge_bases_id": kb_row.knowledge_bases_id,
        "deleted_records": deleted,
        "index_deleted": bool(ok),
    }
    if idx_error:
        resp["index_error"] = idx_error
    return resp

@app.post("/api/v1/jobs/{job_id}/save", summary="保存/更新任务的 document 字段 - 需JWT")
async def save_job_document(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
    job_id: str = PathParam(..., description="jobs.job_id"),
    payload: JobSaveIn = Body(...),
):
    # 1) 查 job
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 2) 仅允许本人修改
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this job")

    # 3) 更新 document 字段
    job.document = payload.content
    await db.commit()
    # 如需返回最新对象，可 refresh
    await db.refresh(job)

    try:
        new_wenhao = await increment_user_wenhao(current_user.id)
    except Exception:
        new_wenhao = None  # 不阻断主流程

    return {
        "ok": True,
        "job_id": job.job_id,
        "project_name": job.project_name,
        "document_size": len(job.document or ""),
        "wenhao_after_increment": new_wenhao,
        "message": "document updated",
    }

@app.post("/api/v1/templates/upload", summary="上传模板文件，解析生成JSON并入库 - 需JWT")
async def upload_template_file(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
    payload: TemplateUploadIn = Body(...),
):
    """
    1) 用 JWT 拿到用户ID
    2) 从 MEDIA_ROOT 拼出源文件路径
    3) generate_template_from_file(target_file) 解析为 dict
    4) 以模板数据内容的 SHA256 生成文件名 <hash>.json 写入 FORMWORK_ROOT（默认 static/formwork）
    5) 在 templates 表中写入记录：name/owner/content（相对路径 static/formwork/<hash>.json）
    """
    # 1) 当前用户
    owner_id = current_user.id

    # 2) 组装源文件路径并校验存在
    target_file = (MEDIA_ROOT / payload.file_id).resolve()
    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"源文件不存在: {target_file}")

    # 3) 解析模板数据
    try:
        template_data = generate_template_from_file(str(target_file))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析模板失败: {e}")

    if not isinstance(template_data, dict):
        raise HTTPException(status_code=500, detail="解析结果不是有效的 JSON 对象（dict）")

    # 4) 以模板数据内容计算哈希，生成文件名
    try:
        json_text_for_hash = json.dumps(template_data, ensure_ascii=False, sort_keys=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模板数据序列化失败: {e}")

    file_hash = sha256_hex(json_text_for_hash)
    json_filename = f"{file_hash}.json"

    # 5) 写入到 FORMWORK_ROOT（物理路径）；content 存相对路径以便后续读取
    out_path = FORMWORK_ROOT / json_filename
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模板JSON写入失败: {e}")

    # 数据库存 content 建议使用相对路径（示例要求：static/formwork/hash.json）
    content_rel = f"static/formwork/{file_hash}.json"

    # 6) 入库 templates
    row = Template(
        name=payload.name.strip(),
        owner=owner_id,
        content=content_rel,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    # 7) 返回结果
    return {
        "ok": True,
        "template": {
            "id": row.id,
            "name": row.name,
            "owner": row.owner,
            "content": row.content,  # 例如：static/formwork/<hash>.json
        }
    }

@app.delete("/api/v1/templates/{template_id}", summary="删除模板（删DB记录+删JSON文件）- 需JWT")
async def delete_template(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auth())],
    template_id: int = PathParam(..., description="模板ID，自增主键"),
):
    # 1. 查模板
    tmpl = await db.get(Template, template_id)
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")

    # 2. 权限校验
    if tmpl.owner != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this template")

    # 3. 缓存路径，避免 commit 后对象失效
    file_path = Path(tmpl.content).resolve()

    # 4. 先删 DB 记录
    try:
        await db.delete(tmpl)
        await db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库删除失败: {e}")

    # 5. 再删 JSON 文件（非强制）
    fs_removed, fs_error = False, None
    try:
        if file_path.exists():
            file_path.unlink()
            fs_removed = True
    except Exception as e:
        fs_error = str(e)

    resp = {
        "ok": True,
        "id": template_id,
        "fs_removed": fs_removed,
    }
    if fs_error:
        resp["fs_error"] = fs_error
    return resp

@app.patch("/api/v1/docnumber", summary="更新当前用户的文号（重置为指定值）- 需JWT")
async def update_user_docnumber(
    current_user: Annotated[User, Depends(require_auth())],
    payload: DocNumberUpdateIn = Body(...),
):
    # 1) 解析输入：支持 full 或 prefix+number_part
    if payload.full:
        try:
            prefix, number_part = _split_full_wenhao(payload.full)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"full 解析失败：{e}")
    else:
        if not (payload.prefix and payload.number_part):
            raise HTTPException(status_code=400, detail="必须提供 full，或提供 prefix 与 number_part")
        prefix = payload.prefix.strip()
        number_part = payload.number_part.strip()
        if not prefix or not number_part:
            raise HTTPException(status_code=400, detail="prefix 与 number_part 不能为空")

    # 2) 调用文号系统：删除旧记录 -> 创建新记录
    try:
        async with DocNumSession() as ddb:
            # 删除旧记录（如果没有也不会报错）
            try:
                await delete_document_number(ddb, current_user.id)
            except Exception:
                # 忽略删除异常，继续尝试创建（有些实现可能自动覆盖）
                pass

            # 创建新文号
            await create_user_document_number(ddb, current_user.id, prefix, number_part)

            # 读回确认
            rec = await get_document_number_by_user_id(ddb, current_user.id)
            if not rec:
                raise HTTPException(status_code=500, detail="创建后未能读取到文号记录")
            full_text = f"{rec.prefix}{rec.number_part}"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新文号失败：{e}")

    # 3) 返回最新文号
    return {
        "ok": True,
        "user_id": current_user.id,
        "prefix": prefix,
        "number_part": number_part,
        "wenhao": full_text,  # 例如：测试公司(2025)第207号
        "message": "文号已更新",
    }

#5个审查模型接口