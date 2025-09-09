# models.py
import os
from pathlib import Path
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import Column, Integer, String

# 放在项目根：__file__ 是 docnumber/models.py
BASE_DIR = Path(__file__).resolve().parents[1]   # 上一级=项目根目录
DB_PATH = (BASE_DIR / "app.db").as_posix()

DATABASE_URL = "mysql+aiomysql://agentusercenter:wdfamzwdfamz@127.0.0.1:3306/document_agent?charset=utf8mb4"

engine = create_async_engine(DATABASE_URL, echo=True, future=True)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class UserDocumentNumber(Base):
    __tablename__ = 'user_document_numbers'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, nullable=False, index=True)
    prefix = Column(String(255), nullable=False, comment="文号前缀 (公司简称+时间)")
    number_part = Column(String(50), nullable=False, comment="文号数字部分 (X号)")

    def __repr__(self) -> str:
        # 使用 __dict__ 直接取已在内存中的字段，避免触发懒加载/数据库 I/O
        d = object.__getattribute__(self, "__dict__")
        user_id = d.get("user_id", "?")
        prefix = d.get("prefix", "")
        number_part = d.get("number_part", "")
        return f"<UserDocumentNumber(user_id={user_id}, full_number='{prefix}{number_part}')>"

    __str__ = __repr__

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
