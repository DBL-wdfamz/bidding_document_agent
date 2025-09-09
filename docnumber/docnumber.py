# crud.py
import re
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .models import UserDocumentNumber

# --- 核心操作函数（与同步版本保持相同的名称与返回约定） ---

async def create_user_document_number(
    db: AsyncSession,
    user_id: int,
    prefix: str,
    number_part: str
) -> Optional[UserDocumentNumber]:
    """
    为新用户创建一条文号记录。
    成功返回对象；若 user_id 已存在（唯一约束），返回 None（保持原有行为）。
    """
    new_doc_num = UserDocumentNumber(user_id=user_id, prefix=prefix, number_part=number_part)
    try:
        db.add(new_doc_num)
        await db.commit()
        await db.refresh(new_doc_num)
        return new_doc_num
    except IntegrityError:
        await db.rollback()
        print(f"Error: Document number for user_id {user_id} already exists.")
        return None


async def get_document_number_by_user_id(
    db: AsyncSession,
    user_id: int
) -> Optional[UserDocumentNumber]:
    """
    根据用户ID查询其文号记录；不存在返回 None。
    """
    stmt = select(UserDocumentNumber).where(UserDocumentNumber.user_id == user_id)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


async def update_document_number(
    db: AsyncSession,
    user_id: int,
    new_prefix: str,
    new_number_part: str
) -> Optional[UserDocumentNumber]:
    """
    更新指定用户的完整文号信息（包括前缀和数字部分）。
    返回更新后的对象；若用户不存在返回 None。
    """
    doc_num_record = await get_document_number_by_user_id(db, user_id)
    if doc_num_record:
        doc_num_record.prefix = new_prefix
        doc_num_record.number_part = new_number_part
        await db.commit()
        await db.refresh(doc_num_record)
        return doc_num_record
    return None


async def delete_document_number(db: AsyncSession, user_id: int) -> bool:
    """
    删除指定用户的文号记录。
    成功返回 True；不存在返回 False。
    """
    doc_num_record = await get_document_number_by_user_id(db, user_id)
    if doc_num_record:
        await db.delete(doc_num_record)
        await db.commit()
        return True
    return False


# --- 业务逻辑封装函数（严格保持原有行为） ---

async def get_and_increment_document_number(
    db: AsyncSession,
    user_id: int
) -> Optional[str]:
    """
    获取用户当前的文号，将其数字部分加一并更新数据库，最后返回新的完整文号。
    - 与原逻辑一致：通过调用 update_document_number 来持久化。
    - 若 number_part 中无数字，则直接返回当前完整文号（不报错）。
    - 若记录不存在，返回 None。
    """
    doc_num_record = await get_document_number_by_user_id(db, user_id)
    if not doc_num_record:
        return None

    current_number_part = doc_num_record.number_part
    match = re.search(r'\d+', current_number_part)

    if match:
        num = int(match.group(0))
        next_num = num + 1
        start, end = match.span(0)
        new_number_part = current_number_part[:start] + str(next_num) + current_number_part[end:]

        # ★ 保持原有逻辑：通过 update_document_number 持久化（而不是直接改对象）
        await update_document_number(
            db=db,
            user_id=user_id,
            new_prefix=doc_num_record.prefix,   # 保持前缀不变
            new_number_part=new_number_part     # 更新数字部分
        )
        return doc_num_record.prefix + new_number_part
    else:
        # 没有数字则不自增，直接返回当前完整文号
        return doc_num_record.prefix + current_number_part
