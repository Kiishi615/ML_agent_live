import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import ForeignKey, String, Text, create_engine, func
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship)

from config import load_config

AppConfig= load_config()


DB__DIR = AppConfig.database.directory
DB_NAME = AppConfig.database.name

DATABASE_DIR = Path(DB__DIR)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = AppConfig.database.connection_string

engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass

def utc_now():
    return datetime.now(timezone.utc)


def generate_file_hash(filepath: str) -> str:
    """Reads a file in chunks and generates a SHA-256 hash."""
    
    hasher = hashlib.sha256()
    
    # Open the file in 'rb' (read binary) mode
    try:
        with open(filepath, 'rb') as file:
            chunk = file.read(65536)
            while len(chunk) > 0:
                hasher.update(chunk)
                chunk = file.read(65536)
                
        return hasher.hexdigest()
        
    except FileNotFoundError:
        return "file_not_found_error"

class Dataset(Base):
    __tablename__ = "datasets"  

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(200), default="Untitled")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan",
        order_by="DatasetVersion.version_number"
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, title='{self.filename}')>"

class DatasetVersion(Base):
    __tablename__ = "dataset_versions" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique =True)
    version_number: Mapped[int] = mapped_column(nullable= False)
    row_count: Mapped[int] = mapped_column(nullable= False)
    column_count: Mapped[int] = mapped_column(nullable= False)
    columns_json: Mapped[str] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(default=utc_now)

    dataset: Mapped["Dataset"] = relationship(
        back_populates="versions",
    )
    sessions: Mapped[list["AnalysisSession"]] = relationship(
        back_populates="dataset_version",
    )

    def __repr__(self) -> str:
        return f"<DatasetVersion(id={self.id},  version_number = {self.version_number})>"

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    completed_at: Mapped[datetime| None] = mapped_column(default=None, nullable= True)
    
    dataset_version: Mapped["DatasetVersion"] = relationship(
        back_populates="sessions",
    )

    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
        order_by="AuditLog.created_at"
    )
    
    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id},  completed_at = {self.completed_at})>"

class AuditLog(Base):
    __tablename__ = "audit_logs" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("analysis_sessions.id"))
    event_type: Mapped[str] = mapped_column(String(20))
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable= True)
    content: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable= True)

    session: Mapped["AnalysisSession"] = relationship(back_populates="audit_logs")
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id},  event_type= {self.event_type}, content= {self.content})>"


def create_tables() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)

def get_session() -> Session:
    return Session(bind=engine)


def create_or_get_dataset(filename: str = "Untitled") -> Dataset:
    session = get_session()
    try:
        dataset = session.query(Dataset).filter(Dataset.filename == filename).first()
        if dataset:
            return dataset
        else:
            dataset = Dataset(filename=filename)
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_version(dataset_id: int, file_hash: str,  row_count: int, column_count: int, columns_json: str) -> DatasetVersion:
    """Creates a new version record for an uploaded file."""
    session = get_session()
    try:
        existing_version = session.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.file_hash == file_hash
        ).first()
        if existing_version:
            return existing_version

        current_version_count = session.query(func.count(DatasetVersion.id)).filter(
            DatasetVersion.dataset_id == dataset_id
        ).scalar()

        next_version_number = current_version_count +1

        version = DatasetVersion(
            dataset_id=dataset_id,
            file_hash=file_hash,
            version_number = next_version_number,
            row_count=row_count,
            column_count=column_count,
            columns_json=columns_json
        )
        session.add(version)
        session.commit()
        session.refresh(version)
        return version
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def start_session(version_id: int) -> AnalysisSession:
    """Starts a new analysis chat session."""
    session = get_session()
    try:
        new_session = AnalysisSession(
            version_id=version_id,
        )
        session.add(new_session)
        session.commit()
        session.refresh(new_session)
        return new_session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_event(session_id, event_type, content, tool_name=None, result=None):
    session = get_session()
    try:
        entry = AuditLog(
            session_id=session_id,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            result=result,
        )
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def complete_session(session_id: int) -> None:
    session = get_session()
    try:
        analysis_session = session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if analysis_session:
            analysis_session.completed_at = utc_now()
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


