import functools
import inspect
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import tools
from config import load_config
from database import (complete_session, create_or_get_dataset, create_tables,
                      create_version, generate_file_hash, log_event,
                      start_session)
from logging_setup import setup_logging

# ═══════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════

load_dotenv()
AppConfig = load_config()
log_file = setup_logging(AppConfig.logging.level, AppConfig.logging.log_dir)
logger = logging.getLogger(__name__)
logger.info("API starting up")

create_tables()

model = init_chat_model(model="gpt-5-mini")

EXCLUDED = {"get_df", "check_state"}
tool_functions = [
    obj for name, obj in inspect.getmembers(tools, inspect.isfunction)
    if name not in EXCLUDED
    and not name.startswith("_")
    and obj.__module__ == "tools"
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ML Intern Agent API")

# ═══════════════════════════════════════════════════════════
# SESSION CACHE
# ═══════════════════════════════════════════════════════════

_SESSIONS = {}  # session_id -> {"agent", "config", "filepath", "filename"}


def make_logged_tool(func, session_id):
    @functools.wraps(func)
    def wrapper(**kwargs):
        logger.info(f"Tool called: {func.__name__} | Input: {kwargs}")
        result = func(**kwargs)
        if isinstance(result, dict) and "error" in result:
            logger.warning(f"Tool failed: {func.__name__} | Error: {result['error']}")
        else:
            logger.info(f"Tool success: {func.__name__}")
        log_event(
            session_id=session_id,
            event_type="tool_call",
            content=str(kwargs),
            tool_name=func.__name__,
            result=str(result),
        )
        return result
    return wrapper


def build_agent(session_id: int, files: list[str] = None):
    """Build and cache agent for a session."""
    
    session_dir = DATA_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    agent_tools = []
    for func in tool_functions:
        logged_func = make_logged_tool(func, session_id)
        tool = StructuredTool.from_function(
            func=logged_func,
            name=func.__name__,
            description=func.__doc__ or f"Run {func.__name__}",
        )
        agent_tools.append(tool)

    files_list = "\n".join([f"        - {f}" for f in (files or [])])
    file_context = f"""- Uploaded files:
{files_list if files else "        (No files uploaded yet)"}
        - session_id: {session_id}
        - Call load_dataset(filepath="...", session_id={session_id}, df_name="...") for each file."""

    agent = create_agent(
        model=model,
        system_prompt=f"""
                            You are an ML pipeline agent. You build machine learning models from CSV files, step by step.

                            SESSION RULES:
                            {file_context}
                            - ALWAYS pass session_id = {session_id} to EVERY tool call.
                            - ALWAYS save outputs to data/session_{session_id}/ directory.
                              For example: save_model(session_id={session_id}, output_path="data/session_{session_id}/model.joblib")
                              For example: generate_report(session_id={session_id}, output_path="data/session_{session_id}/report.txt")
                              Default output paths already point there, so you usually don't need to specify output_path.

                            PHASE 1: LOAD & UNDERSTAND
                            1. load_dataset (Do this for all available files if needed, using different df_names)
                            2. get_basic_info
                            3. identify_target_column

                            PHASE 2: CLEAN
                            4. drop_missing_target_rows
                            5. drop_high_cardinality_columns
                            6. handle_missing_features
                            7. encode_categorical

                            PHASE 3: MODEL
                            8. separate_features_and_target
                            9. split_data
                            10. train_single_model
                            11. generate_predictions

                            HARD RULES:
                            - READ every tool output before calling the next tool.
                            - If a tool returns an error, FIX IT.
                            - NEVER call split_data before separate_features_and_target.
                            - NEVER call train_single_model if non-numeric columns exist.
                        """,
        checkpointer=InMemorySaver(),
        tools=agent_tools,
    )

    config = {"configurable": {"thread_id": str(session_id)}}

    return {"agent": agent, "config": config}


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {"status": "alive"}


@app.post("/session")
async def create_new_session():
    db_session = start_session(version_id=None)
    _SESSIONS[db_session.id] = {
        **build_agent(db_session.id, []),
        "files": [],
    }
    logger.info(f"Blank Session {db_session.id} created")
    return {"session_id": db_session.id}


@app.post("/session/{session_id}/upload")
async def upload_datasets(session_id: int, files: list[UploadFile] = File(...)):
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    uploaded_files = []
    session_dir = DATA_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files):
        if not file.filename.endswith(".csv"):
            continue

        save_path = session_dir / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            df = pd.read_csv(save_path)
            row_count = len(df)
            column_count = len(df.columns)
            columns_json = json.dumps(df.columns.tolist())
        except Exception as e:
            logger.error(f"Failed to read CSV {file.filename}: {e}")
            continue

        file_hash = generate_file_hash(str(save_path))
        dataset = create_or_get_dataset(file.filename)
        version = create_version(
            dataset_id=dataset.id,
            file_hash=file_hash,
            row_count=row_count,
            column_count=column_count,
            columns_json=columns_json,
        )

        # Load first df into ACTIVE_DATAFRAMES as "main" for backward compatibility
        # Load subsequent ones as their filenames
        if session_id not in tools.ACTIVE_DATAFRAMES:
            tools.ACTIVE_DATAFRAMES[session_id] = {}
        
        df_name = "main" if i == 0 and "main" not in tools.ACTIVE_DATAFRAMES[session_id] else file.filename.split('.')[0]
        tools.ACTIVE_DATAFRAMES[session_id][df_name] = df.copy()

        uploaded_files.append(str(save_path))

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid CSV files uploaded")

    # Rebuild agent with updated files context
    all_files = [str(f) for f in session_dir.glob("*.csv")]
    _SESSIONS[session_id]["agent"] = build_agent(session_id, all_files)["agent"]
    _SESSIONS[session_id]["files"].extend([Path(f).name for f in uploaded_files])

    logger.info(f"Added files to session {session_id}: {uploaded_files}")

    return {
        "session_id": session_id,
        "files_added": len(uploaded_files),
        "filenames": [Path(f).name for f in uploaded_files]
    }


class ChatRequest(BaseModel):
    session_id: int
    message: str


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    if request.session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    session = _SESSIONS[request.session_id]

    logger.info(f"User input: {request.message}")
    log_event(
        session_id=request.session_id,
        event_type="message",
        content=request.message,
    )

    response = session["agent"].invoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config=session["config"],
    )

    reply = response["messages"][-1].content
    logger.info(f"Agent response: {reply[:100]}")

    # Collect generated files for this session
    session_dir = DATA_DIR / f"session_{request.session_id}"
    files = []
    if session_dir.exists():
        files = [f.name for f in session_dir.iterdir() if f.is_file()]

    return {
        "session_id": request.session_id,
        "response": reply,
        "files": files,
    }


@app.post("/session/end")
async def end_session(session_id: int):
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    complete_session(session_id)
    del _SESSIONS[session_id]
    logger.info(f"Session {session_id} ended")

    return {"status": "ended", "session_id": session_id}


# ═══════════════════════════════════════════════════════════
# FILE MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/session/{session_id}/files")
async def list_session_files(session_id: int):
    """List all generated artifacts for a session."""
    session_dir = DATA_DIR / f"session_{session_id}"
    if not session_dir.exists():
        return {"files": []}
    files = []
    for f in sorted(session_dir.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "type": f.suffix.lstrip("."),
            })
    return {"files": files}


@app.get("/session/{session_id}/files/{filename}")
async def download_file(session_id: int, filename: str):
    """Download a specific generated artifact."""
    file_path = DATA_DIR / f"session_{session_id}" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


# ═══════════════════════════════════════════════════════════
# SERVE FRONTEND
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def serve_frontend():
    return FileResponse("app/index.html")


# Mount static files LAST (catch-all)
app.mount("/app", StaticFiles(directory="app"), name="static")