import asyncio
import uuid
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

class OISSTRequest(BaseModel):
    mode: str = "single"
    region: str = "north_atlantic"
    theme: str = "light"
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    baseline_date: Optional[str] = "4/18/2015"
    remove_global_mean: bool = True
    show_oceanic_indices: bool = False
    show_pct_overlay: bool = False
    show_inset_map: bool = False

class ERSSTRequest(BaseModel):
    events: list = []
    region: str = "global"
    theme: str = "dark"

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/generate/oisst")
def generate_oisst(req: OISSTRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "image": None}
    thread = threading.Thread(target=run_oisst_job, args=(job_id, req))
    thread.start()
    return {"job_id": job_id}

@app.post("/api/generate/ersst")
def generate_ersst(req: ERSSTRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "image": None}
    thread = threading.Thread(target=run_ersst_job, args=(job_id, req))
    thread.start()
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job["status"] == "done":
        return {"status": "done"}
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "unknown error")}
    return {"status": job["status"]}

@app.get("/api/jobs/{job_id}/image")
def get_image(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return Response(status_code=404)
    return Response(content=job["image"], media_type="image/png")

def run_oisst_job(job_id, req):
    try:
        from runners import run_oisst
        img_bytes = run_oisst(req)
        jobs[job_id] = {"status": "done", "image": img_bytes}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

def run_ersst_job(job_id, req):
    try:
        from runners import run_ersst
        img_bytes = run_ersst(req)
        jobs[job_id] = {"status": "done", "image": img_bytes}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}