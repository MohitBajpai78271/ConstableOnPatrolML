import pandas as pd
import numpy as np
import requests
import seaborn as sns
import os
from dotenv import load_dotenv
import uvicorn
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import os
from typing import List, Optional, Dict, Any

import httpx
import numpy as np
import pandas as pd
import requests
import openai
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.cluster import DBSCAN

app = FastAPI(
    title="Crime + Patrol API",
    description="Provides crime hotspot clustering and GPT-based patrol suggestions.",
    version="1.2",
)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def root():
    return {"message": "Crime + Patrol API is live"}

class CrimeRecord(BaseModel):
    _id: str
    lat: str
    long: str
    crime: Optional[str]
    beat: Optional[str]
    date: Optional[str]
    month: Optional[str]
    year: Optional[str]
    __v: Optional[int]

class Hotspot(BaseModel):
    centroid_lat: float
    centroid_lon: float
    crime_count: int

class PatrolSuggestion(BaseModel):
    summary: str
    suggestion: str


def find_hotspots(
    records: List[CrimeRecord],
    eps_km: float = 0.4,
    min_samples: int = 10
) -> List[Hotspot]:

    df = pd.DataFrame([r.dict() for r in records])
    df[["lat", "long"]] = df[["lat", "long"]].astype(float)

    # Filter within Delhi area
    df = df[df["lat"].between(28.4, 28.88) & df["long"].between(76.84, 77.35)]
    df.drop_duplicates(subset=["lat", "long"], inplace=True)
    if df.empty:
        return []

    coords = np.radians(df[["lat", "long"]])
    eps_rad = eps_km / 6371.0  # Earth radius km
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    df["cluster"] = db.fit_predict(coords)
    hot = df[df["cluster"] >= 0]

    centroids = (
        hot.groupby("cluster")[['lat', 'long']].mean()
        .rename(columns={'lat': 'centroid_lat', 'long': 'centroid_lon'})
        .assign(crime_count=hot['cluster'].value_counts().sort_index().values)
        .sort_values('crime_count', ascending=False)
        .reset_index(drop=True)
        .head(8)
    )

    return [Hotspot(**row) for row in centroids.to_dict(orient='records')]


def get_patrol_suggestion(lat: float, lon: float) -> PatrolSuggestion:

    overpass_q = f"""
    [out:json];
    (
      node(around:50,{lat},{lon})["amenity"];
      node(around:50,{lat},{lon})["shop"];
      way(around:50,{lat},{lon})["highway"];
    );
    out center;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": overpass_q})
    if r.status_code != 200:
        raise HTTPException(502, f"Overpass API error: {r.status_code}")
    elements = r.json().get("elements", [])

    features = []
    for el in elements:
        for k, v in el.get("tags", {}).items():
            features.append(f"{k}: {v}")
    summary = "\n".join(features) or "No features found."

    prompt = (
        f"Nearby features within 50 meters:\n{summary}\n\n"
        "Based on this, provide a short and clear recommendation for where to set up a police patrol. "
        "Only include what is necessary for officers on a mobile app — no extra explanation."
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages = [{"role": "user", "content": prompt}]
        )

        suggestion = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, f"OpenAI API error: {e}")

    return PatrolSuggestion(summary=summary, suggestion=suggestion)

@app.get("/hotspots/", response_model=Dict[str, Any])
async def hotspots_endpoint(
    source_url: str = Query(..., description="URL returning JSON array of crime records"),
    eps_km: float = Query(0.4, description="DBSCAN radius in km"),
    min_samples: int = Query(10, description="DBSCAN min samples per cluster")
):

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(source_url)
    if resp.status_code != 200:
        raise HTTPException(502, f"Failed to fetch {source_url}: {resp.status_code}")
    try:
        records = [CrimeRecord(**item) for item in resp.json()]
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON schema: {e}")

    hotspots = find_hotspots(records, eps_km=eps_km, min_samples=min_samples)
    return {"hotspots": [h.dict() for h in hotspots]}

@app.get("/analyze/", response_model=Dict[str, Any])
async def analyze_endpoint(
    source_url: str = Query(..., description="URL returning JSON array of crime records"),
    eps_km: float = Query(0.4, description="DBSCAN radius in km"),
    min_samples: int = Query(10, description="DBSCAN min samples per cluster")
):

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(source_url)
    if resp.status_code != 200:
        raise HTTPException(502, f"Failed to fetch {source_url}: {resp.status_code}")
    try:
        records = [CrimeRecord(**item) for item in resp.json()]
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON schema: {e}")

    hotspots = find_hotspots(records, eps_km=eps_km, min_samples=min_samples)

    results = []
    for hs in hotspots:
        patrol = get_patrol_suggestion(hs.centroid_lat, hs.centroid_lon)
        results.append({
            "centroid_lat": hs.centroid_lat,
            "centroid_lon": hs.centroid_lon,
            "crime_count": hs.crime_count,
            "suggestion": patrol.dict()
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)