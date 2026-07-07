import os
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

KAGGLE_NGROK_URL = "https://9d68-35-196-254-107.ngrok-free.app" 


app = FastAPI(title="Local UI Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def proxy_health():
    target_url = f"{KAGGLE_NGROK_URL.rstrip('/')}/health"
    headers = {"ngrok-skip-browser-warning": "true"}
    try:
        response = requests.get(target_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deshadow")
async def proxy_deshadow(file: UploadFile = File(...)):
    target_url = f"{KAGGLE_NGROK_URL.rstrip('/')}/deshadow"
    
    headers = {"ngrok-skip-browser-warning": "true"}
    
    try:
        file_bytes = await file.read()
        files = {"file": (file.filename, file_bytes, file.content_type)}
        
        print(f"[*] Image transfer to ({target_url})...")
        response = requests.post(target_url, files=files, headers=headers)
        
        if response.status_code != 200:
            print(f"[-] Kaggle trả về lỗi {response.status_code}: {response.text}")
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404
                )
                
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"[-] Error connecting to ... : {e}")
        raise HTTPException(status_code=500, detail=f"Cannot connect to Server: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print("🚀 BẬT SERVER GIAO DIỆN LOCAL")
    print(f"👉 Link truy cập Web UI: http://127.0.0.1:8080")
    print(f"👉 Đang kết nối tới GPU Kaggle tại: {KAGGLE_NGROK_URL}")
    print("=========================================================")
    uvicorn.run(app, host="127.0.0.1", port=8080)