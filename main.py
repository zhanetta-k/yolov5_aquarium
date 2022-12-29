import json
from app_utils import get_yolov5, get_image_from_bytes
from fastapi import FastAPI, File
from starlette.responses import Response
import io
from PIL import Image


model = get_yolov5()


app = FastAPI(
    title="Custom YOLOv5 API",
    description="""Obtain object value out of image
                    and return json result and image.""",
    version="0.0.1")


@app.post("/object-to-json")
async def detect_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    # JSON img1 predictions
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    # updates results.imgs with boxes and labels
    results.render()

    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
