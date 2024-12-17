import cv2
import os
import time
import io
import importlib
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, Response
from typing import Optional
from functools import lru_cache
from PIL import Image
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from io import BytesIO
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR, PPStructure, sorted_layout_boxes, \
    convert_info_docx, save_structure_res

if True:
    ppocr = importlib.import_module("ppocr", "paddleocr")
    from ppocr.utils.utility import check_and_read


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa: E501


@lru_cache(maxsize=1)
def load_pps_model():
    return PPStructure()


def invoke_ocr_structure(content):
    engine = load_pps_model()

    with TemporaryDirectory() as tmpdir:
        img_name = "a"
        img_path = os.path.join(tmpdir, img_name + ".pdf")
        with open(img_path, "wb") as f:
            f.write(content)
        img, _, _ = check_and_read(img_path)

        img_paths = []
        for index, pdf_img in enumerate(img):
            os.makedirs(os.path.join(tmpdir, img_name), exist_ok=True)
            pdf_img_path = os.path.join(
                tmpdir, img_name, img_name + "_" + str(index) + ".jpg"
            )
            cv2.imwrite(pdf_img_path, pdf_img)
            img_paths.append([pdf_img_path, pdf_img])

        all_res = []
        for index, (_, img) in enumerate(img_paths):
            print(f"processing {index + 1}/{len(img_paths)} page:")
            result = engine(img, img_idx=index)
            save_structure_res(result, tmpdir, img_name, index)
            all_res += sorted_layout_boxes(result, img.shape[1])

        convert_info_docx(img, all_res, tmpdir, img_name)

        with open(os.path.join(tmpdir, img_name+"_ocr.docx"), "rb") as f:
            return f.read()


@lru_cache(maxsize=1)
def load_ocr_model():
    # model = PaddleOCR(use_angle_cls=True, lang='en')
    model = PaddleOCR(use_angle_cls=True)
    return model


def merge_data(values):
    data = []
    for idx in range(len(values)):
        data.append([values[idx][1][0]])
        # print(data[idx])

    return data


def invoke_ocr(doc, content_type):
    worker_pid = os.getpid()
    print(f"Handling OCR request with worker PID: {worker_pid}")
    start_time = time.time()

    model = load_ocr_model()

    bytes_img = io.BytesIO()
    format_img = "JPEG"
    if content_type == "image/png":
        format_img = "PNG"

    doc.save(bytes_img, format=format_img)
    bytes_data = bytes_img.getvalue()
    bytes_img.close()

    result = model.ocr(bytes_data, cls=True)

    values = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            values.append(line)

    values = merge_data(values)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"OCR done, worker PID: {worker_pid}")

    return values, processing_time


router = APIRouter()


@router.post("/inference")
async def inference(file: UploadFile = File(None),
                    image_url: Optional[str] = Form(None)):
    result = None
    if file:
        if file.content_type in ["image/jpeg", "image/jpg", "image/png"]:
            doc = Image.open(BytesIO(await file.read()))
            result, processing_time = invoke_ocr(doc, file.content_type)
        elif file.content_type == "application/pdf":
            pdf_bytes = await file.read()
            pages = convert_from_bytes(pdf_bytes, 300)
            all_results = []
            total_processing_time = 0
            for page in pages:
                page_result, processing_time = invoke_ocr(page, "image/jpeg")
                all_results.extend(page_result)
                total_processing_time += processing_time
            result = all_results
            print(f"Total time: {total_processing_time:.2f} seconds")
        else:
            return {"error": "Invalid file type. Only JPG/PNG images and PDF are allowed."}  # noqa: E501
    elif image_url:
        headers = {"User-Agent": "Mozilla/5.0"}  # to avoid 403 error
        req = Request(image_url, headers=headers)
        with urlopen(req) as response:
            content_type = response.info().get_content_type()

            if content_type in ["image/jpeg", "image/jpg", "image/png"]:
                doc = Image.open(BytesIO(response.read()))
                result, processing_time = invoke_ocr(doc, content_type)
            elif content_type in ["application/pdf", "application/octet-stream"]:  # noqa: E501
                pdf_bytes = response.read()
                pages = convert_from_bytes(pdf_bytes, 300)
                all_results = []
                total_processing_time = 0
                for page in pages:
                    page_result, processing_time = invoke_ocr(
                        page, "image/jpeg")
                    all_results.extend(page_result)
                    total_processing_time += processing_time
                result = all_results
                print(f"Total time: {total_processing_time:.2f} seconds")
            else:
                return {"error": "Invalid file type. Only JPG/PNG images and PDF are allowed."}  # noqa: E501
    else:
        result = {"info": "No input provided"}

    if result is None:
        raise HTTPException(
            status_code=400, detail="Failed to process the input.")

    return JSONResponse(status_code=status.HTTP_200_OK, content=result)


@router.post("/doc")
async def doc(file: UploadFile = File()):
    if file.content_type in ["application/pdf", "application/octet-stream"]:
        pdf_bytes = await file.read()
        result = invoke_ocr_structure(pdf_bytes)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. (JPG/PNG/PDF)",
        )

    return Response(
        status_code=status.HTTP_200_OK,
        content=result,
        media_type=DOCX_MEDIA_TYPE,
    )
