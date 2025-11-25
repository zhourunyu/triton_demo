# Object Detection Tab
import gradio as gr
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput
from clients import triton_client as client
import cv2
import json

with open("examples/coco-names.json") as f:
    NAMES: list[str] = json.load(f)

conf_thres = 0.25
iou_thres = 0.45
input_shape = (640, 640)

def preprocess(image: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
    img_h, img_w, _ = image.shape
    input_h, input_w = input_shape

    # 1. Letterbox
    r = min(input_h / img_h, input_w / img_w)
    resized_w, resized_h = int(img_w * r), int(img_h * r)

    # resize and pad image
    resized_img = cv2.resize(image, (resized_w, resized_h))
    padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    top_pad = (input_h - resized_h) // 2
    left_pad = (input_w - resized_w) // 2
    padded_img[top_pad:top_pad + resized_h, left_pad:left_pad + resized_w] = resized_img

    # 2. HWC to CHW, BGR to RGB, Normalization
    padded_img = padded_img[:, :, ::-1] 
    padded_img = padded_img.transpose(2, 0, 1) 
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0

    return padded_img, (r, top_pad, left_pad)

def postprocess(output: np.ndarray, meta: tuple[float, int, int]) -> tuple[list, list, list]:
    # filter confidence
    obj_conf = output[:, 4]
    predictions = output[obj_conf > conf_thres]
    if len(predictions) == 0:
        return [], [], []

    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    max_class_scores = np.max(class_scores, axis=1)
    keep = max_class_scores > conf_thres
    predictions = predictions[keep]
    class_ids = class_ids[keep]
    if len(predictions) == 0:
        return [], [], []

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes = predictions[:, :4]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2)
    y1 = (cy - h / 2)
    x2 = (cx + w / 2)
    y2 = (cy + h / 2)

    # NMS
    # cv2.dnn.NMSBoxes requires [x, y, w, h] for boxes
    boxes_for_nms = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
    scores_for_nms = predictions[:, 4].tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []

    # restore coordinates
    final_boxes, final_scores, final_class_ids = [], [], []
    ratio, top_pad, left_pad = meta
    for i in indices:
        box_x1 = (x1[i] - left_pad) / ratio
        box_y1 = (y1[i] - top_pad) / ratio
        box_x2 = (x2[i] - left_pad) / ratio
        box_y2 = (y2[i] - top_pad) / ratio

        final_boxes.append([int(box_x1), int(box_y1), int(box_x2), int(box_y2)])
        final_scores.append(scores_for_nms[i])
        final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids

def draw_results(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{NAMES[class_id]}: {score:.2f}"

        # draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # draw label
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1, label_size[1])
        cv2.rectangle(image, (x1, top - label_size[1]), (x1 + label_size[0], top + base_line), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def detect(image: np.ndarray | None, model_name: str) -> np.ndarray | None:
    if image is None:
        return None

    input, meta = preprocess(image)
    input = np.expand_dims(input, axis=0)

    inputs = [InferInput("input", input.shape, "FP32")]
    inputs[0].set_data_from_numpy(input)
    outputs = [InferRequestedOutput("output")]

    result = client.infer(model_name, inputs=inputs, outputs=outputs)
    if result is None:
        raise gr.Error("Inference request failed.")
    output: np.ndarray | None = result.as_numpy("output")
    if output is None:
        raise gr.Error("Failed to get output from inference result.")

    boxes, scores, class_ids = postprocess(output[0], meta)
    return draw_results(image, boxes, scores, class_ids)

def create():
    with gr.TabItem("Object Detection"):
        gr.Markdown("# Object Detection")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", height=600)
                model = gr.Dropdown(["yolov5s"], value="yolov5s", label="Model")
                examples = gr.Examples("examples/detection", image)

            with gr.Column():
                output = gr.Image(type="numpy", height=560)

    image.change(fn=detect, inputs=[image, model], outputs=output)
    model.change(fn=detect, inputs=[image, model], outputs=output)
