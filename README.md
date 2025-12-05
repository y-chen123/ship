# AIS × YOLO 船舶長寬估算 

本專案示範如何利用：

- AIS 資料（經緯度、船長 len、船寬 width）
- YOLO 物件偵測模型（偵測影像中的船舶 bounding box）

來推估船舶的實際長度與寬度，並計算與真實值的誤差。

流程概念：

1. 從兩個時間點的 AIS 資料計算「真實位移距離」（公尺）。
2. 從對應兩張影像計算「像素位移距離」（px）。
3. 真實距離 ÷ 像素距離 → 取得比例尺（幾公尺／每像素）。
4. 用 YOLO 偵測框出船舶 bounding box（w, h），乘上比例尺。
5. 與 AIS 給的真實船長（len）、船寬（width）做誤差比較。

---
2. 安裝環境與相依套件
2.1 建議 Python 版本

Python 3.9 ~ 3.11 皆可（你目前在 Windows 下可以正常執行即可）

2.2 安裝必要套件

在專案資料夾下執行：

pip install ultralytics opencv-python numpy


如要寫成 requirements.txt，可以這樣：

ultralytics
opencv-python
numpy


之後也可以用：

pip install -r requirements.txt

3. 各程式檔用途與程式碼說明
3.1 detect_yolo.py：負責載入 YOLO，對影像偵測船舶 bbox

這支檔案做的事：匯入 YOLO 模型。

一開始就載入一次 yolov8n.pt（小模型，速度較快）。

定義 detect_ship_bbox(image_path) 函式：

對單張圖片做 YOLO 偵測。

找出面積最大的偵測框，當作目標船舶。

回傳中心點與寬高（像素）。

# detect_yolo.py
from ultralytics import YOLO

# 一開始載入一次 YOLO 模型，之後呼叫 detect_ship_bbox() 就直接重用
# 先用小模型 yolov8n.pt，跑比較快；之後效果 OK 再換 yolov8s.pt 等
model = YOLO("yolov8n.pt")


def detect_ship_bbox(image_path):
    """
    對單張圖片做 YOLO 偵測，回傳:
      {
        "cx": 中心 x 座標,
        "cy": 中心 y 座標,
        "w":  寬度(像素),
        "h":  高度(像素)
      }
    若偵測不到船舶，回傳 None
    """
    # 用已載入的 model 對這張圖做推論，results[0] 是第一張影像的結果
    results = model(image_path)[0]
    
    # 直接顯示 YOLO 畫好框的影像視窗（方便目視 debug）
    results.show()
    
    # 取得所有偵測到的 bounding boxes
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        # 沒偵測到任何東西
        return None

    # 簡單策略：從所有偵測框裡面，挑「面積最大的」當成目標船舶
    best_box = None
    best_area = 0

    for box in boxes.xyxy:   # xyxy 代表 [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    # 中心點座標
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 寬與高（像素）
    w = x2 - x1
    h = y2 - y1

    return {"cx": cx, "cy": cy, "w": w, "h": h}

3.2 ship_utils.py：距離計算工具（真實距離 + 像素距離）

這支檔案做的事：

haversine(lon1, lat1, lon2, lat2)
→ 用 Haversine 公式計算兩個經緯度之間的「真實距離（公尺）」。

pixel_distance(p1, p2)
→ 計算兩個像素座標點（例如兩張圖中的船中心點）之間的距離（px）。

# ship_utils.py
import math

def haversine(lon1, lat1, lon2, lat2):
    """用 Haversine 公式計算兩個經緯度之間的真實距離（公尺）"""
    R = 6371000  # 地球半徑（公尺）

    # 經緯度轉成弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine 公式
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # 回傳距離（公尺）


def pixel_distance(p1, p2):
    """計算兩個像素座標點之間的距離（px）
    參數 p1, p2 格式: {"x": ..., "y": ...}
    """
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return math.sqrt(dx * dx + dy * dy)

3.3 run_example.py：主程式（從 JSON → YOLO → 長寬估算）

這支程式是整個流程的主控：

讀取 data/samples_1.json。

取出兩筆樣本（同一艘船、不同時間）。

用 AIS 經緯度算真實距離 d_meter。

用 JSON 給的影像座標算像素距離 d_pixel。

scale = d_meter / d_pixel 得到比例尺（m/px）。

對兩張影像跑 YOLO，取得 bounding box。

用第二張圖的 bbox 推估船長與船寬。

與 AIS 真實長寬比較，印出誤差。

# run_example.py
import json
from ship_utils import haversine, pixel_distance
from detect_yolo import detect_ship_bbox

# 指定要使用哪一個樣本 JSON 檔
#DATA_PATH = "data/samples_2.json"
DATA_PATH = "data/samples_1.json"


def load_samples(path):
    """讀取 JSON 檔，回傳 Python list/dict"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    # 讀取測試樣本（裡面應該有至少 2 筆資料）
    samples = load_samples(DATA_PATH)

    # 這裡假設 samples[0], samples[1] 就是你那兩筆：同船不同時間點
    s1 = samples[0]
    s2 = samples[1]

    # ---------- Step 1：真實距離（AIS → Haversine） ----------
    lon1, lat1 = s1["data"]["lon"], s1["data"]["lat"]
    lon2, lat2 = s2["data"]["lon"], s2["data"]["lat"]

    d_meter = haversine(lon1, lat1, lon2, lat2)
    print(f"真實距離 d_meter = {d_meter:.3f} m")

    # ---------- Step 2：像素距離（先用 JSON 的 pos 測流程） ----------
    # pos 是你事先標在 JSON 裡的影像座標（兩張圖中船的中心點）
    p1 = s1["pos"]
    p2 = s2["pos"]
    d_pixel = pixel_distance(p1, p2)
    print(f"像素距離 d_pixel = {d_pixel:.3f} px")

    # ---------- Step 3：比例尺 ----------
    scale_m_per_px = d_meter / d_pixel
    print(f"比例尺 scale = {scale_m_per_px:.5f} m/px")

    # ---------- Step 4：用 YOLO 偵測 bbox，估計長寬 ----------
    # 從 JSON 讀出兩張影像的路徑
    img1_path = s1["image"]
    img2_path = s2["image"]

    # 對兩張影像分別跑 YOLO
    bbox1 = detect_ship_bbox(img1_path)
    bbox2 = detect_ship_bbox(img2_path)

    if bbox1 is None or bbox2 is None:
        print("YOLO 沒有偵測到船舶，請檢查影像或模型設定。")
        return

    print("YOLO 偵測結果：")
    print("  圖1:", bbox1)
    print("  圖2:", bbox2)

    # 簡化處理：用第二張圖的 bbox 當估計長寬的依據
    w_px = bbox2["w"]
    h_px = bbox2["h"]

    # 假設船的「長邊」對應 bbox 的較長那一邊
    bbox_long_px  = max(w_px, h_px)
    bbox_short_px = min(w_px, h_px)

    # 用比例尺把像素換成公尺
    length_est = bbox_long_px * scale_m_per_px
    beam_est   = bbox_short_px * scale_m_per_px

    print(f"估計長度 Length_est = {length_est:.3f} m")
    print(f"估計寬度 Beam_est   = {beam_est:.3f} m")

    # ---------- Step 5：與 AIS 真實長寬比較誤差 ----------
    # 真實值從 AIS JSON 裡讀出來
    true_len  = s1["data"]["len"]
    true_beam = s1["data"]["width"]

    # 相對誤差 = |估計值 - 真實值| / 真實值
    length_err = abs(length_est - true_len) / true_len
    beam_err   = abs(beam_est - true_beam) / true_beam

    print(f"長度相對誤差 = {length_err:.3%}")
    print(f"寬度相對誤差 = {beam_err:.3%}")


if __name__ == "__main__":
    main()
