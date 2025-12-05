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


pip install ultralytics opencv-python numpy


3. 各程式檔用途與程式碼說明
3.1 detect_yolo.py：負責載入 YOLO，對影像偵測船舶 bbox

這支檔案做的事：匯入 YOLO 模型。

一開始就載入一次 yolov8n.pt（小模型，速度較快）。

定義 detect_ship_bbox(image_path) 函式：

對單張圖片做 YOLO 偵測。

找出面積最大的偵測框，當作目標船舶。

回傳中心點與寬高（像素）。


3.2 ship_utils.py：距離計算工具（真實距離 + 像素距離）

這支檔案做的事：

haversine(lon1, lat1, lon2, lat2)
→ 用 Haversine 公式計算兩個經緯度之間的「真實距離（公尺）」。

pixel_distance(p1, p2)
→ 計算兩個像素座標點（例如兩張圖中的船中心點）之間的距離（px）。

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

