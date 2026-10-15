# 作業一 HW1：經典影像處理 Classical Image Processing

**課程**：智慧醫療影像與訊號處理（115-1）
**對應單元**：W3 經典影像處理
**執行環境**：Google Colab（免安裝，瀏覽器即可）
**繳交**：Colab notebook（`.ipynb` 或「任何人可檢視」分享連結）＋ 報告 PDF
**配分**：Part A 手算 25% ＋ Part B 程式 35% ＋ Part C 迷你專題 25% ＋ **Part D 逆向工程 15%**（兩題，各 7.5%）

---

## 學習目標

做完這份作業，你能：
1. 用手算理解卷積、窗寬窗位、直方圖等化、形態學、頻域濾波、PSNR。
2. 用 `scikit-image` 在醫學影像上完成增強、去噪、邊緣、分割、頻域處理，並量化評估。
3. 串出一條完整影像處理 pipeline，並說明每步的取捨。

## 環境設定（Colab）

- Colab 已內建 `numpy` / `scipy` / `matplotlib` / `scikit-image`。若遇版本問題，在第一格執行 `!pip install -U scikit-image`。
- **影像來源**：直接用 scikit-image 內建醫學影像，不必另外下載：
  ```python
  from skimage import data
  vol = data.brain()          # 腦部 MRI（10 張切片）
  img = vol[5]                # 取一張切片
  # 或 CT 模擬：data.shepp_logan_phantom()
  ```
- Starter notebook：`HW1_starter.ipynb`（已附各題骨架與 TODO）。

---

## Part A — 手算題（25%，不用電腦，寫在報告內）

**A1 卷積（5%）**
給定 5×5 影像塊與 3×3 均值濾波 kernel（值皆為 1/9）：
```
I = 10 10 10 80 80        K = 1/9 1/9 1/9
    10 10 10 80 80            1/9 1/9 1/9
    10 10 10 80 80            1/9 1/9 1/9
    10 10 10 80 80
    10 10 10 80 80
```
(a) 計算中心 3×3 區域的卷積輸出（邊界像素略過）。
(b) 從結果說明：為什麼均值濾波會讓「邊緣」變模糊？

**A2 窗寬窗位 Windowing（5%）**
某 CT 影像像素 HU 值為 {−1000, 0, 40, 300, 1000}。使用軟組織窗 window level = 40、width = 400。
(a) 寫出把 HU 映射到顯示灰階 0–255 的線性公式（提示：window 範圍 = [L−W/2, L+W/2]，範圍外飽和）。
(b) 算出每個 HU 對應的灰階值。

**A3 直方圖等化 Histogram Equalization（5%）**
一張 3-bit（灰階 0–7）影像，各灰階像素數如下：
```
灰階 g:     0   1   2   3   4   5   6   7
像素數:    100 90  80  0   0   0   30  0    （總計 300）
```
計算 CDF，並用 **g' = round(7 × CDF(g))** 求等化後每個灰階的新對應值。（另一種教科書慣例 g' = round(7 × (CDF−CDF_min)/(1−CDF_min)) 也給分，但答案不同，須寫明你用的公式。）

**A4 形態學 Morphology（5%）**
下面二值影像（1 = 前景），結構元素為 3×3 全 1 方形：
```
0 0 0 0 0 0
0 1 1 1 0 0
0 1 1 1 1 0
0 1 1 1 0 0
0 0 1 0 0 0
0 0 0 0 0 0
```
畫出 erosion、dilation、opening、closing 的結果（各一張格子圖）。

**A5 頻域 Frequency Domain（5%）**
(a) 兩個頻域 mask：M1 只保留頻譜「中心」低頻、M2 只保留「外圍」高頻。分別說明是 low-pass 還 high-pass，套用後影像會變模糊還是強調邊緣。
(b) 用一句話解釋「空間域的高斯平滑」為何等價於「頻域的低通濾波」。

**A6 影像品質 PSNR（5%）**
兩張 8-bit（最大值 255）影像的 MSE = 100。
(a) 計算 PSNR（dB）。公式：PSNR = 10·log₁₀(MAX² / MSE)。
(b) 若改善到 MSE = 25，PSNR 變為多少？差幾 dB？

---

## Part B — 程式題（35%，Colab，用 `HW1_starter.ipynb`）

每小題：填完程式 → 貼出結果圖 → 寫一句觀察。

- **B1 載入與檢視（5%）**：載入 `data.brain()` 一張切片（或 phantom），顯示，並印出 `shape`、`dtype`、最小/最大值。
- **B2 直方圖等化（5%）**：用 `exposure.equalize_hist`，並排顯示原圖／等化圖與兩者的直方圖。
- **B3 去噪比較（10%）**：用 `util.random_noise` 加高斯雜訊，套用 mean、Gaussian、median 三種濾波；對每個結果算 **PSNR 與 SSIM**（相對乾淨原圖），做成表格，說明哪個最好、為什麼。
- **B4 邊緣偵測（5%）**：套用 `filters.sobel` 與 `feature.canny`；調整 canny 的 `sigma`（至少 3 個值），觀察邊緣數量如何變化。
- **B5 分割（8%）**：`threshold_otsu` 二值化 → `remove_small_objects` 清理 → `label` → `label2rgb` 疊圖顯示。
- **B6 頻域濾波（7%）**：對影像做 2D FFT，顯示 log 頻譜；自行設計 low-pass 與 high-pass mask，各自 `ifft2` 重建並並排顯示。

---

## Part C — 迷你專題 Pipeline（25%）

自選一張醫學影像（scikit-image 內建，或老師提供的影像組），設計一條處理 pipeline，**至少包含：增強 → 去噪 → （邊緣偵測 或 分割）**。

要求：
1. 說明你的 pipeline 每一步做什麼、參數怎麼選。
2. 量化評估：對「去噪」用 PSNR/SSIM。做法：把選定的乾淨影像**自行加合成雜訊**再去噪，才有乾淨參考可算 PSNR/SSIM（對來路不明的原圖是算不出來的）。對「分割」可用覆蓋率或視覺比對（沒有真值時，覆蓋率只是描述統計，不是效能指標）。
3. 討論（200–400 字）：每步的 tradeoff；若把某一步換成深度學習方法，你預期會有什麼差異（連到後面 W6–W7）。

---

## Part D — 逆向工程（15%，兩題各 7.5%）

**規則（五份作業共用）**：每題給你一組輸入與我們處理後的結果（`reverse/` 資料夾，格式見 `manifest.json`）。處理鏈與參數不公開，只公開「可能用到的操作家族」與搜尋刻度。你要推回我們做了什麼，寫成一個函式 `reproduce_D1(inputs) -> outputs`（D2 同理），對每一張公開影像達到 manifest 標的門檻。助教會用**你沒看過的保留測例**再呼叫一次你的函式，所以硬編碼公開答案不會過。報告要有：推定的處理鏈與參數、至少兩個假說的比較（哪個被證據排除、為什麼）、支持判斷的圖或數字、以及你認為無法辨識的部分。可以討論觀念，但函式與推理自己寫。

**D1 灰階與模糊的先後之謎 Intensity-Blur Ordering（7.5%）**
三張 128×128 影像（兩張腦部切片、一張 phantom，已 min-max 到 [0,1]）與我們的處理結果。已知我們只用了三種操作：線性窗位 `clip((x−lo)/(hi−lo), 0, 1)`、power-law（gamma）、Gaussian 平滑（`scipy.ndimage.gaussian_filter`，`truncate=3`、`mode='reflect'`）；順序與參數要你找。搜尋刻度：窗端點 0.05、gamma 0.05、sigma 0.2。
- 門檻：每張 SSIM ≥ 0.98 且 MAE ≤ 0.005（`data_range=1`）。
- 提示只有一句：增強與平滑不交換。比較飽和區邊界附近的灰階剖面，兩種順序留下的痕跡不一樣。
- 報告額外要求：畫出「交換兩步」的結果與目標在邊界處的差異圖。

**D2 頻譜上的手術 Spectral Filter Forensics（7.5%）**
同三張影像被加上週期性條紋（輸入未裁切，值可能略超出 [0,1]），以及我們濾波後的結果。已知我們在 2D DFT 域（`fft2` + `fftshift`，不另外正規化）做了一個 Gaussian 低通，再加最多兩對共軛 notch，取反變換的實部。sigma 以 DFT index 為單位、步長 1；notch 中心在整數 bin，半徑 ∈ {0, 1, 2}。
- 門檻：每張 MAE ≤ 0.0008 且 PSNR ≥ 53 dB（range 1）。門檻刻意很緊：只做低通會讓條紋變淡但不會消失，sigma 差一格也過不了。
- 提示：先看殘差 `target − 你的結果` 的頻譜，能量集中在哪幾個 bin。非 notch 位置的 |T|/|X| 就是低通的 H，取 log 對 u²+v² 做線性擬合可以直接讀出 sigma。
- 報告額外要求：附上你用的頻譜 mask 圖、被移除的頻率清單、以及條紋方向與 notch 位置的對應說明。

---

## 繳交規範

- **Notebook**：`HW1_學號.ipynb`，或 Colab 分享連結（權限設「知道連結的任何人可檢視」）。確認從頭到尾 **Runtime → Run all** 可無錯執行。
- **報告 PDF**：`HW1_學號.pdf`，含 Part A 手算過程（可手寫拍照）＋ Part C 討論。
- **工具庫（重要，HW2 起要用）**：把 Part B 寫過的函式整理成三個檔案，存到自己的 Google Drive 並隨作業附上：`mypreproc.py`（normalization、去噪、augmentation）、`mymetrics.py`（PSNR、SSIM）、`myviz.py`（並排顯示、疊圖）。HW2–HW5 每份都會直接 `import` 這些檔案（starter 最後一格有打包示範）。
- **Part D**：`reproduce_D1(inputs)`、`reproduce_D2(inputs)` 兩個函式放在 notebook 最後一節，助教會用你沒看過的保留測例呼叫它們；報告含 Part D 的推理段落。
- 截止：見課程公告。

## 評分 Rubric

| 項目 | 配分 | 重點 |
|---|---|---|
| Part A 手算 | 25% | 觀念正確、過程完整 |
| Part B 程式 | 35% | 可執行、結果正確、每題有觀察 |
| Part C 專題 | 25% | pipeline 合理、有量化評估、討論有深度 |
| Part D 逆向工程 | 15% | 兩題各 7.5%：門檻達標 4 分、推理與假說比較 2.5 分、保留測例通過 1 分 |
| 加分 | +5% | 程式整潔、額外探索（如比較更多濾波、參數掃描） |

> 誠信：可討論觀念，但程式與報告需自行完成；引用外部程式碼須註明出處。
