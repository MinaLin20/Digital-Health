# 作業二 HW2：CNN 醫學影像分類 CNN Classification

**課程**：智慧醫療影像與訊號處理（115-1）
**對應單元**：10/12 CNN 與遷移學習
**解鎖論文**：CheXNet（Rajpurkar et al., 2017，胸片肺炎偵測的 CNN，並延伸到 14 類胸部疾病）
**執行環境**：Google Colab（免費層即可，建議開 GPU：執行階段 → 變更執行階段類型 → T4 GPU）
**繳交**：Colab notebook（`.ipynb` 或「任何人可檢視」分享連結）＋ 報告 PDF
**配分**：Part A 計算與推導 25% ＋ Part B 程式 35% ＋ Part C 迷你專題 25% ＋ **Part D 逆向工程 15%**（兩題，各 7.5%）

---

## 學習目標

做完這份作業，你能：
1. 計算卷積輸出尺寸、感受野、softmax 與交叉熵，理解 CNN 為何逐層擴大視野。
2. 在真實醫學影像上訓一個小型 CNN 分類器，用 **AUROC 與混淆矩陣**（不是只看 accuracy）誠實評估，並在類別不平衡下解讀。
3. 用 **CAM** 看模型「看哪裡」，並比較「從零訓練」與「遷移學習」的差異，體會 CheXNet 的核心做法。

## 環境設定（Colab，輕量）

- **資料集**：用 [`MedMNIST`](https://medmnist.com/)（超小、免下載大檔，Colab 幾秒載完）。第一格執行：
  ```python
  !pip install -q "medmnist>=3.0"
  from medmnist import PneumoniaMNIST, INFO
  train = PneumoniaMNIST(split="train", download=True, size=28)   # 胸片肺炎二分類，28x28
  ```
  下載失敗時（課堂上多人同時下載偶爾會慢），把本作業資料夾 `data/` 裡的 `pneumoniamnist.npz` 放進你的 Drive，改用 `PneumoniaMNIST(split=..., root='<該資料夾>', download=False, size=28)`。資料來源：MedMNIST（Yang et al., Scientific Data 2023），CC BY 4.0。PneumoniaMNIST 來自 Kermany 等人的 5,856 張兒童胸片，不是 CheXNet 用的 ChestX-ray14，所以本作業是 CheXNet 肺炎任務的縮小類比，不是重現。
- **算力提醒（重要）**：本作業刻意設計成 Colab 免費層可跑完。**用 28×28 小圖、小模型、≤10 epoch**，全程訓練應在數分鐘內。不要換大模型或高解析度，重點是「跑通一條正確的分類 pipeline」不是刷 SOTA。
- **重用你的工具庫**：把 HW1 寫的 `mypreproc.py`（normalization / augmentation）`import` 進來當前處理；本次會在 `mymetrics.py` 新增 **AUROC**，並把你的網路定義存成 `mymodels.py`。
- Starter notebook：`HW2_starter.ipynb`（已附各題骨架與 TODO）。

---

## Part A — 計算與推導題（25%，寫在報告內，需列出過程）

**A1 卷積輸出尺寸與參數量（5%）**
一張 28×28 單通道影像，經過 kernel 5×5、stride 2、padding 1 的卷積層，輸出 8 個 feature map。
(a) 輸出的空間尺寸（高×寬）是多少？寫出公式 `out = ⌊(in + 2p − k)/s⌋ + 1`，並說明 floor 在這裡的作用。
(b) 若接一個 2×2、stride 2 的 max pooling，輸出尺寸又變多少？
(c) 這一層卷積共有幾個可學參數（含 bias）？

**A2 感受野 Receptive Field（5%）**
(a) 一個小網路依序為：3×3 卷積（stride 1）→ 2×2 max pooling（stride 2）→ 3×3 卷積（stride 1）。最後一層一個單元在原圖上的感受野邊長是多少？列出逐層推算過程。
(b) 若只用 stride 1 的 3×3 卷積、不加 pooling，最少要疊幾層，感受野邊長才 ≥ 28（涵蓋整張 28×28 影像）？用 (a) 的結果說明 pooling 為何讓所需層數大幅減少。

**A3 Softmax 與交叉熵（5%）**
三分類，模型輸出 logits `z = [0.5, 2.0, −1.0]`，正確類別是第 2 類（index 1）。
(a) 計算 softmax 機率（列出三個值）。
(b) 計算交叉熵損失 `L = −ln(p_correct)`（自然對數）。
(c) 若正確類別改成第 3 類（index 2），損失變大還變小？說明為何。

**A4 混淆矩陣與指標（5%）**
某二分類器在測試集的混淆矩陣：TP=80、FN=20、FP=30、TN=870。
(a) 算 accuracy、precision、recall（sensitivity）、specificity、F1（定義：precision = TP/(TP+FP)、recall = TP/(TP+FN)、specificity = TN/(TN+FP)、F1 = 2·precision·recall/(precision+recall)）。
(b) 若這是「1000 人中 100 個病人」的篩檢，一個「全部猜沒病」的模型 accuracy 多少？說明為何本題不能只看 accuracy。

**A5 AUROC 的意義（5%）**
(a) 用一句話寫出 AUROC 的機率定義（提示：隨機一個陽性 vs 隨機一個陰性）。
(b) 給 4 個樣本的真實標籤與分數：`(y=1, s=0.9) (y=0, s=0.8) (y=1, s=0.4) (y=0, s=0.2)`，用「陽性分數 > 陰性分數的配對比例」計算 AUROC（共 2×2=4 個配對）。

---

## Part B — 程式題（35%，Colab，用 `HW2_starter.ipynb`）

每小題：填完程式 → 貼出結果圖／數字 → 寫一句觀察。

- **B1 載入與檢視（4%）**：載入 `PneumoniaMNIST`，印出 train/val/test 筆數、影像 shape、類別分布（是否不平衡？哪一類是少數？），並顯示每類幾張範例圖。
- **B2 前處理與 DataLoader（4%）**：`import` HW1 的 `mypreproc` 做 normalization；用 `torchvision.transforms` 加簡單 augmentation：小角度旋轉（±10° 內）、亮度或對比擾動。**胸片不可上下翻轉**；左右翻轉若要用，請在報告說明理由（見 10/12 augmentation 頁）。HW1 的 `augment()` 若預設會翻轉，這裡要關掉。建立 batch DataLoader。
- **B3 建一個小 CNN（7%）**：用 PyTorch 建 2–3 個卷積區塊 + GAP + 線性分類頭的小網路，參數量 < 1M。**區塊數或通道數請和 10/12 投影片「Putting it together」的範例不同**。附一張你自己網路的逐層表（output shape、參數量）：先用公式算，再用 `sum(p.numel() for p in model.parameters())` 與實際 forward 的 shape 核對。
- **B4 訓練（6%）**：用交叉熵 + Adam 訓 ≤10 epoch，畫 train/val loss 與 accuracy 曲線。**若看到 val loss 回升就是過擬合（10/12「Overfitting, and how to see it」頁），寫一句你觀察到什麼**。**訓完把權重 `torch.save` 到 Google Drive（如 `hw2_cnn.pt`）**：HW5 要載回這個分類器當 FID 特徵抽取器與閉環 baseline，別只留在 Colab 暫存區。並把你的網路類別定義存成 Drive 上的 `mymodels.py`（和 `mypreproc.py` 同資料夾），HW5 會 `from mymodels import SmallCNN` 再載入權重。
- **B5 誠實評估（9%）**：在 test set 算 **AUROC**（分數用 p(pneumonia) = `softmax(logits)[:, 1]` 的連續值，不可先切成 0/1；陽性為 label 1 = pneumonia）、**混淆矩陣**（門檻用 argmax，報告註明）與 precision/recall/F1。在 `mymetrics.py` 自己實作 `auroc(y_true, score)`（可用 A5 的配對法），再用 `sklearn.metrics.roc_auc_score` 核對兩者一致。做一張混淆矩陣圖，並列出「全部猜多數類」的 baseline accuracy。**討論：accuracy 和 AUROC 說的故事一樣嗎？**
- **B6 CAM（5%）**：對 2–3 張測試影像畫 **CAM** 熱圖，疊在原圖上。你的 B3 小 CNN 是「GAP + 線性頭」，正好是 10/12 教的 CAM 直接適用的架構（把送進 GAP 的最後特徵圖，用分類頭對應類別的權重加權即可）。**觀察：模型看的是病灶區還是別的地方？（連 10/12 CAM 頁：熱點是分數的證據，不是病灶的證明；捷徑學習與 Zech 2018 的跨院案例會在 11/23、11/30 兩講展開）**
  > 註：CAM 是 10/12 教的、需要 GAP+線性頭；Grad-CAM 是 11/23 評估與可解釋性那講對它的推廣（用梯度加權，不限架構）。本題用 CAM 即可，行有餘力可在加分題改用 Grad-CAM 比較兩者。

---

## Part C — 迷你專題：從零 vs 遷移（25%）

在 PneumoniaMNIST 上做一個對照實驗（想用 PathMNIST 可以，但它是 RGB、9 類，in_channels、n_cls 要改，AUROC 用 macro one-vs-rest，且務必開 GPU）：

1. **兩個模型**：① 你 B3 的小 CNN 從零訓練；② 一個 ImageNet 預訓練的輕量模型（如 `torchvision` 的 `resnet18(weights=...)`）做遷移學習。兩者用**相同資料、相同 epoch 數**。
   > 提示：ResNet 是為 224×224 三通道彩圖設計的。28×28 灰階要先處理：單通道 repeat 成 3 通道、像素轉成 [0,1] 浮點後用 ImageNet 的平均與標準差正規化，再用 `F.interpolate` 放大到 64×64 或 96×96。兩個模型都從同一份 28×28 資料出發，不要改載 `size=64` 的原生高解析版本，否則比較會混入解析度差異。ResNet 請整網微調（與 CheXNet 相同）。原因：ResNet 的 stem（7×7 stride 2 + maxpool）會先把 28×28 降到 7×7，之後每個 stage 再對半降，最後只剩 1×1，空間資訊所剩無幾，對照實驗的結論會失真。
2. **比較**：報兩者的 AUROC 與收斂速度（loss 曲線），做成一張對照表／圖。
3. **討論（200–400 字）**：遷移學習有沒有贏？預訓練在什麼情況下有幫助、什麼情況下效果有限（連 10/12「When transfer helps, and when to doubt it」頁：Raghu 等人只用 5,000 張時，準確率增益主要出現在大模型）？你的結果符合哪一種情況？你的 ResNet 是凍結 backbone 只訓分類頭，還是整網微調？兩者各適合什麼情況（連 10/12「Freeze or fine-tune」頁）？你的 CAM 有沒有暴露任何 shortcut？如果要當成 CheXNet 那樣的臨床工具，你還缺哪些驗證（連 10/5「Cross-validation」頁：交叉驗證取代不了另一家醫院的外部測試集；校準在 11/23 評估那講才講，可先列為待辦）？

---

## Part D — 逆向工程（15%，兩題各 7.5%）

**規則（五份作業共用）**：每題給你一組輸入與我們處理後的結果（`reverse/` 資料夾，格式見 `manifest.json`）。處理鏈與參數不公開，只公開「可能用到的操作家族」與搜尋刻度。你要推回我們做了什麼，寫成 `reproduce_D1(...)`、`reproduce_D2(...)` 兩個函式，對公開測例達到門檻；助教會用**你沒看過的保留測例**再呼叫一次，硬編碼公開答案不會過。報告要有：推定的處理鏈與參數、至少兩個假說的比較、支持判斷的圖或數字、以及你認為無法辨識的部分。

**D1 這張 CAM 是怎麼畫出來的 CAM Rendering Forensics（7.5%）**
給你我們訓好的教師小 CNN（`cnn.pt` 與公開定義 `teacher_cnn.py`：兩個卷積區塊、GAP、線性頭，最後卷積特徵是 14×14）、八張測試影像與標籤，以及我們畫出的 28×28 浮點 CAM 和疊圖。你要把同一張 CAM 算出來。要決定的事：用哪一類的線性頭權重（預測類？固定某類？）、要不要加線性頭的 bias、負值怎麼處理、正規化方式、正規化和放大的先後，以及放大的插值方式（候選見 manifest）。特徵一律用 `TeacherCNN.features(x)`（ReLU 後、GAP 前）。
- 門檻：每張 CAM 的 MAE ≤ 0.003 且 SSIM ≥ 0.98。
- 提示：被預測為陰性的那幾張影像上，我們的熱圖特別值得看。先正規化再放大，和先放大再正規化，結果差得剛好過不了門檻。放大一律以 `torch.nn.functional.interpolate` 的定義為準；用其他函式庫（例如 `skimage.transform.resize` 預設的 `mode='reflect'`）做同名插值，邊界數值會不同。
- 報告額外要求：比較「用預測類別」與「固定一類」兩種 CAM 在同一張影像上的差異；說明為什麼 CAM 把某區域畫亮，不能直接當成病灶定位正確的證據（連 10/12 CAM 頁）。

**D2 消失的訓練配方 Training Recipe from Weight Updates（7.5%）**
只訓線性頭。給你教師 CNN 對 64 張訓練影像算好的 GAP 特徵與標籤、線性頭的初始權重 `D2_head_init.pt`、更新後的 `D2_head_final.pt`，以及 21 個點的 `D2_loss.csv`（更新前一次、每步更新後一次）。已知每一步都用同一整批資料、更新次數固定 20 次、optimizer 是 SGD 家族、損失是交叉熵，可能有類別權重（候選比例 {1,2,3,4}）；學習率、momentum 與類別權重要你從曲線和權重位移推回來（搜尋刻度見 manifest）。momentum 與類別權重一律依 `torch.optim.SGD` 與 `nn.CrossEntropyLoss(weight=...)` 的定義（見 manifest）。
- 門檻：loss 曲線 RMSE ≤ 1e-4，且最終權重最大絕對誤差 ≤ 1e-3。
- 提示：loss 是一個數字，權重更新的方向包含更多資訊。用 Adam 可以把曲線調得很像，但權重位移的方向對不上。
- 報告額外要求：列出至少兩組「曲線相近但權重不同」的假說，說明你靠什麼證據排除；用你推出的配方在保留特徵上再跑一次，確認曲線與權重都能對上。

---

## 繳交規範

- **Notebook**：`HW2_學號.ipynb`，或 Colab 分享連結（權限「知道連結的任何人可檢視」）。確認 **Runtime → Run all** 可無錯跑完（含 GPU 關掉也能跑，只是慢一點）。
- **報告 PDF**：`HW2_學號.pdf`，含 Part A 計算與推導過程＋ Part C 對照表與討論。
- **工具庫**：更新後的 `mymetrics.py`（新增 AUROC）與 `mymodels.py`（你的網路定義）一併附上，HW3 會繼續用；分類器權重 `hw2_cnn.pt` 存 Drive，HW5 會用。
- **Part D**：`reproduce_D1(inputs)`、`reproduce_D2(features, labels, head_init_state_dict)` 兩個函式（簽名以 starter 與 manifest 為準）放在 notebook 最後一節，助教會用你沒看過的保留測例呼叫它們；報告含 Part D 的推理段落。
- 截止：11/2（一），時間以課程公告為準。

## 評分 Rubric

| 項目 | 配分 | 重點 |
|---|---|---|
| Part A 計算與推導 | 25% | 卷積/感受野/softmax/交叉熵/混淆矩陣/AUROC 觀念正確、過程完整 |
| Part B 程式 | 35% | 可執行、評估用 AUROC 而非只看 accuracy、每題有觀察、CAM 正確 |
| Part C 專題 | 25% | 對照實驗設計合理、有量化比較、討論觸及遷移學習與臨床驗證 |
| Part D 逆向工程 | 15% | 兩題各 7.5%：門檻達標 4%、推理與假說比較 2.5%、保留測例通過 1% |
| 加分 | +5% | 額外探索（多分類任務、class weight 處理不平衡、更多 CAM/Grad-CAM 分析） |

> **接續**：本作業的「載入資料 → 訓練 → 驗證 → 畫指標」骨架與 `mymetrics`（含 AUROC）會在 HW3 分割任務直接重用，只換 model 與 loss/metric（交叉熵→Dice）。
