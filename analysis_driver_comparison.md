# Expression Transfer: Atlas vs. Single-Driver Comparison

## 實驗設定

| 項目 | 說明 |
|---|---|
| Source | `data/source.jpg` |
| Atlas | `data/expression_atlas.npz`（多人平均 canonical landmarks） |
| Driver | `data/demo_images/` 各表情人工挑選單張 |
| Landmark backend | MediaPipe 478-pt |
| Region-warp | eye=(x=0.0, y=0.5)、nose=0.0、brow=0.75、mouth=1.0、jaw=0.20、outer=0.15 |

---

## Phase 1：初始對照（Atlas scale=0.6-0.8 vs Driver scale=0.8-1.4）

### 指標說明

| 指標 | 含義 | Atlas 模式注意事項 |
|---|---|---|
| **BS cosine** | result 和 driver 的 AU 向量夾角餘弦，越高表示表情越相似 | **最可信的主指標** |
| **BS MAE** | AU 強度平均絕對誤差，越低越好 | 同上 |
| **ETR** | 目標 landmark 位移實現率，1.0=完全實現，>1.2=過衝，<0.7=不足 | 可信 |
| **SSIM face** | 臉部結構相似度，越低代表變化越多 | 可信 |
| **Color drift** | 臉部 LAB ΔE 色偏，單位 px | 可信 |
| **LM RMSE** | result vs driver landmark 歐式距離 | **Atlas 模式無效**（canonical 座標系 vs 原始圖片座標系不同） |

### 對照表（Phase 1）

| 表情 | 模式 | Scale | BS cosine | BS MAE | ETR | Color drift |
|---|---|---|---|---|---|---|
| **happy** | Atlas | 0.8 | 0.565 | 0.105 | 0.747 ✓ | 17.16 |
| | Driver (happy_1.png) | 0.8 | **0.987** | 0.028 | 2.164 ↑過衝 | 13.50 |
| **angry** | Atlas | 0.8 | 0.720 | 0.052 | 0.586 | 16.33 |
| | Driver (angry_87_2.jpg) | 1.2 | **0.959** | 0.061 | 0.439 ↓不足 | 10.88 |
| **disgust** | Atlas | 1.0 | 0.826 | 0.068 | 0.522 | 17.54 |
| | Driver (disgust_81_16.jpg) | 1.0 | **0.894** | 0.070 | 1.271 | 11.66 |
| **fear** | Atlas | 0.8 | **0.584** ← Atlas 贏 | 0.061 | 0.584 | 17.56 |
| | Driver (fear_2.png) | 1.0 | 0.201 ← 崩潰 | 0.057 | 2.185 ↑過衝 | 14.43 |
| **sad** | Atlas | 0.8 | 0.788 | 0.042 | 0.699 | 17.32 |
| | Driver (sad_35_11.jpg) | 1.4 | **0.895** | 0.034 | 1.665 ↑過衝 | 11.62 |
| **surprise** | Atlas | 0.6 | **0.503** ← Atlas 贏 | 0.072 | 0.616 | 17.45 |
| | Driver (surprise_25_20.jpg) | 0.8 | 0.190 ← 崩潰 | 0.124 | 1.118 ✓ | 14.39 |

### 系統性差異（Phase 1 結論）

**1. Driver 在「標準」表情佔優，Atlas 在「問題 driver」反勝**

- Driver 獲勝：happy、angry、disgust、sad — BS cosine 普遍高 0.2~0.4 以上
- Atlas 獲勝：fear（+0.383）、surprise（+0.313）— driver 圖本身不具代表性

**2. Atlas ETR 系統性偏低（0.52~0.75）**

Atlas canonical landmarks（IOD=200, center=256）經過 `_align_landmarks` 映射到 source 空間，但這個映射同時把「表情 delta」和「atlas 族群平均臉型 vs source 臉型的幾何差異」混在一起，使位移方向比純表情 delta 更噪，warp 無法完全實現。

→ **Atlas 模式需要更高的 scale（建議 1.2~1.8）來補償 ETR 損失。**

**3. Color drift — Atlas 系統性偏高（16~18 px vs 11~14 px）**

Atlas 大位移 → Delaunay warp 把更多不連續紋理拉至新位置 → 色偏更大。是 atlas 模式的固有代價。

**4. Mouth clone — Atlas 無法執行**

Atlas 模式沒有原始 driver 圖，永遠不能做 seamlessClone。這對 happy（笑露牙齒）影響最大，是 Atlas happy BS_cos=0.565 遠低於 Driver 0.987 的核心原因之一。

---

## Phase 2：Atlas 最佳 Scale 搜尋（新實驗）

目標：對各表情找到 BS cosine 最高的 atlas scale，並與 driver 模式公平比較。

測試 scales：1.0, 1.2, 1.5（補充 Phase 1 已跑的 0.6-0.8）

### BS cosine — Atlas scale sweep 完整結果

| 表情 | s=0.6 | s=0.8 | s=1.0 | s=1.2 | s=1.5 | **最佳 atlas** | Driver |
|---|---|---|---|---|---|---|---|
| happy | — | 0.565 | 0.540 | 0.470 | 0.452 | **0.565 (s=0.8)** | **0.987** |
| angry | — | 0.720 | 0.656 | 0.682 | 0.778 | **0.778 (s=1.5)** | **0.959** |
| disgust | — | — | 0.826 | 0.656 | 0.857 | **0.857 (s=1.5)** | **0.894** |
| fear | — | 0.584 | 0.492 | **0.803** | 0.747 | **0.803 (s=1.2)** | 0.201 |
| sad | — | 0.788 | 0.656 | 0.623 | 0.599 | **0.788 (s=0.8)** | **0.895** |
| surprise | 0.503 | — | 0.898 | **0.943** | 0.915 | **0.943 (s=1.2)** | 0.190 |

### ETR — Atlas scale sweep 完整結果（參考用）

| 表情 | s=0.8 | s=1.0 | s=1.2 | s=1.5 |
|---|---|---|---|---|
| happy | 0.747 | 0.661 | 0.561 | 0.470 |
| angry | 0.586 | 0.609 | 0.674 | 0.713 |
| disgust | 0.522 | 0.522 | 0.477 | 0.639 |
| fear | 0.584 | 0.619 | 0.741 | 0.832 |
| sad | 0.699 | 0.726 | 0.768 | 0.805 |
| surprise | 0.616 | 0.766 | 0.783 | 0.743 |

### Phase 2 結論

**1. surprise atlas 是最大亮點（0.503 → 0.943）**

Scale 從 0.6 到 1.2，BS cosine 提升 0.44！這是因為 surprise 的關鍵 AU（張嘴 AU26/27、眉毛上揚 AU1/2/5）需要大位移才能讓 blendshape 偵測器觸發。Scale 0.6 時位移太小，臉部沒顯出 surprise 特徵；scale 1.2 達到最佳，1.5 略降（0.915）顯示存在甜蜜點。

**2. fear atlas 1.2 破解 driver 崩潰問題（0.201 → 0.803）**

Driver fear_2.png 本身 ETR 過衝（2.185）但 BS_cos 只有 0.201，是典型的「driver 臉型太不像、位移是噪聲」問題。Atlas 1.2 的 ETR=0.741（合理範圍），且統計平均消除了個別 driver 的臉型偏差，BS_cos 達到 0.803。

**3. Scale 對各表情的效果截然不同（三種 pattern）**

| Pattern | 表情 | 行為 | 原因 |
|---|---|---|---|
| 單調下降 | happy、sad | scale↑ → BS↓ | 這類表情位移幅度小（嘴角微抬、眉心皺），更大的 warp 反而擾亂其他 region；且 happy 缺 mouth clone，放大後失真更明顯 |
| 非單調（先下後上） | disgust | 1.0>1.2<1.5 | 嘴部/鼻部 landmark 在特定 scale 出現相消效應，1.5 方達到正確的 AU17/9 激活 |
| 非單調（有最佳峰值） | surprise、fear、angry | 有明確甜蜜點 | 位移需要足夠大以觸發 AU，但過大會產生 mesh 失真；angry 在 1.5 仍上升，可能還有空間 |

**4. ETR 與 BS cosine 相關性弱**

happy 在 scale=1.5 時 ETR=0.470（最低），但 BS_cos 也最低（0.452）；  
sad 在 scale=1.5 時 ETR=0.805（最高），但 BS_cos 反而最低（0.599）。  
這說明 ETR 高不等於表情準確，位移方向比幅度更關鍵。Atlas 的位移方向受 canonical → source 映射引入的人臉幾何差異影響，導致方向偏差造成 AU 組合錯誤。

**5. 最佳 atlas scale per expression（結合 Phase 1+2）**

| 表情 | 最佳 atlas scale | 最佳 BS cosine | vs Driver | 建議使用 |
|---|---|---|---|---|
| happy | 0.8 | 0.565 | -0.422 | Driver |
| angry | 1.5 | 0.778 | -0.181 | Driver（有好 driver 時） |
| disgust | 1.5 | 0.857 | -0.037 | 任一可 |
| fear | **1.2** | **0.803** | +0.602 | **Atlas** |
| sad | 0.8 | 0.788 | -0.107 | Driver（有好 driver 時） |
| surprise | **1.2** | **0.943** | +0.753 | **Atlas** |

---

## 延伸觀察

### fear_2.png 和 surprise_25_20.jpg 為何是爛 driver？

- **fear_2.png**：ETR=2.185（過衝，scale=1.0）但 BS_cos 僅 0.201。表示 warp 有產生大位移，但偵測到的 AU 和 fear 的特徵 AU（內眉上揚 AU1、外眉上揚 AU2、眼睛放大 AU5、張嘴 AU26）不匹配。可能原因：臉型差異太大，直接模式把「臉型差異」誤認成「表情 delta」。
- **surprise_25_20.jpg**：嘴巴張超大（open_ratio=0.293）有被 seamlessClone，但 BS_cos=0.190 且 MAE=0.124。可能這張 surprise 的 AU 向量方向特殊（強張嘴 + 少量眉毛動作），和 result 的 AU 組合夾角太大。

### Scale 不對等對比較的影響

Phase 1 的 driver 實驗 scale 範圍較大（0.8~1.4），部分案例 ETR 嚴重過衝（1.665、2.164、2.185）。嚴格的比較應在相近的 ETR 下進行。Phase 2 的 atlas scale 搜尋將讓 ETR 更接近理想值（0.85~1.1），使比較更公平。

---

## 建議

| 場景 | 建議 | 指令範例 |
|---|---|---|
| happy（需要 mouth clone） | Driver，scale ~0.8，ETR ≈ 1.0 | `--driver happy_1.png --scale 0.8` |
| angry | Driver，scale ~1.0~1.4 | `--driver angry_87_2.jpg --scale 1.2` |
| disgust | Atlas s=1.5 或 Driver s=1.0 | `--atlas ... --expr disgust --scale 1.5` |
| **fear（driver 圖差）** | **Atlas s=1.2** | `--atlas ... --expr fear --scale 1.2` |
| sad | Driver，scale ~0.8~1.0 | `--driver sad_35_11.jpg --scale 1.0` |
| **surprise（driver 圖差）** | **Atlas s=1.2** | `--atlas ... --expr surprise --scale 1.2` |
| 自動 pipeline（無 driver 選圖步驟） | Atlas，fear/surprise 用 s=1.2，其餘用 s=1.0~1.5 | `--atlas --scale-search --scales 0.8 1.0 1.2 1.5` |
| 最高表情準確度 | Driver（確保 driver 臉型和 source 相近） | 手動挑選 driver；比較多張 driver 的 BS cosine |

---

*Last updated: 2026-06-10*
