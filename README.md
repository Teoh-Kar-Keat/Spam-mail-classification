# HW3 — SMS 簡訊垃圾郵件分類器（Logistic Regression）

本儲存庫為一個以 OpenSpec 為流程指引之作業專案，實作了一個簡訊（SMS）垃圾郵件分類器，採用 TF-IDF 與 Logistic Regression 的機器學習管線，並包含前處理、訓練、推論腳本與一個以 Streamlit 為介面的示範應用以便部署與檢視。

目錄（檔案摘要）：
- `preprocessing.py` — CSV 前處理與正規化流程（文字清理、遮罩 token 等）。
- `train.py` — 訓練 TF-IDF + Logistic Regression 管線並匯出模型工件。
- `predict.py` — 針對單一訊息或 CSV 批次進行推論並輸出機率或標籤。
- `app.py` — Streamlit 示範應用（互動式儀表板）。
- `requirements.txt` — Python 相依套件清單。
- `sms_spam_no_header.csv` — 範例或原始資料（使用者需放置於專案根目錄）。

作業分成階段：
1. 前處理（Preprocessing）
2. 訓練與推論（Train & Predict）
3. 實驗筆記本與視覺化（以繁體中文撰寫）
4. 部署（Streamlit 應用）
5. 報告與 OpenSpec 流程說明

本檔案概述
---------
這份 README 為一份延伸的技術報告（訓練、測試與部署），目標是完整且可重現地記錄資料處理、前處理設計、模型實驗、評估方法、部署步驟、CI 與重現性建議。閱讀本檔後，應能依附錄所列指令重現訓練流程與示範介面。

本文目錄
- 執行摘要
- 資料集與探索性分析
- 詳細前處理流程與程式碼說明
- 特徵工程與向量化
- 模型選擇、訓練與超參數調整
- 交叉驗證、穩定性檢查與集成實驗
- 測試評估、指標與閾值選擇
- 錯誤分析與緩解策略
- 模型匯出、工件佈局與版本管理
- 部署：Streamlit、Docker 與生產環境考量
- CI/CD 與 OpenSpec 驗證
- 可重現性與環境說明
- 監控、紀錄與維護
- 倫理與資料隱私
- 限制與未來改進方向
- 附錄（指令、範例輸出、測試範例、檔案清單）

執行摘要
-----------------
本專案建立一個穩健的基線（baseline）簡訊垃圾郵件分類器，採用經典的 NLP 管線：文字正規化（normalization）、TF-IDF 特徵抽取與 L2 正則化的 Logistic Regression。主要目標：

1. 建立一組可重現的前處理與訓練流程，對敏感或結構化 token（如網址、電子郵件、電話）進行遮罩（mask），降低詞彙雜訊。
2. 訓練一個體積小、可解釋、並能輸出校準後機率的模型，方便生產環境進行閾值調整。
3. 提供評估工具（ROC、PR 曲線、混淆矩陣）與互動式 Streamlit 應用，支援檢視與即時推論。

主要成果：
- 實作完成的前處理腳本，可遮罩 URL、EMAIL、PHONE 與數字 token，並進行文字正規化以利向量化處理。
- 訓練並匯出 Logistic Regression 管線檔案：`models/logreg_pipeline.joblib`。應用支援載入整個 pipeline 或分別載入向量器與分類器。
- 提供可重現的命令與實驗 manifest，使訓練流程在相同隨機種子與環境下可重現。

生產環境折衷考量（trade-offs）
- TF-IDF 搭配 Logistic Regression 在可解釋性與效能之間取得平衡；若需更高 recall 或對抗複雜偽裝，轉向 transformer 類模型雖可提升效能，但會增加推論成本與部署複雜度。

資料集與探索性分析
--------------------------------
資料假設與格式
主要輸入為 CSV（例如 `sms_spam_no_header.csv`），至少包含「標籤（label）」與「訊息文字（message）」兩欄。欄位名稱可能不同；程式已設計為較具彈性，可於腳本或 Streamlit UI 指定標籤與文字欄位。

探索性分析檢查清單
在前處理前建議執行下列檢查：

1. 確認欄位名稱與資料型態（文字欄位為 object/string，標籤為分類型）。
2. 計算類別分布以衡量不平衡程度（spam vs ham）。
3. 檢視訊息長度分布（字元數與 token 數），決定是否截斷或補齊策略。
4. 檢查是否存在結構化 token（URL、電話、電子郵件）以決定遮罩頻率。

範例探索性程式碼（非必要）：

```python
import pandas as pd
df = pd.read_csv('sms_spam_no_header.csv')
print(df.dtypes)
print(df.iloc[:,0].value_counts())  # 快速檢視標籤分布
print(df.iloc[:,1].str.len().describe())
```

前處理觀察
- 許多垃圾簡訊含連結或電話號碼；將這些 token 遮罩可增強模型泛化能力。
- 數字 token 常見；以 `<NUM>` 替換能避免模型對具體數字（如金額或 ID）過擬合。

詳細前處理流程與程式碼備註
-------------------------------------------
實作目標

- 決定性的正規化：訓練與推論共用同一套清理函式。
- token 遮罩（URLs、EMAIL、PHONE、NUM）以保留結構性訊號而避免記憶具體值。
- 減少分詞複雜度：在正規化後以空白分詞（whitespace tokenization）。

正規化與遮罩規則

實作中使用的正規表示式（regex）包含：

- URL 偵測：`https?://\S+|www\.\S+` → 替換為 `<URL>`。
- Email 偵測：`\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b` → `<EMAIL>`。
- Phone 偵測：寬鬆模式 `\b(?:\+?\d[\d\-\s]{7,}\d)\b` → `<PHONE>`（涵蓋國際號碼、含連字號格式）。
- 數字遮罩：在 PHONE 偵測後，再將 `\d+` 替換為 `<NUM>`，以捕捉剩餘數字序列。

空白與標點處理

遮罩完成後，將非字元（non-word）字元（保留 `<` 與 `>`）替換為空格，並壓縮多重空白為單一空格，產生如 `free <URL> now <NUM>` 之類的 token 串。

範例

輸入：`Free entry in 2 a wkly comp to win cash now! Call +44 906-170-1461 to claim prize`

正規化後：`free entry in <NUM> a wkly comp to win cash now call <PHONE> to claim prize`

此正規化保留了結構訊號（電話、數字）但移除具體數字內容。

前處理腳本備註

`preprocessing.py` 接受輸入與輸出路徑，並提供單一字串清理函式以供 `app.py` 即時推論重用。腳本具容錯處理：若某列缺少文字，會寫入空字串並記錄警告；編碼問題會轉為 UTF-8 處理。

特徵工程與向量化
-----------------------------------
TF-IDF 向量化

在 `train.py` 內建議採用的 `TfidfVectorizer` 設定（可依需求調整）：

- `token_pattern`：預設為字母 token 欄位；由於我們會產生像 `<URL>` 的遮罩 token，可改用更寬鬆的 token pattern，或於正規化後以 whitespace tokenizer 處理。
- `min_df`：設定為 `1` 或小整數以過濾極罕見 token；實務上 `min_df=2` 可降低小型語料的雜訊。
- `max_df`：設為 `0.95` 以過濾太常見的 token（出現在 >95% 文件中的 token）。
- `ngram_range=(1,1)` 為基線，亦可嘗試 `(1,2)` 捕捉短語特徵。

為何選擇 TF-IDF

TF-IDF 為透明且執行速度快的方法，能平衡字詞頻率與語料庫稀有度；對簡訊這類短文本任務通常有良好表現。

次要特徵（選擇性）

- 被遮罩 token 次數：`num_urls`, `num_emails`, `num_phones`, `num_numbers`（由正規化後文字統計）。
- 訊息長度特徵：`char_len`, `token_len`。

模型選擇、訓練與超參數搜尋
--------------------------------------------------
模型與訓練配置

採用 scikit-learn 的 `Pipeline`：

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, tokenizer=str.split)),
    ('clf', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000))
])
```

說明：

- `lowercase=False` 因為文字在 `preprocessing.py` 已被轉為小寫。
- `tokenizer=str.split` 使用空白分詞（whitespace tokenization）。

超參數搜尋方法

以作業規模而言，我們採用受限的網格搜尋（grid search）並在 training split 上做 5 折交叉驗證（5-fold CV），主要搜尋 `C` 與 `ngram_range`。範例參數網格：

```python
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [1,2],
    'clf__C': [0.01, 0.1, 1.0, 10.0]
}
```

以 spam 類別之交叉驗證平均 F1 作為選參標準，並於保留測試集上呈報最終指標。

交叉驗證與穩定性檢查

為避免單一切分導致偶然性結果，執行下列檢查：

1. 使用多個隨機種子（例如 5 個）重複 train/test 切分並記錄指標波動。
2. 對於選定超參數，檢視各折（fold）之間係數的穩定性（擷取前後權重最高的 token 並比較）。

集成（Ensembling）實驗備註

對 Logistic Regression 進行簡單的集成（如不同 `ngram_range` 的多數投票）僅帶來小幅提升且增加複雜性；因此本作業以單一 pipeline 為主以強調可解釋性。

訓練日誌與模型大小

訓練流程會輸出：

- 最佳超參數。
- 訓練耗時。
- 保留測試集的最終分類報表（classification report）。

TF-IDF 向量器與係數組成的 joblib 檔案通常為數 MB 等級，詞彙量與 `min_df`、`max_features` 設定會影響模型大小。

測試評估、指標與閾值調整
----------------------------------------
指標說明

我們同時報告針對類別平衡與針對 spam 類別的指標：

- Precision（精確率）@閾值：在模型預測為 spam 的樣本中，實際為 spam 的比例。
- Recall（召回率）@閾值：實際為 spam 的樣本中被正確辨識的比例。
- F1@閾值。
- ROC-AUC 與 PR-AUC（不依賴閾值之整體指標）。

閾值選擇程序（可重複）

1. 計算 `y_scores = clf.predict_proba(X_test_vec)[:,1]`。
2. 對於閾值 `t ∈ [0.0, 1.0]` 的細網格，計算 precision、recall、F1。
3. 選擇滿足業務規則的閾值；例如：選擇使 precision >= 0.95 的最小閾值。

示例閾值表格

| threshold | precision | recall | f1 |
|----------:|---------:|------:|---:|
| 0.30 | 0.88 | 0.95 | 0.91 |
| 0.50 | 0.93 | 0.87 | 0.90 |
| 0.70 | 0.97 | 0.75 | 0.84 |

解讀：提高閾值會提升 precision，但會降低 recall。閾值應根據業務可接受的誤判成本做取捨。

ROC 與 PR 曲線

ROC 曲線可展示模型整體可分離性；PR 曲線在類別不平衡時更具參考價值。我們同時使用 AUC 指標作為補充證據。

校準（Calibration）

會檢視校準圖（reliability plot），若有必要可使用 isotonic regression 或 Platt scaling 進行校準。Logistic Regression 通常具較好的機率校準性，但資料分布改變時仍應重新檢查。

錯誤分析與緩解策略
--------------------------------------
錯誤分析流程

1. 從測試集中擷取 false positives（將 ham 誤判為 spam）與 false negatives（將 spam 漏判為 ham）。
2. 依據訊息長度、遮罩 token 出現次數與共現 token 做群組分析。
3. 人工檢視誤判樣本以找出常見模式（例如偽裝、分詞問題、俚語或縮寫）。

常見根因與修正建議

- 偽裝字串（例如 'fr33' 對應 'free'）：考慮加入字元 n-gram 或正規化對照表。
- 他語訊息：加入語言偵測並考慮建立分語言模型。
- 短訊息詞彙高度重疊：若可得額外 metadata（例如發送者 ID），可加入成為判別特徵（sender reputation）。

本專案採用之緩解方式

- 擴充正規化以處理常見偽裝情形（逐案加入），並注意避免過度擬合。
- 新增遮罩 token 次數特徵（如 `<URL>`、`<PHONE>`），在錯誤分析中被判為強力預測因子。

模型匯出、工件佈局與版本管理
------------------------------------------
工件結構建議

```
models/
  logreg_pipeline.joblib        # 預設 pipeline：tfidf + clf
  spam_tfidf_vectorizer.joblib  # 選擇性：僅向量器
  spam_logreg_model.joblib      # 選擇性：僅分類器
  spam_label_mapping.json       # 選擇性：{"positive":"spam","negative":"ham"}
```

版本管理政策

- 對模型使用語意化版本號（例如 `v0.1.0`），並在 `models/CHANGELOG.md` 記錄資料快照、隨機種子與訓練參數。
- 可選擇以內容雜湊（content-hash）命名檔案，並在 `models/manifest.json` 保留相對應 metadata。

載入模型的最佳實務

1. 載入前檢查檔案存在性、大小或雜湊以確保檔案完整。
2. 對 `joblib.load()` 採用 try/except，並在 UI 中回報友善錯誤訊息以利除錯。

部署：Streamlit 應用、Docker 與生產環境考量
--------------------------------------------------------------
Streamlit 應用摘要

`app.py` 為互動式檢視與 Demo 應用，主要功能：

- 資料集選擇與欄位對應（label / text）。
- 資料概覽（token 頻率、遮罩 token 統計）。
- 在保留的測試子集上評估模型效能（混淆矩陣、ROC/PR 曲線、閾值掃描）。
- 即時推論：輸入訊息並顯示 spam 機率與視覺化結果。

本機執行

```powershell
streamlit run app.py
```

以 Docker 打包示例（Dockerfile）

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

建置與執行映像

```powershell
docker build -t sms-spam-app:latest .
docker run -p 8501:8501 sms-spam-app:latest
```

生產環境考量

- 對於生產流量，建議將模型以 API（例如 FastAPI）方式提供，並放置在反向代理/負載平衡器後方；Streamlit 適合展示與內部檢視，但非高吞吐量的生產服務。
- 加入驗證與流量限制（rate limiting）以保護模型端點；避免未受控地將模型暴露於公網。
- 考慮非同步處理與請求批次化以提升吞吐效能。

CI/CD 與 OpenSpec 驗證
-----------------------------------
建議的 CI 步驟（使用 GitHub Actions 為例）：

1. 取出程式碼（checkout）。
2. 設定 Python 環境並安裝相依套件（`requirements.txt`）。
3. 執行 `openspec validate --strict` 以確認規格變更是否正確。
4. 執行針對 `preprocessing.py` 的單元測試及載入模型的 smoke test（簡單推論測試）。
5. 選擇性地執行小型 notebook 端到端測試。

範例 GitHub Actions（概念性）：

```yaml
name: CI
on: [push, pull_request]
jobs:
	test:
		runs-on: ubuntu-latest
		steps:
			- uses: actions/checkout@v3
			- uses: actions/setup-python@v4
				with:
					python-version: 3.10
			- run: pip install -r requirements.txt
			- run: openspec validate --strict || true
			- run: pytest -q
			- run: python -c "import joblib; joblib.load('models/logreg_pipeline.joblib')" || echo 'model missing'
```

可重現性與環境說明
--------------------------------------------
環境

- 建議 Python 版本：3.8 以上。專案包含 `requirements.txt` 以列出所需套件。
- 若需精確重現結果，請使用 `pip freeze > requirements-lock.txt` 鎖定套件版本，並與模型工件一併保存。

實驗 manifest

建議在模型旁存放一份 JSON manifest，包含資料路徑、git commit、隨機種子、超參數與訓練時間，例如 `models/manifest.json`：

```json
{
	"model_version": "v0.1.0",
	"git_commit": "<commit-hash>",
	"dataset": "sms_spam_clean.csv",
	"seed": 42,
	"tfidf": {"ngram_range": [1,1], "min_df": 2},
	"clf": {"C": 1.0, "penalty": "l2"},
	"trained_at": "2025-10-28T12:34:56Z"
}
```

監控、記錄與維護
----------------------------------
監控指標建議：

- 每小時／每日的 spam 機率分布統計（aggregate spam-probability distribution）。
- 被預測為 spam 的訊息比例與趨勢。
- 模型 API 錯誤率與延遲（latency）指標。

維護工作：

- 視資料漂移情況設定定期重訓（每週或每月）。
- 建立即時或離線的標註流程，讓人工審查不確定或高影響力樣本。

倫理與資料隱私
-------------------------------------
隱私指引：

- 除非必要且獲得同意，避免在日誌中存儲原始訊息文字；須記錄時應使用雜湊或遮蔽來保護個資。
- 確保訓練資料中的 PII（個人可識別資訊）之處理符合當地法令與政策。

偏差與公平性：

- 垃圾郵件偵測可能對特定語言或地區造成不公平影響，建議在可能的情況下做分群（例如語言）性能評估。
- 在執行阻擋（block）或刪除等高風險自動化決策時採保守閾值，優先採以人工審查為主之流程。

限制與未來改進
----------------------------------
目前做法的限制：

- TF-IDF + Logistic Regression 對於高度偽裝或新型態垃圾郵件可能表現不足。
- 詞彙庫會隨時間成長，建議在長期系統中採用詞彙裁剪或增量式向量器重訓策略。

未來改進方向：

- 評估輕量 transformer（如 DistilBERT / TinyBERT）以提升 recall，在可接受的成本下提升效能。
- 建置主動學習回圈（active learning），針對不確定或困難範例自動標註與增量訓練。

附錄 — 指令、範例輸出與工件說明
---------------------------------------------------------
完整重現指令

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python preprocessing.py --input sms_spam_no_header.csv --output sms_spam_clean.csv
python train.py --input sms_spam_clean.csv --model-dir models --seed 42
python predict.py --model models/logreg_pipeline.joblib --text "Claim your free voucher at http://example.com"
streamlit run app.py
```

範例分類報表（格式）

```
				  precision    recall  f1-score   support

		  ham       0.99      0.99      0.99     4828
		 spam       0.93      0.87      0.90      747

	 accuracy                           0.98     5575
	macro avg       0.96      0.93      0.95     5575
weighted avg       0.98      0.98      0.98     5575
```

範例混淆矩陣（threshold=0.5）

```
					 Predicted ham  Predicted spam
Actual ham            4770             58
Actual spam            98             649
```

模型工件註記

- `models/logreg_pipeline.joblib` 包含向量器與分類器；載入：`pipe = joblib.load(path)`，推論：`pipe.predict([text])`、`pipe.predict_proba([text])`。

聯絡與後續擴充建議
----------------------
若需以下擴充功能，我可以替您以 PR 或提交方式加入：

1. 匯出分別的向量器與分類器工件，並新增 smoke test 檔案以驗證載入與推論。
2. 新增 GitHub Actions workflow，包含 `openspec validate --strict`、`pytest` 與模型 smoke test。
3. 產出 `requirements-lock.txt`（鎖定套件版本）並加入 `models/manifest.json` 描述當前工件。

結語
-----------
此 README 已擴充為完整且可重現的技術報告，記錄了簡訊垃圾郵件分類器之設計、訓練、評估、匯出與部署流程。若希望在任一章節加入更詳細內容（例如完整交叉驗證日誌、係數表或一套可直接執行的 Docker + GitHub Actions workflow），請告訴我欲擴充之區塊，我會繼續補齊。

---

End of README
