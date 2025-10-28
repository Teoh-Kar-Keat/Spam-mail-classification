# HW3 — SMS 簡訊垃圾郵件分類器（Logistic Regression）

本儲存庫為一個以 OpenSpec 為流程指引之作業專案，實作了一個簡訊（SMS）垃圾郵件分類器，採用 TF-IDF 與 Logistic Regression 的機器學習管線，並包含前處理、訓練、推論腳本與一個以 Streamlit 為介面的示範應用以便部署與檢視。

目錄（檔案摘要）：
- `preprocessing.py` — CSV 前處理與正規化流程（文字清理、遮罩 token 等）。
- `train.py` — 訓練 TF-IDF + Logistic Regression 管線並匯出模型工件。
- `predict.py` — 針對單一訊息或 CSV 批次進行推論並輸出機率或標籤。
- `app.py` — Streamlit 示範應用（互動式儀表板）。
- `requirements.txt` — Python 相依套件清單。
- `sms_spam_no_header.csv` — 範例或原始資料（使用者需放置於專案根目錄）。
-  `sms_spam_clean.csv` - 清洗後的資料集
-  `hw3_spam_classification.ipynb` - 資料前處理、訓練、和儲存模型之完整流程
-  `Streamlit app` -  https://spam-mail-classification-hw3.streamlit.app/

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
  
使用 OpenSpec 管理本作業
---------------------------------

本專案以 OpenSpec 作為規格與變更流程控制工具。以下為建議的實作與協作流程，能讓作業變更、CI 驗證、與審查更具可追溯性與一致性。

核心原則

- 所有影響專案行為或介面的變更（包括前處理規則、模型工件格式、API/介面變更、或 CI 設定），應以 OpenSpec 變更提案（change proposal）描述、提交與審查。
- 變更提案與規格存放於 `openspec/` 標準目錄：`openspec/project.md`、`openspec/specs/`、以及 `openspec/changes/`。

快速上手（本地開發）

1. 撰寫變更提案目錄：

	 - 建立目錄 `openspec/changes/<YYYY-MM-DD>-short-name/`。
	 - 在該目錄新增 `proposal.md`（摘要、目的、影響範圍、關聯檔案）與 `tasks.md`（要做的步驟）。

	 範例資料夾結構：

	 ```text
	 openspec/
		 changes/
			 2025-10-28-add-openspec-ci/
				 proposal.md
				 tasks.md
		 specs/
			 ci/spec.md
		 project.md
	 ```

2. proposal.md 範本（建議欄位）

   	範例（proposal.md）：

	 ```md
	Follow instructions in openspec-proposal.prompt.md.
	im doing HW3 , the aim of this project is to classify spam email and ham email, using logistic regression machine learning method, i seperated in to 5 different phase to done my homework:
	
	phase 1 will be the preproceesing process create a preprocessing.py, which try to clean up the data set D:\test\HW3\sms_spam_no_header.csv and save the clean dataset in D:\test\HW3\sms_spam_clean.csv
	pahse 2 create a train.py and predict.py train and predict the clean data set using logistic regression and save the model.
	phase 3 according the work before, create a complete ipynb file (from preproccesing to predict), to talk about how to using logistic regression to train a spam email classifiy model using logistic regression. Beside that you also need to visuallise data like data overview(class distribution, token replacements in cleaned text(approximate)),top tokens by class, model performance(text), ROC, Precision-Recall , threshold sweep(precision/recall/f1）, and so on, just wrote anything you think is important)), For, Each code cell, need a markdown cell before it to explain what actually the code cell done, please wrote it in traditional chinese, and executed every cell .
	phase 4 will be deployment, i will like to create a app.py and push to github, and using streamlit.app to demo, the left side will have some parameter that i can adjust, include (dataset CSV, label colomn, text column, models dir, textsize, seed, decision threshold,)and need display data data overview(class distribution, token replacements in cleaned text(approximate)),top tokens by class +graph , model performance(text)confusion matrix, , ROC, Precision-Recall , threshold sweep(precision/recall/f1, live inference (with 2 button, use spam example, use ham example, a box to fill in message, and a predict button), it will display a spam probalility graph after after predict.
	phase 5 wrote a complete report with title recoreding how im using openspec to done a spam mail classified streamlit app
	 ```
  
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
- Confusion Matrix (混淆矩陣)
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

混淆矩陣
<img width="652" height="473" alt="af50b376-2f3e-47a3-aec9-482f20e34d9a" src="https://github.com/user-attachments/assets/e45139dd-0a4a-44bc-a0a2-2f3cc6c46ab2" />

ROC 與 PR 曲線
<img width="695" height="473" alt="a4ef9b5f-b34e-4a98-845d-11e66d18c9c9" src="https://github.com/user-attachments/assets/0b501e16-6bee-4ba2-8dba-f624eef2be1e" />
<img width="695" height="473" alt="629be7f4-a774-4940-a529-a6c0f67898dc" src="https://github.com/user-attachments/assets/cf00d5f4-5983-46cf-bd23-a836eb8f6aa0" />

ROC 曲線可展示模型整體可分離性；PR 曲線在類別不平衡時更具參考價值。我們同時使用 AUC 指標作為補充證據。


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
<img width="1906" height="912" alt="image" src="https://github.com/user-attachments/assets/c9eaf999-e7df-46dc-92eb-61d8b6bcbc26" />


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

---

End of README
