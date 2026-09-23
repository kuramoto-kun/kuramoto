import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from datetime import datetime, timedelta

# --- 設定値（GitHubのSecretsから環境変数として安全に読み込みます） ---
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- 東証プライム 主要100銘柄 ＋ 各セクター代表銘柄（計200銘柄） ---
top200_tickers = {
    # 【主要大型・時価総額上位トップ100】
    "7203.T": "トヨタ自動車", "6758.T": "ソニーグループ", "9984.T": "ソフトバンクグループ", "8306.T": "三菱UFJフィナンシャル・グループ", "9432.T": "日本電信電話 (NTT)",
    "6861.T": "キーエンス", "8035.T": "東京エレクトロン", "4385.T": "メルカリ", "6501.T": "日立製作所", "6902.T": "デンソー",
    "4063.T": "信越化学工業", "7974.T": "任天堂", "6098.T": "リクルートホールディングス", "4502.T": "武田薬品工業", "8411.T": "みずほフィナンシャルグループ",
    "6920.T": "レーザーテック", "9101.T": "日本郵船", "5401.T": "日本製鉄", "7011.T": "三菱重工業", "2914.T": "日本たばこ産業 (JT)",
    "8316.T": "三井住友フィナンシャルグループ", "8473.T": "SBIホールディングス", "6762.T": "TDK", "7741.T": "HOYA", "6367.T": "ダイキン工業",
    "4543.T": "テルモ", "4568.T": "第一三共", "6981.T": "村田製作所", "6594.T": "ニデック", "9020.T": "東日本旅客鉄道 (JR東日本)",
    "7267.T": "本田技研工業", "6503.T": "三菱電機", "8801.T": "三井不動産", "8802.T": "三菱地所", "8766.T": "東京海上ホールディングス",
    "3382.T": "セブン＆アイ・ホールディングス", "8001.T": "伊藤忠商事", "8031.T": "三井物産", "8053.T": "双日", "8058.T": "三菱商事",
    "4661.T": "オリエンタルランド", "6301.T": "小松製作所", "7012.T": "川崎重工業", "9107.T": "川崎汽船", "9104.T": "商船三井",
    "5020.T": "ENEOSホールディングス", "3407.T": "旭化成", "4188.T": "三菱ケミカルグループ", "4901.T": "富士フイルムホールディングス", "4452.T": "花王",
    "4519.T": "中外製薬", "4578.T": "大塚ホールディングス", "6273.T": "SMC", "6146.T": "ディスコ", "6857.T": "アドバンテスト",
    "6723.T": "ルネサスエレクトロニクス", "6526.T": "ソシオネクスト", "6971.T": "京セラ", "7751.T": "キヤノン", "6702.T": "富士通",
    "6701.T": "日本電気 (NEC)", "7272.T": "ヤマハ発動機", "7832.T": "バンダイナムコホールディングス", "7936.T": "アシックス", "9735.T": "セコム",
    "4689.T": "LINEヤフー", "9433.T": "KDDI", "9434.T": "ソフトバンク", "9501.T": "東京電力ホールディングス", "9503.T": "関西電力",
    "9531.T": "東京ガス", "9022.T": "東海旅客鉄道 (JR東海)", "9021.T": "西日本旅客鉄道 (JR西日本)", "9202.T": "ANAホールディングス", "9301.T": "三菱倉庫",
    "8267.T": "イオン", "3038.T": "神戸物産", "9843.T": "ニトリホールディングス", "9983.T": "ファーストリテイリング", "3099.T": "三越伊勢丹ホールディングス",
    "8591.T": "オリックス", "8725.T": "MS&ADインシュアランス", "8750.T": "第一生命ホールディングス", "8830.T": "住友不動産", "1925.T": "大和ハウス工業",
    "1928.T": "積水ハウス", "1801.T": "大成建設", "1802.T": "大林組", "1812.T": "鹿島建設", "5201.T": "AGC",
    "5332.T": "TOTO", "5713.T": "住友金属鉱山", "5711.T": "三菱マテリアル", "5802.T": "住友電気工業", "6326.T": "クボタ",

    # 【各セクター代表銘柄（追加100銘柄）】
    "1332.T": "日本水産 (ニッスイ)", "1605.T": "INPEX", "1963.T": "日揮ホールディングス", "2002.T": "日清製粉グループ本社", "2269.T": "明治ホールディングス",
    "2282.T": "日本ハム", "2502.T": "アサヒグループホールディングス", "2503.T": "キリンホールディングス", "2801.T": "キッコーマン", "2802.T": "味の素",
    "3101.T": "東洋紡", "3103.T": "ユニチカ", "3401.T": "帝人", "3405.T": "クラレ", "3861.T": "王子ホールディングス",
    "3863.T": "日本製紙", "4005.T": "住友化学", "4042.T": "東ソー", "4208.T": "UBE", "4272.T": "日本化薬",
    "4503.T": "アステラス製薬", "4507.T": "塩野義製薬", "4523.T": "エーザイ", "4911.T": "資生堂", "5019.T": "出光興産",
    "5108.T": "ブリヂストン", "5191.T": "住友ゴム工業", "5301.T": "東海カーボン", "5333.T": "日本碍子", "5411.T": "JFEホールディングス",
    "5406.T": "神戸製鋼所", "5631.T": "日本製鋼所", "5706.T": "三井金属鉱業", "5801.T": "古河電気工業", "6103.T": "オークマ",
    "6113.T": "アマダ", "6201.T": "豊田自動織機", "6305.T": "日立建機", "6361.T": "荏原製作所", "6471.T": "日本精工",
    "6473.T": "ジェイテクト", "6479.T": "ミネベアミツミ", "6504.T": "富士電機", "6506.T": "安川電機", "6752.T": "パナソニック ホールディングス",
    "6753.T": "シャープ", "6770.T": "アルプスアルパイン", "6841.T": "横河電機", "6952.T": "カシオ計算機", "7013.T": "IHI",
    "7182.T": "ゆうちょ銀行", "7186.T": "コンコルディア", "7201.T": "日産自動車", "7211.T": "三菱自動車工業", "7270.T": "SUBARU",
    "7731.T": "ニコン", "7733.T": "オリンパス", "7752.T": "リコー", "7860.T": "エイベックス", "7911.T": "トッパンホールディングス",
    "7912.T": "大日本印刷", "7951.T": "ヤマハ", "8002.T": "丸紅", "8015.T": "豊田通商", "8233.T": "高島屋",
    "8252.T": "丸井グループ", "8308.T": "りそなホールディングス", "8331.T": "千葉銀行", "8354.T": "静岡フィナンシャルグループ", "8439.T": "東京都競馬",
    "8572.T": "アコム", "8601.T": "大和証券グループ本社", "8604.T": "野村ホールディングス", "8628.T": "松井証券", "8703.T": "カブドットコム証券（auカブコム）",
    "8795.T": "T&Dホールディングス", "8804.T": "東京建物", "8803.T": "平和不動産", "9001.T": "東武鉄道", "9005.T": "東急",
    "9007.T": "小田急電鉄", "9009.T": "京成電鉄", "9041.T": "近鉄グループホールディングス", "9042.T": "阪急阪神ホールディングス", "9062.T": "日本通運 (NXHD)",
    "9064.T": "ヤマトホールディングス", "9143.T": "SGホールディングス", "9201.T": "日本航空 (JAL)", "9364.T": "上組", "9502.T": "中部電力",
    "9506.T": "東北電力", "9532.T": "大阪ガス", "9602.T": "東宝", "9766.T": "コナミグループ", "9984.T": "ソフトバンクグループ"
}

def predict_stock(ticker, market_df):
    start_date = "2020-01-01"
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 300:
        return None

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required_columns):
        return None

    df = df[required_columns].copy()
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not market_df.empty and "Close" in market_df.columns:
        df["Market_Close"] = market_df["Close"].ffill()
        df["Market_Return_1"] = df["Market_Close"].pct_change(1)
        df["Market_Return_5"] = df["Market_Close"].pct_change(5)
    else:
        df["Market_Close"] = np.nan
        df["Market_Return_1"] = np.nan
        df["Market_Return_5"] = np.nan

    df.dropna(inplace=True)

    data = df.copy()
    for n in [1, 2, 3, 5, 10, 20, 60]:
        data[f"return_{n}"] = data["Close"].pct_change(n)

    for n in [5, 10, 20, 50, 100, 200]:
        data[f"MA_{n}"] = data["Close"].rolling(n).mean()
        data[f"MA_ratio_{n}"] = data["Close"] / data[f"MA_{n}"] - 1

    data["high_low"] = data["High"] / data["Low"] - 1
    data["close_high"] = data["Close"] / data["High"] - 1
    data["close_low"] = data["Close"] / data["Low"] - 1

    data["MA_200_slope"] = data["MA_200"].pct_change(20)
    data["Donchian_High_20"] = data["High"].rolling(20).max()
    data["Donchian_Low_20"] = data["Low"].rolling(20).min()
    data["Donchian_Position_20"] = (data["Close"] - data["Donchian_Low_20"]) / (data["Donchian_High_20"] - data["Donchian_Low_20"]).replace(0, np.nan)

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    data["VWAP_20"] = (typical_price * data["Volume"]).rolling(20).sum() / data["Volume"].rolling(20).sum().replace(0, np.nan)
    data["Close_to_VWAP_ratio"] = data["Close"] / data["VWAP_20"] - 1

    raw_money_flow = typical_price * data["Volume"]
    price_diff = typical_price.diff()
    positive_flow = np.where(price_diff > 0, raw_money_flow, 0)
    negative_flow = np.where(price_diff < 0, raw_money_flow, 0)
    positive_mf = pd.Series(positive_flow, index=data.index).rolling(14).sum()
    negative_mf = pd.Series(negative_flow, index=data.index).rolling(14).sum()
    mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
    data["MFI_14"] = 100 - (100 / (1 + mfi_ratio))

    data["volume_change"] = data["Volume"].pct_change()
    for n in [5, 20, 60]:
        data[f"volume_ratio_{n}"] = data["Volume"] / data["Volume"].rolling(n).mean()
    data["volume_price_impact"] = data["volume_change"] * data["return_1"]

    prev_close = data["Close"].shift(1)
    tr = pd.concat([data["High"] - data["Low"], (data["High"] - prev_close).abs(), (data["Low"] - prev_close).abs()], axis=1).max(axis=1)
    data["ATR_5"] = tr.rolling(5).mean()
    data["ATR_20"] = tr.rolling(20).mean()
    data["ATR_ratio_short_long"] = data["ATR_5"] / data["ATR_20"].replace(0, np.nan)
    data["ATR_ratio"] = data["ATR_20"] / data["Close"]

    hl_spread = data["High"] - data["Low"]
    data["HL_spread_ratio"] = hl_spread / hl_spread.rolling(20).mean()

    rolling_cov = data["return_1"].rolling(20).cov(data["Market_Return_1"])
    rolling_mkt_var = data["Market_Return_1"].rolling(20).var()
    data["market_beta_20"] = rolling_cov / rolling_mkt_var.replace(0, np.nan)
    data["market_diff_1"] = data["return_1"] - data["Market_Return_1"]
    data["market_diff_5"] = data["return_5"] - data["Market_Return_5"]

    for n in [5, 10, 20, 60]:
        data[f"volatility_{n}"] = data["return_1"].rolling(n).std()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0.0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    data["RSI"] = 100 - 100 / (1 + rs)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    ma20 = data["Close"].rolling(20).mean()
    std20 = data["Close"].rolling(20).std()
    data["BB_position"] = (data["Close"] - (ma20 - 2 * std20)) / ((ma20 + 2 * std20) - (ma20 - 2 * std20))

    data["MA_5_20"] = data["MA_5"] / data["MA_20"] - 1
    data["MA_20_50"] = data["MA_20"] / data["MA_50"] - 1
    data["MA_50_200"] = data["MA_50"] / data["MA_200"] - 1

    features = [
        "return_1", "return_2", "return_3", "return_5", "return_10", "return_20", "return_60",
        "MA_ratio_5", "MA_ratio_10", "MA_ratio_20", "MA_ratio_50", "MA_ratio_100", "MA_ratio_200",
        "high_low", "close_high", "close_low",
        "MA_200_slope", "Donchian_Position_20",
        "Close_to_VWAP_ratio", "MFI_14",
        "volume_change", "volume_ratio_5", "volume_ratio_20", "volume_ratio_60", "volume_price_impact",
        "ATR_ratio_short_long", "HL_spread_ratio",
        "Market_Return_1", "Market_Return_5", "market_beta_20", "market_diff_1", "market_diff_5",
        "volatility_5", "volatility_10", "volatility_20", "volatility_60",
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "ATR_ratio", "BB_position",
        "MA_5_20", "MA_20_50", "MA_50_200",
    ]

    subset = data.tail(504).copy()
    subset["target_return"] = subset["Close"].shift(-5) / subset["Close"] - 1
    subset["target_up"] = (subset["target_return"] > 0).astype(int)
    train_data = subset[features + ["target_return", "target_up"]].dropna().copy()

    if len(train_data) < 100:
        return None

    X_train = train_data[features]

    reg = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)

    reg.fit(X_train, train_data["target_return"])
    clf.fit(X_train, train_data["target_up"])

    latest_features = data[features].dropna().iloc[[-1]]
    current_price = float(data["Close"].iloc[-1])
    
    predicted_return = float(reg.predict(latest_features)[0])
    predicted_price = current_price * (1 + predicted_return)
    probability = float(clf.predict_proba(latest_features)[0, 1])

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return": predicted_return,
        "probability": probability
    }

def send_line_message(message):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("LINEへの通知に成功しました！")
    else:
        print(f"通知失敗: {response.status_code}, {response.text}")

def update_and_append_log(top5_prob, top5_return):
    log_file = "forward_test_log.csv"
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_dt = datetime.strptime(today_str, "%Y-%m-%d")

    expected_columns = ["Date", "Ticker", "Name", "Condition", "EntryPrice", "TargetReturn5d", "ActualReturn5d", "Status"]

    # 1. 既存ログの読み込みと勝敗判定（Openのものをチェック）
    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file)
            for col in expected_columns:
                if col not in df_log.columns:
                    df_log[col] = ""
        except Exception:
            df_log = pd.DataFrame(columns=expected_columns)
    else:
        df_log = pd.DataFrame(columns=expected_columns)

    print(f"--- 既存ログの勝敗判定チェック開始 (全 {len(df_log)} 行) ---")
    
    # ステータスがOpenの行について、5営業日以上経過していれば実績株価を取得して判定
    for idx, row in df_log.iterrows():
        if str(row["Status"]) == "Open":
            entry_date_str = str(row["Date"])
            print(f"チェック対象: Ticker={row['Ticker']}, Date={entry_date_str}")
            try:
                entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
                days_passed = (today_dt - entry_dt).days
                print(f" -> 経過日数: {days_passed}日")

                if days_passed >= 7:
                    ticker = row["Ticker"]
                    entry_price = float(row["EntryPrice"])
                    
                    hist = yf.download(ticker, start=entry_date_str, auto_adjust=True, progress=False)
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    
                    print(f" -> 取得できた株価データ数: {len(hist)} 行")
                    # 変更前：if len(hist) >= 6:
                    # 変更後：
                    if len(hist) >= 5:
                        exit_page_idx = 4 if len(hist) == 5 else 5
                        exit_price = float(hist["Close"].iloc[exit_page_idx])
                        actual_return = (exit_price - entry_price) / entry_price
                        
                        df_log.loc[idx, "ActualReturn5d"] = round(actual_return, 4)
                        df_log.loc[idx, "Status"] = "Win" if actual_return > 0 else "Lose"
                        print(f" ==> 判定完了！ Status: {df_log.loc[idx, 'Status']} (ActualReturn: {actual_return:.4f})")
                    else:
                        print(" ==> 5営業日分のデータがまだ揃っていません。")
                else:
                    print(" ==> 経過日数が7日未満のためスキップします。")
            except Exception as e:
                print(f" ==> エラー発生: {e}")

    # 2. 本日の新規データを追加
    new_logs = []
    for _, row in top5_prob.reset_index(drop=True).iterrows():
        new_logs.append({
            "Date": today_str,
            "Ticker": row["ticker"],
            "Name": row["name"],
            "Condition": "Probability_TOP5",
            "EntryPrice": row["current_price"],
            "TargetReturn5d": round(row["predicted_return"], 4),
            "ActualReturn5d": "",
            "Status": "Open"
        })

    for _, row in top5_return.reset_index(drop=True).iterrows():
        new_logs.append({
            "Date": today_str,
            "Ticker": row["ticker"],
            "Name": row["name"],
            "Condition": "Return_TOP5_Prob60",
            "EntryPrice": row["current_price"],
            "TargetReturn5d": round(row["predicted_return"], 4),
            "ActualReturn5d": "",
            "Status": "Open"
        })

    df_new = pd.DataFrame(new_logs)
    df_log = df_log[df_log["Date"] != today_str]
    df_combined = pd.concat([df_log, df_new], ignore_index=True)
    
    df_combined.to_csv(log_file, index=False)
    print("forward_test_log.csv の保存が完了しました！")
    # 条件Aの追加
    for _, row in top5_prob.reset_index(drop=True).iterrows():
        new_logs.append({
            "Date": today_str,
            "Ticker": row["ticker"],
            "Name": row["name"],
            "Condition": "Probability_TOP5",
            "EntryPrice": row["current_price"],
            "TargetReturn5d": round(row["predicted_return"], 4),
            "ActualReturn5d": "",
            "Status": "Open"
        })

    # 条件Bの追加（重複を避けるため、すでに条件Aに入っていないか確認しつつ追加も可能ですが、そのまま素直に記録します）
    for _, row in top5_return.reset_index(drop=True).iterrows():
        new_logs.append({
            "Date": today_str,
            "Ticker": row["ticker"],
            "Name": row["name"],
            "Condition": "Return_TOP5_Prob60",
            "EntryPrice": row["current_price"],
            "TargetReturn5d": round(row["predicted_return"], 4),
            "ActualReturn5d": "",
            "Status": "Open"
        })

    df_new = pd.DataFrame(new_logs)

    # 同日のデータが既に存在する場合は重複追加しないように整理
    df_log = df_log[df_log["Date"] != today_str]
    df_combined = pd.concat([df_log, df_new], ignore_index=True)
    
    df_combined.to_csv(log_file, index=False)
    print("forward_test_log.csv の勝敗判定と本日の結果追加を完了しました！")

def run_screening():
    print("日経平均データ取得中...")
    market_df = yf.download("^N225", start="2020-01-01", auto_adjust=True, progress=False)
    if isinstance(market_df.columns, pd.MultiIndex):
        market_df.columns = market_df.columns.get_level_values(0)

    unique_tickers = list(dict.fromkeys(top200_tickers.items()))
    
    results = []
    print(f"全 {len(unique_tickers)} 銘柄の予測計算を開始します...")
    for ticker, name in unique_tickers:
        print(f"分析中: {name} ({ticker})")
        res = predict_stock(ticker, market_df)
        if res:
            results.append({
                "ticker": ticker,
                "name": name,
                "current_price": res["current_price"],
                "predicted_price": res["predicted_price"],
                "predicted_return": res["predicted_return"],
                "probability": res["probability"]
            })

    if not results:
        print("有効な予測結果が得られませんでした。")
        return

    df_res = pd.DataFrame(results)

    # ランキング1: 単純に上昇確率が高い順 TOP5
    top5_prob = df_res.sort_values(by="probability", ascending=False).head(5)

    # ランキング2: 上昇確率60%以上の中で、予測上昇幅が大きい順 TOP5
    df_filtered = df_res[df_res["probability"] >= 0.60].copy()
    if df_filtered.empty:
        df_filtered = df_res.copy()
    top5_return = df_filtered.sort_values(by="predicted_return", ascending=False).head(5)

    # ログの更新とCSV保存処理の実行
    update_and_append_log(top5_prob, top5_return)

    # メッセージの組み立て
    msg = "【📈 AI株価予測 朝のスクリーニング結果（200銘柄）】\n\n"

    msg += "【🎯 上昇確率が高い順 TOP5】\n"
    for i, row in top5_prob.reset_index(drop=True).iterrows():
        msg += f"■ {i+1}. {row['name']} ({row['ticker']})\n"
        msg += f"・現在値: {row['current_price']:,.1f} 円\n"
        msg += f"・5日後予測: {row['predicted_price']:,.1f} 円 ({row['predicted_return']*100:+.2f}%)\n"
        msg += f"・上昇確率: {row['probability']*100:.1f} %\n\n"

    msg += "【🚀 予測上昇幅が大きい順（確率60%以上）TOP5】\n"
    for i, row in top5_return.reset_index(drop=True).iterrows():
        msg += f"■ {i+1}. {row['name']} ({row['ticker']})\n"
        msg += f"・現在値: {row['current_price']:,.1f} 円\n"
        msg += f"・5日後予測: {row['predicted_price']:,.1f} 円 ({row['predicted_return']*100:+.2f}%)\n"
        msg += f"・上昇確率: {row['probability']*100:.1f} %\n\n"

    msg += "※投資は自己責任でお願いします。"

    print("\n--- LINEに送信するメッセージ ---")
    print(msg)
    print("--------------------------------")

    send_line_message(msg)

if __name__ == "__main__":
    run_screening()