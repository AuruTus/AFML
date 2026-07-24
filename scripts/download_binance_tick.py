import pandas as pd
import numpy as np
import urllib.request
import os

# 0. 确保输出目录 ./Sample_data/ 存在
OUTPUT_DIR = "./Sample_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_binance_tick_data(symbol="BNBBTC", date_str="2024-01-01"):
    """下载并加载 Binance 逐笔数据"""
    zip_filename = f"./Sample_data/{symbol}-aggTrades-{date_str}.zip"
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{zip_filename}"

    if not os.path.exists(zip_filename):
        print(f"正在下载 {symbol} {date_str} 的数据...")
        urllib.request.urlretrieve(url, zip_filename)
        print("下载完成！")
    else:
        print(f"找到本地缓存: {zip_filename}")

    headers = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
    df = pd.read_csv(zip_filename, compression="zip", names=headers)

    df["date_time"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["dollar_value"] = df["price"] * df["quantity"]
    return df


def generate_standard_bars(df, bar_type="dollar", threshold=10.0):
    """
    通用 Standard Bar 生成器
    :param bar_type: 'tick', 'volume', 或 'dollar'
    :param threshold: 触发阈值
    """
    bars = []
    cum_val = 0.0
    bar_ticks = []

    for row in df.itertuples():
        if bar_type == "tick":
            cum_val += 1.0
        elif bar_type == "volume":
            cum_val += row.quantity
        elif bar_type == "dollar":
            cum_val += row.dollar_value
        else:
            raise ValueError("bar_type 必须是 'tick', 'volume', 或 'dollar'")

        bar_ticks.append(row)

        if cum_val >= threshold:
            prices = [t.price for t in bar_ticks]
            volumes = [t.quantity for t in bar_ticks]
            dollars = [t.dollar_value for t in bar_ticks]

            bars.append(
                {
                    "date_time": bar_ticks[-1].date_time,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": sum(volumes),
                    "dollar_value": sum(dollars),
                    "ticks_count": len(bar_ticks),
                }
            )

            cum_val = 0.0
            bar_ticks = []

    return pd.DataFrame(bars)


# --- 主程序运行与文件保存 ---
if __name__ == "__main__":
    # 1. 下载并加载逐笔数据
    df_ticks = load_binance_tick_data("BNBBTC", "2024-01-01")
    print(f"成功载入逐笔交易数据，共 {len(df_ticks)} 条记录。\n")

    # 2. 生成三种 Bar (阈值可根据需要自行微调)
    print("正在生成 Tick Bars...")
    df_tb = generate_standard_bars(df_ticks, bar_type="tick", threshold=1000)

    print("正在生成 Volume Bars...")
    df_vb = generate_standard_bars(df_ticks, bar_type="volume", threshold=500.0)

    print("正在生成 Dollar Bars...")
    df_db = generate_standard_bars(df_ticks, bar_type="dollar", threshold=10.0)

    # 3. 保存至 ./Sample_data/ 目录下对应的 .txt 文件
    tb_path = os.path.join(OUTPUT_DIR, "tick_bars.txt")
    vb_path = os.path.join(OUTPUT_DIR, "volume_bars.txt")
    db_path = os.path.join(OUTPUT_DIR, "dollar_bars.txt")

    # 保存为逗号分隔文本 (支持以文本编辑器或 pd.read_csv 读取)
    df_tb.to_csv(tb_path, sep=",", index=False)
    df_vb.to_csv(vb_path, sep=",", index=False)
    df_db.to_csv(db_path, sep=",", index=False)

    print("\n所有数据文件保存完毕！")
    print(f"1. {tb_path} ({len(df_tb)} 行)")
    print(f"2. {vb_path} ({len(df_vb)} 行)")
    print(f"3. {db_path} ({len(df_db)} 行)")
