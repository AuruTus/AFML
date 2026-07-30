import pandas as pd
import numpy as np
import urllib.request
import os

# 0. 确保输出目录 ./sample-data/ 存在
OUTPUT_DIR = "./sample-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_binance_tick_data(symbol="BNBBTC", date_str="2024-01-01"):
    """下载并加载 Binance 逐笔数据"""
    zip_basename = f"{symbol}-aggTrades-{date_str}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_basename)
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{zip_basename}"

    if not os.path.exists(zip_path):
        print(f"正在下载 {symbol} {date_str} 的数据...")
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成！")
    else:
        print(f"找到本地缓存: {zip_path}")

    headers = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
    df = pd.read_csv(zip_path, compression="zip", names=headers)

    df["date_time"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["dollar_value"] = df["price"] * df["quantity"]
    return df


def generate_standard_bars(df, bar_type="dollar", threshold=10.0):
    """
    通用 Standard Bar 生成器 (向量化实现)
    :param bar_type: 'tick', 'volume', 或 'dollar'
    :param threshold: 触发阈值
    """
    if bar_type == "tick":
        values = np.ones(len(df))
    elif bar_type == "volume":
        values = df["quantity"].values
    elif bar_type == "dollar":
        values = df["dollar_value"].values
    else:
        raise ValueError("bar_type 必须是 'tick', 'volume', 或 'dollar'")

    cumsum = np.cumsum(values)
    n_bars = int(cumsum[-1] // threshold)
    if n_bars == 0:
        return pd.DataFrame()

    # 用 searchsorted 找到每个 threshold 边界对应的索引
    boundaries = np.searchsorted(cumsum, threshold * np.arange(1, n_bars + 1))
    starts = np.concatenate([[0], boundaries[:-1] + 1])
    ends = boundaries

    # 向量化构建 bars
    price = df["price"].values
    quantity = df["quantity"].values
    dollar_val = df["dollar_value"].values
    date_time = df["date_time"].values

    bar_data = {
        "date_time": date_time[ends],
        "open": price[starts],
        "high": np.maximum.reduceat(price, starts),
        "low": np.minimum.reduceat(price, starts),
        "close": price[ends],
        "volume": np.add.reduceat(quantity, starts),
        "dollar_value": np.add.reduceat(dollar_val, starts),
        "ticks_count": ends - starts + 1,
    }

    return pd.DataFrame(bar_data)


# --- 主程序运行与文件保存 ---
if __name__ == "__main__":
    # 1. 下载并加载逐笔数据 (多天合并，确保至少覆盖数周)
    symbol = "BTCUSDT"
    date_range = pd.date_range("2024-01-01", "2024-02-04", freq="D")
    dfs = []
    for d in date_range:
        date_str = d.strftime("%Y-%m-%d")
        try:
            df_day = load_binance_tick_data(symbol, date_str)
            dfs.append(df_day)
            print(f"  {date_str}: {len(df_day)} 条记录")
        except Exception as e:
            print(f"  {date_str}: 跳过 ({e})")

    df_ticks = pd.concat(dfs, ignore_index=True).sort_values("transact_time").reset_index(drop=True)
    print(
        f"\n成功载入逐笔交易数据，共 {len(df_ticks)} 条记录，"
        f"日期范围: [{df_ticks.date_time.min()}, {df_ticks.date_time.max()}]\n"
    )

    # 2. 生成三种 Bar (阈值调整为 BTCUSDT 的量级, 每种约产出 1000 条以便比较)
    print("正在生成 Tick Bars...")
    df_tb = generate_standard_bars(df_ticks, bar_type="tick", threshold=10000)

    print("正在生成 Volume Bars...")
    df_vb = generate_standard_bars(df_ticks, bar_type="volume", threshold=300.0)

    print("正在生成 Dollar Bars...")
    df_db = generate_standard_bars(df_ticks, bar_type="dollar", threshold=13_000_000.0)

    # 3. 保存至 ./sample-data/ 目录下对应的 .txt 文件
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
