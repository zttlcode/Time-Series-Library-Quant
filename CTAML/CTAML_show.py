import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./ctaml_results/equity_curves/equity_BOLL_RSI_A_002311.csv")
df['time'] = pd.to_datetime(df['time'])

plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['return_signal'], label='Original signal')
plt.plot(df['time'], df['return_ctaml'], label='CTAML filtered')
plt.plot(df['time'], df['return_buy_hold'], label='Buy and Hold')
plt.axhline(0, color='black', linewidth=0.8)
plt.legend()
plt.title("002311 cumulative return comparison")
plt.show()