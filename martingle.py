import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prediction_score_calculator(x):
    return (x['martingle_prediction'] - x['target']) ** 2


def main():
    data_address = os.path.join(os.getcwd(), 'BTC_DATA/BTCUSDT-1d-data.csv')

    df = pd.read_csv(data_address)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['open', 'high', 'low', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av',
                     'ignore'], inplace=True)
    df['diff'] = df.rolling(2).agg({"close": lambda x: x.iloc[1] - x.iloc[0]})
    trend = df['diff'].apply(lambda x: int(x > 0))
    df['martingle_prediction'] = trend
    df['target'] = trend.shift(-1)
    df.dropna(inplace=True)
    df['prediction_score'] = df.apply(prediction_score_calculator, axis=1)
    batch_size = int(np.floor(df.prediction_score.count() / 20))
    results = []
    for i in range(20):
        start = i * batch_size
        end = (i + 1) * batch_size
        result = (df['prediction_score'].iloc[start:end].sum()) / batch_size
        results.append(result)
    plt.plot(results, '*-')
    plt.ylim([0, 1])
    plt.show()
    print('results:', results)
    print('accuracy_mean: {}, std: {}'.format(np.mean(results), np.std(results)))


if __name__ == '__main__':
    main()
