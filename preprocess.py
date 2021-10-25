import pandas as pd


if __name__ == "__main__":
    time_index = "수집시각"
    data = pd.read_csv(
        "POWER_M1_FILL copy.csv",
        parse_dates=[time_index]
    )
    
    # data['수집시각'] = data['수집시각'].astype(str)
    # data['수집시각'] = data['수집시각'].apply(
    #     lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]} {x[8:10]}:{x[10:12]}:{x[12:14]}"
    # # )
    # data[time_index] = pd.to_datetime(data[time_index])
    timegap = data[time_index][1] - data[time_index][0]
    # data = data.set_index(time_index)

    origin_len = len(data)
    for i in range(1, len(data)):
        if data[time_index][i] - data[time_index][i - 1] != timegap:
            start_time = data[time_index][i - 1] + timegap
            end_time = data[time_index][i] - timegap

            # Fill time gap
            for _ in range(1 + (end_time - start_time) // timegap):
                time = start_time + _ * timegap
                data = data.append({time_index: time}, ignore_index=True)
                
    data = data.set_index(time_index).sort_index().reset_index()
    
    filled_len = len(data)
    print(f"Filling Time Gap :{filled_len - origin_len} : timegap : {timegap}")

    data = data[::3]
    data.to_csv("POWER_M1_GAP-2.csv", index=False)
    print(data)