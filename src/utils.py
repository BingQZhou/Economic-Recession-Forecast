from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def investing_convert_time(time_str):
    # convert string month to numeric
    mon_dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    time_str = str(mon_dict[time_str[:3]]) + time_str[3:]
    # convert to prefered format
    num_time = datetime.strptime(time_str, '%m %d, %Y')
    return num_time.strftime('%Y%m%d')

def yield_convert_time(time):
    return time.strftime('%Y%m%d')

def normalize(ser):
    return (ser - np.mean(ser)) / np.std(ser)

def plot_pred(true, pred, title, x_title, y_title):
    # line plot of observed vs predicted
    plt.plot(true)
    plt.plot(pred)
    blue_patch = mpatches.Patch(color='blue', label='raw_values')
    orange_patch = mpatches.Patch(color='orange', label='predictions')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.title(title)
    plt.show()
    
def yield_rate_convert(yield_used):
    # difference between 2 Yr and 10 Yr
    diff = np.array(yield_used["10 Yr"]) - np.array(yield_used["2 Yr"])
    # calculate percentiles
    five_perc = np.nanpercentile(diff, 5, axis = 0)
    ten_perc = np.nanpercentile(diff, 10, axis = 0)
    twenty_perc = np.nanpercentile(diff, 20, axis = 0)
    thirty_perc = np.nanpercentile(diff, 30, axis = 0)
    # convert accroding to the percentiles
    conversion_rate = []
    for i in range(len(yield_used)):
        cur_yield = yield_used.iloc[i]
        cur_diff = cur_yield['10 Yr'] - cur_yield['2 Yr']
        if cur_diff <= five_perc:
            conversion_rate.append(0.8)
        elif cur_diff <= ten_perc:
            conversion_rate.append(0.87)
        elif cur_diff <= twenty_perc:
            conversion_rate.append(0.93)
        elif cur_diff <= thirty_perc:
            conversion_rate.append(0.97)
        else:
            conversion_rate.append(1.0)
    return conversion_rate