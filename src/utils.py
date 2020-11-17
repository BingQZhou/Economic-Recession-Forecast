from datetime import datetime
import numpy as np

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