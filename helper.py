# -*- coding: utf-8 -*-
import re
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def extract_val(txt):
    deaths = 0
    cases = 0
    txt = str(txt).replace(" ", "").replace("(", " ").replace(")", " ").replace("–", "-")
    try:
        # there are deaths reported
        extract = re.match(r'(?P<cases>.*) (?P<deaths>.*) ', txt)
        cases = int(extract.group('cases'))
        deaths = int(extract.group('deaths'))
    except:
        try:
            # there are deaths reported
            extract = re.match(r' (?P<deaths>.*) (?P<cases>.*)', txt)
            cases = int(extract.group('cases'))
            deaths = int(extract.group('deaths'))
        except:
            try:
                cases = int(float(txt))
            except ValueError:
                # no cases reported
                pass
    return [cases, deaths]
    
def organize_df(df):
    index = pd.DatetimeIndex(df["Date"].unique())
    columns = df.columns
    cases_data = []
    death_data = []
    for index_i, row in df.iterrows():
        cases_row = []
        deaths_row = []
        for col_name in columns:
            if col_name != "Date":
                [cases, deaths] = extract_val(row[col_name])
                cases_row.append(cases)
                deaths_row.append(deaths)
            else:
                cases_row.append(row[col_name])
                deaths_row.append(row[col_name])
        cases_data.append(cases_row)
        death_data.append(deaths_row)
    death_df = pd.DataFrame(data=death_data, columns=columns)
    cases_df = pd.DataFrame(data=cases_data, columns=columns)
    try:
        death_df = death_df.groupby("Date").sum()
        cases_df = cases_df.groupby("Date").sum()
    except:
        pass
    cases_df.index = index
    death_df.index = index
    return cases_df, death_df


def get_data(filepath):
    df = pd.read_csv(filepath).fillna(0)
    return organize_df(df)

def get_time_to_die(cases_ts, deaths_ts):
    cases_ts, deaths_ts
    count_to_death = 0
    for item in deaths_ts:
        count_to_death += 1
        if item > 0:
            break

    if count_to_death == len(deaths_ts):
        count_to_death = 0

    cases_count = 0
    time_to_die = 0
    for item in cases_ts[:count_to_death]:
        cases_count += item
        if item > 0:
            time_to_die += 1
    return time_to_die, cases_count

def gen_pareto(data, total, verbose=False):
    """
    This function creates a distribution following pareto principle, in which 20% of data
    corresponds to 80% of the total. The returned data is a list of values which sum up roughly
    to the value of 'total'
    
    Paremeters
    ----------
     - data (array): list of zeros, this will help to store the values
     - total (int): parameter which tells what the sum of all values should be
     - verbose (bool): if user needs to see the how function is behaving 
    
    Returns
    -------
     - list: same size as 'data' with sum roughty equal to 'total', data is not sorted
    """
    data_size = len(data)
    if verbose:
        print "calling function"
        print "data size:", data_size, "total value:", total
    new_data = []
    if len(data) == 1:
        new_data.append(total)
        return new_data
    if len(data) == 2:
        new_data.append(total*0.2)
        new_data.append(total*0.8)
        return new_data
    if len(data) > 2:
        size_20perc = int(math.floor(data_size*0.8))
        if verbose:
            print "20perc:", size_20perc
        data_20 = data[:size_20perc]
        data_80 = data[size_20perc:]
        new_data = gen_pareto(data_20, total*0.2)
        new_data += gen_pareto(data_80, total*0.8)
        return new_data
    
def simulate_region(density, population, **kwargs):
    """
    This function creates a distribution roughly following a pareto distribution
    we take in a count that a region has very populated areas and very sparse areas as well
    Therefore we should see a percentage of empty areas and a small fraction densily populated
    
    Parameters
    ----------
     - density (int): number of people per squared km
     - population (int): total number of people in the region
     - min_pop (int): smallest number allowed per person in an occupied squared km
     
    Returns
    -------
     - np.array: array of people distributed in the given region parameters
    """
    # we consider the least habitated area with one person per block
    min_pop = kwargs.get("min_pop", 1)
    area = population/density
    # we consider every squared km as a block in the region
    data = np.zeros(area) 
    # now we distribute people across these blocks following a pareto distribution
    ts = pd.Series(gen_pareto(data, population))
    ts = ts.sort_values() # we must sort the values
    sum_min_pop = sum(ts[ts<=min_pop]) # sum of people below min threshold
    ts[ts.size-1] += sum_min_pop #moving these people to the highest occupied area
    ts[ts<=min_pop] = 0 # defining empty space
    return ts.get_values()

def plot_cdf(data, title):
    data_sum = float(data.sum())
    data_mean = data_sum/len(data)
    cdf = data.cumsum(0)/data_sum
    ccdf = 1-cdf
    ax = plt.figure(figsize=(10,6))
    plt.plot(data,cdf,'-o', markersize=2.5, linewidth=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([np.partition(np.unique(data), 1)[1]/2, data[-1]])
    plt.ylim([0,5])
    plt.ylabel('CDF')
    plt.title(title)
    plt.grid(True, which='major', color='0.65', linestyle='--')
    plt.figtext(.7, .6, "mean = %2.f" % data_mean, backgroundcolor='w')
    plt.figtext(.7, .5, "sum = %d" % data_sum, backgroundcolor='w')
    plt.figtext(.5, .2, "author: ricardosilveira@poli.ufrj.br", fontsize=8, fontstyle="italic", backgroundcolor='w')
    return ax

def plot_cdfs(data_list, title, labels, lines, filename):
    ax = plt.figure(figsize=(10,6))
    min_axis = []
    max_axis = []
    for i, data in enumerate(data_list):
        data_sum = float(data.sum())
        data_mean = data_sum/len(data)
        cdf = data.cumsum(0)/data_sum
        ccdf = 1-cdf
        plt.plot(data,cdf, lines[i], label=labels[i], markersize=2.5, linewidth=0.5)
        min_axis.append(np.partition(np.unique(data), 1)[1]/2)
        max_axis.append(data[-1])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([min(min_axis), max(max_axis)])
    plt.ylim([0,5])
    plt.ylabel('CDF')
    plt.title(title)
    plt.grid(True, which='major', color='0.65', linestyle='--')
    #plt.figtext(.7, .6, "mean = %2.f" % data_mean, backgroundcolor='w')
    #plt.figtext(.7, .5, "sum = %d" % data_sum, backgroundcolor='w')
    plt.figtext(.5, .2, "author: ricardosilveira@poli.ufrj.br", fontsize=8, fontstyle="italic", backgroundcolor='w')
    plt.savefig(filename, dpi = 300)
    return ax