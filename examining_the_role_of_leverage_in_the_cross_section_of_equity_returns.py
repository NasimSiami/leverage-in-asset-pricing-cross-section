# -*- coding: utf-8 -*-
"""Examining the Role of Leverage in the Cross-Section of Equity Returns.ipynb
This is the coding of my supervised project with the following abstract:

The purpose of this study is to evaluate the performance of asset pricing models, such as the Capital Asset Pricing Model (CAPM) and the Fama-French three-factor model, in accounting for the effects of leverage. It draws inspiration from the research conducted by Doshi, Jacobs et al. (2019) and builds upon the existing theoretical asset pricing literature. We aim to investigate the role of market risk as well as anomalies such as size and book-to-market (BTM) ratio in explaining the cross-section of unlevered equity returns. Additionally, understanding the influence of these factors on the cross-section of unlevered equity returns can contribute to a better understanding of asset valuation and pricing dynamics. We observe that the presence of heteroskedasticity in returns due to leverage can distort the observed patterns and affect the valuation of assets. Unlevering returns helps to remove this influence, resulting in the disappearance of the value premium and the size effect seen in unlevered returns.

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

"""Compustat"""

compustat_df01 = pd.read_csv('yourcompustatdata.csv')

compustat_df01

# Convert 'datadate' column to datetime format
compustat_df01['datadate'] = pd.to_datetime(compustat_df01['datadate'])
#compustat_df01['book_equity'] = compustat_df01['atq'] - compustat_df01['ltq']
compustat_df01['book_equity'] = compustat_df01['ceqq']
#+ compustat_df01['txdbq'].fillna(0) - compustat_df01['pstkq'].fillna(0)
# Sort the dataframe by 'gvkey' and 'datadate'
compustat_df01 = compustat_df01.sort_values(by=['gvkey', 'datadate'])

compustat_df01[(compustat_df01['gvkey'] == 12141) &
                         (compustat_df01['datadate'] >= '2000-01-01') &
                         (compustat_df01['datadate'] <= '2001-12-31')]

"""Debt data Compustat"""

compustat_l = compustat_df01.copy()
compustat_l = compustat_l.drop(columns=['atq', 'cshoq', 'book_equity'])

new_rows = []  # List to store the new rows

# Iterate over each unique gvkey
for gvkey, group in compustat_l.groupby('gvkey'):

    # Placeholders for values
    previous_ltq = None
    #next_year_atq_start_date = None
    #next_year_atq_end_date = None

    # Use the value of the quarter for the subsequent months
    for _, row in group.iterrows():

        # Create 3 new rows for the subsequent months
        for month_delta in range(1, 4):
            new_row = row.copy()
            new_date = row['datadate'] + pd.DateOffset(months=month_delta)
            new_row['datadate'] = new_date

            # Logic to assign ltq based on previous data's ltq
            if row['datadate'].month in [12, 3, 6, 9]:
                previous_ltq = row['ltq']

            new_row['ltq'] = previous_ltq



            new_rows.append(new_row)

# Create a new dataframe from the list of new rows (Don't concatenate with the original dataframe)
expanded_df = pd.DataFrame(new_rows)

# Sort the expanded dataframe by 'gvkey' and 'datadate'
expanded_df = expanded_df.sort_values(by=['gvkey', 'datadate'])

"""Asset data Compustat"""

compustat_a = compustat_df01.copy()
compustat_a = compustat_a.drop(columns=['ltq', 'dlcq','dlttq' ])

# Step 1: Create a helper function to generate monthly rows for each gvkey
def make_monthly(group):
    # Create a date range from the minimum to maximum 'datadate' of the group with monthly frequency
    monthly_dates = pd.date_range(start=group['datadate'].min(), end=group['datadate'].max(), freq='M')

    # Create a new dataframe with these monthly dates
    monthly_df = pd.DataFrame({'datadate': monthly_dates})

    # Merge this dataframe with the group to fill in the data for known months and NaN for others
    merged = pd.merge(monthly_df, group, on='datadate', how='left')

    # Forward fill the gvkey values
    merged['gvkey'] = merged['gvkey'].ffill()

    return merged

# Step 2: Group the dataframe by gvkey and apply the helper function to each group
monthly_groupsadata = compustat_a.groupby('gvkey').apply(make_monthly)

# Reset the index to get rid of the multi-level index introduced by the groupby operation
df_a_monthly = monthly_groupsadata.reset_index(drop=True)

def update_atq(group):
    # Identify the value of `atq` for every June in this group
    june_values = group.set_index('datadate').resample('A-JUN').last().reset_index()[['datadate', 'atq']]

    for index, row in june_values.iterrows():
        june_value = row['atq']

        # Add 12 months to the June date for the start date (July of next year)
        start_date = row['datadate'] + pd.DateOffset(months=13)  # This will be July of next year
        end_date = start_date + pd.DateOffset(months=12)  # This will be June of the year after

        # Update the `atq` values for this date range
        mask = (group['datadate'] >= start_date) & (group['datadate'] <= end_date)
        group.loc[mask, 'atq'] = june_value

    return group

# Apply the function on each `gvkey` group
df_a_monthly_updated = df_a_monthly.groupby('gvkey').apply(update_atq).reset_index(drop=True)

def update_book_equity(group):
    # Identify the value of `book_equity` for every June in this group
    june_values = group.set_index('datadate').resample('A-JUN').last().reset_index()[['datadate', 'book_equity']]

    for index, row in june_values.iterrows():
        june_value = row['book_equity']

        # Add 12 months to the June date for the start date (July of next year)
        start_date = row['datadate'] + pd.DateOffset(months=13)  # This will be July of next year
        end_date = start_date + pd.DateOffset(months=12)  # This will be June of the year after

        # Update the `book_equity` values for this date range
        mask = (group['datadate'] >= start_date) & (group['datadate'] <= end_date)
        group.loc[mask, 'book_equity'] = june_value

    return group

# Apply the function on each `gvkey` group for `book_equity`
df_a_monthly_updated = df_a_monthly_updated.groupby('gvkey').apply(update_book_equity).reset_index(drop=True)

df_a_monthly_updated[(df_a_monthly_updated['book_equity'] == 15647)]

df_a_monthly_updated[(df_a_monthly_updated['gvkey'] == 12141) &
                         (df_a_monthly_updated['datadate'] >= '2000-01-01') &
                         (df_a_monthly_updated['datadate'] <= '2001-12-31')]

"""Merging Compustat"""

# Create 'year_month' column for both dataframes
expanded_df['year_month'] = expanded_df['datadate'].dt.to_period('M')
df_a_monthly_updated['year_month'] = df_a_monthly_updated['datadate'].dt.to_period('M')

# Merge based on 'gvkey' and 'year_month'
compustat_df = pd.merge(expanded_df, df_a_monthly_updated, on=['gvkey', 'year_month'])

# If you want to drop the 'year_month' column after merging:
compustat_df.drop('year_month', axis=1, inplace=True)

# Drop 'datadate_x' column
compustat_df = compustat_df.drop('datadate_x', axis=1)

# Reorder columns to put 'datadate_y' after 'gvkey'
columns_order = ['gvkey', 'datadate_y'] + [col for col in compustat_df.columns if col not in ['gvkey', 'datadate_y']]
compustat_df = compustat_df[columns_order]

# If you want to rename the 'datadate_y' column to 'datadate' or something else
compustat_df = compustat_df.rename(columns={'datadate_y': 'datadate'})

compustat = compustat_df[['datadate', 'gvkey', 'ltq', 'atq', 'book_equity', 'cshoq']].copy()

compustat['count']=compustat.groupby(['gvkey']).cumcount()

compustat[
    (compustat['gvkey'] == 12141) &
    (compustat['datadate'] >= '2000-01-01') &
    (compustat['datadate'] <= '2001-12-31')
]

from pandas.tseries.offsets import YearEnd, MonthEnd

"""CRSP"""

crsp_df01 = pd.read_csv('yourcrspdata.csv')

#crsp_df01 = crsp_df01[crsp_df01['SHRCD'].isin([10, 11])]

crsp_df01 = crsp_df01[crsp_df01['EXCHCD'].isin([1, 2, 3])]

# Convert 'date' to datetime format and set it as index
crsp_df01['date'] = pd.to_datetime(crsp_df01['date'])

# Replace alphabets in 'RET' column with NaN
crsp_df01['RETX'] = pd.to_numeric(crsp_df01['RETX'], errors='coerce')
crsp_df01['RET'] = pd.to_numeric(crsp_df01['RET'], errors='coerce')

crsp_df01['RETX'] = np.where(crsp_df01['RETX'] < -1, np.nan, crsp_df01['RETX'])

crsp_df01.dropna(subset=['RET', 'RETX'], inplace=True)

#crsp_df01['RET']=crsp_df01['RET'].fillna(0)
#crsp_df01['RETX']=crsp_df01['RETX'].fillna(0)

crsp_df01['date']=crsp_df01['date']+MonthEnd(0)

crsp_df01['market_equity'] = np.abs(crsp_df01['PRC']) * crsp_df01['SHROUT'] / 1000

crsp_df01.dropna(subset=['market_equity'], inplace=True)

# Group by 'PERMCO' and 'date', then count the unique 'PERMNO'
groupedt = crsp_df01.groupby(['PERMCO', 'date'])['PERMNO'].nunique().reset_index()

# Filter rows where the unique count of 'PERMNO' is greater than 1
duplicatest = groupedt[groupedt['PERMNO'] > 1]

# Merge the result with the original DataFrame to get the actual rows
duplicate_rows = pd.merge(crsp_df01, duplicatest[['PERMCO', 'date']], on=['PERMCO', 'date'], how='inner')

if not duplicate_rows.empty:
    print("Rows with the same PERMCO and date but different PERMNO:")
    print(duplicate_rows)
else:
    print("No such entries found.")

crsp_df01[
    (crsp_df01['PERMCO'] == 1728  ) &
    (crsp_df01['date'] >= '1986-01-31')
].sort_values(by='date')

# Aggregate ME by permco and date
agg_me = crsp_df01.groupby(['PERMCO', 'date'])['market_equity'].sum().reset_index()

# Find the permno with the largest ME for each permco and date
max_me_permno = crsp_df01.loc[crsp_df01.groupby(['PERMCO', 'date'])['market_equity'].idxmax()]

# Keep all columns from max_me_permno and merge with aggregated ME values
result000 = pd.merge(max_me_permno, agg_me.rename(columns={'market_equity': 'agg_market_equity'}), on=['PERMCO', 'date'])

# Drop 'market_equity' column
result000.drop('market_equity', axis=1, inplace=True)

# Rename 'agg_market_equity' to 'market_equity'
result000.rename(columns={'agg_market_equity': 'market_equity'}, inplace=True)

# Step 4: Sort and drop duplicates
crsp = result000.sort_values(by=['PERMNO', 'date']).drop_duplicates()
#crsp = crsp_df01.sort_values(by=['PERMNO', 'date']).drop_duplicates()

# keep December market cap
crsp['year']=crsp['date'].dt.year
crsp['month']=crsp['date'].dt.month

# cumret by stock
crsp['1+retx']=1+crsp['RETX']
crsp['cumretx']=crsp.groupby(['PERMNO','year'])['1+retx'].cumprod()
# lag cumret
crsp['lcumretx']=crsp.groupby(['PERMNO'])['cumretx'].shift(1)

# Create a new column 'meLag' by shifting the 'market_equity' column by 1 period
crsp['meLag'] = crsp.groupby('PERMNO')['market_equity'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
crsp['count']=crsp.groupby(['PERMNO']).cumcount()
crsp['meLag']=np.where(crsp['count']==0, crsp['market_equity']/crsp['1+retx'], crsp['meLag'])

tttest = crsp.copy()
# Extract year and month from the 'date' column
tttest['year'] = tttest['date'].dt.year
tttest['month'] = tttest['date'].dt.month

# Group by 'PERMCO', 'year', and 'month', then count unique 'PERMNO'
grouped_counts = tttest.groupby(['PERMCO', 'date'])['PERMNO'].nunique()

# Filter out groups with only one unique 'PERMNO'
multi_permno_groups = grouped_counts[grouped_counts > 1]

print(multi_permno_groups)

crsp.set_index('date', inplace=True)

rf_df0 = pd.read_csv('yourkenwebsitedata.csv')

# Rename the 'Unnamed' column to 'date'
rf_df0.rename(columns={'yyyymm': 'date'}, inplace=True)

# Parse the date column as datetime with the correct format
rf_df0['date'] = pd.to_datetime(rf_df0['date'], format='%Y%m')

# Select data from 1971 to 2012
start_date_rf = pd.to_datetime('1971-01-01')
end_date_rf = pd.to_datetime('2012-12-31')
rf_df0 = rf_df0[(rf_df0['date'] >= start_date_rf) & (rf_df0['date'] <= end_date_rf)]
rf_df0.set_index('date', inplace=True)

rf_df0

# Create a year_month column in each dataframe
crsp['year_month'] = crsp.index.strftime('%Y-%m')
rf_df0['year_month'] = rf_df0.index.strftime('%Y-%m')

# Merge on year_month and PERMNO
ret_merged_df = pd.merge(crsp.reset_index(), rf_df0.reset_index(), on='year_month', how='left')

# Drop the year_month column if it's no longer needed
ret_merged_df.drop('year_month', axis=1, inplace=True)

# Set date from crsp_df01 as the index again
#ret_merged_df.set_index('date_x', inplace=True)
ret_merged_df.rename(columns={'date_x': 'date'}, inplace=True)

crsp_rf_merged = ret_merged_df.drop(columns=['date_y'])

crsp_rf_merged['Excess_RET'] = crsp_rf_merged['RET'] - crsp_rf_merged['rfFFWebsite']

crsp_rf_merged['rfFFWebsite'].mean()

crsp_rf_merged[
    (crsp_rf_merged['PERMCO'] == 1728  ) &
    (crsp_rf_merged['date'] >= '1986-01-31')
]

# Create a mask for December
mask_dec = crsp_rf_merged['date'].dt.month == 12

# Copy market equity for December to a temporary 'dec_values' column
crsp_rf_merged['dec_values'] = crsp_rf_merged.loc[mask_dec, 'market_equity']

# Shift the values in 'dec_values' by 6 months (or half a year)
# to get the value in the subsequent July
# (assuming your dataframe is sorted by date within each PERMNO group)
crsp_rf_merged['shifted_dec'] = crsp_rf_merged.groupby('PERMNO')['dec_values'].shift(+7)

# Fill 'medec' column with 'shifted_dec' values starting from July
crsp_rf_merged['medec'] = crsp_rf_merged['shifted_dec']

# Forward fill the values in 'medec' for each PERMNO until June of the following year
crsp_rf_merged['medec'] = crsp_rf_merged.groupby('PERMNO')['medec'].ffill(limit=12)

# Drop temporary columns 'dec_values' and 'shifted_dec'
crsp_rf_merged.drop(['dec_values', 'shifted_dec', 'count'], axis=1, inplace=True)

# Create a new column 'meLag' by shifting the 'market_equity' column by 1 period
#crsp_rf_merged['meLag'] = crsp_rf_merged.groupby('PERMNO')['market_equity'].shift(1)

# Ensure the 'date_x' column is in datetime format
crsp_rf_merged['date'] = pd.to_datetime(crsp_rf_merged['date'])

# Create a mask for June
mask_j0 = crsp_rf_merged['date'].dt.month == 6

# Copy market equity for June to a temporary 'june_values' column
crsp_rf_merged['june_values'] = crsp_rf_merged.loc[mask_j0, 'market_equity']

# Shift the values in 'june_values' by 1 year (assuming your dataframe is sorted by date within each PERMNO group)
crsp_rf_merged['shifted_june'] = crsp_rf_merged.groupby('PERMNO')['june_values'].shift()

# Fill 'mejun' column with 'shifted_june' values
crsp_rf_merged['mejun'] = crsp_rf_merged['shifted_june']

# Forward fill the values in 'mejun' for each PERMNO
crsp_rf_merged['mejun'] = crsp_rf_merged.groupby('PERMNO')['mejun'].ffill()

# Drop temporary columns 'june_values' and 'shifted_june'
crsp_rf_merged.drop(['june_values', 'shifted_june'], axis=1, inplace=True)

crsp_rf_merged[
    (crsp_rf_merged['PERMNO'] == 10107) &
    (crsp_rf_merged['date'] >= '2000-01-01') &
    (crsp_rf_merged['date'] <= '2001-12-31')
]

crsp_rf_merged[
    (crsp_rf_merged['PERMNO'] == 10107) &
    (crsp_rf_merged['date'] >= '2000-01-01') &
    (crsp_rf_merged['date'] <= '2001-12-31')
]

crsp_rf_merged['rfFFWebsite'].mean()

"""CCM key"""

merge_guide = pd.read_csv('yourmergeguide.csv')

ccm = merge_guide.copy()
# Filter the link_df dataframe based on given conditions
ccm = ccm[
    ccm['linktype'].str.startswith('L') &
    ccm['linkprim'].isin(['P', 'C'])
]
# Convert linkdt and linkenddt to datetime format
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])

# if linkenddt is missing then set to today date
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# Line up date to be end of month
#compustat['datadate']=compustat['datadate']+MonthEnd(0)

ccm_compustat0 = pd.merge(compustat, ccm, on=['gvkey'], how='left')

# Filter rows based on year-month representations
ccm_compustat = ccm_compustat0[(ccm_compustat0['datadate'] >= ccm_compustat0['linkdt']) &
                      (ccm_compustat0['datadate'] <= ccm_compustat0['linkenddt'])]

ccm_compustat.rename(columns={'datadate': 'date'}, inplace=True)

crsp_rf_merged.rename(columns={'PERMNO': 'permno'}, inplace=True)

# Create year-month columns for both dataframes
#ccm_compustat['year_month'] = ccm_compustat['datadate'].dt.to_period('M')
#crsp_rf_merged['year_month'] = crsp_rf_merged['date'].dt.to_period('M')

# Merge the resulting dataframe with compustat on 'gvkey' and 'year_month'
final_df = pd.merge(ccm_compustat, crsp_rf_merged, on=['permno','date' ], how='inner')

# Drop the 'year_month' columns if you no longer need them
#final_df.drop('year_month', axis=1, inplace=True)

# Optional: Drop unnecessary columns if needed
columns_to_drop = ['permco', 'linkdt', 'linkenddt']
final_df.drop(columns_to_drop, axis=1, inplace=True)

final_df = final_df[['date', 'permno', 'EXCHCD','PERMCO', 'PRC', 'RET', 'RETX', 'SHROUT','SHRCD', 'rfFFWebsite', 'Excess_RET', 'gvkey', 'ltq', 'atq','book_equity','count','mejun','medec', 'market_equity', 'meLag']]

# compute the leverage ratio
final_df['Leverage'] = (final_df['ltq'] / (final_df['ltq'] + final_df['market_equity']))

# shift the leverage ratio one period forward to represent Li(t-1)
final_df['Leverage_shifted'] = final_df.groupby('permno')['Leverage'].shift(1)

# compute the scaled returns
final_df['scaled_returns'] = final_df['Excess_RET'] * (1 - final_df['Leverage_shifted'])

final_df[(final_df['permno'] == 10107) &
                         (final_df['date'] >= '2000-01-01') &
                         (final_df['date'] <= '2001-12-31')]

final_df[(final_df['permno'] == 10107) &
                         (final_df['date'] >= '2000-01-01') &
                         (final_df['date'] <= '2001-12-31')]

df_tabl_1 = final_df.copy()

#df_tabl_1 = df_tabl_1[df_tabl_1['date'].dt.month == 6]

#df_tabl_1 = df_tabl_1[(df_tabl_1['count']>=1)]

#df_tabl_1['medec'].replace(0, np.nan, inplace=True)

#df_tabl_1 = df_tabl_1.dropna(subset=['book_equity', 'market_equity','mejun', 'medec', 'Excess_RET'])

#df_tabl_1 = df_tabl_1[(df_tabl_1['book_equity'] > 0)]

#df_tabl_1 = df_tabl_1[(df_tabl_1['market_equity'] > 0)]

df_tabl_1['BTM'] = df_tabl_1['book_equity'] / df_tabl_1['medec']
df_tabl_1 = df_tabl_1.dropna(subset=['BTM'])

#df_tabl_1 = df_tabl_1.dropna(subset=['ltq', 'atq', 'market_equity', 'book_equity', 'meLag',	'mejun', 'medec'])

df_tabl_1 = df_tabl_1[df_tabl_1['SHRCD'].isin([10, 11])]

nyse_stocks = df_tabl_1[(df_tabl_1['EXCHCD']==1) & (df_tabl_1['book_equity']>0) & (df_tabl_1['BTM'] > 0) & (df_tabl_1['count']>=1) & ((df_tabl_1['SHRCD']==10) | (df_tabl_1['SHRCD']==11))]

me_breakpoints = nyse_stocks['mejun'].quantile([0.2, 0.4, 0.6, 0.8]).values
btm_breakpoints = nyse_stocks['BTM'].quantile([0.2, 0.4, 0.6, 0.8]).values

def assign_me_portfolio(market_equity):
    if market_equity <= me_breakpoints[0]:
        return 1
    elif market_equity <= me_breakpoints[1]:
        return 2
    elif market_equity <= me_breakpoints[2]:
        return 3
    elif market_equity <= me_breakpoints[3]:
        return 4
    else:
        return 5

def assign_btm_portfolio(btm):
    if btm <= btm_breakpoints[0]:
        return 1
    elif btm <= btm_breakpoints[1]:
        return 2
    elif btm <= btm_breakpoints[2]:
        return 3
    elif btm <= btm_breakpoints[3]:
        return 4
    else:
        return 5

df_tabl_1['size_quintile'] = df_tabl_1['mejun'].apply(assign_me_portfolio)
df_tabl_1['BTM_quintile'] = df_tabl_1['BTM'].apply(assign_btm_portfolio)

#df_tabl_1[(df_tabl_1['BTM_quintile'] == 2)]

# Create BTM quintile labels
#df_tabl_1['size_quintile'] = pd.qcut(df_tabl_1['mejun'], 5, labels=[1, 2, 3, 4, 5])

# Convert the column to numeric
#df_tabl_1['BTM'] = pd.to_numeric(df_tabl_1['BTM'], errors='coerce')

# Then, create the quintiles again
#df_tabl_1['BTM_quintile'] = pd.qcut(df_tabl_1['BTM'], 5, labels=[1, 2, 3, 4, 5])

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

vwret=df_tabl_1.groupby(['date','size_quintile', 'BTM_quintile']).apply(wavg, 'Excess_RET','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})

portfolios = vwret.groupby(['size_quintile', 'BTM_quintile']).agg({'vwret': 'mean'}).reset_index()

portfolio_pivot = portfolios.pivot_table(index='size_quintile', columns=['BTM_quintile'], values='vwret')*100

# Compute 5-1 differences for sizes and BTMs
portfolio_pivot['5-1'] = portfolio_pivot[5] - portfolio_pivot[1]

# Place the 5-1 BTM differences into the table rather than appending
btm_diff1 = (portfolio_pivot.loc[5] - portfolio_pivot.loc[1])
portfolio_pivot.loc['5-1'] = btm_diff1

# Remove the value in the last row and column which is the difference of both size and BTM
portfolio_pivot.loc['5-1', '5-1'] = np.nan

for size in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 1)]['Excess_RET']
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 5)]['Excess_RET']

import numpy as np
import pandas as pd
import statsmodels.api as sm

def compute_t_stat(series1, series2):
    # Preprocess the data: Drop NaN and infinite values
    series1 = series1.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series2 = series2.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Stack the two series
    stacked_series = pd.concat([series1, series2], ignore_index=True)

    # Create interaction term: 1 for series1, -1 for series2
    interaction = [-1] * len(series1) + [1] * len(series2)

    # Run regression
    X_stacked = sm.add_constant(interaction)
    model_stacked = sm.OLS(stacked_series, X_stacked).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Extract the difference in means (coefficient on interaction) and its standard error
    diff_mu = model_stacked.params[1]
    se_diff_mu = model_stacked.HC0_se[1]

    # Calculate t-statistic for the difference in means
    t_stat = diff_mu / se_diff_mu
    return t_stat


# Assuming you have already defined and populated 'vwret', the rest remains largely unchanged.
# Compute t-stats for each size conditional on BTM
for btm in range(1, 6):
    series1 = vwret[(vwret['size_quintile'] == 1) & (vwret['BTM_quintile'] == btm)]['vwret']
    series5 = vwret[(vwret['size_quintile'] == 5) & (vwret['BTM_quintile'] == btm)]['vwret']
    t_stat = compute_t_stat(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = vwret[(vwret['size_quintile'] == size) & (vwret['BTM_quintile'] == 1)]['vwret']
    series5 = vwret[(vwret['size_quintile'] == size) & (vwret['BTM_quintile'] == 5)]['vwret']
    t_stat = compute_t_stat(series1, series5)

def compute_t_stat_for_difference(series1, series2):
    # Preprocess the data: Drop NaN and infinite values
    series1 = series1.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series2 = series2.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Compute weighted average differences
    differences = series2 - series1
    differences = differences.dropna()

    # Regress the differences on a constant of one
    X = np.ones_like(differences)
    model = sm.OLS(differences, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Extract coefficient (mean difference) and its standard error
    diff_mu = model.params[0]
    se_diff_mu = model.HC0_se[0]

    # Calculate t-statistic for the difference in means
    t_stat = diff_mu / se_diff_mu
    return t_stat

for btm in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == 1) & (df_tabl_1['BTM_quintile'] == btm)].groupby('date').apply(wavg, 'Excess_RET','meLag')
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == 5) & (df_tabl_1['BTM_quintile'] == btm)].groupby('date').apply(wavg, 'Excess_RET','meLag')
    t_stat = compute_t_stat(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 1)].groupby('date').apply(wavg, 'Excess_RET','meLag')
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 5)].groupby('date').apply(wavg, 'Excess_RET','meLag')
    t_stat = compute_t_stat_for_difference(series1, series5)

# Initialize a row for the t-stats in your pivot table
portfolio_pivot.loc['t-stat'] = np.nan

# Calculate and assign t-stats for BTM to the pivot table
btm_t_stats = []
for btm in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == 1) & (df_tabl_1['BTM_quintile'] == btm)]['Excess_RET']
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == 5) & (df_tabl_1['BTM_quintile'] == btm)]['Excess_RET']
    t_stat = compute_t_stat(series1, series5)
    portfolio_pivot.loc['t-stat', btm] = t_stat
    btm_t_stats.append(t_stat)  # Collect for later use if needed

# Calculate and assign t-stats for size to the pivot table
size_t_stats = []
for size in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 1)].groupby('date').apply(wavg, 'Excess_RET', 'meLag')
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 5)].groupby('date').apply(wavg, 'Excess_RET', 'meLag')
    t_stat = compute_t_stat_for_difference(series1, series5)
    portfolio_pivot.loc[size, 't-stat'] = t_stat
    size_t_stats.append(t_stat)  # Collect for later use if needed

# Make sure the 5-1 t-stat is NaN
portfolio_pivot.loc['5-1', 't-stat'] = np.nan

# Calculate average value-weighted stock returns for BTM quintiles
btm_returns_table1 = df_tabl_1.groupby(['date','BTM_quintile']).apply(wavg, 'Excess_RET','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})
btm_returns_table1 = btm_returns_table1.groupby(['BTM_quintile']).agg({'vwret': 'mean'}).reset_index()
btm_returns_table1.name = 'BTM'
btm_returns_table1 = btm_returns_table1.pivot_table(columns=['BTM_quintile'], values='vwret')*100

# Calculate average value-weighted stock returns for Size quintiles
size_returns_table1 = df_tabl_1.groupby(['date','size_quintile']).apply(wavg, 'Excess_RET','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})
size_returns_table1 = size_returns_table1.groupby(['size_quintile']).agg({'vwret': 'mean'}).reset_index()
size_returns_table1.name = 'Size'
size_returns_table1 = size_returns_table1.pivot_table(columns=['size_quintile'], values='vwret')*100

# Calculate t-stat for BTM (5th quintile vs 1st quintile)
btm_series1 = vwret[(vwret['BTM_quintile'] == 1) ]['vwret']
#df_tabl_1[df_tabl_1['BTM_quintile'] == 1].groupby('date').apply(wavg, 'Excess_RET','meLag')
btm_series5 = vwret[(vwret['BTM_quintile'] == 5) ]['vwret']
#df_tabl_1[df_tabl_1['BTM_quintile'] == 5].groupby('date').apply(wavg, 'Excess_RET','meLag')
btm_t_stat = compute_t_stat(btm_series1, btm_series5)

# Calculate t-stat for Size (5th quintile vs 1st quintile)
size_series1 = vwret[(vwret['size_quintile'] == 1) ]['vwret']
#df_tabl_1[df_tabl_1['size_quintile'] == 1].groupby('date').apply(wavg, 'Excess_RET','meLag')

size_series5 = vwret[(vwret['size_quintile'] == 5) ]['vwret']
#df_tabl_1[df_tabl_1['size_quintile'] == 5].groupby('date').apply(wavg, 'Excess_RET','meLag')
size_t_stat = compute_t_stat(size_series1, size_series5)

import pandas as pd

# Assuming you've already computed btm_returns_table1 and size_returns_table1 as given

# Compute 5-1 differences
btm_diffpnlb = btm_returns_table1[5] - btm_returns_table1[1]
size_diffpnlb = size_returns_table1[5] - size_returns_table1[1]

# Create the desired DataFrame
pnlB0 = {
    'Ptf 1': [btm_returns_table1[1].values[0], size_returns_table1[1].values[0]],
    'Ptf 2': [btm_returns_table1[2].values[0], size_returns_table1[2].values[0]],
    'Ptf 3': [btm_returns_table1[3].values[0], size_returns_table1[3].values[0]],
    'Ptf 4': [btm_returns_table1[4].values[0], size_returns_table1[4].values[0]],
    'Ptf 5': [btm_returns_table1[5].values[0], size_returns_table1[5].values[0]],
    '5 − 1': [btm_diffpnlb.values[0], size_diffpnlb.values[0]],
    't-stat': ['-', '-'] # Placeholder for now
}

pnlB = pd.DataFrame(pnlB0, index=['BTM', 'Size'])
pnlB['t-stat'] = [btm_t_stat, size_t_stat]
pnlB

tbl1_pnlC = df_tabl_1.copy()
# Compute ln_E
tbl1_pnlC['lnme'] = tbl1_pnlC['mejun'].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)

# Set book_equity values <= 0 to NaN
tbl1_pnlC.loc[tbl1_pnlC['book_equity'] <= 0, 'book_equity'] = np.nan

# Compute ln_BE/E and replace infinite values with NaN
tbl1_pnlC['lnbeme'] = (tbl1_pnlC['book_equity'] / tbl1_pnlC['medec']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC.loc[np.isinf(tbl1_pnlC['lnbeme']).abs(), 'lnbeme'] = np.nan

# Compute ln_BA/E and replace infinite values with NaN
tbl1_pnlC['lnbame'] = (tbl1_pnlC['atq'] / tbl1_pnlC['medec']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC.loc[np.isinf(tbl1_pnlC['lnbame']).abs(), 'lnbame'] = np.nan

# Compute ln_BA/BE and replace infinite values with NaN
tbl1_pnlC['lnbabe'] = (tbl1_pnlC['atq'] / tbl1_pnlC['book_equity']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC.loc[np.isinf(tbl1_pnlC['lnbabe']).abs(), 'lnbabe'] = np.nan

mktr = pd.read_csv('yourmarketdata.csv')
# Rename the 'Unnamed' column to 'date'
mktr.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
# Parse the date column as datetime with the correct format
mktr['date'] = pd.to_datetime(mktr['date'], format='%Y%m')

# Select data from 1971 to 2012
start_date_mkt = pd.to_datetime('1971-01-01')
end_date_mkt = pd.to_datetime('2012-12-31')
mktr = mktr[(mktr['date'] >= start_date_mkt) & (mktr['date'] <= end_date_mkt)]
# Drop the 'SMB', 'HML', and 'RF' columns
mktr = mktr.drop(columns=['SMB', 'HML', 'RF'])

# Extract year and month from the 'date' column for both dataframes
tbl1_pnlC['year_month'] = tbl1_pnlC['date'].dt.to_period('M')
mktr['year_month'] = mktr['date'].dt.to_period('M')
mktbetadf = pd.merge(tbl1_pnlC, mktr, on='year_month', how='inner')
mktbetadf.rename(columns={'date_x': 'date'}, inplace=True)

june_datac = mktbetadf[(mktbetadf['date'].dt.month == 6) & (mktbetadf['count'] > 2)]

# Group data by permno
grouped = june_datac.groupby('permno')

# Define function to get beta for each stock
def get_beta(group):
    y = group['Excess_RET']
    X = group['Mkt-RF']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    beta = model.params['Mkt-RF']
    return beta

# Apply function to each group
june_datac = june_datac.groupby(['permno', 'date']).apply(get_beta)

betas = pd.DataFrame(june_datac)
betas = betas.reset_index()
betas = betas.rename(columns={0: 'beta'})

mkt_pnlc = pd.merge(mktbetadf, betas[['permno', 'date', 'beta']], on=['permno', 'date'], how='left')

mkt_pnlc.drop(['date_y', 'year_month'], axis=1, inplace=True)

def update_beta(group):
    # Identify the value of `beta` for every June in this group
    june_values = group.set_index('date').resample('A-JUN').last().reset_index()[['date', 'beta']]

    for index, row in june_values.iterrows():
        june_value = row['beta']

        # Add 12 months to the June date for the start date (July of next year)
        start_date = row['date'] + pd.DateOffset(months=13)  # This will be July of next year
        end_date = start_date + pd.DateOffset(months=12)  # This will be June of the year after

        # Update the `beta` values for this date range
        mask = (group['date'] >= start_date) & (group['date'] <= end_date)
        group.loc[mask, 'beta'] = june_value

    return group

# Apply the function on each `permno` group
beta_updated = mkt_pnlc.groupby('permno').apply(update_beta).reset_index(drop=True)

print(beta_updated[['Excess_RET', 'beta', 'lnme', 'lnbeme', 'lnbame', 'lnbabe']].dtypes)

beta_updated = beta_updated.dropna(subset=['beta', 'lnme', 'lnbeme', 'lnbame', 'lnbabe', 'Excess_RET'])
beta_updated['beta'] = pd.to_numeric(beta_updated['beta'], errors='coerce')

import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Assuming you've loaded your dataframe as beta_updated
beta_updated_C = beta_updated.copy()



# 2. Run cross-sectional regressions
resultsc1 = []

def regressionfm(group):
    X = add_constant(group[['beta', 'lnme', 'lnbeme', 'lnbame', 'lnbabe']])
    y = group['RET']
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return model.params

resultspc1 = beta_updated_C.groupby('date').apply(regressionfm)
# Reshape the Series into a DataFrame
df_resultspc1 = resultspc1.unstack()

# Calculate mean coefficients
mean_coefficientsc1 = df_resultspc1.mean()

# Calculate t-statistics
t_statisticsc1 = df_resultspc1.mean() / df_resultspc1.std()

results_df_c1 = pd.DataFrame({
    'coefficients': mean_coefficientsc1*10,
    't-stats': t_statisticsc1*10
})

# Drop the row corresponding to the constant
results_df_c1 = results_df_c1.drop('const')

beta_updated.columns

import matplotlib.pyplot as plt
import seaborn as sns

df_plot = df_tabl_1.copy()

# Set up the subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plotting leverage vs. scaled returns
sns.scatterplot(data=df_plot, x="Leverage_shifted", y="Excess_RET", ax=axes[0])
axes[0].set_title("Panel A: Historical Stock Returns")
locs = axes[0].get_yticks()
axes[0].set_ylabel("Monthly Returns (Percentages)")  # Renaming y-axis for the first plot
axes[0].set_yticklabels([f"{int(item*100)}" for item in locs])

# Plotting leverage vs. unlevered return
sns.scatterplot(data=df_plot, x="Leverage_shifted", y="scaled_returns", ax=axes[1])  # assuming excess return is stored in 'excess ret' column
axes[1].set_title("Panel B: Unlevered Return")
locs = axes[1].get_yticks()
axes[1].set_ylabel("Monthly Returns (Percentages)")  # Renaming y-axis for the second plot
axes[1].set_yticklabels([f"{int(item*100)}" for item in locs])

# Setting the shared x-axis label
axes[1].set_xlabel("Leverage")  # Renaming x-axis

plt.tight_layout()  # Adjusts the layout so that plots don't overlap
plt.show()

df_tabl_3 = final_df.copy()

df_tabl_3['BTM'] = df_tabl_3['book_equity'] / df_tabl_3['medec']
df_tabl_3 = df_tabl_3.dropna(subset=['BTM'])
df_tabl_3 = df_tabl_3[df_tabl_3['SHRCD'].isin([10, 11])]
nyse_stocks3 = df_tabl_3[(df_tabl_3['EXCHCD']==1) & (df_tabl_3['book_equity']>0) & (df_tabl_3['BTM'] > 0) & (df_tabl_3['count']>=1) & ((df_tabl_3['SHRCD']==10) | (df_tabl_3['SHRCD']==11))]

me_breakpoints3 = nyse_stocks3['mejun'].quantile([0.2, 0.4, 0.6, 0.8]).values
btm_breakpoints3 = nyse_stocks3['BTM'].quantile([0.2, 0.4, 0.6, 0.8]).values

def assign_me_portfolio3(market_equity):
    if market_equity <= me_breakpoints[0]:
        return 1
    elif market_equity <= me_breakpoints[1]:
        return 2
    elif market_equity <= me_breakpoints[2]:
        return 3
    elif market_equity <= me_breakpoints[3]:
        return 4
    else:
        return 5

def assign_btm_portfolio3(btm):
    if btm <= btm_breakpoints[0]:
        return 1
    elif btm <= btm_breakpoints[1]:
        return 2
    elif btm <= btm_breakpoints[2]:
        return 3
    elif btm <= btm_breakpoints[3]:
        return 4
    else:
        return 5

df_tabl_3['size_quintile'] = df_tabl_3['mejun'].apply(assign_me_portfolio3)
df_tabl_3['BTM_quintile'] = df_tabl_3['BTM'].apply(assign_btm_portfolio3)

vwret3 = df_tabl_3.groupby(['date','size_quintile', 'BTM_quintile']).apply(wavg, 'scaled_returns','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})

portfolios3 = vwret3.groupby(['size_quintile', 'BTM_quintile']).agg({'vwret': 'mean'}).reset_index()
portfolio_pivot3 = portfolios3.pivot_table(index='size_quintile', columns=['BTM_quintile'], values='vwret')*100
# Compute 5-1 differences for sizes and BTMs
portfolio_pivot3['5-1'] = portfolio_pivot3[5] - portfolio_pivot3[1]

# Place the 5-1 BTM differences into the table rather than appending
btm_diff3 = (portfolio_pivot3.loc[5] - portfolio_pivot3.loc[1])
portfolio_pivot3.loc['5-1'] = btm_diff3

# Remove the value in the last row and column which is the difference of both size and BTM
portfolio_pivot3.loc['5-1', '5-1'] = np.nan

import numpy as np

# Initialize the row for t-stats
portfolio_pivot3.loc['t-stat', :] = np.nan

# Calculate t-stats for BTM
for btm in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == 1) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == 5) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    t_stat = compute_t_stat(series1, series5)
    portfolio_pivot3.loc['t-stat', btm] = t_stat

# Calculate t-stats for size
for size in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 1)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 5)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    t_stat = compute_t_stat(series1, series5)
    portfolio_pivot3.loc[size, 't-stat'] = t_stat

# Add a t-stat for the 5-1 size and BTM difference
size1_series = df_tabl_3[df_tabl_3['size_quintile'] == 1].groupby('date').apply(wavg, 'scaled_returns', 'meLag')
size5_series = df_tabl_3[df_tabl_3['size_quintile'] == 5].groupby('date').apply(wavg, 'scaled_returns', 'meLag')
portfolio_pivot3.loc['t-stat', '5-1'] = compute_t_stat(size1_series, size5_series)
portfolio_pivot3.loc['5-1', 't-stat'] = np.nan
portfolio_pivot3.loc['t-stat', '5-1'] = np.nan

for btm in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == 1) & (df_tabl_3['BTM_quintile'] == btm)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == 5) & (df_tabl_3['BTM_quintile'] == btm)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    t_stat = compute_t_stat(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 1)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 5)].groupby('date').apply(wavg, 'scaled_returns','meLag')
    t_stat = compute_t_stat(series1, series5)

# Calculate average value-weighted stock returns for BTM quintiles
btm_returns_table3 = df_tabl_1.groupby(['date','BTM_quintile']).apply(wavg, 'scaled_returns','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})
btm_returns_table3 = btm_returns_table3.groupby(['BTM_quintile']).agg({'vwret': 'mean'}).reset_index()
btm_returns_table3.name = 'BTM'
btm_returns_table3 = btm_returns_table3.pivot_table(columns=['BTM_quintile'], values='vwret')*100

# Calculate average value-weighted stock returns for Size quintiles
size_returns_table3 = df_tabl_3.groupby(['date','size_quintile']).apply(wavg, 'scaled_returns','meLag').to_frame().reset_index().rename(columns={0: 'vwret'})
size_returns_table3 = size_returns_table3.groupby(['size_quintile']).agg({'vwret': 'mean'}).reset_index()
size_returns_table3.name = 'Size'
size_returns_table3 = size_returns_table3.pivot_table(columns=['size_quintile'], values='vwret')*100

# Calculate t-stat for BTM (5th quintile vs 1st quintile)
btm_series13 = df_tabl_3[df_tabl_3['BTM_quintile'] == 1].groupby('date').apply(wavg, 'scaled_returns','meLag')
btm_series53 = df_tabl_3[df_tabl_3['BTM_quintile'] == 5].groupby('date').apply(wavg, 'scaled_returns','meLag')
btm_t_stat3 = compute_t_stat(btm_series13, btm_series53)

# Calculate t-stat for Size (5th quintile vs 1st quintile)
size_series13 = df_tabl_3[df_tabl_3['size_quintile'] == 1].groupby('date').apply(wavg, 'scaled_returns','meLag')
size_series53 = df_tabl_3[df_tabl_3['size_quintile'] == 5].groupby('date').apply(wavg, 'scaled_returns','meLag')
size_t_stat3 = compute_t_stat(size_series13, size_series53)

import pandas as pd

# Assuming you've already computed btm_returns_table1 and size_returns_table1 as given

# Compute 5-1 differences
btm_diffpnlb3 = btm_returns_table3[5] - btm_returns_table3[1]
size_diffpnlb3 = size_returns_table3[5] - size_returns_table3[1]

# Create the desired DataFrame
pnlB03 = {
    'Ptf 1': [btm_returns_table3[1].values[0], size_returns_table3[1].values[0]],
    'Ptf 2': [btm_returns_table3[2].values[0], size_returns_table3[2].values[0]],
    'Ptf 3': [btm_returns_table3[3].values[0], size_returns_table3[3].values[0]],
    'Ptf 4': [btm_returns_table3[4].values[0], size_returns_table3[4].values[0]],
    'Ptf 5': [btm_returns_table3[5].values[0], size_returns_table3[5].values[0]],
    '5 − 1': [btm_diffpnlb3.values[0], size_diffpnlb3.values[0]],
    't-stat': ['-', '-'] # Placeholder for now
}

pnlB3 = pd.DataFrame(pnlB03, index=['BTM', 'Size'])
pnlB3['t-stat'] = [btm_t_stat3, size_t_stat3]
pnlB3

tbl1_pnlC3 = df_tabl_3.copy()
# Compute ln_E
tbl1_pnlC3['lnme'] = tbl1_pnlC3['mejun'].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)

# Set book_equity values <= 0 to NaN
tbl1_pnlC3.loc[tbl1_pnlC3['book_equity'] <= 0, 'book_equity'] = np.nan

# Compute ln_BE/E and replace infinite values with NaN
tbl1_pnlC3['lnbeme'] = (tbl1_pnlC3['book_equity'] / tbl1_pnlC3['medec']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC3.loc[np.isinf(tbl1_pnlC3['lnbeme']).abs(), 'lnbeme'] = np.nan

# Compute ln_BA/E and replace infinite values with NaN
tbl1_pnlC3['lnbame'] = (tbl1_pnlC3['atq'] / tbl1_pnlC3['medec']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC3.loc[np.isinf(tbl1_pnlC3['lnbame']).abs(), 'lnbame'] = np.nan

# Compute ln_BA/BE and replace infinite values with NaN
tbl1_pnlC3['lnbabe'] = (tbl1_pnlC3['atq'] / tbl1_pnlC3['book_equity']).apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
tbl1_pnlC3.loc[np.isinf(tbl1_pnlC3['lnbabe']).abs(), 'lnbabe'] = np.nan
mktbetadf3 = mktbetadf.copy()
june_datac3 = mktbetadf3[(mktbetadf3['date'].dt.month == 6) & (mktbetadf3['count'] > 2)]
# Group data by permno
grouped3 = june_datac3.groupby('permno')

# Define function to get beta for each stock
def get_beta3(group):
    y = group['scaled_returns']
    X = group['Mkt-RF']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    beta3 = model.params['Mkt-RF']
    return beta3

# Apply function to each group
june_datac3 = june_datac3.groupby(['permno', 'date']).apply(get_beta3)

betas3 = pd.DataFrame(june_datac3)
betas3 = betas3.reset_index()
betas3 = betas3.rename(columns={0: 'beta'})
mkt_pnlc3 = pd.merge(mktbetadf3, betas3[['permno', 'date', 'beta']], on=['permno', 'date'], how='left')
mkt_pnlc3.drop(['date_y', 'year_month'], axis=1, inplace=True)
def update_beta3(group):
    # Identify the value of `beta` for every June in this group
    june_values3 = group.set_index('date').resample('A-JUN').last().reset_index()[['date', 'beta']]

    for index, row in june_values3.iterrows():
        june_value3 = row['beta']

        # Add 12 months to the June date for the start date (July of next year)
        start_date = row['date'] + pd.DateOffset(months=13)  # This will be July of next year
        end_date = start_date + pd.DateOffset(months=12)  # This will be June of the year after

        # Update the `beta` values for this date range
        mask = (group['date'] >= start_date) & (group['date'] <= end_date)
        group.loc[mask, 'beta'] = june_value3

    return group

# Apply the function on each `permno` group
beta_updated3 = mkt_pnlc3.groupby('permno').apply(update_beta3).reset_index(drop=True)

beta_updated3 = beta_updated3.dropna(subset=['beta', 'lnme', 'lnbeme', 'lnbame', 'lnbabe', 'Excess_RET'])
beta_updated3['beta'] = pd.to_numeric(beta_updated3['beta'], errors='coerce')
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Assuming you've loaded your dataframe as beta_updated
beta_updated_C3 = beta_updated3.copy()



# 2. Run cross-sectional regressions
resultsc13 = []

def regressionfm3(group):
    X = add_constant(group[['beta', 'lnme', 'lnbeme', 'lnbame', 'lnbabe']])
    y = group['scaled_returns']
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return model.params

resultspc13 = beta_updated_C3.groupby('date').apply(regressionfm3)
# Reshape the Series into a DataFrame
#df_resultspc13 = resultspc13.unstack()


# Calculate mean coefficients
mean_coefficientsc13 = resultspc13.mean()

# Calculate t-statistics
t_statisticsc13 = resultspc13.mean() / resultspc13.std()

results_df_c13 = pd.DataFrame({
    'coefficients': mean_coefficientsc13*10,
    't-stats': t_statisticsc13*10
})

# Drop the row corresponding to the constant
results_df_c13 = results_df_c13.drop('const')

import pandas as pd
import statsmodels.api as sm

def compute_t_stat(series1, series2):
    # Preprocess the data: Drop NaN and infinite values
    series1 = series1.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series2 = series2.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Stack the two series
    stacked_series = pd.concat([series1, series2], ignore_index=True)

    # Create interaction term: 1 for series1, -1 for series2
    interaction = [-1] * len(series1) + [1] * len(series2)

    # Run regression
    X_stacked = sm.add_constant(interaction)
    model_stacked = sm.OLS(stacked_series, X_stacked).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Extract the difference in means (coefficient on interaction) and its standard error
    diff_mu = model_stacked.params[1]
    se_diff_mu = model_stacked.HC0_se[1]

    # Calculate t-statistic for the difference in means
    t_stat = diff_mu / se_diff_mu
    return t_stat

# Compute t-stats for each size conditional on BTM
for btm in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == 1) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == 5) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    t_stat = compute_t_stat(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 1)]['scaled_returns']
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 5)]['scaled_returns']
    t_stat = compute_t_stat(series1, series5)

import numpy as np
import pandas as pd
import statsmodels.api as sm

def compute_t_stat(series1, series2):
    # Preprocess the data: Drop NaN and infinite values
    series1 = series1.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series2 = series2.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Stack the two series
    stacked_series = pd.concat([series1, series2], ignore_index=True)

    # Create interaction term: 1 for series1, -1 for series2
    interaction = [-1] * len(series1) + [1] * len(series2)

    # Run regression
    X_stacked = sm.add_constant(interaction)
    model_stacked = sm.OLS(stacked_series, X_stacked).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Extract the difference in means (coefficient on interaction) and its standard error
    diff_mu = model_stacked.params[1]
    se_diff_mu = model_stacked.HC0_se[1]

    # Calculate t-statistic for the difference in means
    t_stat = diff_mu / se_diff_mu
    return t_stat

# Compute t-stats for each size conditional on BTM
for btm in range(1, 6):
    series1 = df_tabl_3[(df_tabl_1['size_quintile'] == 1) & (df_tabl_1['BTM_quintile'] == btm)]['Excess_RET']
    series5 = df_tabl_3[(df_tabl_1['size_quintile'] == 5) & (df_tabl_1['BTM_quintile'] == btm)]['Excess_RET']
    t_stat = compute_t_stat(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 1)]['Excess_RET']
    series5 = df_tabl_1[(df_tabl_1['size_quintile'] == size) & (df_tabl_1['BTM_quintile'] == 5)]['Excess_RET']
    t_stat = compute_t_stat(series1, series5)

# Calculate average value-weighted stock returns for BTM quintiles
btm_returns_table1 = df_tabl_1.groupby('BTM_quintile').apply(lambda x: (x['Excess_RET'] * x['meLag']).sum() / x['meLag'].sum()) * 100
btm_returns_table1.name = 'BTM'

# Calculate average value-weighted stock returns for Size quintiles
size_returns_table1 = df_tabl_1.groupby('size_quintile').apply(lambda x: (x['Excess_RET'] * x['meLag']).sum() / x['meLag'].sum()) * 100
size_returns_table1.name = 'Size'


# Calculate the difference 5-1 for BTM and Size
btm_returns_table1['5-1'] = btm_returns_table1[5] - btm_returns_table1[1]
size_returns_table1['5-1'] = size_returns_table1[5] - size_returns_table1[1]


# Combine results into a panel B matrix
panel_B_matrix_table1 = pd.DataFrame({
    'BTM': btm_returns_table1.values,
    'Size': size_returns_table1.values
}).T

# Correct column names to match the given format
panel_B_matrix_table1.columns = ['Ptf 1', 'Ptf 2', 'Ptf 3', 'Ptf 4', 'Ptf 5', '5-1']

# Extract the Excess_RET for each BTM and size portfolio
btm_portfolios = {}
size_portfolios = {}

for btm in range(1, 6):
    btm_portfolios[btm] = df_tabl_1[df_tabl_1['BTM_quintile'] == btm]['Excess_RET']

for size in range(1, 6):
    size_portfolios[size] = df_tabl_1[df_tabl_1['size_quintile'] == size]['Excess_RET']

# Compute t-stat for the difference between Portfolio 5 and Portfolio 1 for BTM quintiles
btm_t_stat = compute_t_stat(btm_portfolios[1], btm_portfolios[5])

# Compute t-stat for the difference between Portfolio 5 and Portfolio 1 for Size quintiles
size_t_stat = compute_t_stat(size_portfolios[1], size_portfolios[5])

# Add these t-stat values to your panel_B_matrix_table1
panel_B_matrix_table1['t-stat'] = [btm_t_stat, size_t_stat]

#Panel C
df_tabl_1_panelC = df_tabl_1.copy()

df_tabl_3 = final_df.copy()

df_tabl_1['BTM'] = df_tabl_1['book_equity'] / df_tabl_1['medec']
df_tabl_1 = df_tabl_1.dropna(subset=['BTM'])

df_tabl_3 = df_tabl_3[(df_tabl_3['market_equity'] > 0)]

df_tabl_3['BTM'] = df_tabl_3['book_equity']/ (df_tabl_3['medec'] )

df_tabl_3 = df_tabl_3.dropna(subset=['ltq', 'atq', 'market_equity', 'book_equity', 'meLag',	'mejun', 'medec', 'scaled_returns'])

nyse_stocks3 = df_tabl_3[(df_tabl_3['EXCHCD']==1) & (df_tabl_3['book_equity']>0) & (df_tabl_3['BTM'] > 0)]

me_breakpoints3 = nyse_stocks3['mejun'].quantile([0.2, 0.4, 0.6, 0.8]).values
btm_breakpoints3 = nyse_stocks3['BTM'].quantile([0.2, 0.4, 0.6, 0.8]).values

def assign_me_portfolio3(market_equity):
    if market_equity <= me_breakpoints3[0]:
        return 1
    elif market_equity <= me_breakpoints3[1]:
        return 2
    elif market_equity <= me_breakpoints3[2]:
        return 3
    elif market_equity <= me_breakpoints3[3]:
        return 4
    else:
        return 5

def assign_btm_portfolio3(btm):
    if btm <= btm_breakpoints3[0]:
        return 1
    elif btm <= btm_breakpoints3[1]:
        return 2
    elif btm <= btm_breakpoints3[2]:
        return 3
    elif btm <= btm_breakpoints3[3]:
        return 4
    else:
        return 5

df_tabl_3['size_quintile'] = df_tabl_3['mejun'].apply(assign_me_portfolio3)
df_tabl_3['BTM_quintile'] = df_tabl_3['BTM'].apply(assign_btm_portfolio3)

# Group by size and BTM quintiles, compute value-weighted average returns
portfolio_returns3 = df_tabl_3.groupby(['size_quintile', 'BTM_quintile']).apply(
    lambda x: (x['scaled_returns'] * x['meLag']).sum() / x['meLag'].sum()
)

portfolio_matrix3 = portfolio_returns3.unstack(level=-1) *100

# Compute 5-1 differences for sizes and BTMs
portfolio_matrix3['5-1'] = portfolio_matrix3[5] - portfolio_matrix3[1]

# Place the 5-1 BTM differences into the table rather than appending
btm_diff3 = (portfolio_matrix3.loc[5] - portfolio_matrix3.loc[1])
portfolio_matrix3.loc['5-1'] = btm_diff3

# Remove the value in the last row and column which is the difference of both size and BTM
portfolio_matrix3.loc['5-1', '5-1'] = np.nan

import numpy as np
import pandas as pd
import statsmodels.api as sm

def compute_t_stat3(series1, series2):
    # Preprocess the data: Drop NaN and infinite values
    series1 = series1.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series2 = series2.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Stack the two series
    stacked_series = pd.concat([series1, series2], ignore_index=True)

    # Create interaction term: 1 for series1, -1 for series2
    interaction = [1] * len(series1) + [-1] * len(series2)

    # Run regression
    X_stacked = sm.add_constant(interaction)
    model_stacked = sm.OLS(stacked_series, X_stacked).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Extract the difference in means (coefficient on interaction) and its standard error
    diff_mu = model_stacked.params[1]
    se_diff_mu = model_stacked.HC0_se[1]

    # Calculate t-statistic for the difference in means
    t_stat = diff_mu / se_diff_mu
    return t_stat

# Compute t-stats for each size conditional on BTM
for btm in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == 1) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == 5) & (df_tabl_3['BTM_quintile'] == btm)]['scaled_returns']
    t_stat = compute_t_stat3(series1, series5)


# Compute t-stats for each BTM conditional on size
for size in range(1, 6):
    series1 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 1)]['scaled_returns']
    series5 = df_tabl_3[(df_tabl_3['size_quintile'] == size) & (df_tabl_3['BTM_quintile'] == 5)]['scaled_returns']
    t_stat = compute_t_stat3(series1, series5)

# Calculate average value-weighted stock returns for BTM quintiles
btm_returns3 = df_tabl_3.groupby('BTM_quintile').apply(lambda x: (x['scaled_returns'] * x['meLag']).sum() / x['meLag'].sum()) * 100
btm_returns3.name = 'BTM'

# Calculate average value-weighted stock returns for Size quintiles
size_returns3 = df_tabl_3.groupby('size_quintile').apply(lambda x: (x['scaled_returns'] * x['meLag']).sum() / x['meLag'].sum()) * 100
size_returns3.name = 'Size'


# Calculate the difference 5-1 for BTM and Size
btm_returns3['5-1'] = btm_returns3[5] - btm_returns3[1]
size_returns3['5-1'] = size_returns3[5] - size_returns3[1]


# Combine results into a panel B matrix
panel_B_matrix3 = pd.DataFrame({
    'BTM': btm_returns3.values,
    'Size': size_returns3.values
}).T

# Correct column names to match the given format
panel_B_matrix3.columns = ['Ptf 1', 'Ptf 2', 'Ptf 3', 'Ptf 4', 'Ptf 5', '5-1']
