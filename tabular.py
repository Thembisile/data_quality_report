# ---------------------------------------------------------------
# load packages
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html
from sklearn.model_selection import KFold
import string
import datetime
from datetime import date
from dateutil.parser import parse
#
#

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------







# ---------------------------------------------------------------
# Functions
# ---------------------------------------------------------------

def hot_encode_categoricals(data_frame, theta=0.003):
    """
    Decription:
     hot encode categorical variables within a pandas.DataFrame
     Look for all categories which have the dtype encode all of them
     drop the originally encoded features
     Return the pandas.DataFrame
     
     Parameters
     __________
     data  : pandas.DataFrame
        theta : threshold
     Returns
     _______
     pandas.DataFrame
    """
    data = detect_column_types(data_frame, theta)
    cols = data.select_dtypes(include='category').columns.tolist()
    data = pd.concat([data, pd.get_dummies(data[cols], drop_first=True)], axis=1)
    data.drop(cols, axis=1, inplace=True)
    
    return data
def convert_dates_to_integer(date):
    """
    Desription:
     Convert dates to integer values within a pandas data frame
     
     Parameters
     __________
        date : (YYYY-MM-DD or any format)
     Returns
     _______
     Return only converted dates (int)
     """
    dates = pd.to_datetime(date)
    year, month, day = dates.year, dates.month, dates.day
    return 10000*year + 100*month + day

def convert_string_dates_to_datetime(date):
    return datetime.datetime.strptime(date ,'%d-%m-%y')

def hot_encode_free_text(data_frame, col_list, remove_headings):
    """
    Description:
    Hot encode several columns of free text and handle the overlapping
    words from different columns within a pandas data frame.

    Parameters
    ----------
    data : pandas.DataFrame with columns to hot encode
        columns : list, default None
        remove_headings : bool, default False
    Returns
    -------
    pandas.DataFrame
     
    Raise ValueError
    ________________
    Column labels to use when ``orient='index'``. 
    Raises a ValueError if used with ``orient='columns'``.
    """
    if isinstance(data_frame, pd.DataFrame):
        try:
            data_frame[col_list] = data_frame[col_list].applymap(remove_punctuation)
            if remove_headings == True:
                return pd.get_dummies(data_frame[col_list],\
                       prefix=['Column_'+ str(i) for i in range(1, (len(col_list)+1))])
            return pd.get_dummies(data_frame[col_list], drop_first=False)
        except Exception as error:
            print("type error: " + str(error))
    else:
        raise ValueError('Invalid Input: Not a dataframe')
        

def remove_punctuation(x):
    """
    Helper function to remove punctuation from a string
    x: string
    
    returns
    
    x string
    """
    exclude = set(string.punctuation)
    try:
        return ''.join(ch for ch in x if ch not in exclude)
    except:
        return x        
def hot_encode_response(data, column):
    """
    Description:
    Hot encode a categorial or numeric response from a pandas.DataFrame.
        
    Parameters
    __________
    data   : pandas.DataFrame
        column : the response variable which is to be encoded
    Returns
    _______
    pandas.DataFrame
    """
    distinct = data[column].unique()
    number_of_distinct = len(data[column].unique())
    temp = pd.DataFrame(index=distinct, data=np.arange(number_of_distinct))
    data[column] = data[column].map(temp.to_dict()[0])
    
    return data
    
def bucket_response(data_frame, column, alpha, distribution=False):
    """
    Group the numeric response into a some equaly sized or separatedbuckets
    
    Paramaters
    __________
    data_frame  : pandas.DataFrame
        col   : list , default None
        alpha : int , number of sized buckets 
        distribution : boolean , False if the range will be equally sliced  and True for equally sliced elements
    Returns
    _______
    pandas.DataFrame
    """
    if isinstance(data_frame, pd.DataFrame) & isinstance(alpha, int):
        try:
            if distribution == False:
                data_frame['buckets'] = pd.cut(data_frame[column], alpha)
                return pd.get_dummies(data_frame, columns=['buckets'], dummy_na=True)
            else:
                data_frame['buckets equal dist'] = pd.qcut(data_frame[column], alpha)
                return pd.get_dummies(data_frame, columns=['buckets equal dist'], dummy_na=True)
        except Exception as error:
            print("type error: " + str(error))
    else:
        raise ValueError('Invalid Input: Not a dataframe or Not an integer')
def drop_rows(data_frame):
    """
    Description:
    Drop rows with missing data with a pandas data frame

    Paramters
    _________
        data_frame: pandas.DataFrame
    Returns
    _______
    pandas.DataFrame
    """
    return data_frame.dropna(axis=0, inplace=True)
def drop_static_column(data_frame, column_name, alpha):
    """
    Description:
    Drop columns with static values within a pandas.DataFrame.

    Parameters
    __________
    data_frame  : pandas.DataFrame
        column_name : column name
        alpha       : significance level (float)
    Returns
    _______
    pandas.DataFrame
    """
    dff = data_frame[column_name].value_counts()
    data_frame_1 = pd.DataFrame(dff)
    data_frame_1['Percent'] = (data_frame_1[column_name]/sum(data_frame_1[column_name]))*100
    if max(data_frame_1.Percent) > alpha:
        return data_frame_1.drop(column_name, axis=1, inplace=True)
    return data_frame

def drop_static_columns(data_frame, alpha):
    """
    Description:
    Drop columns with static values within a pandas data frame.

    Parameters
    __________
        data_frame : pandas.DataFrame
    Returns
    _______
    pandas.DataFrame
    """
    data_frame.dropna(axis=1, inplace=True)
    new_data_frame = data_frame.columns.to_series().apply(lambda x: drop_static_column(data_frame, x, alpha))
    return data_frame[new_data_frame.index.values.tolist()]
def imputer(data_frame, col, method):
    """
   Description:
       Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

   params:
       data_frame: data frame
       col       : Column
       method    : impute method e.g mean, median 

   returns:
       data_frame
     """
    try:  
        if isinstance(data_frame[col], float) | isinstance(data_frame[col], int):
            if method == 'median':
                return data_frame[col].fillna(data_frame[col].median(), inplace=True)

            elif method == 'mean':
                return data_frame[col].fillna(data_frame[col].mean(), inplace=True)

            elif method == 'mode':
                return data_frame[col].fillna(data_frame[col].mode(), inplace=True)
        else:
            if method == 'bfill':
                return data_frame[col].fillna(method='bfill', inplace=True)

            elif method == 'ffill':
                return data_frame[col].fillna(method='ffill', inplace=True)

            elif method == 'most_frequent':
                return data_frame[col].value_counts().index[0]
    except Exception:
        print('Unable to impute column, please check if there are any values in the column')
        
def detect_column_types(data_frame, alpha, beta):
    """
    Description:
        given a pandas data frame detect what kind of data is contained within each column exmaples will include
                text           - words, that are mostly unique over rows
                categorical    - string or number that indicate categorical data
                date           - a string matching any date format
                integer        - a numeric integer value
                continous      - a numeric continous value
    params:
        data_frame: data frame
        alpha: float, 
        beta: float, threshold for categorical
        
    returns:
        dict
    """
    data_col = data_frame.columns
    types = data_frame.dtypes.tolist()
    aDict = {}
    
    for column, column_type in zip(data_col, types):
        if column_type == np.object or column_type == np.int64:
            if column_type == np.int64:
                if data_frame[column].nunique() > data_frame.shape[0]*alpha and monotonic(data_frame[data_col].values) == True:
                    aDict[column] = 'Numerical_id'
                else:
                    aDict[column] = column_type
            elif data_frame[column].nunique() / len(data_frame[column]) < beta:
                aDict[column] = 'Category'
            elif column_type == np.object and is_date(data_frame[data_col].iloc[0]) == True:
                aDict[column] = 'Date'
            else:
                aDict[column] = 'Text'
        elif column_type == np.float64:
            if data_frame[column].nunique() > data_frame.shape[0]*alpha:
                aDict[column] = 'Numerical'
        else:
            aDict[column] = column_type
    return aDict
def monotonic(series):
    """
    Description:
    checks if a series of values is monotonically increasing
    
    Parameters
    __________
        x : pd.Series
    Returns
    _______
    bool : True if series is monotonically increasing False otherwise 
    """
    if isinstance(series, int):
        try: 
            d_x = np.diff(series)
            return np.all(d_x >= 0)
        except ValueError:
            return False
    else:
        return False

def is_date(string):
    """
    Description:
    checks if a string could represent a date
    
    Parameters
    __________
        string : str
    Returns
    _______
    bool : True if string could represent a date and False otherwise 
    """
    if isinstance(string, str):
        try: 
            parse(string)
            return True
        except ValueError:
            return False
    else:
        return False
        
def data_split_cv(x_set, y_set, folds):
    """
    Description:
    produce a generator or dictionary with each cross validation cross fold
    
    Parameters
    __________
    X     : predictor variables
        y     : response variable
        folds : number of KFolds (10 or 5 folds preffered)
    Returns
    ______
    set with train and test set
    """
    k_fold = KFold(n=len(x_set), n_folds=folds)
    for train, test in k_fold:
        yield ((x_set[train], y_set[train]), (x_set[test], y_set[test]))
def timer(func, *args, **kwargs):
            
    """
    Description:
    @decorator timer function
    
    Parameters
    __________
    func     : function to  be timed
    *args    : functions arguments (non-keyworded variable length argument)
    **Kwargs : function arguments (keyworded variable length of arguments)
    Returns
    ____________
    inner function
    """
    
    def inner():
        return func(*args, **kwargs)
    return inner
# ---------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------

class DataQualityWorkflow:
    """
    Description:
    Produce a data quality report
    """
    def __init__(self, data):
        
        self.data = data
        
    def row_analysis(self):
        """
        Produce a Table containing analysis of row counts and
        row-wise nan values found in the dataset.
    
        PARAMETER
        _________
        data : pd.DataFrame
        """
        data = self.data
        total_rows = data.shape[0]
        row_w_any_nans = []
        for i in range(len(data)):
            if data.loc[i].isnull().sum() > 0:
                row_w_any_nans.append(i)
        row_w_only_nans = []
        for i in range(len(data)):
            if data.loc[i].isnull().sum() == len(data.loc[i]):
                row_w_only_nans.append(i)
        total_duplicate_rows = data.duplicated().sum()
        total_unique_rows = total_rows - total_duplicate_rows
        row_dict = {'Total_rows': total_rows,
                    'Row_w_ANY_nans':len(row_w_any_nans),
                    'Row_w_ONLY_nans':len(row_w_only_nans),
                    'Total_duplicate_rows' :total_duplicate_rows,
                    'Total_unique_rows':total_unique_rows}
        return pd.DataFrame(row_dict, index=['Count']).T.reset_index().rename(columns={'index':'Info'})
    def column_analysis(self):
        """
        Produce a table containing analysis of column counts and
        column-wise nan values found in the dataset.
        
        PARAMETER
        ___________
        data : pd.DataFrame
        """
        data = self.data
        total_columns = data.shape[1]
        columns_exl_only_nans = []
        for i in data.columns:
            if data[i].isnull().sum() > 0:
                columns_exl_only_nans.append(i)
        columns_w_only_nans = []
        for i in data.columns:
            if data[i].isnull().sum() == len(data[i]):
                columns_w_only_nans.append(i)    
        total_duplicate_columns = data.columns.duplicated().sum()
        total_unique_columns = total_columns - total_duplicate_columns
        column_dict = {'TOTAL_COLUMNS': total_columns,
                       'COLUMNS_EXCL_ONLY_NANS':len(columns_exl_only_nans),
                       'COLUMNS_W_ONLY_NANS':len(columns_w_only_nans),
                       'TOTAL_DUPLICATE_COLUMNS':total_duplicate_columns,
                       'TOTAL_UNIQUE_COLUMNS':total_unique_columns}
        return pd.DataFrame(column_dict, index=['Count']).T.reset_index().rename(columns={'index':'Info'})
    def number_of_columns_per_types(self):
        """
        Produce a table that contains the number of columns for each datatype.
        PARAMETERS
        __________
        data : pd.DataFrame
        """
        data = self.data 
        return pd.DataFrame(data.columns.to_series().groupby(data.dtypes).count(), columns=['Counts']).reset_index().rename(columns={'index':'Info'})
    def colums_grouped_per_datatype(self):
        """
        Produce a table that contains the columns by name grouped into their datatypes.
        
        PARAMETERS
        __________
        data : pd.DataFrame
        """
        data = self.data
        return pd.DataFrame([x for x in data.columns.groupby(data.dtypes).values()]\
                            , index=data.columns.groupby(data.dtypes).keys()).transpose().replace(np.nan, '')
    
    def display_side_by_side(*args):
        """ 
        Function to display pd.DataFrames side by side
        PARAMETERS
        ___________
        *args : pd.DataFrames
        """
        html_str = ''
        for data_frame in args:
            html_str += data_frame.to_html(index = None)
        display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)
    
    def column_compsition(self):
        """
        For each column within a dataframe provide the top 5 unique values
        and their density in percentage of total values

        parameters
        -------------
        data : pd.DataFrame()
        """
        data = self.data
        top_n=5
        result = {}
        for column in data.columns:
            unique_values = data[column].value_counts()
            top = unique_values.nlargest(top_n)

            row = {}
            row['total_unique'] = unique_values.count()
            for i in range(min(top_n,len(top))):
                row['top_{}'.format(i+1)] =  top.index[i]
                row['%top_{}'.format(i+1)] =  100*top.values[i]/unique_values.sum()

                result[column] = row
        return pd.DataFrame(result).T.reset_index().rename(columns={'index':'columns'})
    def statistical_quartile_analysis(self):
        """Produce a table contains the 5 number summary for each column which can be coerced in to a
        integer/float datatype
        PARAMETERS
        _________
        data : pd.DataFrame
        """
        data = self.data
        data_frame = pd.DataFrame(data.columns.to_series().groupby(data.dtypes).keys).reset_index()\
                     .rename(columns={'index': 'columns', 0 : 'DataTypes'})
        data_frame = data_frame[data_frame['DataTypes'] != 'object']
    
        statistical_dict = {
            'Minimum': np.percentile(data[data_frame[data_frame['DataTypes'] != 'object']['columns']], 0, axis=0, interpolation='linear'),
            'Lower_quartile' : np.percentile(data[data_frame[data_frame['DataTypes'] != 'object']['columns']], 25, axis=0, interpolation='linear'),
            'Mean' : np.array(data[data_frame[data_frame['DataTypes'] != 'object']['columns']].mean()),
            'Upper_quartile' : np.percentile(data[data_frame[data_frame['DataTypes'] != 'object']['columns']], 75, axis=0, interpolation='linear'),
            'Maximum' : np.percentile(data[data_frame[data_frame['DataTypes'] != 'object']['columns']], 100, axis=0, interpolation='linear')}
        return pd.DataFrame(statistical_dict, index=data[data_frame[data_frame['DataTypes'] != 'object']['columns']].columns).reset_index().rename(columns={'index':'Columns'})
    def mode_values_per_column(self):
        """
        Produce a table that contains the mode for each integer 
        and string type column in data_set
        PARAMETERS
        ___________
        data : pd.DataFrame
        """
        data = self.data
        data_frame = pd.DataFrame(data.columns.to_series().groupby(data.dtypes).keys).reset_index()\
                     .rename(columns={'index':'columns', 0 : 'DataTypes'})
        return pd.DataFrame(data[data_frame['columns']].mode().transpose()[0]).reset_index()\
               .rename(columns={'index': 'Column_name', 0 :'Mode'})
    def nulls(self):
        """
        Procude a table with potential problematic columns,
        where more than x% of columns are nulls
        
        PARAMETERS
        -------------
        data : pd.DataFrame
        """
        data = self.data
        data_frame = pd.DataFrame(data.columns.to_series().groupby(data.dtypes).keys).reset_index().rename(columns={'index': 'columns', 0 : 'DataTypes'})
        null_dict = {'Null%':list((data[data_frame['columns']].isnull().sum()/data.shape[0])*100)}
    
        data_frame = pd.DataFrame(null_dict, index=data[data_frame['columns']].columns).reset_index()\
            .rename(columns={'index': 'Column_name', 0 :'Mode'}).sort_values(by='Null%', ascending=False)\
    
        return data_frame[data_frame['Null%']!=0]
    def quartiles(self, column):
        """ 
        Produce  a tables that contain quartile Analysis for Numeric Metric Columns
        PARAMETERS
        -------------
        data : pd.DataFrame
        column : column in the data
        function : Function to display pd.DataFrame side by side
        """
        data = self.data
        quartiles = {'MAX': data[column].max(), 'UPPER_QUARTILE' : np.percentile(data[column], 75, axis=0, interpolation='linear'),\
                     'MEAN' : data[column].mean(), 'LOWER_QUARTILE' : np.percentile(data[column], 25, axis=0, interpolation='linear')\
                     , 'MIN' : data[column].min()}

        return pd.DataFrame(quartiles, index=[0]).transpose().reset_index().rename(columns={'index': 'QUARTILES', 0:''})
    def fence(self, column):
        """
        Produce a table that contains fence Analysis for Numeric Metric Columns
        PARAMETERS
        -------------
        column : Numetric Metric of interest
        """
        data=self.data
        IQR = np.percentile(data[column], 75, axis=0) - np.percentile(data[column], 25, axis=0, interpolation='linear')
    
                       
        fence = {'UPPER_OUTER' : (np.percentile(data[column], 75, axis=0, interpolation='linear') + 3*IQR),\
                 "UPPER_INNER" : (np.percentile(data[column], 75, axis=0, interpolation='linear') + 1.5*IQR),\
                 'LOWER_INNER' : (np.percentile(data[column], 25, axis=0, interpolation='linear') - 1.5*IQR),\
                 'LOWER_OUTER' : (np.percentile(data[column], 25, axis=0, interpolation='linear') - 3*IQR)}
            
        return pd.DataFrame(fence, index=[0]).transpose().reset_index().rename(columns={'index': 'FENCE', 0 :''})
    def outliers(self):
        """
        Produce a table that contains Outlier Analysis for Numeric Metric Columns
        PARAMETERS
        -------------
        column : Numetric Metric of interest
        """
        data = self.data
        data_frame = pd.DataFrame(data.columns.to_series().groupby(data.dtypes).keys).reset_index().rename(columns={'index': 'columns', 0 : 'DataTypes'})
        data_frame = data_frame[(data_frame['DataTypes'] != 'object')]

        IQR = []
        MILD = []
        EXTREME = []
        
        for i in data[data_frame['columns']].columns:
            IQR.append(np.percentile(data[i] ,75, axis= 0) - np.percentile(data[i] ,25, axis= 0))
        
        for i, j in zip(range(len(data[data_frame['columns']].columns)) , data[data_frame['columns']].columns):
            MILD.append(len([x for x in data[j] if (x < (np.percentile(data[j], 25, axis=0) - 3*IQR[i])) or (x > (np.percentile(data[j], 25, axis=0) + 3*IQR[i]))]))
            EXTREME.append(len([x for x in data[j] if x < (np.percentile(data[j], 25, axis=0) - 3*IQR[i]) or x > (np.percentile(data[j], 75, axis=0)) + 3*IQR[i]]))
        
        return pd.DataFrame({'COLUMNS': data[data_frame['columns']].columns , 'MILD OUTLIERS': MILD, 'EXTREME OUTLIERS' : EXTREME })
    def box_plots(self, column):
        """
        Produce a box plot of a Numeric Metric
        PARAMETERS
        -------------
        column : Numetric Metric of interest
        """
        data = self.data
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        axes[0].boxplot(data[column], 0, 'gD')
        axes[0].set_title('{} WITH OUTLIERS'.format(column.upper()))
        axes[1].boxplot(data[column], 0, '')
        axes[1].set_title('{} WITH NO OUTLIERS'.format(column.upper()))
        plt.close()
            
        return fig
    def value_bucketing(self, col, alpha=5):
        """
        Group the numeric response into a some equaly sized or separated
        buckets  and procude a visual of alpha buckets

        PARAMETERS
        __________
        data: pd.DataFrame
        col_list: list of column names
        alpha: number of equaly sized buckets
        """
        data = self.data
        if isinstance(data, pd.DataFrame) & isinstance(alpha, int):
            try:
                custom_bucket_array = np.linspace(data[col].min(), data[col].max(), (alpha+1))
                data['buckets'] = pd.cut(data[col], custom_bucket_array)
                data_frame = pd.get_dummies(data, columns=['buckets'], dummy_na=True)
                data_frame = data_frame[data_frame.columns[-alpha:]]
                cols = data_frame.columns
            
                fig,axes = plt.subplots(1,1,figsize=(20,10))
                axes = sns.barplot(x=cols, y=data_frame.sum().values)
                plt.tight_layout()
                plt.title('{}'.format(col.upper()))
                plt.close()
                return fig
            except Exception as error:
                print("type error: " + str(error))
        else:
            raise ValueError('Invalid Input: Not a dataframe or Not an integer')
            
    def time_series(self, date_column, metric, start_date, end_date):
        """
        Plot each numeric metric to determine whether the periodicity is consistent throughout.
        This will inform on the quality of the data.
        
        PARAMETERS
        __________
        data        : pd.DataFrame
        date_column : date column on which time series is performed
        metric      : metric of interest
        start_date  : start the time-series at  datetime(year, month, day)
        end_date    : end the time-series at  datetime(year, month, day)
        
        """
        
        def int_to_date(argdate: int) -> date_column:
            year = int(argdate / 10000)
            month = int((argdate % 10000)/100)
            day = int(argdate % 100)
            return date(year, month, day)
        
        data = self.data
        #self.int_to_date=int_to_date()
        if data[date_column].dtype == object:
            data = pd.DataFrame(data[[date_column, metric]])
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            
            fig,axes = plt.subplots(1,1,figsize=(25,10))
            axes = data[(data.index >= start_date) & (data.index <= end_date)].plot(grid=True)
                           
            plt.title('{}'.format(metric.upper()))
    
        elif data[date_column].dtype == 'int64':
            data = pd.DataFrame(data[[date_column, metric]])
            data[date_column] = pd.to_datetime(data[date_column].apply(int_to_date))
            data.set_index(date_column, inplace=True)
            data[(data.index >= start_date) & (data.index <= end_date)].plot(figsize=(25, 10), grid=True)
            plt.title('{}'.format(metric.upper()))

        else:
            data = pd.DataFrame(data[[date_column, metric]])
            data.set_index(date_column, inplace=True)
            data[(data.index >= start_date) & (data.index <= end_date)].plot(figsize=(25, 10), grid=True)
            plt.title('{}'.format(metric.upper()))
            
    def datetime_information(self, column, date):
        """
        produce Datetime summary information
        
        PARAMETERS
        __________
        data        : pd.DataFrame
        column      : site_no
        date        : date column
        """
        data = self.data
        date_dict = {'Earliest Date' : min(data[date]),
                     'Latest Date'   : max(data[date]), 
                     'Available (unique years)' : len(set([i.year for i in data[date]])),
                     'Available (unique months)': len(np.unique(data[date].map(lambda x: 100*x.year + x.month))),
                     'Available (unique weeks)' : len(np.unique(data[date].map(lambda x: 100*x.year + x.isocalendar()[1]))),
                     'Available (unique days)'  : len(np.unique(data[date])),
                     'Available (unique hours)' : sum(Counter([i.hour for i in data[date]]).values()),
                     'Maximum Datetime stamps per site' : len(data.groupby(column).max()[date]),
                     'Average Datetime stamps per site' : data.groupby(column).count().mean()[date],
                     'Minimum Datetime stamps per site' : len(data.groupby(column).min()[date]),
                     'Number of Sites' : len(np.unique(data[date]))}
        return pd.DataFrame(date_dict, index=[column])
