import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns

from sklearn.preprocessing import Imputer, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import ShuffleSplit, train_test_split, validation_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import fbeta_score, recall_score, make_scorer
from itertools import combinations

pd.options.display.max_rows = 160
pd.options.display.max_columns = 200

import warnings


# 
def ModelComplexityRF(X, y, crit='entropy', p_range=[50, 80, 90], max_depth=1):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    warnings.filterwarnings('always')
    
    # Calculate the training and testing scores
    clf = RandomForestClassifier(criterion=crit, class_weight='balanced', random_state=7, max_depth=max_depth, max_features=10)
    
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    train_scores, test_scores = validation_curve(clf, X, y, \
                                             param_name = 'max_features', \
                                             param_range = p_range, \
                                             cv = 3, 
                                             scoring = 'recall')
    

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    mx = np.argmax(test_mean)
    print("Test score for param=", p_range[mx], "max_depth=", max_depth, "recall=", 
          round(test_mean[mx], 3), "std=", round(test_std[mx],3))
    
    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Random Forest Classifier Complexity Performance')
    plt.plot(p_range, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(p_range, test_mean, 'o-', color = 'g', label = 'Test Score')
    plt.fill_between(p_range, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(p_range, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Features')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()
    
def ModelComplexity(X, y, crit='entropy', p_range=[1, 2, 3, 4, 5, 10, 15]):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    import warnings
    warnings.filterwarnings('always')
    # Vary the max_depth parameter from 1 to 15

    # Calculate the training and testing scores
    clf = DecisionTreeClassifier(criterion=crit, class_weight='balanced', random_state=7)
       
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    train_scores, test_scores = validation_curve(clf, X, y, \
                                             param_name = 'max_depth', \
                                             param_range = p_range, \
                                             cv = 3, 
                                             scoring = 'recall')
    

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    mx = np.argmax(test_mean)
    print("Test score for param=", p_range[mx], "recall=", 
          round(test_mean[mx], 3), "std=", round(test_std[mx],3))
    
    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Classifier Complexity Performance')
    plt.plot(p_range, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(p_range, test_mean, 'o-', color = 'g', label = 'Test Score')
    plt.fill_between(p_range, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(p_range, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df

def scale(df):
    x_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(df)
    #x_scaled = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df) 

    dfr = pd.DataFrame(x_scaled)
    dfr.columns = df.columns.values
    dfr.index = df.index.values
    return dfr

def plot_box_hist2(df, col, title='', **kwargs):
    '''
    Show a box plot and a histogram of the column values.
    '''
    print("Missing values for", col, df[col].isnull().sum())
    f, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(20, 9), sharex=True);
    df.boxplot(ax=ax1, column=col, by='loan_status', vert=False);
    df[df['loan_status'] == 1][col].plot.hist(ax=ax2, color='r', title='Charge-off ' + title + ' (' + col + ')', **kwargs);
    df[df['loan_status'] == 0][col].plot.hist(ax=ax3, color='g', title='Good loan ' + title + ' (' + col + ')', **kwargs);
    
    plt.show()
    
def get_missing(df, feature_list):
    '''
    Get a table of the featues and the number of missing values
    '''
    fm = [] 
    feat = []
    for feature in feature_list:
        sn = sum(df[feature].isnull())
        if sn > 0:
            fm.append(sn)
            feat.append(feature)
    return pd.DataFrame({'feature': feat, 'Missing': fm})

def chargeoff_rank(df, col):
    gls = df.groupby([col, 'loan_status']).size()
    glsx = gls.unstack('loan_status', fill_value=0)
    glsx['chargeoff_rate'] = round (glsx[1] / glsx[0] * 100, 2)
    glsx = glsx.reset_index()

    glsx.drop(columns=[0,1], inplace=True)
    glsx.sort_values(by='chargeoff_rate', inplace=True)#.unstack('zip_code')#.index #[['chargeoff_rate']]
    glsx.reset_index(drop=True, inplace=True)
    glsx.index.name = 'Index'
    glsx.columns = [col, 'chargeoff_rate']
    glsx = glsx.set_index(col)
    return glsx

def corr_heatmap(df, col):
    '''
    Print correlation coefficient and heat map
    '''
    cr = chargeoff_rank(df, col)
    crr = cr.reset_index()
    print("Correlation of {} and charge-off: {}".format(col, round(np.corrcoef(crr[col], crr.chargeoff_rate)[0,1], 3)))
    plt.figure(figsize=(20,1)); sns.heatmap(cr.T);

def analyze_col(df, col, showtable=True, showgroups=False, piesize=(6, 6)):
    '''
    Display a table, a pie chart showing relative percentages, and piecharts as good-to-chargeoff ratios.
    '''
    misv = df[col].isnull().sum()
    print("Missing values:", misv, " Missing percent:", misv / len(df))
    vc = DF(df[col].value_counts()).sort_values(by=col)
    if showtable:
        display(vc);
    n = len(vc.index) 
    plt.figure(figsize=piesize); plt.pie( vc, #df[col].value_counts(), 
                                    labels=vc.index.values, autopct="%.2f %%"); 
    plt.suptitle('Percentage of ' + col, fontsize=16);

    if not showgroups:
        return
    
    # pie chart for each group 
    gls = df.groupby([col, 'loan_status']).size()
    glsx = gls.unstack('loan_status')
    glsx['ratio'] = glsx[1] / glsx[0]
    sorted_ind = glsx.sort_values(by='ratio').index.values

    max_col = min(n, 4)
    n_rows = ((n - 1) // max_col)+1

    f, axs = plt.subplots(n_rows, max_col, figsize=(max_col * 6, n_rows * 6))
    for i, v in enumerate(sorted_ind): # vc.index.values):
        i += 1
        rows = ((i - 1) // max_col)+1
        cols = (i - 1) % max_col + 1
        
        plt.subplot(n_rows, max_col, i); 
        gls1 = gls.xs(v, level=0); gls1.index = ['Good' if v1 == 0 else 'Charge-off' for v1 in gls1.index.values]
        gls1.plot.pie(labels=gls1.index, autopct='%.2f',colors=['g', 'r'], title='Charge-off rate for ' + col + ' "' + str(v) + '"'); plt.ylabel('');
        if i == 1:
            plt.legend(labels= gls1.index, loc=1);
            
    # hide extra subplot axes
    for ax in axs.ravel():
        ax.axis('off')            

def percent_col(df, col, figsize=(8, 6)):
    '''
    Create a stacked percentage bar chart, indicating good to charge-off loan ratio.
    '''
    if df is None:
        # Data
        raw_data = {'Good': [20, 1.5, 7, 10, 5], 'Charge-off': [5, 15, 5, 10, 15]}
        df = pd.DataFrame(raw_data)
        col = 'color'
    else:
        df = df[['loan_status', col]].groupby(by=['loan_status', col]).size().unstack('loan_status')
        df.fillna(0, inplace=True)
        df.columns = ['Good', 'Charge-off']
        
    df['ratio'] = df['Charge-off'] / df['Good'] 
    df = df.sort_values(by='ratio')
    r = df.index.values

    # From raw value to percentage
    totals = [i+j for i, j in zip(df['Good'], df['Charge-off'])]
    greenBars  = [i / j * 100 for i, j in zip(df['Good'], totals)]
    orangeBars = [i / j * 100 for i, j in zip(df['Charge-off'], totals)]

    # plot
    f, ax = plt.subplots(figsize=figsize)
    barWidth = 0.85
    #names = ('A','B','C','D','E')
    # Create green Bars
    rects1 = plt.bar(r, greenBars, color='g', edgecolor='white', width=barWidth)
    # Create orange Bars
    rects2 = plt.bar(r, orangeBars, bottom=greenBars, color='r', edgecolor='white', width=barWidth)
    
    # Custom x axis
    #
    plt.xlabel(col)
    plt.xticks(r, rotation='vertical')

    plt.suptitle('Percent Charge-off vs ' + col)
    plt.legend(labels=['Good', 'Charge-off'], loc=2, bbox_to_anchor=(1, 1))
    
    
    def autolabel(rects):
        #Attach a text label above each bar displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + 0.45*height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Show graphic
    plt.show()    

def col_stats(df):
    '''
    Display null percent, min, max median, squew and unique count statistics on all features.
    '''
    nsd = pd.DataFrame()
    nsd['dtypes'] = df.dtypes
    ns = df.isnull().sum()
    #ns = ns.sort_values(ascending=False)
    nsd['Null Count'] = ns
    nsd['Null Percent'] = ns / len(df) *100
    nsd.index.name = 'columns'
    nsd.sort_values(by='Null Percent' ,inplace=True, ascending=False)

    m = df.min(0)
    m.name = 'Min'
    nsd = nsd.join(m)

    m = df.max(0)
    m.name = 'Max'
    nsd = nsd.join(m)
    nsd

    m = df.median(0)
    m.name = 'Med'
    nsd = nsd.join(m)
    nsd
    
    m = df.skew(0)
    m.name = 'Sqew'
    nsd = nsd.join(m)
    nsd

    unique_counts = []
    for col_name in nsd.index.values:
        unique_counts.append(len(df[col_name].unique()))
        #print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
    nsd['unique_counts'] = unique_counts
    return nsd

usecols = ['loan_status', 'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length', 
           'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'purpose', 
           'zip_code', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 
           'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'emp_title',
           
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 
           'collections_12_mths_ex_med', 
           'mths_since_last_major_derog', 
           'application_type', 'annual_inc_joint', 'dti_joint', 
           'verification_status_joint', 'acc_now_delinq', 
           'tot_coll_amt', 
           'tot_cur_bal', 
           'open_acc_6m', 
           'open_act_il', 
           'open_il_12m', 'open_il_24m', 
           'mths_since_rcnt_il', 'total_bal_il', 'il_util',
           'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 
           'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 
           'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 
           'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 
           'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 
           #'mths_since_recent_bc_dlq', 
           'mths_since_recent_inq', 
           #'mths_since_recent_revol_delinq', 
           'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 
           'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 
           'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 
           'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 
           'total_il_high_credit_limit']

def get_usecols():
    return usecols


# dummies
dummy_list = ['verification_status', 'home_ownership', 'verification_status_joint', 'purpose', 'emp_length']

# converters
term_c = lambda x: 0 if x.lstrip() == '36 months' else 1 # np.int32(x.lstrip()[:2]) # keep the numerical part of term and convert to int
pct_c = lambda x: x if pd.isnull(x) else np.float32(x.strip('%')) # strip the percent sign from interest rate
grade_d = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F':2, 'G': 1}
grade_c = lambda x: grade_d.get(x)
loan_status_c = lambda x: 1 if ((x == 'Charged Off') | (x == 'Default')) else 0
emp_title_c = lambda x: 1 if x == '' else 0
log10_c = lambda x: np.log10(1 if x == 0 else x)
application_type_d = {'Individual': 1, 'Joint App': 0}
application_type_c = lambda x: application_type_d.get(x)
initial_list_status_c = lambda x: 0 if x == 'w' else 1
collections_12_mths_ex_med_c = lambda x: x if x <= 3 else 3

conv = {'term': term_c, 
        'int_rate':pct_c, 
        'grade': grade_c, 
        'loan_status': loan_status_c, 
        'emp_title': emp_title_c, 
        'application_type': application_type_c, 
        'revol_util': pct_c, 
        'initial_list_status': initial_list_status_c}

def get_conv():
    return conv

def dummies(df, col, prefix):
    '''
    One-hot encode the indicated column. You still need to assign df to the returned object.
    '''
    df = df.join(pd.get_dummies(df[[col]], prefix=prefix))
    df.drop(columns=col, inplace=True)
    return df

# preprocess
def preprocess(df, impute_strategy='mean', impute=True):
    '''
    Data cleanup steps; fill null valuess, remove some outliers, drop columns and impute mean.
    '''
    pd.options.mode.chained_assignment = None  # default='warn'

    warnings.filterwarnings('always')

    col = 'home_ownership'
    df.loc[df[col] == 'NONE', col] = 'ANY'

    col = 'annual_inc_joint'
    df.loc[:, col].fillna(0, inplace=True)
    
    col = 'annual_inc'
    df = df[df.annual_inc < 1000000]

    col = 'zip_code'
    # most common zip code
    mcz = df.zip_code.value_counts().sort_values(ascending=False).index.values[0]
    df.loc[:, col].fillna(mcz, inplace=True)
    # encode:
    df.loc[:, col] = [int(s[:2]) for s in df.zip_code]

    col = 'dti'
    df.loc[:, col].fillna(0, inplace=True)
    
    col = 'dti_joint'
    df.loc[:, col].fillna(0, inplace=True)
    
    col = 'cr_history'
    is_d = pd.to_datetime(df.issue_d, format='%b-%Y')
    ecl = pd.to_datetime(df.earliest_cr_line, format='%b-%Y')
    # create new feature as the length of credit history in years
    cr_history = (is_d - ecl).astype('timedelta64[Y]')
    df.loc[:, col] = cr_history

    col = 'fico_range_mean'
    df.loc[:, col] = df[['fico_range_high', 'fico_range_low']].mean(1)

    col = 'collections_12_mths_ex_med'
    df.loc[:, col] = df[col].apply(collections_12_mths_ex_med_c)
    
    for col in dummy_list:
        df = dummies(df, col, col)
    
            
    ### Drop column not needed but referenced so far
    drop_cols = ['issue_d', 'dti', 
                    'earliest_cr_line',
                    'fico_range_high', 'fico_range_low',                     
                    'mths_since_last_major_derog']
    df.drop(columns=drop_cols, inplace=True)
    
    if impute:
        imp = Imputer(missing_values='NaN', strategy=impute_strategy, axis=0)
        imp.fit(df)
        df = pd.DataFrame(data=imp.transform(df) , columns=df.columns)
    df = df.astype(np.float32)

    return df

def selectKBest(X, y, k):
    '''
    Select and return the k best features
    '''
    from sklearn.feature_selection import SelectKBest
    select = SelectKBest(k=k)
    selected_features = select.fit(X, y)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]
    X = X[colnames_selected]
    return X

def remove_outliers(df):
    df = df[df.delinq_2yrs < 5]
    df = df[df.open_acc < 30]
    df = df[df.pub_rec < 10]
    df = df[df.revol_bal > 2 ] 
    df = df[df.revol_bal < 1000000 ] 
    df = df[df.total_acc < 60]
    df = df[df.tot_coll_amt < 200000 ] # outlier!
    df = df[df.tot_cur_bal < 200000]
    df = df[df.open_act_il < 10]
    df = df[df.open_il_24m < 10]
    df = df[df.mths_since_rcnt_il < 200]
    df = df[df.total_bal_il < 200000]
    df = df[df.il_util < 200]
    df = df[df.open_rv_12m < 10]
    df = df[df.open_rv_24m < 20]
    df = df[df.max_bal_bc < 100000]
    df = df[df.total_rev_hi_lim < 500000 ]
    df = df[df.inq_fi < 15]
    df = df[df.total_cu_tl < 10]
    df = df[df.inq_last_12m < 10]
    df = df[df.acc_open_past_24mths < 20]
    df = df[df.avg_cur_bal < 100000]
    df = df[df.bc_open_to_buy < 50000]
    df = df[df.delinq_amnt < 10000]
    df = df[df.mo_sin_old_il_acct < 300]
    df = df[df.mo_sin_old_rev_tl_op < 600]
    df = df[df.mo_sin_rcnt_rev_tl_op < 100]
    df = df[df.mo_sin_rcnt_tl < 50]
    df = df[df.mort_acc < 10]
    df = df[df.mths_since_recent_bc < 200]
    df = df[df.num_accts_ever_120_pd < 10]
    df = df[df.num_actv_bc_tl < 15]
    df = df[df.num_actv_rev_tl < 20]
    df = df[df.num_bc_sats < 20]
    df = df[df.num_bc_tl < 30]
    df = df[df.num_il_tl < 40]
    df = df[df.num_op_rev_tl < 30]
    df = df[df.num_rev_accts < 60]
    df = df[df.num_rev_tl_bal_gt_0 < 20]
    df = df[df.num_sats < 30]
    df = df[df.num_tl_op_past_12m < 10]
    df = df[df.pct_tl_nvr_dlq > 50]
    df = df[df.tot_hi_cred_lim < 200000]
    df = df[df.total_bal_ex_mort < 500000]
    df = df[df.total_bc_limit < 100000]
    df = df[df.total_il_high_credit_limit < 400000]
    return df

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   