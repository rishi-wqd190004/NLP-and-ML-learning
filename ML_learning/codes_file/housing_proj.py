import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_pandas import DataFrameMapper

#various codes parameters
pd.set_option('max_columns', None)

# reading the data
housing_data = pd.read_csv("/home/richie/NLP-and-ML-learning/ML_learning/datasets/housing.csv")
print(housing_data.head(5))
print(housing_data.describe())
print("total rows: ", len(housing_data))

# plot the graph as you wanna see the values
#housing_data.hist(bins=50, figsize=(16,8))
housing_data["median_income"].hist(bins=50)
#plt.show()

# split the data into test and train (now common approach is to use train test split method from sklearn.model_selection) but here we use different approach
def split_train_test(data, test_ratio):
    '''
    picking some instances randomly using np.random.permutation
    '''
    all_shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = all_shuffled_indices[:test_set_size]
    trian_indices = all_shuffled_indices[test_set_size:]
    return data.iloc[trian_indices], data.iloc[test_indices]
# showing how to use the above function
train_set, test_set = split_train_test(housing_data, 0.33)
print(len(train_set), "train +", len(test_set), "test")

# lets say the housing data will get updated so we can use each instance's identifier to decide whether or not it should go in the test dataset
## so we have 2 options:
## a) save the test set of data on first run and then load it for subsequent runs
## b) set a seed(np.random.seed(42)) before you call the np.random.permutation()

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_data_with_id = housing_data.reset_index()
train_set, test_set = split_train_test_by_id(housing_data_with_id, 0.2, "index")

## but we are smart will create another id using longitude and latitude

# sorry but this was just to showcase you can create your own identifier or ways to use the train test split

# using sklearn.model_selection.train_test_split
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

# now to mention we create a housing category
housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)
# using StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    start_train_set = housing_data.loc[train_index]
    start_test_set = housing_data.loc[test_index]
# checking the split made
print(housing_data["income_cat"].value_counts() / len(housing_data))

# so all this was done to showcase setting up the data which we will use later too during the validation sets too
for set in (start_train_set, start_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# so now we create a copy of the test set
housing = start_train_set.copy()

# moving on to visualizing to showcase the data
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) # adding alpha displays density of data points
#plt.show()

# more clear visualization
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
#plt.show()

# looking for correlations (Pearson's r)
corr_matrix = housing.corr()
print("correlation for median_house_value\n", corr_matrix["median_house_value"].sort_values(ascending=True))

# another way of correlation we can use pandas scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(16,8))
#plt.show()

# now the median income is having the maximum correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)
#plt.show()

# lets create attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# check the correlations
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# preping the data
housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()

# time to clean the data
# as we know throwing off the data or the attribute is never a good practice (my learning)
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # imputer works on numerical attributes only
imputer.fit(housing_num) # fitting the data
print(f"median of all attributes \n", imputer.statistics_)
# time to apply it on the training data
X = imputer.transform(housing_num)
# changing X into a dataframe
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# handling text and categorical data
## for handling textual data like under ocean_proximity will change it to numerical data
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print("Encoded categorical data\n", housing_cat_encoded)
print(encoder.classes_) # printing various features present in ocean_proximity
# implementing one hot encoder
one_hot_encoder = OneHotEncoder()
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# applying LabelBinarizer to process text category to integer category and then finally to one hot vectors
label_binarizer = LabelBinarizer() # to output a sparse matrix just mention sparse_output=True
housing_cat_1hot = label_binarizer.fit_transform(housing_cat)

#adding BaseEstimator and TransformerMixin to get set_params and get_param for hyperparameter tuning
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attributes = attr_adder.transform(housing.values)

# feature scaling
# use sklearn.preprocessing.Minmaxscaler or use sklearn.preprocessing.standardscaler
## using pipelines for transformation
# creating DataFrameSelector class
class DataFrameSelector(BaseEstimator, TransformerMixin): # if not comfortable can use sklearn_pandas.DataFrameMapper()
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values
# creating a new label_binarizer so as to resolve the fit_transform issue mentioned below
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
# adding num_attributes and cat_attributes to create a union
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)), # if getting TypeError: fit_transform() takes 2 positional arguments but 3 were given --> create your own new label
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(True)),
        ('std_scaler', StandardScaler()),
        ])
#housing_num_tr = num_pipeline.fit_transform(housing_num)
# this above pipeline was good for numerical numbers but now we need a pipeline for LabelBinarizer on categorical data
# Thanks to sklearn provides with FeatureUnion class
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)), # be patient will define this DataFrameSelector class below
        ('label_binarizer', MyLabelBinarizer()),
        ])
# if adding sklearn_pandas.DataFrameMapper():
# cat_pipeline = Pipeline([
#     ('label_binarizer', DataFrameMapper([(cat_attribs, LabelBinarizer())])),
# ])
# adding the full pipeline using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ])
# running the complete pipeline
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)
