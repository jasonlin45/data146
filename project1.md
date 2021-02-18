# Project 1

## Packages and Libraries 
A package is a collection of various python modules.  It's easy to think of a package as a 'folder' or namespace which can contain modules, while also being a module itself.
Packages can contain subpackages within it as well. More specifically, a module containing the `__path__` attribute is a package.  
These modules contain definitions and statements, where the statements are executed on first import, and the definitions can be used.

Libraries conceptually define a collection of modules and libraries.  While package can also be seen as a collection, packages are again more akin to a 'folder', whereas a library is a
collection of modules and packages.  These libraries are meant to provide generic functionality which can be easily reused in many different applications and scripts.  Some examples
of commonly used libraries in data science would be `pandas` and `numpy`.

### Installing and Importing
The first steps to take when using a library would be to install the package containing the library.  Typically, a package manager is used for this task, however it is also
possible to build from binaries.  In our example, we'll be using the pip package manager.

First, install the packages in the terminal.  We are using `pandas` and `numpy`.  
```
pip install pandas
pip intall numpy
```

Then, for any Python script in our environment, we can import these libraries as such:

```python
import pandas as pd
import numpy as np
```

Notice that we used an 'alias' for both of the packages.  It's beneficial to use an alias, since it's much quicker and more efficient to type `pd` as opposed to `pandas`.
In general, go for convenience with the alias name.  We can really alias them as whatever we want.  I could have aliased `pandas` as `bigElephant`, but that doesn't really help me.

## Dataframes

Dataframes are a two dimensional data structure consisting of rows and columns.  In Python, the `pandas` library facilitates the use of dataframes as well as operations involving dataframes.  

### Reading data

Pandas offers a variety of ways to read in data.  In our example, we will be reading in a file where the separator is a tab instead of a comma.  To do this, we will use the `read_csv` function, and pass it two arguments.  First, we need the relative filepath on the system we're working on.  In this example, we'll put it next to the script we are working out of, so it should just be the filename.  Second, we want to specify how the data is delimited.  In our case, it is delimited with a tab (\t) instead of a comma, which is the default.  So we will need to pass it a `sep` parameter to specify.  Our final command looks like:

```python
data = pd.read_csv("gapminder.tsv", sep="\t")
```

Different formats will require different arguments, and a full list of arguments as well as data reading methods can be found comprehensively in the pandas documentation.  

### Creating data

We can construct our own dataframe as well.  For example, if we use a Python dictionary:

```python
data = pd.Dataframe({'col1': [1, 2, 3], 'col2': [1, 2, 3], 'col3': [1, 2, 3])
```

This will create a dataframe with three columns named `col1`, `col2` and `col3`, with three rows containing numbers.  

### Analyzing a Dataframe

It's very useful to look at the dimensions, or the number of rows and columns in a dataframe.  Given dataframe `data`, we can look at the dimensions using `data.shape`.  Rather than looking at dimensions, we can also think of it more as rows and columns, and extract the number of rows using `len(data)`, then the number of columns using `len(data.columns)`.  Rows can also be called observations, and columns can also be called variables.  

## Gapminder Analysis

### Read Dataset
Let's read in our dataset using the above knowledge.  

```python
import pandas as pd

data = pd.read_csv("gapminder.tsv", sep="\t")
```

### Years Analysis

We then extract the years and look at the unique values to get an understanding of what we're looking at.
```python
years = data['year']
years.unique()
```
Output:
```
array([1952, 1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002,
       2007], dtype=int64)
```

The variable for years appears to be at a regular interval for every 5 years, starting from 1952 and ending at 2007.

Thus, the most logical next step for updating the data would be to add years in 5 year intervals from 2007 until the current year (2021).  Let's add 2012 and 2017.  How many outcomes would we be adding if we added these two years?

```python
lengths = [len(data[data['year'] == x]) for x in years.unique()]
lengths
```
Output:
```
[142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142]
```

In the case of our data, it appears that there are 142 outcomes for each year.  Thus, we would be adding 2 years * 142 outcomes to get 284 additional outcomes.  

### Lowest Life Expectancy

Let's extract the country with the lowest life expectancy.

```python
min_lifeExp = min(data['lifeExp'])
data[data['lifeExp'] == min_lifeExp]
```
Output:
```
	   country	continent  year	 lifeExp  	  pop	 gdpPercap
1292	Rwanda	  Africa   1992	  23.599  7290203	737.068595
```

In 1992, Rwanda had the lowest life expectancy in our dataset.  After a cursory investigation into why this might be, we find that during this time, Rwanda was going through a civil war.  We can estimate that due to the ongoing conflict, the life expectancy dropped as a result of casualties from the war.

### Gross Domestic Product

Let's make a new variable analyzing GDP by multiplying population and gdp per capita together.  From there, we'll look at Germany, France, Italy, and Spain in 2007.  

```python
data['gdp'] = data['pop'] * data['gdpPercap']

countries = ['Germany', 'France', 'Italy', 'Spain']

# filter
gdp_filtered = data[(data['country'].isin(countries)) & (data['year']==2007)]

# order by gdp
gdp_filtered = gdp_filtered.sort_values(by=')
```

|      | country   | continent   |   year |   lifeExp |      pop |   gdpPercap |         gdp |
|-----:|:----------|:------------|-------:|----------:|---------:|------------:|------------:|
| 1427 | Spain     | Europe      |   2007 |    80.941 | 40448191 |     28821.1 | 1.16576e+12 |
|  779 | Italy     | Europe      |   2007 |    80.546 | 58147733 |     28569.7 | 1.66126e+12 |
|  539 | France    | Europe      |   2007 |    80.657 | 61083916 |     30470   | 1.86123e+12 |
|  575 | Germany   | Europe      |   2007 |    79.406 | 82400996 |     32170.4 | 2.65087e+12 |


Now let's look at which countries experienced the biggest growth from the previous 5 years.  Let's make a dataframe for the 2002 data, then check how much they increased by.  From there, we find that Germany has had the greatest GDP increase in the past 5 years.

## Logical Operators

`==` is a binary operator which returns `True` if both operands are equivalent or contain equivalent values.  The behavior can be overridden for objects.  If the objects are not equivalent, it returns `False`.

`&`, `|`, and `^` are bitwise operators which represent the bitwise and, or, and xor respectively.  They are designed to operate on binary numbers, but the behavior can be overriden for objects.  Since Python will evaluate booleans as well as integers, they can be used somewhat interchangeably with the logical operators `and` and `or`.  However, statements will not benefit from short circuiting, reducing flexibility and efficiency.  In the context of data science, the bitwise operators must be used to chain additional conditions when working with dataframes in `pandas`.  

Below are the tables representing each operator.

`|`, bitwise or operator

|   | 1 | 0 |
|---|---|---|
| 1 | 1 | 1 |
| 0 | 1 | 0 |

`&`, bitwise and operator

|   | 1 | 0 |
|---|---|---|
| 1 | 1 | 0 |
| 0 | 0 | 0 |

`^`, bitwise xor operator

|   | 1 | 0 |
|---|---|---|
| 1 | 0 | 1 |
| 0 | 1 | 0 |

## iloc vs loc

`iloc` and `loc` accomplish similar things, but function differently.  `loc` takes in names of columns and/or rows that we want to filter out, and `iloc` takes in an integer index.  Think of the i in `iloc` as standing for index.  

Using `iloc` lets select a series of continuous observations very easily.  Lets go ahead and select a subset of 5 observations from our dataframe:

```python
data.iloc[0:5]
```
Alternatively, we could extract all observations from a subset of columns:
```python
data.iloc[:, 0:3]
```

## APIs

API stands for application programming interface, and is meant as a way to join various components of software together, or delivering data.  In short, an API receives requests, and responds to those requests.  An example would be a REST API.  

In the following example code, we will use the `requests` library in order to make a GET request to an imaginary REST endpoint in order to retrieve CSV data.  We then write this data to a file, and read it in using `pandas`.

```python
import requests as rq
import pandas as pd

url = "myendpoint"
file_name = "data.csv"

r = rq.get(url)

with open(file_name, 'wb') as f:
    f.write(r.content)

data = pd.read_csv(file_name)
```

## Apply

The `apply` function from the `pandas` library allows us to apply a function to every observation along an axis.  Alternatively, we can use `.iterrows()` and loop over all these observations.  However, the `apply` function is preferable in many cases, since it saves us the trouble of having to use a loop.

## Alternative Filtering

Rather than using `iloc`, there exists a similar alternative to filtering and subsetting dataframes.  Dataframes themselves support indexing and slicing, so we can do the following to select consecutive rows:
```python
data[2:30]
# these are equivalent
data.iloc[2:30]
```
We can also do something similar for columns:
```python
data[data.columns[1:3]]
# these are equivalent
data.iloc[:,1:3]
```
