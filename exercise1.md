# Exercise 1

Importing libraries:
```
import pandas as pd
```

Reading in data:
```
data = pd.read_csv("gapminder.tsv", sep="\t")
```

## 1
```
# exercise 1
# which years are in the dataset, and how many?
unq_yrs = data['year'].unique().tolist()
num_yrs = len(unq_yrs)
```
1952, 1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002, 2007
12 unique years

## 2
```
# exercise 2
# largest pop and where it occurs
largest_pop = data['pop'].max()
largest_pop_data = data[data['pop'] == largest_pop]
```
China, 2007.  Population in 2007: 1,318,683,096.

## 3
```
# exercise 3
# smallest european country in 1952, and its pop in 2007
europe = data[data['continent'] == "Europe"]
min_pop_1952_eu = europe[europe['pop'] == europe[europe['year'] == 1952]['pop'].min()]['country'].tolist()[0]
iceland_2007_pop = europe[(europe['country'] == min_pop_1952_eu) & (europe['year'] == 2007)]
```
Iceland had the smallest population in 1952.  In 2007, its population was 301,931.
