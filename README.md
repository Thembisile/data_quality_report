# Data Quality Report
This is a Python module that generates a data quality report, which can be used to quickly evaluate the quality of a dataset. The report is generated in a Jupyter notebook, which provides an interactive environment for data analysis and visualization.

## Usage
To use this module, simply import it into your Jupyter notebook and call the generate_report function with your dataset as a parameter. The function will generate a report that includes the following information:

## Data type of each column
- Count of missing values in each column
- Summary statistics for numeric columns (mean, median, standard deviation, etc.)
- Distribution of values for categorical columns

## Installation
To install this module, you can use pip:
```Python
pip install data-quality-report
```

## Example
Here's an example of how to use the module:
```Python
import pandas as pd
from data_quality_report import generate_report

# Load dataset
df = pd.read_csv('my_dataset.csv')

# Generate report
generate_report(df)

```

## Contributing
If you'd like to contribute to this module, feel free to fork the repository and submit a pull request. Any contributions are welcome!

## License
This module is released under the MIT License. See LICENSE.md for more information.
