# This code tests the parameters output by the user-written linear
# regression from linear_regression.py against the parameters
# output for linear regression over the same data by the np.polyfit
# function.

# Author: Your name here
# Date created: 
# Date modified: 

import numpy as np
import pytest
from linear_regression import params, df

def test_1():

    # Fit data with linear regression using np.polyfit

    ## YOUR CODE HERE
    test_params = np.polyfit(df['Age_years'], df['Value_thousands_USD'], 1)

    assert np.allclose(params, test_params, atol = 1e-8), f"❌ Test failed: parameters do not match"


    print("✅ Test passed: Parameters are approximately equal.")