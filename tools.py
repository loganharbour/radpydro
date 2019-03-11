from sys import exit

# Ensure value is of type float or integer
def checkNumber(value, error_string):
    if isinstance(value, float) or isinstance(value, int):
        return value
    else:
        sys.exit('{} is not of a number type'.format(error_string))

# Ensure value is of type integer
def checkInteger(value, error_string):
    if isinstance(value, int):
        return value
    else:
        sys.exit('{} is not of type integer'.format(error_string))
