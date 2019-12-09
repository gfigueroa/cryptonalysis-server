from datetime import date, datetime, timedelta


def parse_date(date_str):
    """
    Parse a date string to a datetime object. The string can only have one of
    the following formats:
    - 'YYYY-MM-DD'
    - 'today'
    - 'yesterday'

    >>> parse_date('2019-01-01')
    datetime.date(2019, 1, 1)

    :param date_str: The date string to parse. Accepts 'today' and 'yesterday'
    :return: A date
    :rtype: date
    """
    # Parse init date
    if date_str.lower() == 'today':
        init_date = date.today().replace()
    elif date_str.lower() == 'yesterday':
        init_date = date.today().replace() - timedelta(1)
    else:
        init_date = datetime.strptime(date_str, '%Y-%m-%d').date()

    return init_date


def all_elements_unique(l):
    """
    Returns True if all elements in the given list are unique; False otherwise.

    >>> all_elements_unique([1, 2, 3, 4])
    True
    >>> all_elements_unique([1, 2, 3, 1])
    False
    >>> all_elements_unique([{'a': 1}, {'a': 1}])
    False

    :param l: a list of elements
    :type l: list
    :return: True if all elements of the list are unique.
    :rtype: bool
    """
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] == l[j]:
                return False
    return True
