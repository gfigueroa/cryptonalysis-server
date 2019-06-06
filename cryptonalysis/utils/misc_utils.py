from datetime import date, datetime, timedelta


def parse_date(date_str):
    """
    Parse a date string to a datetime object. The string can only have one of
    the following formats:
    - 'YYYY-MM-DD'
    - 'today'
    - 'yesterday'
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
