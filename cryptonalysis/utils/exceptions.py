"""
Commonplace exceptions in the application.
"""


class ExternalUnavailabilityError(Exception):
    def __init__(self, msg=None, original_exception=None):
        if original_exception:
            super(ExternalUnavailabilityError, self).__init__("{0}, caused by {1}".format(
                msg, repr(original_exception))
            )
        else:
            super(ExternalUnavailabilityError, self).__init__("{0}".format(msg))
