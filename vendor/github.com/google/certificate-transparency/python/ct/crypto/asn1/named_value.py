"""Named values."""


class NamedValue(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "%r(%s, %s)" % (self.__class__.__name__, self.name, self.value)

    def __str__(self):
        return self.name
