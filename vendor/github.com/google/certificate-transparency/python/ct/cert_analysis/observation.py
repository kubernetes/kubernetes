import logging

class Observation(object):
    """Describes certificate observation."""
    def __init__(self, description, reason=None, details=None):
        self.description = description
        self.reason = reason
        self.details = details

    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__,
                                   repr(self.description),
                                   repr(self.reason),
                                   repr(self.details))

    def __str__(self):
        """Returns observation representation as: description (reason) [details]

        (if some field is unavailable then even brackets aren't returned)
        """
        ret = self.description
        if self.reason:
            ret = "%s (%s)" % (ret, self.reason)
        if self.details:
            ret = "%s [%s]" % (ret, self._format_details())
        return ret

    def _format_details(self):
        """Convenience method, so it's easy to override how details have to be
        printed without overriding whole __str__.
        """
        try:
            if isinstance(self.details, str):
                return unicode(self.details, 'utf-8')
            else:
                return unicode(self.details)
        except Exception as e:
            logging.warning("Unprintable observation %r" % self.details)
            return "UNPRINTABLE " + str(e)

    def details_to_proto(self):
        """Specifies how details should be written to protobuf."""
        if self.details:
            return self._format_details().encode('utf-8')
        else:
            return ''

