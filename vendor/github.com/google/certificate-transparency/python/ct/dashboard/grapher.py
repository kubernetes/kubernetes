from datetime import datetime
import gviz_api

class GvizGrapher(object):
    """A grapher that produces JSON output for the Google Visualizaton
    Datatable.
    """

    def __init__(self):
        pass

    def _get_nested_attr(self, item, attr):
        attributes = attr.split(".")
        ret = item
        for a in attributes:
            ret = getattr(ret, a)
        return ret

    def _convert_value(self, item, attr):
        """Convert the attribute of an item to a gviz value type."""
        value = self._get_nested_attr(item, attr[0])
        if len(attr) > 1 and attr[1] == "timestamp_ms":
            value = datetime.fromtimestamp(value/1000)
        return value

    def _convert_attr(self, attr):
        """Convert the attribute types accepted by us to ones recognised by
        gviz."""
        if len(attr) < 1 or attr[1] != "timestamp_ms":
            return attr
        tmp = list(attr)
        tmp[1] = "datetime"
        return tuple(tmp)

    def make_table(self, generator, columns, order_by=()):
        """A data table maker.
        Args:
           generator: A generator that yields the input data objects
           columns: A list of object attributes. Each attribute to be included
           as a column in the table must be represented as a tuple
           (attr [,data_type [,label [,custom_properties]]])
           Types supported by gviz are "string", "number", "boolean", "date",
           "datetime" or "timeofday".
           Additionally we've added "timestamp_ms" for millisecond since epoch,
           which we convert to "datetime".
        Returns:
           A JSON string containing the data
        """
        data = []
        for item in generator:
            row = [self._convert_value(item, attr) for attr in columns]
            data.append(row)

        data_table = gviz_api.DataTable(
            [self._convert_attr(attr) for attr in columns])
        data_table.LoadData(data)

        json = data_table.ToJSon(order_by=order_by)
        return json
