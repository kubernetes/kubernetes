# Line Protocol

The line protocol is a text based format for writing points to InfluxDB.  Each line defines a single point. 
Multiple lines must be separated by the newline character `\n`. The format of the line consists of three parts:

```
[key] [fields] [timestamp]
```

Each section is separated by spaces.  The minimum required point consists of a measurement name and at least one field. Points without a specified timestamp will be written using the server's local timestamp. Timestamps are assumed to be in nanoseconds unless a `precision` value is passed in the query string.

## Key

The key is the measurement name and any optional tags separated by commas.  Measurement names, tag keys, and tag values must escape any spaces or commas using a backslash (`\`). For example: `\ ` and `\,`.  All tag values are stored as strings and should not be surrounded in quotes. 

Tags should be sorted by key before being sent for best performance. The sort should match that from the Go `bytes.Compare` function (http://golang.org/pkg/bytes/#Compare).

### Examples

```
# measurement only
cpu

# measurment and tags
cpu,host=serverA,region=us-west

# measurment with commas
cpu\,01,host=serverA,region=us-west

# tag value with spaces
cpu,host=server\ A,region=us\ west
```

## Fields

Fields are key-value metrics associated with the measurement.  Every line must have at least one field.  Multiple fields must be separated with commas and not spaces.

Field keys are always strings and follow the same syntactical rules as described above for tag keys and values. Field values can be one of four types.  The first value written for a given field on a given measurement defines the type of that field for all series under that measurement.

* _integer_ - Numeric values that do not include a decimal.  (e.g. 1, 345, 2015, -10)
* _float_ - Numeric values that include a decimal.  (e.g. 1.0, -3.14, 6.0+e5).  Note that all values _must_ have a decimal even if the decimal value is zero (1 is an _integer_, 1.0 is a _float_).
* _boolean_ - A value indicating true or false.  Valid boolean strings are (t, T, true, TRUE, f, F, false, and FALSE).
* _string_ - A text value.  All string values _must_ be surrounded in double-quotes `"`.  If the string contains
a double-quote, it must be escaped with a backslash, e.g. `\"`.


```
# integer value
cpu value=1

# float value
cpu_load value=1.2

# boolean value
error fatal=true

# string value
event msg="logged out"

# multiple values
cpu load=10.0,alert=true,reason="value above maximum threshold"
```

## Timestamp

The timestamp section is optional but should be specified if possible.  The value is an integer representing nanoseconds since the epoch. If the timestamp is not provided the point will inherit the server's local timestamp.

Some write APIs allow passing a lower precision.  If the API supports a lower precision, the timestamp may also be
an integer epoch in microseconds, milliseconds, seconds, minutes or hours.

## Full Example
A full example is shown below.
```
cpu,host=server01,region=uswest value=1.0 1434055562000000000
cpu,host=server02,region=uswest value=3.0 1434055562000010000
```
In this example the first line shows a `measurement` of "cpu", there are two tags "host" and "region, the `value` is 1.0, and the `timestamp` is 1434055562000000000. Following this is a second line, also a point in the `measurement` "cpu" but belonging to a different "host".
```
cpu,host=server\ 01,region=uswest value=1.0,msg="all systems nominal"
cpu,host=server\ 01,region=us\,west value_int=1
```
In these examples, the "host" is set to `server 01`. The field value associated with field key `msg` is double-quoted, as it is a string. The second example shows a region of `us,west` with the comma properly escaped. In the first example `value` is written as a floating point number. In the second, `value_int` is an integer. 

# Distributed Queries

