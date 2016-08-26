/*
Package influxql implements a parser for the InfluxDB query language.

InfluxQL is a DML and DDL language for the InfluxDB time series database.
It provides the ability to query for aggregate statistics as well as create
and configure the InfluxDB server.

Selecting data

The SELECT query is used for retrieving data from one or more series. It allows
for a list of columns followed by a list of series to select from.

	SELECT value FROM cpu_load

You can also add a a conditional expression to limit the results of the query:

	SELECT value FROM cpu_load WHERE host = 'influxdb.com'

Two or more series can be combined into a single query and executed together:

	SELECT cpu0.value + cpu1.value
	FROM cpu_load AS cpu0 INNER JOIN cpu_load cpu1 ON cpu0.host = cpu1.host

Limits and ordering can be set on selection queries as well:

	SELECT value FROM cpu_load LIMIT 100 ORDER DESC;


Removing data

The DELETE query is available to remove time series data points from the
database. This query will delete "cpu_load" values older than an hour:

	DELETE FROM cpu_load WHERE time < now() - 1h


Continuous Queries

Queries can be run indefinitely on the server in order to generate new series.
This is done by running a "SELECT INTO" query. For example, this query computes
the hourly mean for cpu_load and stores it into a "cpu_load" series in the
"daily" shard space.

	SELECT mean(value) AS value FROM cpu_load GROUP BY 1h
	INTO daily.cpu_load

If there is existing data on the source series then this query will be run for
all historic data. To only execute the query on new incoming data you can append
"NO BACKFILL" to the end of the query:

	SELECT mean(value) AS value FROM cpu_load GROUP BY 1h
	INTO daily.cpu_load NO BACKFILL

Continuous queries will return an id that can be used to remove them in the
future. To remove a continous query, use the DROP CONTINUOUS QUERY statement:

	DROP CONTINUOUS QUERY 12

You can also list all continuous queries by running:

	LIST CONTINUOUS QUERIES

*/
package influxql
