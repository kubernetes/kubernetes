# Continuous Queries

This document lays out continuous queries and a proposed architecture for how they'll work within an InfluxDB cluster.

## Definition of Continuous Queries

Continuous queries serve two purposes in InfluxDB:

1. Combining many series into a single series (i.e. removing 1 or more tag dimensions to make queries more efficient)
2. Aggregating and downsampling series

The purpose of both types of continuous queries is to duplicate or downsample data automatically in the background to make querying thier results fast and efficient. Think of them as another way to create indexes on data.

Generally, there are continuous queries that create copyies of data into another measurement or tagset and queries that downsample and aggregate data. The only difference between the two types is if the query has a `GROUP BY time` clause.

Before we get to the continuous query examples, we need to define the `INTO` syntax of queries.

### INTO

`INTO` is a method for running a query and having it output into either another measurement name, retention policy, or database. The syntax looks like this:

```sql
SELECT *
INTO [<retention policy>.]<measurement> [ON <database>]
FROM <measurement>
[WHERE ...]
[GROUP BY ...]
```

The syntax states that the retention policy, database, where clause, and group by clause are all optional. If a retention policy isn't specified, the database's default retention policy will be written into. If the database isn't specified, the database the query is running from will be written into.

By selecting specific fields, `INTO` can merge many series into one that will go into a new either a new measurement, retention policy, or database. For example:

```sql
SELECT mean(value) as value, region
INTO "1h.cpu_load"
FROM cpu_load
GROUP BY time(1h), region
```

That will give 1h summaries of the mean value of the `cpu_load` for each `region`. Specifying `region` in the `GROUP BY` clause is unnecessary since having it in the `SELECT` clause forces it to be grouped by that tag, we've just included it in the example for clarity.

With `SELECT ... INTO`, fields will be written as fields and tags will be written as tags.

### Continuous Query Syntax

The `INTO` queries run once. Continuous queries will turn `INTO` queries into something that run in the background in the cluster. They're kind of like triggers in SQL.

```sql
CREATE CONTINUOUS QUERY "1h_cpu_load"
ON database_name
BEGIN
  SELECT mean(value) as value, region
  INTO "1h.cpu_load"
  FROM cpu_load
  GROUP BY time(1h), region
END
```

Or chain them together:

```sql
CREATE CONTINUOUS QUERY "10m_event_count"
ON database_name
BEGIN
  SELECT count(value)
  INTO "10m.events"
  FROM events
  GROUP BY time(10m)
END

-- this selects from the output of one continuous query and outputs to another series
CREATE CONTINUOUS QUERY "1h_event_count"
ON database_name
BEGIN
  SELECT sum(count) as count
  INTO "1h.events"
  FROM events
  GROUP BY time(1h)
END
```

Or multiple aggregations from all series in a measurement. This example assumes you have a retention policy named `1h`.

```sql
CREATE CONTINUOUS QUERY "1h_cpu_load"
ON database_name
BEGIN
  SELECT mean(value), percentile(80, value) as percentile_80, percentile(95, value) as percentile_95
  INTO "1h.cpu_load"
  FROM cpu_load
  GROUP BY time(1h), *
END
```

The `GROUP BY *` indicates that we want to group by the tagset of the points written in. The same tags will be written to the output series. The multiple aggregates in the `SELECT` clause (percentile, mean) will be written in as fields to the resulting series.

Showing what continuous queries we have:

```sql
LIST CONTINUOUS QUERIES
```

Dropping continuous queries:

```sql
DROP CONTINUOUS QUERY <name>
ON <database>
```

### Security

To create or drop a continuous query, the user must be an admin.

### Limitations

In order to prevent cycles and endless copying of data, the following limitation is enforced on continuous queries at create time:

*The output of a continuous query must go to either a different measurement or to a different retention policy.*

In theory they'd still be able to create a cycle with multiple continuous queries. We should check for these and disallow.

## Proposed Architecture

Continuous queries should be stored in the metastore cluster wide. That is, they amount to database schema that should be stored in every server in a cluster.

Continuous queries will have to be handled in a different way for two different use cases: those that simply copy data (CQs without a group by time) and those that aggregate and downsample data (those with a group by time).

### No group by time

For CQs that have no `GROUP BY time` clause, they should be evaluated at the data node as part of the write. The single write should create any other writes for the CQ and submit those in the same request to the brokers to ensure that all writes succeed (both the original and the new CQ writes) or none do.

I imagine the process going something like this:

1. Convert the data point into its compact form `<series id><time><values>`
2. For each CQ on the measurement and retention policy without a group by time:
3. Run the data point through a special query engine that will output 0 or 1 data point
4. Goto #1 for each newly generated data point
5. Write all the data points in a single call to the brokers
6. Return success to the user

Note that for the generated data points, we need to go through and run this process against them since they can feed into different retention policies, measurements, and new tagsets. On #3 I mention that the output will either be a data point or not. That's because of `WHERE` clauses on the query. However, it will never be more than a single data point.

I mention that we'll need a special query engine for these types of queries. In this case, they never have an aggregate function. Any query with an aggregate function also has a group by time, and these queries by definition don't have that.

The only thing we have to worry about is which fields are being selected, and what the where clause looks like. We should be able to put the raw data point through a simple transform function that either outputs another raw points or doesn't.

I think this transform function be something separate from the regular query planner and engine. It can be in `influxQL` but it should be something fairly simply since the only purpose of these types of queries is to either filter some data out and output to a new series or transform into a new series by dropping tags.

### Has group by time

CQs that have a `GROUP BY time` (or aggregate CQs) will need to be handled differently.

One key point on continuous queries with a group by time is that all their writes should always be `overwrite = true`. That is, they should only have a single data point for each timestamp. This distinction means that continuous queries for previous blocks of time can be safely run multiple times without duplicating data (i.e. they're idempotent).

There are two different ideas I have for how CQs with group by time could be handled. The first is through periodic updates handled by the Raft Leader. The second would be to expand out writes for each CQ and handle them on the data node.

#### Periodic Updates

In this approach the management of how CQs run in a cluster will be centrally located on the Raft Leader. It will be responsible for orchestrating which data nodes run CQs and when.

The naive approach would be to have the leader hand out each CQ for a block of time periodically. The leader could also rerun CQ for periods of time that have recently passed. This would be an easy way to handle the "lagging data" problem, but it's not precise.

Unfortunately, there's no easy way to tell cluster wide if there were data points written in an already passed window of time for a CQ. We might be able to add this at the data nodes and have them track it, but it would be quite a bit more work.

The easy way would just be to have CQs re-execute for periods that recently passed and have some user-configurable window of time that they stop checking after. Then we could give the user the ability to recalculate CQs ranges of time if they need to correct for some problem that occurred or the loading of a bunch of historical data.

With this approach, we'd have the metadata in the database store the last time each CQ was run. Whenever the Raft leader sent out a command to a data node to handle a CQ, the data node would use this metadata to determine which windows of time it should compute.

This approach is like what exists in 0.8, with the exception that it will automatically catch data that is lagged behind in a small window of time and give the user the ability to force recalculation.

#### Expanding writes

When a write comes into a data node, we could have it evaluated against group by CQs in addition to the non-group by ones. It would then create writes that would then go through the brokers. When the CQ writes arrive at the data nodes, they would have to handle each write differently depending on if it was a write to a raw series or if it was a CQ write.

Let's lay out a concrete example.

```sql
CREATE CONTINUOUS QUERY "10m_cpu_by_region"
ON foo
BEGIN
  SELECT mean(value)
  INTO cpu_by_region
  FROM cpu
  GROUP BY time(10m), region
END
```

In this example we write values into `cpu` with the tags `region` and `host`.

Here's another example CQ:

```sql
CREATE CONTINUOUS QUERY "1h_cpu"
ON foo
BEGIN
  SELECT mean(value)
  INTO "1h.cpu"
  FROM raw.cpu
  GROUP BY time(10m), *
END
```

That would output one series into the `1h` retention policy for the `cpu` measurement for every series from the `raw` retention policy and the `cpu` measurement.

Both of these examples would be handled the same way despite one being a big merge of a bunch of series into one and the other being an aggregation of series in a 1-to-1 mapping.

Say we're collecting data for two hosts in a single region. Then we'd have two distinct series like this:

```
1 - cpu host=serverA region=uswest
2 - cpu host=serverB region=uswest
```

Whenever a write came into a server, we'd look at the continuous queries and see if we needed to create new writes. If we had the two CQ examples above, we'd have to expand a single write into two more writes (one for each CQ).

The first CQ would have to create a new series:

```
3 - cpu_by_region region=uswest
```

The second CQ would use the same series id as the write, but would send it to another retention policy (and thus shard).

We'd need to keep track of which series + retention policy combinations were the result of a CQ. When the data nodes get writes replicated downward, they would have to handle them like this:

1. If write is normal, write through
2. If write is CQ write, compute based on existing values, write to DB

#### Approach tradeoffs

The first approach of periodically running queries would almost certainly be the easiest to implement quickly. It also has the added advantage of not putting additional load on the brokers by ballooning up the number of writes that go through the system.

The second approach is appealing because it would be accurate regardless of when writes come in. However, it would take more work and cause the number of writes going through the brokers to be multiplied by the number of continuous queries, which might not scale to where we need.

Also, if the data nodes write for every single update, the load on the underlying storage engine would go up significantly as well.
