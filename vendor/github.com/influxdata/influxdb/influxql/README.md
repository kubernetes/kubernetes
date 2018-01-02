# The Influx Query Language Specification

## Introduction

This is a reference for the Influx Query Language ("InfluxQL").

InfluxQL is a SQL-like query language for interacting with InfluxDB.  It has
been lovingly crafted to feel familiar to those coming from other SQL or
SQL-like environments while providing features specific to storing and analyzing
time series data.


## Notation

The syntax is specified using Extended Backus-Naur Form ("EBNF").  EBNF is the
same notation used in the [Go](http://golang.org) programming language
specification, which can be found [here](https://golang.org/ref/spec).  Not so
coincidentally, InfluxDB is written in Go.

```
Production  = production_name "=" [ Expression ] "." .
Expression  = Alternative { "|" Alternative } .
Alternative = Term { Term } .
Term        = production_name | token [ "…" token ] | Group | Option | Repetition .
Group       = "(" Expression ")" .
Option      = "[" Expression "]" .
Repetition  = "{" Expression "}" .
```

Notation operators in order of increasing precedence:

```
|   alternation
()  grouping
[]  option (0 or 1 times)
{}  repetition (0 to n times)
```

## Query representation

### Characters

InfluxQL is Unicode text encoded in [UTF-8](http://en.wikipedia.org/wiki/UTF-8).

```
newline             = /* the Unicode code point U+000A */ .
unicode_char        = /* an arbitrary Unicode code point except newline */ .
```

## Letters and digits

Letters are the set of ASCII characters plus the underscore character _ (U+005F)
is considered a letter.

Only decimal digits are supported.

```
letter              = ascii_letter | "_" .
ascii_letter        = "A" … "Z" | "a" … "z" .
digit               = "0" … "9" .
```

## Identifiers

Identifiers are tokens which refer to database names, retention policy names,
user names, measurement names, tag keys, and field keys.

The rules:

- double quoted identifiers can contain any unicode character other than a new line
- double quoted identifiers can contain escaped `"` characters (i.e., `\"`)
- double quoted identifiers can contain InfluxQL keywords
- unquoted identifiers must start with an upper or lowercase ASCII character or "_"
- unquoted identifiers may contain only ASCII letters, decimal digits, and "_"

```
identifier          = unquoted_identifier | quoted_identifier .
unquoted_identifier = ( letter ) { letter | digit } .
quoted_identifier   = `"` unicode_char { unicode_char } `"` .
```

#### Examples:

```
cpu
_cpu_stats
"1h"
"anything really"
"1_Crazy-1337.identifier>NAME👍"
```

## Keywords

```
ALL           ALTER         ANY           AS            ASC           BEGIN
BY            CREATE        CONTINUOUS    DATABASE      DATABASES     DEFAULT
DELETE        DESC          DESTINATIONS  DIAGNOSTICS   DISTINCT      DROP
DURATION      END           EVERY         EXPLAIN       FIELD         FOR
FROM          GRANT         GRANTS        GROUP         GROUPS        IN
INF           INSERT        INTO          KEY           KEYS          KILL
LIMIT         SHOW          MEASUREMENT   MEASUREMENTS  NAME          OFFSET
ON            ORDER         PASSWORD      POLICY        POLICIES      PRIVILEGES
QUERIES       QUERY         READ          REPLICATION   RESAMPLE      RETENTION
REVOKE        SELECT        SERIES        SET           SHARD         SHARDS
SLIMIT        SOFFSET       STATS         SUBSCRIPTION  SUBSCRIPTIONS TAG
TO            USER          USERS         VALUES        WHERE         WITH
WRITE
```

## Literals

### Integers

InfluxQL supports decimal integer literals.  Hexadecimal and octal literals are
not currently supported.

```
int_lit             = ( "1" … "9" ) { digit } .
```

### Floats

InfluxQL supports floating-point literals.  Exponents are not currently supported.

```
float_lit           = int_lit "." int_lit .
```

### Strings

String literals must be surrounded by single quotes. Strings may contain `'`
characters as long as they are escaped (i.e., `\'`).

```
string_lit          = `'` { unicode_char } `'` .
```

### Durations

Duration literals specify a length of time.  An integer literal followed
immediately (with no spaces) by a duration unit listed below is interpreted as
a duration literal.

### Duration units
| Units  | Meaning                                 |
|--------|-----------------------------------------|
| u or µ | microseconds (1 millionth of a second)  |
| ms     | milliseconds (1 thousandth of a second) |
| s      | second                                  |
| m      | minute                                  |
| h      | hour                                    |
| d      | day                                     |
| w      | week                                    |

```
duration_lit        = int_lit duration_unit .
duration_unit       = "u" | "µ" | "ms" | "s" | "m" | "h" | "d" | "w" .
```

### Dates & Times

The date and time literal format is not specified in EBNF like the rest of this document.  It is specified using Go's date / time parsing format, which is a reference date written in the format required by InfluxQL.  The reference date time is:

InfluxQL reference date time: January 2nd, 2006 at 3:04:05 PM

```
time_lit            = "2006-01-02 15:04:05.999999" | "2006-01-02" .
```

### Booleans

```
bool_lit            = TRUE | FALSE .
```

### Regular Expressions

```
regex_lit           = "/" { unicode_char } "/" .
```

**Comparators:**
`=~` matches against
`!~` doesn't match against

> **Note:** Use regular expressions to match measurements and tags.
You cannot use regular expressions to match databases, retention policies, or fields.

## Queries

A query is composed of one or more statements separated by a semicolon.

```
query               = statement { ";" statement } .

statement           = alter_retention_policy_stmt |
                      create_continuous_query_stmt |
                      create_database_stmt |
                      create_retention_policy_stmt |
                      create_subscription_stmt |
                      create_user_stmt |
                      delete_stmt |
                      drop_continuous_query_stmt |
                      drop_database_stmt |
                      drop_measurement_stmt |
                      drop_retention_policy_stmt |
                      drop_series_stmt |
                      drop_shard_stmt |
                      drop_subscription_stmt |
                      drop_user_stmt |
                      grant_stmt |
                      kill_query_statement |
                      show_continuous_queries_stmt |
                      show_databases_stmt |
                      show_field_keys_stmt |
                      show_grants_stmt |
                      show_measurements_stmt |
                      show_queries_stmt |
                      show_retention_policies |
                      show_series_stmt |
                      show_shard_groups_stmt |
                      show_shards_stmt |
                      show_subscriptions_stmt|
                      show_tag_keys_stmt |
                      show_tag_values_stmt |
                      show_users_stmt |
                      revoke_stmt |
                      select_stmt .
```

## Statements

### ALTER RETENTION POLICY

```
alter_retention_policy_stmt  = "ALTER RETENTION POLICY" policy_name on_clause
                               retention_policy_option
                               [ retention_policy_option ]
                               [ retention_policy_option ]
                               [ retention_policy_option ] .
```

> Replication factors do not serve a purpose with single node instances.

#### Examples:

```sql
-- Set default retention policy for mydb to 1h.cpu.
ALTER RETENTION POLICY "1h.cpu" ON "mydb" DEFAULT

-- Change duration and replication factor.
ALTER RETENTION POLICY "policy1" ON "somedb" DURATION 1h REPLICATION 4
```

### CREATE CONTINUOUS QUERY

```
create_continuous_query_stmt = "CREATE CONTINUOUS QUERY" query_name on_clause
                               [ "RESAMPLE" resample_opts ]
                               "BEGIN" select_stmt "END" .

query_name                   = identifier .

resample_opts                = (every_stmt for_stmt | every_stmt | for_stmt) .
every_stmt                   = "EVERY" duration_lit
for_stmt                     = "FOR" duration_lit
```

#### Examples:

```sql
-- selects from DEFAULT retention policy and writes into 6_months retention policy
CREATE CONTINUOUS QUERY "10m_event_count"
ON "db_name"
BEGIN
  SELECT count("value")
  INTO "6_months"."events"
  FROM "events"
  GROUP BY time(10m)
END;

-- this selects from the output of one continuous query in one retention policy and outputs to another series in another retention policy
CREATE CONTINUOUS QUERY "1h_event_count"
ON "db_name"
BEGIN
  SELECT sum("count") as "count"
  INTO "2_years"."events"
  FROM "6_months"."events"
  GROUP BY time(1h)
END;

-- this customizes the resample interval so the interval is queried every 10s and intervals are resampled until 2m after their start time
-- when resample is used, at least one of "EVERY" or "FOR" must be used
CREATE CONTINUOUS QUERY "cpu_mean"
ON "db_name"
RESAMPLE EVERY 10s FOR 2m
BEGIN
  SELECT mean("value")
  INTO "cpu_mean"
  FROM "cpu"
  GROUP BY time(1m)
END;
```

### CREATE DATABASE

```
create_database_stmt = "CREATE DATABASE" db_name
                       [ WITH
                           [ retention_policy_duration ]
                           [ retention_policy_replication ]
                           [ retention_policy_shard_group_duration ]
                           [ retention_policy_name ]
                       ] .
```

> Replication factors do not serve a purpose with single node instances.

#### Examples:

```sql
-- Create a database called foo
CREATE DATABASE "foo"

-- Create a database called bar with a new DEFAULT retention policy and specify the duration, replication, shard group duration, and name of that retention policy
CREATE DATABASE "bar" WITH DURATION 1d REPLICATION 1 SHARD DURATION 30m NAME "myrp"

-- Create a database called mydb with a new DEFAULT retention policy and specify the name of that retention policy
CREATE DATABASE "mydb" WITH NAME "myrp"
```

### CREATE RETENTION POLICY

```
create_retention_policy_stmt = "CREATE RETENTION POLICY" policy_name on_clause
                               retention_policy_duration
                               retention_policy_replication
                               [ retention_policy_shard_group_duration ]
                               [ "DEFAULT" ] .
```

> Replication factors do not serve a purpose with single node instances.

#### Examples

```sql
-- Create a retention policy.
CREATE RETENTION POLICY "10m.events" ON "somedb" DURATION 60m REPLICATION 2

-- Create a retention policy and set it as the DEFAULT.
CREATE RETENTION POLICY "10m.events" ON "somedb" DURATION 60m REPLICATION 2 DEFAULT

-- Create a retention policy and specify the shard group duration.
CREATE RETENTION POLICY "10m.events" ON "somedb" DURATION 60m REPLICATION 2 SHARD DURATION 30m
```

### CREATE SUBSCRIPTION

Subscriptions tell InfluxDB to send all the data it receives to Kapacitor or other third parties.

```
create_subscription_stmt = "CREATE SUBSCRIPTION" subscription_name "ON" db_name "." retention_policy "DESTINATIONS" ("ANY"|"ALL") host { "," host} .
```

#### Examples:

```sql
-- Create a SUBSCRIPTION on database 'mydb' and retention policy 'autogen' that send data to 'example.com:9090' via UDP.
CREATE SUBSCRIPTION "sub0" ON "mydb"."autogen" DESTINATIONS ALL 'udp://example.com:9090'

-- Create a SUBSCRIPTION on database 'mydb' and retention policy 'autogen' that round robins the data to 'h1.example.com:9090' and 'h2.example.com:9090'.
CREATE SUBSCRIPTION "sub0" ON "mydb"."autogen" DESTINATIONS ANY 'udp://h1.example.com:9090', 'udp://h2.example.com:9090'
```

### CREATE USER

```
create_user_stmt = "CREATE USER" user_name "WITH PASSWORD" password
                   [ "WITH ALL PRIVILEGES" ] .
```

#### Examples:

```sql
-- Create a normal database user.
CREATE USER "jdoe" WITH PASSWORD '1337password'

-- Create an admin user.
-- Note: Unlike the GRANT statement, the "PRIVILEGES" keyword is required here.
CREATE USER "jdoe" WITH PASSWORD '1337password' WITH ALL PRIVILEGES
```

> **Note:** The password string must be wrapped in single quotes.

### DELETE

```
delete_stmt = "DELETE" ( from_clause | where_clause | from_clause where_clause ) .
```

#### Examples:

```sql
DELETE FROM "cpu"
DELETE FROM "cpu" WHERE time < '2000-01-01T00:00:00Z'
DELETE WHERE time < '2000-01-01T00:00:00Z'
```

### DROP CONTINUOUS QUERY

```
drop_continuous_query_stmt = "DROP CONTINUOUS QUERY" query_name on_clause .
```

#### Example:

```sql
DROP CONTINUOUS QUERY "myquery" ON "mydb"
```

### DROP DATABASE

```
drop_database_stmt = "DROP DATABASE" db_name .
```

#### Example:

```sql
DROP DATABASE "mydb"
```

### DROP MEASUREMENT

```
drop_measurement_stmt = "DROP MEASUREMENT" measurement .
```

#### Examples:

```sql
-- drop the cpu measurement
DROP MEASUREMENT "cpu"
```

### DROP RETENTION POLICY

```
drop_retention_policy_stmt = "DROP RETENTION POLICY" policy_name on_clause .
```

#### Example:

```sql
-- drop the retention policy named 1h.cpu from mydb
DROP RETENTION POLICY "1h.cpu" ON "mydb"
```

### DROP SERIES

```
drop_series_stmt = "DROP SERIES" ( from_clause | where_clause | from_clause where_clause ) .
```

#### Example:

```sql
DROP SERIES FROM "telegraf"."autogen"."cpu" WHERE cpu = 'cpu8'

```

### DROP SHARD

```
drop_shard_stmt = "DROP SHARD" ( shard_id ) .
```

#### Example:

```
DROP SHARD 1
```

### DROP SUBSCRIPTION

```
drop_subscription_stmt = "DROP SUBSCRIPTION" subscription_name "ON" db_name "." retention_policy .
```

#### Example:

```sql
DROP SUBSCRIPTION "sub0" ON "mydb"."autogen"
```

### DROP USER

```
drop_user_stmt = "DROP USER" user_name .
```

#### Example:

```sql
DROP USER "jdoe"
```

### GRANT

> **NOTE:** Users can be granted privileges on databases that do not exist.

```
grant_stmt = "GRANT" privilege [ on_clause ] to_clause .
```

#### Examples:

```sql
-- grant admin privileges
GRANT ALL TO "jdoe"

-- grant read access to a database
GRANT READ ON "mydb" TO "jdoe"
```

### KILL QUERY

```
kill_query_statement = "KILL QUERY" query_id .
```

#### Examples:

```
--- kill a query with the query_id 36
KILL QUERY 36
```

> **NOTE:** Identify the `query_id` from the `SHOW QUERIES` output.

### SHOW CONTINUOUS QUERIES

```
show_continuous_queries_stmt = "SHOW CONTINUOUS QUERIES" .
```

#### Example:

```sql
-- show all continuous queries
SHOW CONTINUOUS QUERIES
```

### SHOW DATABASES

```
show_databases_stmt = "SHOW DATABASES" .
```

#### Example:

```sql
-- show all databases
SHOW DATABASES
```

### SHOW FIELD KEYS

```
show_field_keys_stmt = "SHOW FIELD KEYS" [ from_clause ] .
```

#### Examples:

```sql
-- show field keys and field value data types from all measurements
SHOW FIELD KEYS

-- show field keys and field value data types from specified measurement
SHOW FIELD KEYS FROM "cpu"
```

### SHOW GRANTS

```
show_grants_stmt = "SHOW GRANTS FOR" user_name .
```

#### Example:

```sql
-- show grants for jdoe
SHOW GRANTS FOR "jdoe"
```

### SHOW MEASUREMENTS

```
show_measurements_stmt = "SHOW MEASUREMENTS" [ with_measurement_clause ] [ where_clause ] [ limit_clause ] [ offset_clause ] .
```

#### Examples:

```sql
-- show all measurements
SHOW MEASUREMENTS

-- show measurements where region tag = 'uswest' AND host tag = 'serverA'
SHOW MEASUREMENTS WHERE "region" = 'uswest' AND "host" = 'serverA'

-- show measurements that start with 'h2o'
SHOW MEASUREMENTS WITH MEASUREMENT =~ /h2o.*/
```

### SHOW QUERIES

```
show_queries_stmt = "SHOW QUERIES" .
```

#### Example:

```sql
-- show all currently-running queries
SHOW QUERIES
```

### SHOW RETENTION POLICIES

```
show_retention_policies = "SHOW RETENTION POLICIES" on_clause .
```

#### Example:

```sql
-- show all retention policies on a database
SHOW RETENTION POLICIES ON "mydb"
```

### SHOW SERIES

```
show_series_stmt = "SHOW SERIES" [ from_clause ] [ where_clause ] [ limit_clause ] [ offset_clause ] .
```

#### Example:

```sql
SHOW SERIES FROM "telegraf"."autogen"."cpu" WHERE cpu = 'cpu8'
```

### SHOW SHARD GROUPS

```
show_shard_groups_stmt = "SHOW SHARD GROUPS" .
```

#### Example:

```sql
SHOW SHARD GROUPS
```

### SHOW SHARDS

```
show_shards_stmt = "SHOW SHARDS" .
```

#### Example:

```sql
SHOW SHARDS
```

### SHOW SUBSCRIPTIONS

```
show_subscriptions_stmt = "SHOW SUBSCRIPTIONS" .
```

#### Example:

```sql
SHOW SUBSCRIPTIONS
```

### SHOW TAG KEYS

```
show_tag_keys_stmt = "SHOW TAG KEYS" [ from_clause ] [ where_clause ] [ group_by_clause ]
                     [ limit_clause ] [ offset_clause ] .
```

#### Examples:

```sql
-- show all tag keys
SHOW TAG KEYS

-- show all tag keys from the cpu measurement
SHOW TAG KEYS FROM "cpu"

-- show all tag keys from the cpu measurement where the region key = 'uswest'
SHOW TAG KEYS FROM "cpu" WHERE "region" = 'uswest'

-- show all tag keys where the host key = 'serverA'
SHOW TAG KEYS WHERE "host" = 'serverA'
```

### SHOW TAG VALUES

```
show_tag_values_stmt = "SHOW TAG VALUES" [ from_clause ] with_tag_clause [ where_clause ]
                       [ group_by_clause ] [ limit_clause ] [ offset_clause ] .
```

#### Examples:

```sql
-- show all tag values across all measurements for the region tag
SHOW TAG VALUES WITH KEY = "region"

-- show tag values from the cpu measurement for the region tag
SHOW TAG VALUES FROM "cpu" WITH KEY = "region"

-- show tag values across all measurements for all tag keys that do not include the letter c
SHOW TAG VALUES WITH KEY !~ /.*c.*/

-- show tag values from the cpu measurement for region & host tag keys where service = 'redis'
SHOW TAG VALUES FROM "cpu" WITH KEY IN ("region", "host") WHERE "service" = 'redis'
```

### SHOW USERS

```
show_users_stmt = "SHOW USERS" .
```

#### Example:

```sql
-- show all users
SHOW USERS
```

### REVOKE

```
revoke_stmt = "REVOKE" privilege [ on_clause ] "FROM" user_name .
```

#### Examples:

```sql
-- revoke admin privileges from jdoe
REVOKE ALL PRIVILEGES FROM "jdoe"

-- revoke read privileges from jdoe on mydb
REVOKE READ ON "mydb" FROM "jdoe"
```

### SELECT

```
select_stmt = "SELECT" fields from_clause [ into_clause ] [ where_clause ]
              [ group_by_clause ] [ order_by_clause ] [ limit_clause ]
              [ offset_clause ] [ slimit_clause ] [ soffset_clause ] .
```

#### Examples:

```sql
-- select mean value from the cpu measurement where region = 'uswest' grouped by 10 minute intervals
SELECT mean("value") FROM "cpu" WHERE "region" = 'uswest' GROUP BY time(10m) fill(0)

-- select from all measurements beginning with cpu into the same measurement name in the cpu_1h retention policy
SELECT mean("value") INTO "cpu_1h".:MEASUREMENT FROM /cpu.*/
```

## Clauses

```
from_clause     = "FROM" measurements .

group_by_clause = "GROUP BY" dimensions fill(fill_option).

into_clause     = "INTO" ( measurement | back_ref ).

limit_clause    = "LIMIT" int_lit .

offset_clause   = "OFFSET" int_lit .

slimit_clause   = "SLIMIT" int_lit .

soffset_clause  = "SOFFSET" int_lit .

on_clause       = "ON" db_name .

order_by_clause = "ORDER BY" sort_fields .

to_clause       = "TO" user_name .

where_clause    = "WHERE" expr .

with_measurement_clause = "WITH MEASUREMENT" ( "=" measurement | "=~" regex_lit ) .

with_tag_clause = "WITH KEY" ( "=" tag_key | "!=" tag_key | "=~" regex_lit | "IN (" tag_keys ")"  ) .
```

## Expressions

```
binary_op        = "+" | "-" | "*" | "/" | "AND" | "OR" | "=" | "!=" | "<>" | "<" |
                   "<=" | ">" | ">=" .

expr             = unary_expr { binary_op unary_expr } .

unary_expr       = "(" expr ")" | var_ref | time_lit | string_lit | int_lit |
                   float_lit | bool_lit | duration_lit | regex_lit .
```

## Other

```
alias            = "AS" identifier .

back_ref         = ( policy_name ".:MEASUREMENT" ) |
                   ( db_name "." [ policy_name ] ".:MEASUREMENT" ) .

db_name          = identifier .

dimension        = expr .

dimensions       = dimension { "," dimension } .

field_key        = identifier .

field            = expr [ alias ] .

fields           = field { "," field } .

fill_option      = "null" | "none" | "previous" | "linear" | int_lit | float_lit .

host             = string_lit .

measurement      = measurement_name |
                   ( policy_name "." measurement_name ) |
                   ( db_name "." [ policy_name ] "." measurement_name ) .

measurements     = measurement { "," measurement } .

measurement_name = identifier | regex_lit .

password         = string_lit .

policy_name      = identifier .

privilege        = "ALL" [ "PRIVILEGES" ] | "READ" | "WRITE" .

query_id         = int_lit .

query_name       = identifier .

retention_policy = identifier .

retention_policy_option      = retention_policy_duration |
                               retention_policy_replication |
                               retention_policy_shard_group_duration |
                               "DEFAULT" .

retention_policy_duration    = "DURATION" duration_lit .

retention_policy_replication = "REPLICATION" int_lit .

retention_policy_shard_group_duration = "SHARD DURATION" duration_lit .

retention_policy_name = "NAME" identifier .

series_id        = int_lit .

shard_id         = int_lit .

sort_field       = field_key [ ASC | DESC ] .

sort_fields      = sort_field { "," sort_field } .

subscription_name = identifier .

tag_key          = identifier .

tag_keys         = tag_key { "," tag_key } .

user_name        = identifier .

var_ref          = measurement .
```

## Query Engine Internals

Once you understand the language itself, it's important to know how these
language constructs are implemented in the query engine. This gives you an
intuitive sense for how results will be processed and how to create efficient
queries.

The life cycle of a query looks like this:

1. InfluxQL query string is tokenized and then parsed into an abstract syntax
   tree (AST). This is the code representation of the query itself.

2. The AST is passed to the `QueryExecutor` which directs queries to the
   appropriate handlers. For example, queries related to meta data are executed
   by the meta service and `SELECT` statements are executed by the shards
   themselves.

3. The query engine then determines the shards that match the `SELECT`
   statement's time range. From these shards, iterators are created for each
   field in the statement.

4. Iterators are passed to the emitter which drains them and joins the resulting
   points. The emitter's job is to convert simple time/value points into the
   more complex result objects that are returned to the client.


### Understanding Iterators

Iterators are at the heart of the query engine. They provide a simple interface
for looping over a set of points. For example, this is an iterator over Float
points:

```
type FloatIterator interface {
    Next() *FloatPoint
}
```

These iterators are created through the `IteratorCreator` interface:

```
type IteratorCreator interface {
    CreateIterator(opt *IteratorOptions) (Iterator, error)
}
```

The `IteratorOptions` provide arguments about field selection, time ranges,
and dimensions that the iterator creator can use when planning an iterator.
The `IteratorCreator` interface is used at many levels such as the `Shards`,
`Shard`, and `Engine`. This allows optimizations to be performed when applicable
such as returning a precomputed `COUNT()`.

Iterators aren't just for reading raw data from storage though. Iterators can be
composed so that they provided additional functionality around an input
iterator. For example, a `DistinctIterator` can compute the distinct values for
each time window for an input iterator. Or a `FillIterator` can generate
additional points that are missing from an input iterator.

This composition also lends itself well to aggregation. For example, a statement
such as this:

```
SELECT MEAN(value) FROM cpu GROUP BY time(10m)
```

In this case, `MEAN(value)` is a `MeanIterator` wrapping an iterator from the
underlying shards. However, if we can add an additional iterator to determine
the derivative of the mean:

```
SELECT DERIVATIVE(MEAN(value), 20m) FROM cpu GROUP BY time(10m)
```


### Understanding Auxiliary Fields

Because InfluxQL allows users to use selector functions such as `FIRST()`,
`LAST()`, `MIN()`, and `MAX()`, the engine must provide a way to return related
data at the same time with the selected point.

For example, in this query:

```
SELECT FIRST(value), host FROM cpu GROUP BY time(1h)
```

We are selecting the first `value` that occurs every hour but we also want to
retrieve the `host` associated with that point. Since the `Point` types only
specify a single typed `Value` for efficiency, we push the `host` into the
auxiliary fields of the point. These auxiliary fields are attached to the point
until it is passed to the emitter where the fields get split off to their own
iterator.


### Built-in Iterators

There are many helper iterators that let us build queries:

* Merge Iterator - This iterator combines one or more iterators into a single
  new iterator of the same type. This iterator guarantees that all points
  within a window will be output before starting the next window but does not
  provide ordering guarantees within the window. This allows for fast access
  for aggregate queries which do not need stronger sorting guarantees.

* Sorted Merge Iterator - This iterator also combines one or more iterators
  into a new iterator of the same type. However, this iterator guarantees
  time ordering of every point. This makes it slower than the `MergeIterator`
  but this ordering guarantee is required for non-aggregate queries which
  return the raw data points.

* Limit Iterator - This iterator limits the number of points per name/tag
  group. This is the implementation of the `LIMIT` & `OFFSET` syntax.

* Fill Iterator - This iterator injects extra points if they are missing from
  the input iterator. It can provide `null` points, points with the previous
  value, or points with a specific value.

* Buffered Iterator - This iterator provides the ability to "unread" a point
  back onto a buffer so it can be read again next time. This is used extensively
  to provide lookahead for windowing.

* Reduce Iterator - This iterator calls a reduction function for each point in
  a window. When the window is complete then all points for that window are
  output. This is used for simple aggregate functions such as `COUNT()`.

* Reduce Slice Iterator - This iterator collects all points for a window first
  and then passes them all to a reduction function at once. The results are
  returned from the iterator. This is used for aggregate functions such as
  `DERIVATIVE()`.

* Transform Iterator - This iterator calls a transform function for each point
  from an input iterator. This is used for executing binary expressions.

* Dedupe Iterator - This iterator only outputs unique points. It is resource
  intensive so it is only used for small queries such as meta query statements.


### Call Iterators

Function calls in InfluxQL are implemented at two levels. Some calls can be
wrapped at multiple layers to improve efficiency. For example, a `COUNT()` can
be performed at the shard level and then multiple `CountIterator`s can be
wrapped with another `CountIterator` to compute the count of all shards. These
iterators can be created using `NewCallIterator()`.

Some iterators are more complex or need to be implemented at a higher level.
For example, the `DERIVATIVE()` needs to retrieve all points for a window first
before performing the calculation. This iterator is created by the engine itself
and is never requested to be created by the lower levels.
