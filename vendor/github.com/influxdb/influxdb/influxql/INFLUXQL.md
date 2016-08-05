# The Influx Query Language Specification

## Introduction

This is a reference for the Influx Query Language ("InfluxQL").

InfluxQL is a SQL-like query language for interacting with InfluxDB.  It has been lovingly crafted to feel familiar to those coming from other SQL or SQL-like environments while providing features specific to storing and analyzing time series data.

## Notation

The syntax is specified using Extended Backus-Naur Form ("EBNF").  EBNF is the same notation used in the [Go](http://golang.org) programming language specification, which can be found [here](https://golang.org/ref/spec).  Not so coincidentally, InfluxDB is written in Go.

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

Letters are the set of ASCII characters plus the underscore character _ (U+005F) is considered a letter.

Only decimal digits are supported.

```
letter              = ascii_letter | "_" .
ascii_letter        = "A" … "Z" | "a" … "z" .
digit               = "0" … "9" .
```

## Identifiers

Identifiers are tokens which refer to database names, retention policy names, user names, measurement names, tag keys, and field names.

The rules:

- double quoted identifiers can contain any unicode character other than a new line
- double quoted identifiers can contain escaped `"` characters (i.e., `\"`)
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
ALL          ALTER        AS           ASC          BEGIN        BY
CREATE       CONTINUOUS   DATABASE     DATABASES    DEFAULT      DELETE
DESC         DROP         DURATION     END          EXISTS       EXPLAIN
FIELD        FROM         GRANT        GROUP        IF           IN
INNER        INSERT       INTO         KEY          KEYS         LIMIT
SHOW         MEASUREMENT  MEASUREMENTS OFFSET       ON           ORDER
PASSWORD     POLICY       POLICIES     PRIVILEGES   QUERIES      QUERY
READ         REPLICATION  RETENTION    REVOKE       SELECT       SERIES
SLIMIT       SOFFSET      TAG          TO           USER         USERS
VALUES       WHERE        WITH         WRITE
```

## Literals

### Integers

InfluxQL supports decimal integer literals.  Hexadecimal and octal literals are not currently supported.

```
int_lit             = ( "1" … "9" ) { digit } .
```

### Floats

InfluxQL supports floating-point literals.  Exponents are not currently supported.

```
float_lit           = int_lit "." int_lit .
```

### Strings

String literals must be surrounded by single quotes. Strings may contain `'` characters as long as they are escaped (i.e., `\'`).

```
string_lit          = `'` { unicode_char } `'`' .
```

### Durations

Duration literals specify a length of time.  An integer literal followed immediately (with no spaces) by a duration unit listed below is interpreted as a duration literal.

```
Duration unit definitions
-------------------------
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

```
duration_lit        = int_lit duration_unit .
duration_unit       = "u" | "µ" | "s" | "h" | "d" | "w" | "ms" .
```

### Dates & Times

The date and time literal format is not specified in EBNF like the rest of this document.  It is specified using Go's date / time parsing format, which is a reference date written in the format required by InfluxQL.  The reference date time is:

InfluxQL reference date time: January 2nd, 2006 at 3:04:05 PM

```
time_lit            = "2006-01-02 15:04:05.999999" | "2006-01-02"
```

### Booleans

```
bool_lit            = TRUE | FALSE .
```

### Regular Expressions

```
regex_lit           = "/" { unicode_char } "/" .
```

## Queries

A query is composed of one or more statements separated by a semicolon.

```
query               = statement { ; statement } .

statement           = alter_retention_policy_stmt |
                      create_continuous_query_stmt |
                      create_database_stmt |
                      create_retention_policy_stmt |
                      create_user_stmt |
                      delete_stmt |
                      drop_continuous_query_stmt |
                      drop_database_stmt |
                      drop_measurement_stmt |
                      drop_retention_policy_stmt |
                      drop_series_stmt |
                      drop_user_stmt |
                      grant_stmt |
                      show_continuous_queries_stmt |
                      show_databases_stmt |
                      show_field_keys_stmt |
                      show_measurements_stmt |
                      show_retention_policies |
                      show_series_stmt |
                      show_tag_keys_stmt |
                      show_tag_values_stmt |
                      show_users_stmt |
                      revoke_stmt |
                      select_stmt .
```

## Statements

### ALTER RETENTION POLICY

```
alter_retention_policy_stmt  = "ALTER RETENTION POLICY" policy_name "ON"
                               db_name retention_policy_option
                               [ retention_policy_option ]
                               [ retention_policy_option ] .

db_name                      = identifier .

policy_name                  = identifier .

retention_policy_option      = retention_policy_duration |
                               retention_policy_replication |
                               "DEFAULT" .

retention_policy_duration    = "DURATION" duration_lit .
retention_policy_replication = "REPLICATION" int_lit
```

#### Examples:

```sql
-- Set default retention policy for mydb to 1h.cpu.
ALTER RETENTION POLICY "1h.cpu" ON mydb DEFAULT;

-- Change duration and replication factor.
ALTER RETENTION POLICY policy1 ON somedb DURATION 1h REPLICATION 4
```

### CREATE CONTINUOUS QUERY

```
create_continuous_query_stmt = "CREATE CONTINUOUS QUERY" query_name "ON" db_name
                               "BEGIN" select_stmt "END" .

query_name                   = identifier .
```

#### Examples:

```sql
-- selects from default retention policy and writes into 6_months retention policy
CREATE CONTINUOUS QUERY "10m_event_count"
ON db_name
BEGIN
  SELECT count(value)
  INTO "6_months".events
  FROM events
  GROUP BY time(10m)
END;

-- this selects from the output of one continuous query in one retention policy and outputs to another series in another retention policy
CREATE CONTINUOUS QUERY "1h_event_count"
ON db_name
BEGIN
  SELECT sum(count) as count
  INTO "2_years".events
  FROM "6_months".events
  GROUP BY time(1h)
END;
```

### CREATE DATABASE

```
create_database_stmt = "CREATE DATABASE" db_name
```

#### Example:

```sql
CREATE DATABASE foo
```

### CREATE RETENTION POLICY

```
create_retention_policy_stmt = "CREATE RETENTION POLICY" policy_name "ON"
                               db_name retention_policy_duration
                               retention_policy_replication
                               [ "DEFAULT" ] .
```

#### Examples

```sql
-- Create a retention policy.
CREATE RETENTION POLICY "10m.events" ON somedb DURATION 10m REPLICATION 2;

-- Create a retention policy and set it as the default.
CREATE RETENTION POLICY "10m.events" ON somedb DURATION 10m REPLICATION 2 DEFAULT;
```

### CREATE USER

```
create_user_stmt = "CREATE USER" user_name "WITH PASSWORD" password
                   [ "WITH ALL PRIVILEGES" ] .
```

#### Examples:

```sql
-- Create a normal database user.
CREATE USER jdoe WITH PASSWORD '1337password';

-- Create a cluster admin.
-- Note: Unlike the GRANT statement, the "PRIVILEGES" keyword is required here.
CREATE USER jdoe WITH PASSWORD '1337password' WITH ALL PRIVILEGES;
```

### DELETE

```
delete_stmt  = "DELETE" from_clause where_clause .
```

#### Example:

```sql
-- delete data points from the cpu measurement where the region tag
-- equals 'uswest'
DELETE FROM cpu WHERE region = 'uswest';
```

### DROP CONTINUOUS QUERY

drop_continuous_query_stmt = "DROP CONTINUOUS QUERY" query_name .

#### Example:

```sql
DROP CONTINUOUS QUERY myquery;
```

### DROP DATABASE

drop_database_stmt = "DROP DATABASE" db_name .

#### Example:

```sql
DROP DATABASE mydb;
```

### DROP MEASUREMENT

```
drop_measurement_stmt = "DROP MEASUREMENT" measurement .
```

#### Examples:

```sql
-- drop the cpu measurement
DROP MEASUREMENT cpu;
```

### DROP RETENTION POLICY

```
drop_retention_policy_stmt = "DROP RETENTION POLICY" policy_name "ON" db_name .
```

#### Example:

```sql
-- drop the retention policy named 1h.cpu from mydb
DROP RETENTION POLICY "1h.cpu" ON mydb;
```

### DROP SERIES

```
drop_series_stmt = "DROP SERIES" [ from_clause ] [ where_clause ]
```

#### Example:

```sql

```

### DROP USER

```
drop_user_stmt = "DROP USER" user_name .
```

#### Example:

```sql
DROP USER jdoe;

```

### GRANT

NOTE: Users can be granted privileges on databases that do not exist.

```
grant_stmt = "GRANT" privilege [ on_clause ] to_clause
```

#### Examples:

```sql
-- grant cluster admin privileges
GRANT ALL TO jdoe;

-- grant read access to a database
GRANT READ ON mydb TO jdoe;
```

### SHOW CONTINUOUS QUERIES

show_continuous_queries_stmt = "SHOW CONTINUOUS QUERIES"

#### Example:

```sql
-- show all continuous queries
SHOW CONTINUOUS QUERIES;
```

### SHOW DATABASES

```
show_databases_stmt = "SHOW DATABASES" .
```

#### Example:

```sql
-- show all databases
SHOW DATABASES;
```

### SHOW FIELD

show_field_keys_stmt = "SHOW FIELD KEYS" [ from_clause ] .

#### Examples:

```sql
-- show field keys from all measurements
SHOW FIELD KEYS;

-- show field keys from specified measurement
SHOW FIELD KEYS FROM cpu;
```

### SHOW MEASUREMENTS

show_measurements_stmt = [ where_clause ] [ group_by_clause ] [ limit_clause ]
                         [ offset_clause ] .

```sql
-- show all measurements
SHOW MEASUREMENTS;

-- show measurements where region tag = 'uswest' AND host tag = 'serverA'
SHOW MEASUREMENTS WHERE region = 'uswest' AND host = 'serverA';
```

### SHOW RETENTION POLICIES

```
show_retention_policies = "SHOW RETENTION POLICIES" db_name .
```

#### Example:

```sql
-- show all retention policies on a database
SHOW RETENTION POLICIES mydb;
```

### SHOW SERIES

```
show_series_stmt = [ from_clause ] [ where_clause ] [ group_by_clause ]
                   [ limit_clause ] [ offset_clause ] .
```

#### Example:

```sql

```

### SHOW TAG KEYS

```
show_tag_keys_stmt = [ from_clause ] [ where_clause ] [ group_by_clause ]
                     [ limit_clause ] [ offset_clause ] .
```

#### Examples:

```sql
-- show all tag keys
SHOW TAG KEYS;

-- show all tag keys from the cpu measurement
SHOW TAG KEYS FROM cpu;

-- show all tag keys from the cpu measurement where the region key = 'uswest'
SHOW TAG KEYS FROM cpu WHERE region = 'uswest';

-- show all tag keys where the host key = 'serverA'
SHOW TAG KEYS WHERE host = 'serverA';
```

### SHOW TAG VALUES

```
show_tag_values_stmt = [ from_clause ] with_tag_clause [ where_clause ]
                       [ group_by_clause ] [ limit_clause ] [ offset_clause ] .
```

#### Examples:

```sql
-- show all tag values across all measurements for the region tag
SHOW TAG VALUES WITH TAG = 'region';

-- show tag values from the cpu measurement for the region tag
SHOW TAG VALUES FROM cpu WITH TAG = 'region';

-- show tag values from the cpu measurement for region & host tag keys where service = 'redis'
SHOW TAG VALUES FROM cpu WITH TAG IN (region, host) WHERE service = 'redis';
```

### SHOW USERS

```
show_users_stmt = "SHOW USERS" .
```

#### Example:

```sql
-- show all users
SHOW USERS;
```

### REVOKE

```
revoke_stmt = privilege [ "ON" db_name ] "FROM" user_name
```

#### Examples:

```sql
-- revoke cluster admin from jdoe
REVOKE ALL PRIVILEGES FROM jdoe;

-- revoke read privileges from jdoe on mydb
REVOKE READ ON mydb FROM jdoe;
```

### SELECT

```
select_stmt = fields from_clause [ into_clause ] [ where_clause ]
              [ group_by_clause ] [ order_by_clause ] [ limit_clause ]
              [ offset_clause ] [ slimit_clause ] [ soffset_clause ].
```

#### Examples:

```sql
-- select mean value from the cpu measurement where region = 'uswest' grouped by 10 minute intervals
SELECT mean(value) FROM cpu WHERE region = 'uswest' GROUP BY time(10m) fill(0);
```

## Clauses

```
from_clause     = "FROM" measurements .

group_by_clause = "GROUP BY" dimensions fill(<option>).

limit_clause    = "LIMIT" int_lit .

offset_clause   = "OFFSET" int_lit .

slimit_clause    = "SLIMIT" int_lit .

soffset_clause   = "SOFFSET" int_lit .

on_clause       = db_name .

order_by_clause = "ORDER BY" sort_fields .

to_clause       = user_name .

where_clause    = "WHERE" expr .
```

## Expressions

```
binary_op        = "+" | "-" | "*" | "/" | "AND" | "OR" | "=" | "!=" | "<" |
                   "<=" | ">" | ">=" .

expr             = unary_expr { binary_op unary_expr } .

unary_expr       = "(" expr ")" | var_ref | time_lit | string_lit | int_lit |
                   float_lit | bool_lit | duration_lit | regex_lit .
```

## Other

```
dimension         = expr .

dimensions        = dimension { "," dimension } .

field            = expr [ alias ] .

fields           = field { "," field } .

measurement      = measurement_name |
                   ( policy_name "." measurement_name ) |
                   ( db_name "." [ policy_name ] "." measurement_name ) .

measurements     = measurement { "," measurement } .

measurement_name = identifier .

password         = identifier .

policy_name      = identifier .

privilege        = "ALL" [ "PRIVILEGES" ] | "READ" | "WRITE" .

series_id        = int_lit .

sort_field       = field_name [ ASC | DESC ] .

sort_fields      = sort_field { "," sort_field } .

user_name        = identifier .
```
