// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.21.0"

import "go.opentelemetry.io/otel/attribute"

// The shared attributes used to report a single exception associated with a
// span or log.
const (
	// ExceptionTypeKey is the attribute Key conforming to the "exception.type"
	// semantic conventions. It represents the type of the exception (its
	// fully-qualified class name, if applicable). The dynamic type of the
	// exception should be preferred over the static type in languages that
	// support it.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'java.net.ConnectException', 'OSError'
	ExceptionTypeKey = attribute.Key("exception.type")

	// ExceptionMessageKey is the attribute Key conforming to the
	// "exception.message" semantic conventions. It represents the exception
	// message.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Division by zero', "Can't convert 'int' object to str
	// implicitly"
	ExceptionMessageKey = attribute.Key("exception.message")

	// ExceptionStacktraceKey is the attribute Key conforming to the
	// "exception.stacktrace" semantic conventions. It represents a stacktrace
	// as a string in the natural representation for the language runtime. The
	// representation is to be determined and documented by each language SIG.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Exception in thread "main" java.lang.RuntimeException: Test
	// exception\\n at '
	//  'com.example.GenerateTrace.methodB(GenerateTrace.java:13)\\n at '
	//  'com.example.GenerateTrace.methodA(GenerateTrace.java:9)\\n at '
	//  'com.example.GenerateTrace.main(GenerateTrace.java:5)'
	ExceptionStacktraceKey = attribute.Key("exception.stacktrace")
)

// ExceptionType returns an attribute KeyValue conforming to the
// "exception.type" semantic conventions. It represents the type of the
// exception (its fully-qualified class name, if applicable). The dynamic type
// of the exception should be preferred over the static type in languages that
// support it.
func ExceptionType(val string) attribute.KeyValue {
	return ExceptionTypeKey.String(val)
}

// ExceptionMessage returns an attribute KeyValue conforming to the
// "exception.message" semantic conventions. It represents the exception
// message.
func ExceptionMessage(val string) attribute.KeyValue {
	return ExceptionMessageKey.String(val)
}

// ExceptionStacktrace returns an attribute KeyValue conforming to the
// "exception.stacktrace" semantic conventions. It represents a stacktrace as a
// string in the natural representation for the language runtime. The
// representation is to be determined and documented by each language SIG.
func ExceptionStacktrace(val string) attribute.KeyValue {
	return ExceptionStacktraceKey.String(val)
}

// Span attributes used by AWS Lambda (in addition to general `faas`
// attributes).
const (
	// AWSLambdaInvokedARNKey is the attribute Key conforming to the
	// "aws.lambda.invoked_arn" semantic conventions. It represents the full
	// invoked ARN as provided on the `Context` passed to the function
	// (`Lambda-Runtime-Invoked-Function-ARN` header on the
	// `/runtime/invocation/next` applicable).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'arn:aws:lambda:us-east-1:123456:function:myfunction:myalias'
	// Note: This may be different from `cloud.resource_id` if an alias is
	// involved.
	AWSLambdaInvokedARNKey = attribute.Key("aws.lambda.invoked_arn")
)

// AWSLambdaInvokedARN returns an attribute KeyValue conforming to the
// "aws.lambda.invoked_arn" semantic conventions. It represents the full
// invoked ARN as provided on the `Context` passed to the function
// (`Lambda-Runtime-Invoked-Function-ARN` header on the
// `/runtime/invocation/next` applicable).
func AWSLambdaInvokedARN(val string) attribute.KeyValue {
	return AWSLambdaInvokedARNKey.String(val)
}

// Attributes for CloudEvents. CloudEvents is a specification on how to define
// event data in a standard way. These attributes can be attached to spans when
// performing operations with CloudEvents, regardless of the protocol being
// used.
const (
	// CloudeventsEventIDKey is the attribute Key conforming to the
	// "cloudevents.event_id" semantic conventions. It represents the
	// [event_id](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#id)
	// uniquely identifies the event.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: '123e4567-e89b-12d3-a456-426614174000', '0001'
	CloudeventsEventIDKey = attribute.Key("cloudevents.event_id")

	// CloudeventsEventSourceKey is the attribute Key conforming to the
	// "cloudevents.event_source" semantic conventions. It represents the
	// [source](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1)
	// identifies the context in which an event happened.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'https://github.com/cloudevents',
	// '/cloudevents/spec/pull/123', 'my-service'
	CloudeventsEventSourceKey = attribute.Key("cloudevents.event_source")

	// CloudeventsEventSpecVersionKey is the attribute Key conforming to the
	// "cloudevents.event_spec_version" semantic conventions. It represents the
	// [version of the CloudEvents
	// specification](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#specversion)
	// which the event uses.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1.0'
	CloudeventsEventSpecVersionKey = attribute.Key("cloudevents.event_spec_version")

	// CloudeventsEventTypeKey is the attribute Key conforming to the
	// "cloudevents.event_type" semantic conventions. It represents the
	// [event_type](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type)
	// contains a value describing the type of event related to the originating
	// occurrence.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'com.github.pull_request.opened',
	// 'com.example.object.deleted.v2'
	CloudeventsEventTypeKey = attribute.Key("cloudevents.event_type")

	// CloudeventsEventSubjectKey is the attribute Key conforming to the
	// "cloudevents.event_subject" semantic conventions. It represents the
	// [subject](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject)
	// of the event in the context of the event producer (identified by
	// source).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'mynewfile.jpg'
	CloudeventsEventSubjectKey = attribute.Key("cloudevents.event_subject")
)

// CloudeventsEventID returns an attribute KeyValue conforming to the
// "cloudevents.event_id" semantic conventions. It represents the
// [event_id](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#id)
// uniquely identifies the event.
func CloudeventsEventID(val string) attribute.KeyValue {
	return CloudeventsEventIDKey.String(val)
}

// CloudeventsEventSource returns an attribute KeyValue conforming to the
// "cloudevents.event_source" semantic conventions. It represents the
// [source](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1)
// identifies the context in which an event happened.
func CloudeventsEventSource(val string) attribute.KeyValue {
	return CloudeventsEventSourceKey.String(val)
}

// CloudeventsEventSpecVersion returns an attribute KeyValue conforming to
// the "cloudevents.event_spec_version" semantic conventions. It represents the
// [version of the CloudEvents
// specification](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#specversion)
// which the event uses.
func CloudeventsEventSpecVersion(val string) attribute.KeyValue {
	return CloudeventsEventSpecVersionKey.String(val)
}

// CloudeventsEventType returns an attribute KeyValue conforming to the
// "cloudevents.event_type" semantic conventions. It represents the
// [event_type](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type)
// contains a value describing the type of event related to the originating
// occurrence.
func CloudeventsEventType(val string) attribute.KeyValue {
	return CloudeventsEventTypeKey.String(val)
}

// CloudeventsEventSubject returns an attribute KeyValue conforming to the
// "cloudevents.event_subject" semantic conventions. It represents the
// [subject](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject)
// of the event in the context of the event producer (identified by source).
func CloudeventsEventSubject(val string) attribute.KeyValue {
	return CloudeventsEventSubjectKey.String(val)
}

// Semantic conventions for the OpenTracing Shim
const (
	// OpentracingRefTypeKey is the attribute Key conforming to the
	// "opentracing.ref_type" semantic conventions. It represents the
	// parent-child Reference type
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Note: The causal relationship between a child Span and a parent Span.
	OpentracingRefTypeKey = attribute.Key("opentracing.ref_type")
)

var (
	// The parent Span depends on the child Span in some capacity
	OpentracingRefTypeChildOf = OpentracingRefTypeKey.String("child_of")
	// The parent Span does not depend in any way on the result of the child Span
	OpentracingRefTypeFollowsFrom = OpentracingRefTypeKey.String("follows_from")
)

// The attributes used to perform database client calls.
const (
	// DBSystemKey is the attribute Key conforming to the "db.system" semantic
	// conventions. It represents an identifier for the database management
	// system (DBMS) product being used. See below for a list of well-known
	// identifiers.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	DBSystemKey = attribute.Key("db.system")

	// DBConnectionStringKey is the attribute Key conforming to the
	// "db.connection_string" semantic conventions. It represents the
	// connection string used to connect to the database. It is recommended to
	// remove embedded credentials.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Server=(localdb)\\v11.0;Integrated Security=true;'
	DBConnectionStringKey = attribute.Key("db.connection_string")

	// DBUserKey is the attribute Key conforming to the "db.user" semantic
	// conventions. It represents the username for accessing the database.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'readonly_user', 'reporting_user'
	DBUserKey = attribute.Key("db.user")

	// DBJDBCDriverClassnameKey is the attribute Key conforming to the
	// "db.jdbc.driver_classname" semantic conventions. It represents the
	// fully-qualified class name of the [Java Database Connectivity
	// (JDBC)](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/)
	// driver used to connect.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'org.postgresql.Driver',
	// 'com.microsoft.sqlserver.jdbc.SQLServerDriver'
	DBJDBCDriverClassnameKey = attribute.Key("db.jdbc.driver_classname")

	// DBNameKey is the attribute Key conforming to the "db.name" semantic
	// conventions. It represents the this attribute is used to report the name
	// of the database being accessed. For commands that switch the database,
	// this should be set to the target database (even if the command fails).
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If applicable.)
	// Stability: stable
	// Examples: 'customers', 'main'
	// Note: In some SQL databases, the database name to be used is called
	// "schema name". In case there are multiple layers that could be
	// considered for database name (e.g. Oracle instance name and schema
	// name), the database name to be used is the more specific layer (e.g.
	// Oracle schema name).
	DBNameKey = attribute.Key("db.name")

	// DBStatementKey is the attribute Key conforming to the "db.statement"
	// semantic conventions. It represents the database statement being
	// executed.
	//
	// Type: string
	// RequirementLevel: Recommended (Should be collected by default only if
	// there is sanitization that excludes sensitive information.)
	// Stability: stable
	// Examples: 'SELECT * FROM wuser_table', 'SET mykey "WuValue"'
	DBStatementKey = attribute.Key("db.statement")

	// DBOperationKey is the attribute Key conforming to the "db.operation"
	// semantic conventions. It represents the name of the operation being
	// executed, e.g. the [MongoDB command
	// name](https://docs.mongodb.com/manual/reference/command/#database-operations)
	// such as `findAndModify`, or the SQL keyword.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If `db.statement` is not
	// applicable.)
	// Stability: stable
	// Examples: 'findAndModify', 'HMSET', 'SELECT'
	// Note: When setting this to an SQL keyword, it is not recommended to
	// attempt any client-side parsing of `db.statement` just to get this
	// property, but it should be set if the operation name is provided by the
	// library being instrumented. If the SQL statement has an ambiguous
	// operation, or performs more than one operation, this value may be
	// omitted.
	DBOperationKey = attribute.Key("db.operation")
)

var (
	// Some other SQL database. Fallback only. See notes
	DBSystemOtherSQL = DBSystemKey.String("other_sql")
	// Microsoft SQL Server
	DBSystemMSSQL = DBSystemKey.String("mssql")
	// Microsoft SQL Server Compact
	DBSystemMssqlcompact = DBSystemKey.String("mssqlcompact")
	// MySQL
	DBSystemMySQL = DBSystemKey.String("mysql")
	// Oracle Database
	DBSystemOracle = DBSystemKey.String("oracle")
	// IBM DB2
	DBSystemDB2 = DBSystemKey.String("db2")
	// PostgreSQL
	DBSystemPostgreSQL = DBSystemKey.String("postgresql")
	// Amazon Redshift
	DBSystemRedshift = DBSystemKey.String("redshift")
	// Apache Hive
	DBSystemHive = DBSystemKey.String("hive")
	// Cloudscape
	DBSystemCloudscape = DBSystemKey.String("cloudscape")
	// HyperSQL DataBase
	DBSystemHSQLDB = DBSystemKey.String("hsqldb")
	// Progress Database
	DBSystemProgress = DBSystemKey.String("progress")
	// SAP MaxDB
	DBSystemMaxDB = DBSystemKey.String("maxdb")
	// SAP HANA
	DBSystemHanaDB = DBSystemKey.String("hanadb")
	// Ingres
	DBSystemIngres = DBSystemKey.String("ingres")
	// FirstSQL
	DBSystemFirstSQL = DBSystemKey.String("firstsql")
	// EnterpriseDB
	DBSystemEDB = DBSystemKey.String("edb")
	// InterSystems Caché
	DBSystemCache = DBSystemKey.String("cache")
	// Adabas (Adaptable Database System)
	DBSystemAdabas = DBSystemKey.String("adabas")
	// Firebird
	DBSystemFirebird = DBSystemKey.String("firebird")
	// Apache Derby
	DBSystemDerby = DBSystemKey.String("derby")
	// FileMaker
	DBSystemFilemaker = DBSystemKey.String("filemaker")
	// Informix
	DBSystemInformix = DBSystemKey.String("informix")
	// InstantDB
	DBSystemInstantDB = DBSystemKey.String("instantdb")
	// InterBase
	DBSystemInterbase = DBSystemKey.String("interbase")
	// MariaDB
	DBSystemMariaDB = DBSystemKey.String("mariadb")
	// Netezza
	DBSystemNetezza = DBSystemKey.String("netezza")
	// Pervasive PSQL
	DBSystemPervasive = DBSystemKey.String("pervasive")
	// PointBase
	DBSystemPointbase = DBSystemKey.String("pointbase")
	// SQLite
	DBSystemSqlite = DBSystemKey.String("sqlite")
	// Sybase
	DBSystemSybase = DBSystemKey.String("sybase")
	// Teradata
	DBSystemTeradata = DBSystemKey.String("teradata")
	// Vertica
	DBSystemVertica = DBSystemKey.String("vertica")
	// H2
	DBSystemH2 = DBSystemKey.String("h2")
	// ColdFusion IMQ
	DBSystemColdfusion = DBSystemKey.String("coldfusion")
	// Apache Cassandra
	DBSystemCassandra = DBSystemKey.String("cassandra")
	// Apache HBase
	DBSystemHBase = DBSystemKey.String("hbase")
	// MongoDB
	DBSystemMongoDB = DBSystemKey.String("mongodb")
	// Redis
	DBSystemRedis = DBSystemKey.String("redis")
	// Couchbase
	DBSystemCouchbase = DBSystemKey.String("couchbase")
	// CouchDB
	DBSystemCouchDB = DBSystemKey.String("couchdb")
	// Microsoft Azure Cosmos DB
	DBSystemCosmosDB = DBSystemKey.String("cosmosdb")
	// Amazon DynamoDB
	DBSystemDynamoDB = DBSystemKey.String("dynamodb")
	// Neo4j
	DBSystemNeo4j = DBSystemKey.String("neo4j")
	// Apache Geode
	DBSystemGeode = DBSystemKey.String("geode")
	// Elasticsearch
	DBSystemElasticsearch = DBSystemKey.String("elasticsearch")
	// Memcached
	DBSystemMemcached = DBSystemKey.String("memcached")
	// CockroachDB
	DBSystemCockroachdb = DBSystemKey.String("cockroachdb")
	// OpenSearch
	DBSystemOpensearch = DBSystemKey.String("opensearch")
	// ClickHouse
	DBSystemClickhouse = DBSystemKey.String("clickhouse")
	// Cloud Spanner
	DBSystemSpanner = DBSystemKey.String("spanner")
	// Trino
	DBSystemTrino = DBSystemKey.String("trino")
)

// DBConnectionString returns an attribute KeyValue conforming to the
// "db.connection_string" semantic conventions. It represents the connection
// string used to connect to the database. It is recommended to remove embedded
// credentials.
func DBConnectionString(val string) attribute.KeyValue {
	return DBConnectionStringKey.String(val)
}

// DBUser returns an attribute KeyValue conforming to the "db.user" semantic
// conventions. It represents the username for accessing the database.
func DBUser(val string) attribute.KeyValue {
	return DBUserKey.String(val)
}

// DBJDBCDriverClassname returns an attribute KeyValue conforming to the
// "db.jdbc.driver_classname" semantic conventions. It represents the
// fully-qualified class name of the [Java Database Connectivity
// (JDBC)](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/) driver
// used to connect.
func DBJDBCDriverClassname(val string) attribute.KeyValue {
	return DBJDBCDriverClassnameKey.String(val)
}

// DBName returns an attribute KeyValue conforming to the "db.name" semantic
// conventions. It represents the this attribute is used to report the name of
// the database being accessed. For commands that switch the database, this
// should be set to the target database (even if the command fails).
func DBName(val string) attribute.KeyValue {
	return DBNameKey.String(val)
}

// DBStatement returns an attribute KeyValue conforming to the
// "db.statement" semantic conventions. It represents the database statement
// being executed.
func DBStatement(val string) attribute.KeyValue {
	return DBStatementKey.String(val)
}

// DBOperation returns an attribute KeyValue conforming to the
// "db.operation" semantic conventions. It represents the name of the operation
// being executed, e.g. the [MongoDB command
// name](https://docs.mongodb.com/manual/reference/command/#database-operations)
// such as `findAndModify`, or the SQL keyword.
func DBOperation(val string) attribute.KeyValue {
	return DBOperationKey.String(val)
}

// Connection-level attributes for Microsoft SQL Server
const (
	// DBMSSQLInstanceNameKey is the attribute Key conforming to the
	// "db.mssql.instance_name" semantic conventions. It represents the
	// Microsoft SQL Server [instance
	// name](https://docs.microsoft.com/en-us/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15)
	// connecting to. This name is used to determine the port of a named
	// instance.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'MSSQLSERVER'
	// Note: If setting a `db.mssql.instance_name`, `server.port` is no longer
	// required (but still recommended if non-standard).
	DBMSSQLInstanceNameKey = attribute.Key("db.mssql.instance_name")
)

// DBMSSQLInstanceName returns an attribute KeyValue conforming to the
// "db.mssql.instance_name" semantic conventions. It represents the Microsoft
// SQL Server [instance
// name](https://docs.microsoft.com/en-us/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15)
// connecting to. This name is used to determine the port of a named instance.
func DBMSSQLInstanceName(val string) attribute.KeyValue {
	return DBMSSQLInstanceNameKey.String(val)
}

// Call-level attributes for Cassandra
const (
	// DBCassandraPageSizeKey is the attribute Key conforming to the
	// "db.cassandra.page_size" semantic conventions. It represents the fetch
	// size used for paging, i.e. how many rows will be returned at once.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 5000
	DBCassandraPageSizeKey = attribute.Key("db.cassandra.page_size")

	// DBCassandraConsistencyLevelKey is the attribute Key conforming to the
	// "db.cassandra.consistency_level" semantic conventions. It represents the
	// consistency level of the query. Based on consistency values from
	// [CQL](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html).
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	DBCassandraConsistencyLevelKey = attribute.Key("db.cassandra.consistency_level")

	// DBCassandraTableKey is the attribute Key conforming to the
	// "db.cassandra.table" semantic conventions. It represents the name of the
	// primary table that the operation is acting upon, including the keyspace
	// name (if applicable).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'mytable'
	// Note: This mirrors the db.sql.table attribute but references cassandra
	// rather than sql. It is not recommended to attempt any client-side
	// parsing of `db.statement` just to get this property, but it should be
	// set if it is provided by the library being instrumented. If the
	// operation is acting upon an anonymous table, or more than one table,
	// this value MUST NOT be set.
	DBCassandraTableKey = attribute.Key("db.cassandra.table")

	// DBCassandraIdempotenceKey is the attribute Key conforming to the
	// "db.cassandra.idempotence" semantic conventions. It represents the
	// whether or not the query is idempotent.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	DBCassandraIdempotenceKey = attribute.Key("db.cassandra.idempotence")

	// DBCassandraSpeculativeExecutionCountKey is the attribute Key conforming
	// to the "db.cassandra.speculative_execution_count" semantic conventions.
	// It represents the number of times a query was speculatively executed.
	// Not set or `0` if the query was not executed speculatively.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 0, 2
	DBCassandraSpeculativeExecutionCountKey = attribute.Key("db.cassandra.speculative_execution_count")

	// DBCassandraCoordinatorIDKey is the attribute Key conforming to the
	// "db.cassandra.coordinator.id" semantic conventions. It represents the ID
	// of the coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'be13faa2-8574-4d71-926d-27f16cf8a7af'
	DBCassandraCoordinatorIDKey = attribute.Key("db.cassandra.coordinator.id")

	// DBCassandraCoordinatorDCKey is the attribute Key conforming to the
	// "db.cassandra.coordinator.dc" semantic conventions. It represents the
	// data center of the coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'us-west-2'
	DBCassandraCoordinatorDCKey = attribute.Key("db.cassandra.coordinator.dc")
)

var (
	// all
	DBCassandraConsistencyLevelAll = DBCassandraConsistencyLevelKey.String("all")
	// each_quorum
	DBCassandraConsistencyLevelEachQuorum = DBCassandraConsistencyLevelKey.String("each_quorum")
	// quorum
	DBCassandraConsistencyLevelQuorum = DBCassandraConsistencyLevelKey.String("quorum")
	// local_quorum
	DBCassandraConsistencyLevelLocalQuorum = DBCassandraConsistencyLevelKey.String("local_quorum")
	// one
	DBCassandraConsistencyLevelOne = DBCassandraConsistencyLevelKey.String("one")
	// two
	DBCassandraConsistencyLevelTwo = DBCassandraConsistencyLevelKey.String("two")
	// three
	DBCassandraConsistencyLevelThree = DBCassandraConsistencyLevelKey.String("three")
	// local_one
	DBCassandraConsistencyLevelLocalOne = DBCassandraConsistencyLevelKey.String("local_one")
	// any
	DBCassandraConsistencyLevelAny = DBCassandraConsistencyLevelKey.String("any")
	// serial
	DBCassandraConsistencyLevelSerial = DBCassandraConsistencyLevelKey.String("serial")
	// local_serial
	DBCassandraConsistencyLevelLocalSerial = DBCassandraConsistencyLevelKey.String("local_serial")
)

// DBCassandraPageSize returns an attribute KeyValue conforming to the
// "db.cassandra.page_size" semantic conventions. It represents the fetch size
// used for paging, i.e. how many rows will be returned at once.
func DBCassandraPageSize(val int) attribute.KeyValue {
	return DBCassandraPageSizeKey.Int(val)
}

// DBCassandraTable returns an attribute KeyValue conforming to the
// "db.cassandra.table" semantic conventions. It represents the name of the
// primary table that the operation is acting upon, including the keyspace name
// (if applicable).
func DBCassandraTable(val string) attribute.KeyValue {
	return DBCassandraTableKey.String(val)
}

// DBCassandraIdempotence returns an attribute KeyValue conforming to the
// "db.cassandra.idempotence" semantic conventions. It represents the whether
// or not the query is idempotent.
func DBCassandraIdempotence(val bool) attribute.KeyValue {
	return DBCassandraIdempotenceKey.Bool(val)
}

// DBCassandraSpeculativeExecutionCount returns an attribute KeyValue
// conforming to the "db.cassandra.speculative_execution_count" semantic
// conventions. It represents the number of times a query was speculatively
// executed. Not set or `0` if the query was not executed speculatively.
func DBCassandraSpeculativeExecutionCount(val int) attribute.KeyValue {
	return DBCassandraSpeculativeExecutionCountKey.Int(val)
}

// DBCassandraCoordinatorID returns an attribute KeyValue conforming to the
// "db.cassandra.coordinator.id" semantic conventions. It represents the ID of
// the coordinating node for a query.
func DBCassandraCoordinatorID(val string) attribute.KeyValue {
	return DBCassandraCoordinatorIDKey.String(val)
}

// DBCassandraCoordinatorDC returns an attribute KeyValue conforming to the
// "db.cassandra.coordinator.dc" semantic conventions. It represents the data
// center of the coordinating node for a query.
func DBCassandraCoordinatorDC(val string) attribute.KeyValue {
	return DBCassandraCoordinatorDCKey.String(val)
}

// Call-level attributes for Redis
const (
	// DBRedisDBIndexKey is the attribute Key conforming to the
	// "db.redis.database_index" semantic conventions. It represents the index
	// of the database being accessed as used in the [`SELECT`
	// command](https://redis.io/commands/select), provided as an integer. To
	// be used instead of the generic `db.name` attribute.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If other than the default
	// database (`0`).)
	// Stability: stable
	// Examples: 0, 1, 15
	DBRedisDBIndexKey = attribute.Key("db.redis.database_index")
)

// DBRedisDBIndex returns an attribute KeyValue conforming to the
// "db.redis.database_index" semantic conventions. It represents the index of
// the database being accessed as used in the [`SELECT`
// command](https://redis.io/commands/select), provided as an integer. To be
// used instead of the generic `db.name` attribute.
func DBRedisDBIndex(val int) attribute.KeyValue {
	return DBRedisDBIndexKey.Int(val)
}

// Call-level attributes for MongoDB
const (
	// DBMongoDBCollectionKey is the attribute Key conforming to the
	// "db.mongodb.collection" semantic conventions. It represents the
	// collection being accessed within the database stated in `db.name`.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'customers', 'products'
	DBMongoDBCollectionKey = attribute.Key("db.mongodb.collection")
)

// DBMongoDBCollection returns an attribute KeyValue conforming to the
// "db.mongodb.collection" semantic conventions. It represents the collection
// being accessed within the database stated in `db.name`.
func DBMongoDBCollection(val string) attribute.KeyValue {
	return DBMongoDBCollectionKey.String(val)
}

// Call-level attributes for SQL databases
const (
	// DBSQLTableKey is the attribute Key conforming to the "db.sql.table"
	// semantic conventions. It represents the name of the primary table that
	// the operation is acting upon, including the database name (if
	// applicable).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'public.users', 'customers'
	// Note: It is not recommended to attempt any client-side parsing of
	// `db.statement` just to get this property, but it should be set if it is
	// provided by the library being instrumented. If the operation is acting
	// upon an anonymous table, or more than one table, this value MUST NOT be
	// set.
	DBSQLTableKey = attribute.Key("db.sql.table")
)

// DBSQLTable returns an attribute KeyValue conforming to the "db.sql.table"
// semantic conventions. It represents the name of the primary table that the
// operation is acting upon, including the database name (if applicable).
func DBSQLTable(val string) attribute.KeyValue {
	return DBSQLTableKey.String(val)
}

// Call-level attributes for Cosmos DB.
const (
	// DBCosmosDBClientIDKey is the attribute Key conforming to the
	// "db.cosmosdb.client_id" semantic conventions. It represents the unique
	// Cosmos client instance id.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '3ba4827d-4422-483f-b59f-85b74211c11d'
	DBCosmosDBClientIDKey = attribute.Key("db.cosmosdb.client_id")

	// DBCosmosDBOperationTypeKey is the attribute Key conforming to the
	// "db.cosmosdb.operation_type" semantic conventions. It represents the
	// cosmosDB Operation Type.
	//
	// Type: Enum
	// RequirementLevel: ConditionallyRequired (when performing one of the
	// operations in this list)
	// Stability: stable
	DBCosmosDBOperationTypeKey = attribute.Key("db.cosmosdb.operation_type")

	// DBCosmosDBConnectionModeKey is the attribute Key conforming to the
	// "db.cosmosdb.connection_mode" semantic conventions. It represents the
	// cosmos client connection mode.
	//
	// Type: Enum
	// RequirementLevel: ConditionallyRequired (if not `direct` (or pick gw as
	// default))
	// Stability: stable
	DBCosmosDBConnectionModeKey = attribute.Key("db.cosmosdb.connection_mode")

	// DBCosmosDBContainerKey is the attribute Key conforming to the
	// "db.cosmosdb.container" semantic conventions. It represents the cosmos
	// DB container name.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (if available)
	// Stability: stable
	// Examples: 'anystring'
	DBCosmosDBContainerKey = attribute.Key("db.cosmosdb.container")

	// DBCosmosDBRequestContentLengthKey is the attribute Key conforming to the
	// "db.cosmosdb.request_content_length" semantic conventions. It represents
	// the request payload size in bytes
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	DBCosmosDBRequestContentLengthKey = attribute.Key("db.cosmosdb.request_content_length")

	// DBCosmosDBStatusCodeKey is the attribute Key conforming to the
	// "db.cosmosdb.status_code" semantic conventions. It represents the cosmos
	// DB status code.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (if response was received)
	// Stability: stable
	// Examples: 200, 201
	DBCosmosDBStatusCodeKey = attribute.Key("db.cosmosdb.status_code")

	// DBCosmosDBSubStatusCodeKey is the attribute Key conforming to the
	// "db.cosmosdb.sub_status_code" semantic conventions. It represents the
	// cosmos DB sub status code.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (when response was received and
	// contained sub-code.)
	// Stability: stable
	// Examples: 1000, 1002
	DBCosmosDBSubStatusCodeKey = attribute.Key("db.cosmosdb.sub_status_code")

	// DBCosmosDBRequestChargeKey is the attribute Key conforming to the
	// "db.cosmosdb.request_charge" semantic conventions. It represents the rU
	// consumed for that operation
	//
	// Type: double
	// RequirementLevel: ConditionallyRequired (when available)
	// Stability: stable
	// Examples: 46.18, 1.0
	DBCosmosDBRequestChargeKey = attribute.Key("db.cosmosdb.request_charge")
)

var (
	// invalid
	DBCosmosDBOperationTypeInvalid = DBCosmosDBOperationTypeKey.String("Invalid")
	// create
	DBCosmosDBOperationTypeCreate = DBCosmosDBOperationTypeKey.String("Create")
	// patch
	DBCosmosDBOperationTypePatch = DBCosmosDBOperationTypeKey.String("Patch")
	// read
	DBCosmosDBOperationTypeRead = DBCosmosDBOperationTypeKey.String("Read")
	// read_feed
	DBCosmosDBOperationTypeReadFeed = DBCosmosDBOperationTypeKey.String("ReadFeed")
	// delete
	DBCosmosDBOperationTypeDelete = DBCosmosDBOperationTypeKey.String("Delete")
	// replace
	DBCosmosDBOperationTypeReplace = DBCosmosDBOperationTypeKey.String("Replace")
	// execute
	DBCosmosDBOperationTypeExecute = DBCosmosDBOperationTypeKey.String("Execute")
	// query
	DBCosmosDBOperationTypeQuery = DBCosmosDBOperationTypeKey.String("Query")
	// head
	DBCosmosDBOperationTypeHead = DBCosmosDBOperationTypeKey.String("Head")
	// head_feed
	DBCosmosDBOperationTypeHeadFeed = DBCosmosDBOperationTypeKey.String("HeadFeed")
	// upsert
	DBCosmosDBOperationTypeUpsert = DBCosmosDBOperationTypeKey.String("Upsert")
	// batch
	DBCosmosDBOperationTypeBatch = DBCosmosDBOperationTypeKey.String("Batch")
	// query_plan
	DBCosmosDBOperationTypeQueryPlan = DBCosmosDBOperationTypeKey.String("QueryPlan")
	// execute_javascript
	DBCosmosDBOperationTypeExecuteJavascript = DBCosmosDBOperationTypeKey.String("ExecuteJavaScript")
)

var (
	// Gateway (HTTP) connections mode
	DBCosmosDBConnectionModeGateway = DBCosmosDBConnectionModeKey.String("gateway")
	// Direct connection
	DBCosmosDBConnectionModeDirect = DBCosmosDBConnectionModeKey.String("direct")
)

// DBCosmosDBClientID returns an attribute KeyValue conforming to the
// "db.cosmosdb.client_id" semantic conventions. It represents the unique
// Cosmos client instance id.
func DBCosmosDBClientID(val string) attribute.KeyValue {
	return DBCosmosDBClientIDKey.String(val)
}

// DBCosmosDBContainer returns an attribute KeyValue conforming to the
// "db.cosmosdb.container" semantic conventions. It represents the cosmos DB
// container name.
func DBCosmosDBContainer(val string) attribute.KeyValue {
	return DBCosmosDBContainerKey.String(val)
}

// DBCosmosDBRequestContentLength returns an attribute KeyValue conforming
// to the "db.cosmosdb.request_content_length" semantic conventions. It
// represents the request payload size in bytes
func DBCosmosDBRequestContentLength(val int) attribute.KeyValue {
	return DBCosmosDBRequestContentLengthKey.Int(val)
}

// DBCosmosDBStatusCode returns an attribute KeyValue conforming to the
// "db.cosmosdb.status_code" semantic conventions. It represents the cosmos DB
// status code.
func DBCosmosDBStatusCode(val int) attribute.KeyValue {
	return DBCosmosDBStatusCodeKey.Int(val)
}

// DBCosmosDBSubStatusCode returns an attribute KeyValue conforming to the
// "db.cosmosdb.sub_status_code" semantic conventions. It represents the cosmos
// DB sub status code.
func DBCosmosDBSubStatusCode(val int) attribute.KeyValue {
	return DBCosmosDBSubStatusCodeKey.Int(val)
}

// DBCosmosDBRequestCharge returns an attribute KeyValue conforming to the
// "db.cosmosdb.request_charge" semantic conventions. It represents the rU
// consumed for that operation
func DBCosmosDBRequestCharge(val float64) attribute.KeyValue {
	return DBCosmosDBRequestChargeKey.Float64(val)
}

// Span attributes used by non-OTLP exporters to represent OpenTelemetry Span's
// concepts.
const (
	// OTelStatusCodeKey is the attribute Key conforming to the
	// "otel.status_code" semantic conventions. It represents the name of the
	// code, either "OK" or "ERROR". MUST NOT be set if the status code is
	// UNSET.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	OTelStatusCodeKey = attribute.Key("otel.status_code")

	// OTelStatusDescriptionKey is the attribute Key conforming to the
	// "otel.status_description" semantic conventions. It represents the
	// description of the Status if it has a value, otherwise not set.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'resource not found'
	OTelStatusDescriptionKey = attribute.Key("otel.status_description")
)

var (
	// The operation has been validated by an Application developer or Operator to have completed successfully
	OTelStatusCodeOk = OTelStatusCodeKey.String("OK")
	// The operation contains an error
	OTelStatusCodeError = OTelStatusCodeKey.String("ERROR")
)

// OTelStatusDescription returns an attribute KeyValue conforming to the
// "otel.status_description" semantic conventions. It represents the
// description of the Status if it has a value, otherwise not set.
func OTelStatusDescription(val string) attribute.KeyValue {
	return OTelStatusDescriptionKey.String(val)
}

// This semantic convention describes an instance of a function that runs
// without provisioning or managing of servers (also known as serverless
// functions or Function as a Service (FaaS)) with spans.
const (
	// FaaSTriggerKey is the attribute Key conforming to the "faas.trigger"
	// semantic conventions. It represents the type of the trigger which caused
	// this function invocation.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Note: For the server/consumer span on the incoming side,
	// `faas.trigger` MUST be set.
	//
	// Clients invoking FaaS instances usually cannot set `faas.trigger`,
	// since they would typically need to look in the payload to determine
	// the event type. If clients set it, it should be the same as the
	// trigger that corresponding incoming would have (i.e., this has
	// nothing to do with the underlying transport used to make the API
	// call to invoke the lambda, which is often HTTP).
	FaaSTriggerKey = attribute.Key("faas.trigger")

	// FaaSInvocationIDKey is the attribute Key conforming to the
	// "faas.invocation_id" semantic conventions. It represents the invocation
	// ID of the current function invocation.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'af9d5aa4-a685-4c5f-a22b-444f80b3cc28'
	FaaSInvocationIDKey = attribute.Key("faas.invocation_id")
)

var (
	// A response to some data source operation such as a database or filesystem read/write
	FaaSTriggerDatasource = FaaSTriggerKey.String("datasource")
	// To provide an answer to an inbound HTTP request
	FaaSTriggerHTTP = FaaSTriggerKey.String("http")
	// A function is set to be executed when messages are sent to a messaging system
	FaaSTriggerPubsub = FaaSTriggerKey.String("pubsub")
	// A function is scheduled to be executed regularly
	FaaSTriggerTimer = FaaSTriggerKey.String("timer")
	// If none of the others apply
	FaaSTriggerOther = FaaSTriggerKey.String("other")
)

// FaaSInvocationID returns an attribute KeyValue conforming to the
// "faas.invocation_id" semantic conventions. It represents the invocation ID
// of the current function invocation.
func FaaSInvocationID(val string) attribute.KeyValue {
	return FaaSInvocationIDKey.String(val)
}

// Semantic Convention for FaaS triggered as a response to some data source
// operation such as a database or filesystem read/write.
const (
	// FaaSDocumentCollectionKey is the attribute Key conforming to the
	// "faas.document.collection" semantic conventions. It represents the name
	// of the source on which the triggering operation was performed. For
	// example, in Cloud Storage or S3 corresponds to the bucket name, and in
	// Cosmos DB to the database name.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'myBucketName', 'myDBName'
	FaaSDocumentCollectionKey = attribute.Key("faas.document.collection")

	// FaaSDocumentOperationKey is the attribute Key conforming to the
	// "faas.document.operation" semantic conventions. It represents the
	// describes the type of the operation that was performed on the data.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	FaaSDocumentOperationKey = attribute.Key("faas.document.operation")

	// FaaSDocumentTimeKey is the attribute Key conforming to the
	// "faas.document.time" semantic conventions. It represents a string
	// containing the time when the data was accessed in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
	// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '2020-01-23T13:47:06Z'
	FaaSDocumentTimeKey = attribute.Key("faas.document.time")

	// FaaSDocumentNameKey is the attribute Key conforming to the
	// "faas.document.name" semantic conventions. It represents the document
	// name/table subjected to the operation. For example, in Cloud Storage or
	// S3 is the name of the file, and in Cosmos DB the table name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'myFile.txt', 'myTableName'
	FaaSDocumentNameKey = attribute.Key("faas.document.name")
)

var (
	// When a new object is created
	FaaSDocumentOperationInsert = FaaSDocumentOperationKey.String("insert")
	// When an object is modified
	FaaSDocumentOperationEdit = FaaSDocumentOperationKey.String("edit")
	// When an object is deleted
	FaaSDocumentOperationDelete = FaaSDocumentOperationKey.String("delete")
)

// FaaSDocumentCollection returns an attribute KeyValue conforming to the
// "faas.document.collection" semantic conventions. It represents the name of
// the source on which the triggering operation was performed. For example, in
// Cloud Storage or S3 corresponds to the bucket name, and in Cosmos DB to the
// database name.
func FaaSDocumentCollection(val string) attribute.KeyValue {
	return FaaSDocumentCollectionKey.String(val)
}

// FaaSDocumentTime returns an attribute KeyValue conforming to the
// "faas.document.time" semantic conventions. It represents a string containing
// the time when the data was accessed in the [ISO
// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
func FaaSDocumentTime(val string) attribute.KeyValue {
	return FaaSDocumentTimeKey.String(val)
}

// FaaSDocumentName returns an attribute KeyValue conforming to the
// "faas.document.name" semantic conventions. It represents the document
// name/table subjected to the operation. For example, in Cloud Storage or S3
// is the name of the file, and in Cosmos DB the table name.
func FaaSDocumentName(val string) attribute.KeyValue {
	return FaaSDocumentNameKey.String(val)
}

// Semantic Convention for FaaS scheduled to be executed regularly.
const (
	// FaaSTimeKey is the attribute Key conforming to the "faas.time" semantic
	// conventions. It represents a string containing the function invocation
	// time in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
	// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '2020-01-23T13:47:06Z'
	FaaSTimeKey = attribute.Key("faas.time")

	// FaaSCronKey is the attribute Key conforming to the "faas.cron" semantic
	// conventions. It represents a string containing the schedule period as
	// [Cron
	// Expression](https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '0/5 * * * ? *'
	FaaSCronKey = attribute.Key("faas.cron")
)

// FaaSTime returns an attribute KeyValue conforming to the "faas.time"
// semantic conventions. It represents a string containing the function
// invocation time in the [ISO
// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
func FaaSTime(val string) attribute.KeyValue {
	return FaaSTimeKey.String(val)
}

// FaaSCron returns an attribute KeyValue conforming to the "faas.cron"
// semantic conventions. It represents a string containing the schedule period
// as [Cron
// Expression](https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm).
func FaaSCron(val string) attribute.KeyValue {
	return FaaSCronKey.String(val)
}

// Contains additional attributes for incoming FaaS spans.
const (
	// FaaSColdstartKey is the attribute Key conforming to the "faas.coldstart"
	// semantic conventions. It represents a boolean that is true if the
	// serverless function is executed for the first time (aka cold-start).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	FaaSColdstartKey = attribute.Key("faas.coldstart")
)

// FaaSColdstart returns an attribute KeyValue conforming to the
// "faas.coldstart" semantic conventions. It represents a boolean that is true
// if the serverless function is executed for the first time (aka cold-start).
func FaaSColdstart(val bool) attribute.KeyValue {
	return FaaSColdstartKey.Bool(val)
}

// Contains additional attributes for outgoing FaaS spans.
const (
	// FaaSInvokedNameKey is the attribute Key conforming to the
	// "faas.invoked_name" semantic conventions. It represents the name of the
	// invoked function.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'my-function'
	// Note: SHOULD be equal to the `faas.name` resource attribute of the
	// invoked function.
	FaaSInvokedNameKey = attribute.Key("faas.invoked_name")

	// FaaSInvokedProviderKey is the attribute Key conforming to the
	// "faas.invoked_provider" semantic conventions. It represents the cloud
	// provider of the invoked function.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	// Note: SHOULD be equal to the `cloud.provider` resource attribute of the
	// invoked function.
	FaaSInvokedProviderKey = attribute.Key("faas.invoked_provider")

	// FaaSInvokedRegionKey is the attribute Key conforming to the
	// "faas.invoked_region" semantic conventions. It represents the cloud
	// region of the invoked function.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (For some cloud providers, like
	// AWS or GCP, the region in which a function is hosted is essential to
	// uniquely identify the function and also part of its endpoint. Since it's
	// part of the endpoint being called, the region is always known to
	// clients. In these cases, `faas.invoked_region` MUST be set accordingly.
	// If the region is unknown to the client or not required for identifying
	// the invoked function, setting `faas.invoked_region` is optional.)
	// Stability: stable
	// Examples: 'eu-central-1'
	// Note: SHOULD be equal to the `cloud.region` resource attribute of the
	// invoked function.
	FaaSInvokedRegionKey = attribute.Key("faas.invoked_region")
)

var (
	// Alibaba Cloud
	FaaSInvokedProviderAlibabaCloud = FaaSInvokedProviderKey.String("alibaba_cloud")
	// Amazon Web Services
	FaaSInvokedProviderAWS = FaaSInvokedProviderKey.String("aws")
	// Microsoft Azure
	FaaSInvokedProviderAzure = FaaSInvokedProviderKey.String("azure")
	// Google Cloud Platform
	FaaSInvokedProviderGCP = FaaSInvokedProviderKey.String("gcp")
	// Tencent Cloud
	FaaSInvokedProviderTencentCloud = FaaSInvokedProviderKey.String("tencent_cloud")
)

// FaaSInvokedName returns an attribute KeyValue conforming to the
// "faas.invoked_name" semantic conventions. It represents the name of the
// invoked function.
func FaaSInvokedName(val string) attribute.KeyValue {
	return FaaSInvokedNameKey.String(val)
}

// FaaSInvokedRegion returns an attribute KeyValue conforming to the
// "faas.invoked_region" semantic conventions. It represents the cloud region
// of the invoked function.
func FaaSInvokedRegion(val string) attribute.KeyValue {
	return FaaSInvokedRegionKey.String(val)
}

// Operations that access some remote service.
const (
	// PeerServiceKey is the attribute Key conforming to the "peer.service"
	// semantic conventions. It represents the
	// [`service.name`](/docs/resource/README.md#service) of the remote
	// service. SHOULD be equal to the actual `service.name` resource attribute
	// of the remote service if any.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'AuthTokenCache'
	PeerServiceKey = attribute.Key("peer.service")
)

// PeerService returns an attribute KeyValue conforming to the
// "peer.service" semantic conventions. It represents the
// [`service.name`](/docs/resource/README.md#service) of the remote service.
// SHOULD be equal to the actual `service.name` resource attribute of the
// remote service if any.
func PeerService(val string) attribute.KeyValue {
	return PeerServiceKey.String(val)
}

// These attributes may be used for any operation with an authenticated and/or
// authorized enduser.
const (
	// EnduserIDKey is the attribute Key conforming to the "enduser.id"
	// semantic conventions. It represents the username or client_id extracted
	// from the access token or
	// [Authorization](https://tools.ietf.org/html/rfc7235#section-4.2) header
	// in the inbound request from outside the system.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'username'
	EnduserIDKey = attribute.Key("enduser.id")

	// EnduserRoleKey is the attribute Key conforming to the "enduser.role"
	// semantic conventions. It represents the actual/assumed role the client
	// is making the request under extracted from token or application security
	// context.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'admin'
	EnduserRoleKey = attribute.Key("enduser.role")

	// EnduserScopeKey is the attribute Key conforming to the "enduser.scope"
	// semantic conventions. It represents the scopes or granted authorities
	// the client currently possesses extracted from token or application
	// security context. The value would come from the scope associated with an
	// [OAuth 2.0 Access
	// Token](https://tools.ietf.org/html/rfc6749#section-3.3) or an attribute
	// value in a [SAML 2.0
	// Assertion](http://docs.oasis-open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'read:message, write:files'
	EnduserScopeKey = attribute.Key("enduser.scope")
)

// EnduserID returns an attribute KeyValue conforming to the "enduser.id"
// semantic conventions. It represents the username or client_id extracted from
// the access token or
// [Authorization](https://tools.ietf.org/html/rfc7235#section-4.2) header in
// the inbound request from outside the system.
func EnduserID(val string) attribute.KeyValue {
	return EnduserIDKey.String(val)
}

// EnduserRole returns an attribute KeyValue conforming to the
// "enduser.role" semantic conventions. It represents the actual/assumed role
// the client is making the request under extracted from token or application
// security context.
func EnduserRole(val string) attribute.KeyValue {
	return EnduserRoleKey.String(val)
}

// EnduserScope returns an attribute KeyValue conforming to the
// "enduser.scope" semantic conventions. It represents the scopes or granted
// authorities the client currently possesses extracted from token or
// application security context. The value would come from the scope associated
// with an [OAuth 2.0 Access
// Token](https://tools.ietf.org/html/rfc6749#section-3.3) or an attribute
// value in a [SAML 2.0
// Assertion](http://docs.oasis-open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html).
func EnduserScope(val string) attribute.KeyValue {
	return EnduserScopeKey.String(val)
}

// These attributes may be used for any operation to store information about a
// thread that started a span.
const (
	// ThreadIDKey is the attribute Key conforming to the "thread.id" semantic
	// conventions. It represents the current "managed" thread ID (as opposed
	// to OS thread ID).
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 42
	ThreadIDKey = attribute.Key("thread.id")

	// ThreadNameKey is the attribute Key conforming to the "thread.name"
	// semantic conventions. It represents the current thread name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'main'
	ThreadNameKey = attribute.Key("thread.name")
)

// ThreadID returns an attribute KeyValue conforming to the "thread.id"
// semantic conventions. It represents the current "managed" thread ID (as
// opposed to OS thread ID).
func ThreadID(val int) attribute.KeyValue {
	return ThreadIDKey.Int(val)
}

// ThreadName returns an attribute KeyValue conforming to the "thread.name"
// semantic conventions. It represents the current thread name.
func ThreadName(val string) attribute.KeyValue {
	return ThreadNameKey.String(val)
}

// These attributes allow to report this unit of code and therefore to provide
// more context about the span.
const (
	// CodeFunctionKey is the attribute Key conforming to the "code.function"
	// semantic conventions. It represents the method or function name, or
	// equivalent (usually rightmost part of the code unit's name).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'serveRequest'
	CodeFunctionKey = attribute.Key("code.function")

	// CodeNamespaceKey is the attribute Key conforming to the "code.namespace"
	// semantic conventions. It represents the "namespace" within which
	// `code.function` is defined. Usually the qualified class or module name,
	// such that `code.namespace` + some separator + `code.function` form a
	// unique identifier for the code unit.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'com.example.MyHTTPService'
	CodeNamespaceKey = attribute.Key("code.namespace")

	// CodeFilepathKey is the attribute Key conforming to the "code.filepath"
	// semantic conventions. It represents the source code file name that
	// identifies the code unit as uniquely as possible (preferably an absolute
	// file path).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/usr/local/MyApplication/content_root/app/index.php'
	CodeFilepathKey = attribute.Key("code.filepath")

	// CodeLineNumberKey is the attribute Key conforming to the "code.lineno"
	// semantic conventions. It represents the line number in `code.filepath`
	// best representing the operation. It SHOULD point within the code unit
	// named in `code.function`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 42
	CodeLineNumberKey = attribute.Key("code.lineno")

	// CodeColumnKey is the attribute Key conforming to the "code.column"
	// semantic conventions. It represents the column number in `code.filepath`
	// best representing the operation. It SHOULD point within the code unit
	// named in `code.function`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 16
	CodeColumnKey = attribute.Key("code.column")
)

// CodeFunction returns an attribute KeyValue conforming to the
// "code.function" semantic conventions. It represents the method or function
// name, or equivalent (usually rightmost part of the code unit's name).
func CodeFunction(val string) attribute.KeyValue {
	return CodeFunctionKey.String(val)
}

// CodeNamespace returns an attribute KeyValue conforming to the
// "code.namespace" semantic conventions. It represents the "namespace" within
// which `code.function` is defined. Usually the qualified class or module
// name, such that `code.namespace` + some separator + `code.function` form a
// unique identifier for the code unit.
func CodeNamespace(val string) attribute.KeyValue {
	return CodeNamespaceKey.String(val)
}

// CodeFilepath returns an attribute KeyValue conforming to the
// "code.filepath" semantic conventions. It represents the source code file
// name that identifies the code unit as uniquely as possible (preferably an
// absolute file path).
func CodeFilepath(val string) attribute.KeyValue {
	return CodeFilepathKey.String(val)
}

// CodeLineNumber returns an attribute KeyValue conforming to the "code.lineno"
// semantic conventions. It represents the line number in `code.filepath` best
// representing the operation. It SHOULD point within the code unit named in
// `code.function`.
func CodeLineNumber(val int) attribute.KeyValue {
	return CodeLineNumberKey.Int(val)
}

// CodeColumn returns an attribute KeyValue conforming to the "code.column"
// semantic conventions. It represents the column number in `code.filepath`
// best representing the operation. It SHOULD point within the code unit named
// in `code.function`.
func CodeColumn(val int) attribute.KeyValue {
	return CodeColumnKey.Int(val)
}

// Semantic Convention for HTTP Client
const (
	// HTTPResendCountKey is the attribute Key conforming to the
	// "http.resend_count" semantic conventions. It represents the ordinal
	// number of request resending attempt (for any reason, including
	// redirects).
	//
	// Type: int
	// RequirementLevel: Recommended (if and only if request was retried.)
	// Stability: stable
	// Examples: 3
	// Note: The resend count SHOULD be updated each time an HTTP request gets
	// resent by the client, regardless of what was the cause of the resending
	// (e.g. redirection, authorization failure, 503 Server Unavailable,
	// network issues, or any other).
	HTTPResendCountKey = attribute.Key("http.resend_count")
)

// HTTPResendCount returns an attribute KeyValue conforming to the
// "http.resend_count" semantic conventions. It represents the ordinal number
// of request resending attempt (for any reason, including redirects).
func HTTPResendCount(val int) attribute.KeyValue {
	return HTTPResendCountKey.Int(val)
}

// The `aws` conventions apply to operations using the AWS SDK. They map
// request or response parameters in AWS SDK API calls to attributes on a Span.
// The conventions have been collected over time based on feedback from AWS
// users of tracing and will continue to evolve as new interesting conventions
// are found.
// Some descriptions are also provided for populating general OpenTelemetry
// semantic conventions based on these APIs.
const (
	// AWSRequestIDKey is the attribute Key conforming to the "aws.request_id"
	// semantic conventions. It represents the AWS request ID as returned in
	// the response headers `x-amz-request-id` or `x-amz-requestid`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '79b9da39-b7ae-508a-a6bc-864b2829c622', 'C9ER4AJX75574TDJ'
	AWSRequestIDKey = attribute.Key("aws.request_id")
)

// AWSRequestID returns an attribute KeyValue conforming to the
// "aws.request_id" semantic conventions. It represents the AWS request ID as
// returned in the response headers `x-amz-request-id` or `x-amz-requestid`.
func AWSRequestID(val string) attribute.KeyValue {
	return AWSRequestIDKey.String(val)
}

// Attributes that exist for multiple DynamoDB request types.
const (
	// AWSDynamoDBTableNamesKey is the attribute Key conforming to the
	// "aws.dynamodb.table_names" semantic conventions. It represents the keys
	// in the `RequestItems` object field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Users', 'Cats'
	AWSDynamoDBTableNamesKey = attribute.Key("aws.dynamodb.table_names")

	// AWSDynamoDBConsumedCapacityKey is the attribute Key conforming to the
	// "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
	// JSON-serialized value of each item in the `ConsumedCapacity` response
	// field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "CapacityUnits": number, "GlobalSecondaryIndexes": {
	// "string" : { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number } }, "LocalSecondaryIndexes": { "string" :
	// { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number } }, "ReadCapacityUnits": number, "Table":
	// { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number }, "TableName": "string",
	// "WriteCapacityUnits": number }'
	AWSDynamoDBConsumedCapacityKey = attribute.Key("aws.dynamodb.consumed_capacity")

	// AWSDynamoDBItemCollectionMetricsKey is the attribute Key conforming to
	// the "aws.dynamodb.item_collection_metrics" semantic conventions. It
	// represents the JSON-serialized value of the `ItemCollectionMetrics`
	// response field.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "string" : [ { "ItemCollectionKey": { "string" : { "B":
	// blob, "BOOL": boolean, "BS": [ blob ], "L": [ "AttributeValue" ], "M": {
	// "string" : "AttributeValue" }, "N": "string", "NS": [ "string" ],
	// "NULL": boolean, "S": "string", "SS": [ "string" ] } },
	// "SizeEstimateRangeGB": [ number ] } ] }'
	AWSDynamoDBItemCollectionMetricsKey = attribute.Key("aws.dynamodb.item_collection_metrics")

	// AWSDynamoDBProvisionedReadCapacityKey is the attribute Key conforming to
	// the "aws.dynamodb.provisioned_read_capacity" semantic conventions. It
	// represents the value of the `ProvisionedThroughput.ReadCapacityUnits`
	// request parameter.
	//
	// Type: double
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedReadCapacityKey = attribute.Key("aws.dynamodb.provisioned_read_capacity")

	// AWSDynamoDBProvisionedWriteCapacityKey is the attribute Key conforming
	// to the "aws.dynamodb.provisioned_write_capacity" semantic conventions.
	// It represents the value of the
	// `ProvisionedThroughput.WriteCapacityUnits` request parameter.
	//
	// Type: double
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedWriteCapacityKey = attribute.Key("aws.dynamodb.provisioned_write_capacity")

	// AWSDynamoDBConsistentReadKey is the attribute Key conforming to the
	// "aws.dynamodb.consistent_read" semantic conventions. It represents the
	// value of the `ConsistentRead` request parameter.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	AWSDynamoDBConsistentReadKey = attribute.Key("aws.dynamodb.consistent_read")

	// AWSDynamoDBProjectionKey is the attribute Key conforming to the
	// "aws.dynamodb.projection" semantic conventions. It represents the value
	// of the `ProjectionExpression` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Title', 'Title, Price, Color', 'Title, Description,
	// RelatedItems, ProductReviews'
	AWSDynamoDBProjectionKey = attribute.Key("aws.dynamodb.projection")

	// AWSDynamoDBLimitKey is the attribute Key conforming to the
	// "aws.dynamodb.limit" semantic conventions. It represents the value of
	// the `Limit` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 10
	AWSDynamoDBLimitKey = attribute.Key("aws.dynamodb.limit")

	// AWSDynamoDBAttributesToGetKey is the attribute Key conforming to the
	// "aws.dynamodb.attributes_to_get" semantic conventions. It represents the
	// value of the `AttributesToGet` request parameter.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'lives', 'id'
	AWSDynamoDBAttributesToGetKey = attribute.Key("aws.dynamodb.attributes_to_get")

	// AWSDynamoDBIndexNameKey is the attribute Key conforming to the
	// "aws.dynamodb.index_name" semantic conventions. It represents the value
	// of the `IndexName` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'name_to_group'
	AWSDynamoDBIndexNameKey = attribute.Key("aws.dynamodb.index_name")

	// AWSDynamoDBSelectKey is the attribute Key conforming to the
	// "aws.dynamodb.select" semantic conventions. It represents the value of
	// the `Select` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'ALL_ATTRIBUTES', 'COUNT'
	AWSDynamoDBSelectKey = attribute.Key("aws.dynamodb.select")
)

// AWSDynamoDBTableNames returns an attribute KeyValue conforming to the
// "aws.dynamodb.table_names" semantic conventions. It represents the keys in
// the `RequestItems` object field.
func AWSDynamoDBTableNames(val ...string) attribute.KeyValue {
	return AWSDynamoDBTableNamesKey.StringSlice(val)
}

// AWSDynamoDBConsumedCapacity returns an attribute KeyValue conforming to
// the "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
// JSON-serialized value of each item in the `ConsumedCapacity` response field.
func AWSDynamoDBConsumedCapacity(val ...string) attribute.KeyValue {
	return AWSDynamoDBConsumedCapacityKey.StringSlice(val)
}

// AWSDynamoDBItemCollectionMetrics returns an attribute KeyValue conforming
// to the "aws.dynamodb.item_collection_metrics" semantic conventions. It
// represents the JSON-serialized value of the `ItemCollectionMetrics` response
// field.
func AWSDynamoDBItemCollectionMetrics(val string) attribute.KeyValue {
	return AWSDynamoDBItemCollectionMetricsKey.String(val)
}

// AWSDynamoDBProvisionedReadCapacity returns an attribute KeyValue
// conforming to the "aws.dynamodb.provisioned_read_capacity" semantic
// conventions. It represents the value of the
// `ProvisionedThroughput.ReadCapacityUnits` request parameter.
func AWSDynamoDBProvisionedReadCapacity(val float64) attribute.KeyValue {
	return AWSDynamoDBProvisionedReadCapacityKey.Float64(val)
}

// AWSDynamoDBProvisionedWriteCapacity returns an attribute KeyValue
// conforming to the "aws.dynamodb.provisioned_write_capacity" semantic
// conventions. It represents the value of the
// `ProvisionedThroughput.WriteCapacityUnits` request parameter.
func AWSDynamoDBProvisionedWriteCapacity(val float64) attribute.KeyValue {
	return AWSDynamoDBProvisionedWriteCapacityKey.Float64(val)
}

// AWSDynamoDBConsistentRead returns an attribute KeyValue conforming to the
// "aws.dynamodb.consistent_read" semantic conventions. It represents the value
// of the `ConsistentRead` request parameter.
func AWSDynamoDBConsistentRead(val bool) attribute.KeyValue {
	return AWSDynamoDBConsistentReadKey.Bool(val)
}

// AWSDynamoDBProjection returns an attribute KeyValue conforming to the
// "aws.dynamodb.projection" semantic conventions. It represents the value of
// the `ProjectionExpression` request parameter.
func AWSDynamoDBProjection(val string) attribute.KeyValue {
	return AWSDynamoDBProjectionKey.String(val)
}

// AWSDynamoDBLimit returns an attribute KeyValue conforming to the
// "aws.dynamodb.limit" semantic conventions. It represents the value of the
// `Limit` request parameter.
func AWSDynamoDBLimit(val int) attribute.KeyValue {
	return AWSDynamoDBLimitKey.Int(val)
}

// AWSDynamoDBAttributesToGet returns an attribute KeyValue conforming to
// the "aws.dynamodb.attributes_to_get" semantic conventions. It represents the
// value of the `AttributesToGet` request parameter.
func AWSDynamoDBAttributesToGet(val ...string) attribute.KeyValue {
	return AWSDynamoDBAttributesToGetKey.StringSlice(val)
}

// AWSDynamoDBIndexName returns an attribute KeyValue conforming to the
// "aws.dynamodb.index_name" semantic conventions. It represents the value of
// the `IndexName` request parameter.
func AWSDynamoDBIndexName(val string) attribute.KeyValue {
	return AWSDynamoDBIndexNameKey.String(val)
}

// AWSDynamoDBSelect returns an attribute KeyValue conforming to the
// "aws.dynamodb.select" semantic conventions. It represents the value of the
// `Select` request parameter.
func AWSDynamoDBSelect(val string) attribute.KeyValue {
	return AWSDynamoDBSelectKey.String(val)
}

// DynamoDB.CreateTable
const (
	// AWSDynamoDBGlobalSecondaryIndexesKey is the attribute Key conforming to
	// the "aws.dynamodb.global_secondary_indexes" semantic conventions. It
	// represents the JSON-serialized value of each item of the
	// `GlobalSecondaryIndexes` request field
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "IndexName": "string", "KeySchema": [ { "AttributeName":
	// "string", "KeyType": "string" } ], "Projection": { "NonKeyAttributes": [
	// "string" ], "ProjectionType": "string" }, "ProvisionedThroughput": {
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number } }'
	AWSDynamoDBGlobalSecondaryIndexesKey = attribute.Key("aws.dynamodb.global_secondary_indexes")

	// AWSDynamoDBLocalSecondaryIndexesKey is the attribute Key conforming to
	// the "aws.dynamodb.local_secondary_indexes" semantic conventions. It
	// represents the JSON-serialized value of each item of the
	// `LocalSecondaryIndexes` request field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "IndexARN": "string", "IndexName": "string",
	// "IndexSizeBytes": number, "ItemCount": number, "KeySchema": [ {
	// "AttributeName": "string", "KeyType": "string" } ], "Projection": {
	// "NonKeyAttributes": [ "string" ], "ProjectionType": "string" } }'
	AWSDynamoDBLocalSecondaryIndexesKey = attribute.Key("aws.dynamodb.local_secondary_indexes")
)

// AWSDynamoDBGlobalSecondaryIndexes returns an attribute KeyValue
// conforming to the "aws.dynamodb.global_secondary_indexes" semantic
// conventions. It represents the JSON-serialized value of each item of the
// `GlobalSecondaryIndexes` request field
func AWSDynamoDBGlobalSecondaryIndexes(val ...string) attribute.KeyValue {
	return AWSDynamoDBGlobalSecondaryIndexesKey.StringSlice(val)
}

// AWSDynamoDBLocalSecondaryIndexes returns an attribute KeyValue conforming
// to the "aws.dynamodb.local_secondary_indexes" semantic conventions. It
// represents the JSON-serialized value of each item of the
// `LocalSecondaryIndexes` request field.
func AWSDynamoDBLocalSecondaryIndexes(val ...string) attribute.KeyValue {
	return AWSDynamoDBLocalSecondaryIndexesKey.StringSlice(val)
}

// DynamoDB.ListTables
const (
	// AWSDynamoDBExclusiveStartTableKey is the attribute Key conforming to the
	// "aws.dynamodb.exclusive_start_table" semantic conventions. It represents
	// the value of the `ExclusiveStartTableName` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Users', 'CatsTable'
	AWSDynamoDBExclusiveStartTableKey = attribute.Key("aws.dynamodb.exclusive_start_table")

	// AWSDynamoDBTableCountKey is the attribute Key conforming to the
	// "aws.dynamodb.table_count" semantic conventions. It represents the the
	// number of items in the `TableNames` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 20
	AWSDynamoDBTableCountKey = attribute.Key("aws.dynamodb.table_count")
)

// AWSDynamoDBExclusiveStartTable returns an attribute KeyValue conforming
// to the "aws.dynamodb.exclusive_start_table" semantic conventions. It
// represents the value of the `ExclusiveStartTableName` request parameter.
func AWSDynamoDBExclusiveStartTable(val string) attribute.KeyValue {
	return AWSDynamoDBExclusiveStartTableKey.String(val)
}

// AWSDynamoDBTableCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.table_count" semantic conventions. It represents the the
// number of items in the `TableNames` response parameter.
func AWSDynamoDBTableCount(val int) attribute.KeyValue {
	return AWSDynamoDBTableCountKey.Int(val)
}

// DynamoDB.Query
const (
	// AWSDynamoDBScanForwardKey is the attribute Key conforming to the
	// "aws.dynamodb.scan_forward" semantic conventions. It represents the
	// value of the `ScanIndexForward` request parameter.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	AWSDynamoDBScanForwardKey = attribute.Key("aws.dynamodb.scan_forward")
)

// AWSDynamoDBScanForward returns an attribute KeyValue conforming to the
// "aws.dynamodb.scan_forward" semantic conventions. It represents the value of
// the `ScanIndexForward` request parameter.
func AWSDynamoDBScanForward(val bool) attribute.KeyValue {
	return AWSDynamoDBScanForwardKey.Bool(val)
}

// DynamoDB.Scan
const (
	// AWSDynamoDBSegmentKey is the attribute Key conforming to the
	// "aws.dynamodb.segment" semantic conventions. It represents the value of
	// the `Segment` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 10
	AWSDynamoDBSegmentKey = attribute.Key("aws.dynamodb.segment")

	// AWSDynamoDBTotalSegmentsKey is the attribute Key conforming to the
	// "aws.dynamodb.total_segments" semantic conventions. It represents the
	// value of the `TotalSegments` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 100
	AWSDynamoDBTotalSegmentsKey = attribute.Key("aws.dynamodb.total_segments")

	// AWSDynamoDBCountKey is the attribute Key conforming to the
	// "aws.dynamodb.count" semantic conventions. It represents the value of
	// the `Count` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 10
	AWSDynamoDBCountKey = attribute.Key("aws.dynamodb.count")

	// AWSDynamoDBScannedCountKey is the attribute Key conforming to the
	// "aws.dynamodb.scanned_count" semantic conventions. It represents the
	// value of the `ScannedCount` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 50
	AWSDynamoDBScannedCountKey = attribute.Key("aws.dynamodb.scanned_count")
)

// AWSDynamoDBSegment returns an attribute KeyValue conforming to the
// "aws.dynamodb.segment" semantic conventions. It represents the value of the
// `Segment` request parameter.
func AWSDynamoDBSegment(val int) attribute.KeyValue {
	return AWSDynamoDBSegmentKey.Int(val)
}

// AWSDynamoDBTotalSegments returns an attribute KeyValue conforming to the
// "aws.dynamodb.total_segments" semantic conventions. It represents the value
// of the `TotalSegments` request parameter.
func AWSDynamoDBTotalSegments(val int) attribute.KeyValue {
	return AWSDynamoDBTotalSegmentsKey.Int(val)
}

// AWSDynamoDBCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.count" semantic conventions. It represents the value of the
// `Count` response parameter.
func AWSDynamoDBCount(val int) attribute.KeyValue {
	return AWSDynamoDBCountKey.Int(val)
}

// AWSDynamoDBScannedCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.scanned_count" semantic conventions. It represents the value
// of the `ScannedCount` response parameter.
func AWSDynamoDBScannedCount(val int) attribute.KeyValue {
	return AWSDynamoDBScannedCountKey.Int(val)
}

// DynamoDB.UpdateTable
const (
	// AWSDynamoDBAttributeDefinitionsKey is the attribute Key conforming to
	// the "aws.dynamodb.attribute_definitions" semantic conventions. It
	// represents the JSON-serialized value of each item in the
	// `AttributeDefinitions` request field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "AttributeName": "string", "AttributeType": "string" }'
	AWSDynamoDBAttributeDefinitionsKey = attribute.Key("aws.dynamodb.attribute_definitions")

	// AWSDynamoDBGlobalSecondaryIndexUpdatesKey is the attribute Key
	// conforming to the "aws.dynamodb.global_secondary_index_updates" semantic
	// conventions. It represents the JSON-serialized value of each item in the
	// the `GlobalSecondaryIndexUpdates` request field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '{ "Create": { "IndexName": "string", "KeySchema": [ {
	// "AttributeName": "string", "KeyType": "string" } ], "Projection": {
	// "NonKeyAttributes": [ "string" ], "ProjectionType": "string" },
	// "ProvisionedThroughput": { "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number } }'
	AWSDynamoDBGlobalSecondaryIndexUpdatesKey = attribute.Key("aws.dynamodb.global_secondary_index_updates")
)

// AWSDynamoDBAttributeDefinitions returns an attribute KeyValue conforming
// to the "aws.dynamodb.attribute_definitions" semantic conventions. It
// represents the JSON-serialized value of each item in the
// `AttributeDefinitions` request field.
func AWSDynamoDBAttributeDefinitions(val ...string) attribute.KeyValue {
	return AWSDynamoDBAttributeDefinitionsKey.StringSlice(val)
}

// AWSDynamoDBGlobalSecondaryIndexUpdates returns an attribute KeyValue
// conforming to the "aws.dynamodb.global_secondary_index_updates" semantic
// conventions. It represents the JSON-serialized value of each item in the the
// `GlobalSecondaryIndexUpdates` request field.
func AWSDynamoDBGlobalSecondaryIndexUpdates(val ...string) attribute.KeyValue {
	return AWSDynamoDBGlobalSecondaryIndexUpdatesKey.StringSlice(val)
}

// Attributes that exist for S3 request types.
const (
	// AWSS3BucketKey is the attribute Key conforming to the "aws.s3.bucket"
	// semantic conventions. It represents the S3 bucket name the request
	// refers to. Corresponds to the `--bucket` parameter of the [S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
	// operations.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'some-bucket-name'
	// Note: The `bucket` attribute is applicable to all S3 operations that
	// reference a bucket, i.e. that require the bucket name as a mandatory
	// parameter.
	// This applies to almost all S3 operations except `list-buckets`.
	AWSS3BucketKey = attribute.Key("aws.s3.bucket")

	// AWSS3KeyKey is the attribute Key conforming to the "aws.s3.key" semantic
	// conventions. It represents the S3 object key the request refers to.
	// Corresponds to the `--key` parameter of the [S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
	// operations.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'someFile.yml'
	// Note: The `key` attribute is applicable to all object-related S3
	// operations, i.e. that require the object key as a mandatory parameter.
	// This applies in particular to the following operations:
	//
	// -
	// [copy-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html)
	// -
	// [delete-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html)
	// -
	// [get-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/get-object.html)
	// -
	// [head-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/head-object.html)
	// -
	// [put-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html)
	// -
	// [restore-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/restore-object.html)
	// -
	// [select-object-content](https://docs.aws.amazon.com/cli/latest/reference/s3api/select-object-content.html)
	// -
	// [abort-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html)
	// -
	// [complete-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html)
	// -
	// [create-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/create-multipart-upload.html)
	// -
	// [list-parts](https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html)
	// -
	// [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
	// -
	// [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html)
	AWSS3KeyKey = attribute.Key("aws.s3.key")

	// AWSS3CopySourceKey is the attribute Key conforming to the
	// "aws.s3.copy_source" semantic conventions. It represents the source
	// object (in the form `bucket`/`key`) for the copy operation.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'someFile.yml'
	// Note: The `copy_source` attribute applies to S3 copy operations and
	// corresponds to the `--copy-source` parameter
	// of the [copy-object operation within the S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html).
	// This applies in particular to the following operations:
	//
	// -
	// [copy-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html)
	// -
	// [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html)
	AWSS3CopySourceKey = attribute.Key("aws.s3.copy_source")

	// AWSS3UploadIDKey is the attribute Key conforming to the
	// "aws.s3.upload_id" semantic conventions. It represents the upload ID
	// that identifies the multipart upload.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'dfRtDYWFbkRONycy.Yxwh66Yjlx.cph0gtNBtJ'
	// Note: The `upload_id` attribute applies to S3 multipart-upload
	// operations and corresponds to the `--upload-id` parameter
	// of the [S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
	// multipart operations.
	// This applies in particular to the following operations:
	//
	// -
	// [abort-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html)
	// -
	// [complete-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html)
	// -
	// [list-parts](https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html)
	// -
	// [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
	// -
	// [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html)
	AWSS3UploadIDKey = attribute.Key("aws.s3.upload_id")

	// AWSS3DeleteKey is the attribute Key conforming to the "aws.s3.delete"
	// semantic conventions. It represents the delete request container that
	// specifies the objects to be deleted.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples:
	// 'Objects=[{Key=string,VersionID=string},{Key=string,VersionID=string}],Quiet=boolean'
	// Note: The `delete` attribute is only applicable to the
	// [delete-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html)
	// operation.
	// The `delete` attribute corresponds to the `--delete` parameter of the
	// [delete-objects operation within the S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html).
	AWSS3DeleteKey = attribute.Key("aws.s3.delete")

	// AWSS3PartNumberKey is the attribute Key conforming to the
	// "aws.s3.part_number" semantic conventions. It represents the part number
	// of the part being uploaded in a multipart-upload operation. This is a
	// positive integer between 1 and 10,000.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3456
	// Note: The `part_number` attribute is only applicable to the
	// [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
	// and
	// [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html)
	// operations.
	// The `part_number` attribute corresponds to the `--part-number` parameter
	// of the
	// [upload-part operation within the S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html).
	AWSS3PartNumberKey = attribute.Key("aws.s3.part_number")
)

// AWSS3Bucket returns an attribute KeyValue conforming to the
// "aws.s3.bucket" semantic conventions. It represents the S3 bucket name the
// request refers to. Corresponds to the `--bucket` parameter of the [S3
// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
// operations.
func AWSS3Bucket(val string) attribute.KeyValue {
	return AWSS3BucketKey.String(val)
}

// AWSS3Key returns an attribute KeyValue conforming to the "aws.s3.key"
// semantic conventions. It represents the S3 object key the request refers to.
// Corresponds to the `--key` parameter of the [S3
// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
// operations.
func AWSS3Key(val string) attribute.KeyValue {
	return AWSS3KeyKey.String(val)
}

// AWSS3CopySource returns an attribute KeyValue conforming to the
// "aws.s3.copy_source" semantic conventions. It represents the source object
// (in the form `bucket`/`key`) for the copy operation.
func AWSS3CopySource(val string) attribute.KeyValue {
	return AWSS3CopySourceKey.String(val)
}

// AWSS3UploadID returns an attribute KeyValue conforming to the
// "aws.s3.upload_id" semantic conventions. It represents the upload ID that
// identifies the multipart upload.
func AWSS3UploadID(val string) attribute.KeyValue {
	return AWSS3UploadIDKey.String(val)
}

// AWSS3Delete returns an attribute KeyValue conforming to the
// "aws.s3.delete" semantic conventions. It represents the delete request
// container that specifies the objects to be deleted.
func AWSS3Delete(val string) attribute.KeyValue {
	return AWSS3DeleteKey.String(val)
}

// AWSS3PartNumber returns an attribute KeyValue conforming to the
// "aws.s3.part_number" semantic conventions. It represents the part number of
// the part being uploaded in a multipart-upload operation. This is a positive
// integer between 1 and 10,000.
func AWSS3PartNumber(val int) attribute.KeyValue {
	return AWSS3PartNumberKey.Int(val)
}

// Semantic conventions to apply when instrumenting the GraphQL implementation.
// They map GraphQL operations to attributes on a Span.
const (
	// GraphqlOperationNameKey is the attribute Key conforming to the
	// "graphql.operation.name" semantic conventions. It represents the name of
	// the operation being executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'findBookByID'
	GraphqlOperationNameKey = attribute.Key("graphql.operation.name")

	// GraphqlOperationTypeKey is the attribute Key conforming to the
	// "graphql.operation.type" semantic conventions. It represents the type of
	// the operation being executed.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'query', 'mutation', 'subscription'
	GraphqlOperationTypeKey = attribute.Key("graphql.operation.type")

	// GraphqlDocumentKey is the attribute Key conforming to the
	// "graphql.document" semantic conventions. It represents the GraphQL
	// document being executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'query findBookByID { bookByID(id: ?) { name } }'
	// Note: The value may be sanitized to exclude sensitive information.
	GraphqlDocumentKey = attribute.Key("graphql.document")
)

var (
	// GraphQL query
	GraphqlOperationTypeQuery = GraphqlOperationTypeKey.String("query")
	// GraphQL mutation
	GraphqlOperationTypeMutation = GraphqlOperationTypeKey.String("mutation")
	// GraphQL subscription
	GraphqlOperationTypeSubscription = GraphqlOperationTypeKey.String("subscription")
)

// GraphqlOperationName returns an attribute KeyValue conforming to the
// "graphql.operation.name" semantic conventions. It represents the name of the
// operation being executed.
func GraphqlOperationName(val string) attribute.KeyValue {
	return GraphqlOperationNameKey.String(val)
}

// GraphqlDocument returns an attribute KeyValue conforming to the
// "graphql.document" semantic conventions. It represents the GraphQL document
// being executed.
func GraphqlDocument(val string) attribute.KeyValue {
	return GraphqlDocumentKey.String(val)
}

// General attributes used in messaging systems.
const (
	// MessagingSystemKey is the attribute Key conforming to the
	// "messaging.system" semantic conventions. It represents a string
	// identifying the messaging system.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'kafka', 'rabbitmq', 'rocketmq', 'activemq', 'AmazonSQS'
	MessagingSystemKey = attribute.Key("messaging.system")

	// MessagingOperationKey is the attribute Key conforming to the
	// "messaging.operation" semantic conventions. It represents a string
	// identifying the kind of messaging operation as defined in the [Operation
	// names](#operation-names) section above.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	// Note: If a custom value is used, it MUST be of low cardinality.
	MessagingOperationKey = attribute.Key("messaging.operation")

	// MessagingBatchMessageCountKey is the attribute Key conforming to the
	// "messaging.batch.message_count" semantic conventions. It represents the
	// number of messages sent, received, or processed in the scope of the
	// batching operation.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If the span describes an
	// operation on a batch of messages.)
	// Stability: stable
	// Examples: 0, 1, 2
	// Note: Instrumentations SHOULD NOT set `messaging.batch.message_count` on
	// spans that operate with a single message. When a messaging client
	// library supports both batch and single-message API for the same
	// operation, instrumentations SHOULD use `messaging.batch.message_count`
	// for batching APIs and SHOULD NOT use it for single-message APIs.
	MessagingBatchMessageCountKey = attribute.Key("messaging.batch.message_count")

	// MessagingClientIDKey is the attribute Key conforming to the
	// "messaging.client_id" semantic conventions. It represents a unique
	// identifier for the client that consumes or produces a message.
	//
	// Type: string
	// RequirementLevel: Recommended (If a client id is available)
	// Stability: stable
	// Examples: 'client-5', 'myhost@8742@s8083jm'
	MessagingClientIDKey = attribute.Key("messaging.client_id")
)

var (
	// publish
	MessagingOperationPublish = MessagingOperationKey.String("publish")
	// receive
	MessagingOperationReceive = MessagingOperationKey.String("receive")
	// process
	MessagingOperationProcess = MessagingOperationKey.String("process")
)

// MessagingSystem returns an attribute KeyValue conforming to the
// "messaging.system" semantic conventions. It represents a string identifying
// the messaging system.
func MessagingSystem(val string) attribute.KeyValue {
	return MessagingSystemKey.String(val)
}

// MessagingBatchMessageCount returns an attribute KeyValue conforming to
// the "messaging.batch.message_count" semantic conventions. It represents the
// number of messages sent, received, or processed in the scope of the batching
// operation.
func MessagingBatchMessageCount(val int) attribute.KeyValue {
	return MessagingBatchMessageCountKey.Int(val)
}

// MessagingClientID returns an attribute KeyValue conforming to the
// "messaging.client_id" semantic conventions. It represents a unique
// identifier for the client that consumes or produces a message.
func MessagingClientID(val string) attribute.KeyValue {
	return MessagingClientIDKey.String(val)
}

// Semantic conventions for remote procedure calls.
const (
	// RPCSystemKey is the attribute Key conforming to the "rpc.system"
	// semantic conventions. It represents a string identifying the remoting
	// system. See below for a list of well-known identifiers.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	RPCSystemKey = attribute.Key("rpc.system")

	// RPCServiceKey is the attribute Key conforming to the "rpc.service"
	// semantic conventions. It represents the full (logical) name of the
	// service being called, including its package name, if applicable.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'myservice.EchoService'
	// Note: This is the logical name of the service from the RPC interface
	// perspective, which can be different from the name of any implementing
	// class. The `code.namespace` attribute may be used to store the latter
	// (despite the attribute name, it may include a class name; e.g., class
	// with method actually executing the call on the server side, RPC client
	// stub class on the client side).
	RPCServiceKey = attribute.Key("rpc.service")

	// RPCMethodKey is the attribute Key conforming to the "rpc.method"
	// semantic conventions. It represents the name of the (logical) method
	// being called, must be equal to the $method part in the span name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'exampleMethod'
	// Note: This is the logical name of the method from the RPC interface
	// perspective, which can be different from the name of any implementing
	// method/function. The `code.function` attribute may be used to store the
	// latter (e.g., method actually executing the call on the server side, RPC
	// client stub method on the client side).
	RPCMethodKey = attribute.Key("rpc.method")
)

var (
	// gRPC
	RPCSystemGRPC = RPCSystemKey.String("grpc")
	// Java RMI
	RPCSystemJavaRmi = RPCSystemKey.String("java_rmi")
	// .NET WCF
	RPCSystemDotnetWcf = RPCSystemKey.String("dotnet_wcf")
	// Apache Dubbo
	RPCSystemApacheDubbo = RPCSystemKey.String("apache_dubbo")
	// Connect RPC
	RPCSystemConnectRPC = RPCSystemKey.String("connect_rpc")
)

// RPCService returns an attribute KeyValue conforming to the "rpc.service"
// semantic conventions. It represents the full (logical) name of the service
// being called, including its package name, if applicable.
func RPCService(val string) attribute.KeyValue {
	return RPCServiceKey.String(val)
}

// RPCMethod returns an attribute KeyValue conforming to the "rpc.method"
// semantic conventions. It represents the name of the (logical) method being
// called, must be equal to the $method part in the span name.
func RPCMethod(val string) attribute.KeyValue {
	return RPCMethodKey.String(val)
}

// Tech-specific attributes for gRPC.
const (
	// RPCGRPCStatusCodeKey is the attribute Key conforming to the
	// "rpc.grpc.status_code" semantic conventions. It represents the [numeric
	// status
	// code](https://github.com/grpc/grpc/blob/v1.33.2/doc/statuscodes.md) of
	// the gRPC request.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	RPCGRPCStatusCodeKey = attribute.Key("rpc.grpc.status_code")
)

var (
	// OK
	RPCGRPCStatusCodeOk = RPCGRPCStatusCodeKey.Int(0)
	// CANCELLED
	RPCGRPCStatusCodeCancelled = RPCGRPCStatusCodeKey.Int(1)
	// UNKNOWN
	RPCGRPCStatusCodeUnknown = RPCGRPCStatusCodeKey.Int(2)
	// INVALID_ARGUMENT
	RPCGRPCStatusCodeInvalidArgument = RPCGRPCStatusCodeKey.Int(3)
	// DEADLINE_EXCEEDED
	RPCGRPCStatusCodeDeadlineExceeded = RPCGRPCStatusCodeKey.Int(4)
	// NOT_FOUND
	RPCGRPCStatusCodeNotFound = RPCGRPCStatusCodeKey.Int(5)
	// ALREADY_EXISTS
	RPCGRPCStatusCodeAlreadyExists = RPCGRPCStatusCodeKey.Int(6)
	// PERMISSION_DENIED
	RPCGRPCStatusCodePermissionDenied = RPCGRPCStatusCodeKey.Int(7)
	// RESOURCE_EXHAUSTED
	RPCGRPCStatusCodeResourceExhausted = RPCGRPCStatusCodeKey.Int(8)
	// FAILED_PRECONDITION
	RPCGRPCStatusCodeFailedPrecondition = RPCGRPCStatusCodeKey.Int(9)
	// ABORTED
	RPCGRPCStatusCodeAborted = RPCGRPCStatusCodeKey.Int(10)
	// OUT_OF_RANGE
	RPCGRPCStatusCodeOutOfRange = RPCGRPCStatusCodeKey.Int(11)
	// UNIMPLEMENTED
	RPCGRPCStatusCodeUnimplemented = RPCGRPCStatusCodeKey.Int(12)
	// INTERNAL
	RPCGRPCStatusCodeInternal = RPCGRPCStatusCodeKey.Int(13)
	// UNAVAILABLE
	RPCGRPCStatusCodeUnavailable = RPCGRPCStatusCodeKey.Int(14)
	// DATA_LOSS
	RPCGRPCStatusCodeDataLoss = RPCGRPCStatusCodeKey.Int(15)
	// UNAUTHENTICATED
	RPCGRPCStatusCodeUnauthenticated = RPCGRPCStatusCodeKey.Int(16)
)

// Tech-specific attributes for [JSON RPC](https://www.jsonrpc.org/).
const (
	// RPCJsonrpcVersionKey is the attribute Key conforming to the
	// "rpc.jsonrpc.version" semantic conventions. It represents the protocol
	// version as in `jsonrpc` property of request/response. Since JSON-RPC 1.0
	// does not specify this, the value can be omitted.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If other than the default
	// version (`1.0`))
	// Stability: stable
	// Examples: '2.0', '1.0'
	RPCJsonrpcVersionKey = attribute.Key("rpc.jsonrpc.version")

	// RPCJsonrpcRequestIDKey is the attribute Key conforming to the
	// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id`
	// property of request or response. Since protocol allows id to be int,
	// string, `null` or missing (for notifications), value is expected to be
	// cast to string for simplicity. Use empty string in case of `null` value.
	// Omit entirely if this is a notification.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '10', 'request-7', ''
	RPCJsonrpcRequestIDKey = attribute.Key("rpc.jsonrpc.request_id")

	// RPCJsonrpcErrorCodeKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_code" semantic conventions. It represents the
	// `error.code` property of response if it is an error response.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If response is not successful.)
	// Stability: stable
	// Examples: -32700, 100
	RPCJsonrpcErrorCodeKey = attribute.Key("rpc.jsonrpc.error_code")

	// RPCJsonrpcErrorMessageKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_message" semantic conventions. It represents the
	// `error.message` property of response if it is an error response.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Parse error', 'User already exists'
	RPCJsonrpcErrorMessageKey = attribute.Key("rpc.jsonrpc.error_message")
)

// RPCJsonrpcVersion returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.version" semantic conventions. It represents the protocol
// version as in `jsonrpc` property of request/response. Since JSON-RPC 1.0
// does not specify this, the value can be omitted.
func RPCJsonrpcVersion(val string) attribute.KeyValue {
	return RPCJsonrpcVersionKey.String(val)
}

// RPCJsonrpcRequestID returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id`
// property of request or response. Since protocol allows id to be int, string,
// `null` or missing (for notifications), value is expected to be cast to
// string for simplicity. Use empty string in case of `null` value. Omit
// entirely if this is a notification.
func RPCJsonrpcRequestID(val string) attribute.KeyValue {
	return RPCJsonrpcRequestIDKey.String(val)
}

// RPCJsonrpcErrorCode returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.error_code" semantic conventions. It represents the
// `error.code` property of response if it is an error response.
func RPCJsonrpcErrorCode(val int) attribute.KeyValue {
	return RPCJsonrpcErrorCodeKey.Int(val)
}

// RPCJsonrpcErrorMessage returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.error_message" semantic conventions. It represents the
// `error.message` property of response if it is an error response.
func RPCJsonrpcErrorMessage(val string) attribute.KeyValue {
	return RPCJsonrpcErrorMessageKey.String(val)
}

// Tech-specific attributes for Connect RPC.
const (
	// RPCConnectRPCErrorCodeKey is the attribute Key conforming to the
	// "rpc.connect_rpc.error_code" semantic conventions. It represents the
	// [error codes](https://connect.build/docs/protocol/#error-codes) of the
	// Connect request. Error codes are always string values.
	//
	// Type: Enum
	// RequirementLevel: ConditionallyRequired (If response is not successful
	// and if error code available.)
	// Stability: stable
	RPCConnectRPCErrorCodeKey = attribute.Key("rpc.connect_rpc.error_code")
)

var (
	// cancelled
	RPCConnectRPCErrorCodeCancelled = RPCConnectRPCErrorCodeKey.String("cancelled")
	// unknown
	RPCConnectRPCErrorCodeUnknown = RPCConnectRPCErrorCodeKey.String("unknown")
	// invalid_argument
	RPCConnectRPCErrorCodeInvalidArgument = RPCConnectRPCErrorCodeKey.String("invalid_argument")
	// deadline_exceeded
	RPCConnectRPCErrorCodeDeadlineExceeded = RPCConnectRPCErrorCodeKey.String("deadline_exceeded")
	// not_found
	RPCConnectRPCErrorCodeNotFound = RPCConnectRPCErrorCodeKey.String("not_found")
	// already_exists
	RPCConnectRPCErrorCodeAlreadyExists = RPCConnectRPCErrorCodeKey.String("already_exists")
	// permission_denied
	RPCConnectRPCErrorCodePermissionDenied = RPCConnectRPCErrorCodeKey.String("permission_denied")
	// resource_exhausted
	RPCConnectRPCErrorCodeResourceExhausted = RPCConnectRPCErrorCodeKey.String("resource_exhausted")
	// failed_precondition
	RPCConnectRPCErrorCodeFailedPrecondition = RPCConnectRPCErrorCodeKey.String("failed_precondition")
	// aborted
	RPCConnectRPCErrorCodeAborted = RPCConnectRPCErrorCodeKey.String("aborted")
	// out_of_range
	RPCConnectRPCErrorCodeOutOfRange = RPCConnectRPCErrorCodeKey.String("out_of_range")
	// unimplemented
	RPCConnectRPCErrorCodeUnimplemented = RPCConnectRPCErrorCodeKey.String("unimplemented")
	// internal
	RPCConnectRPCErrorCodeInternal = RPCConnectRPCErrorCodeKey.String("internal")
	// unavailable
	RPCConnectRPCErrorCodeUnavailable = RPCConnectRPCErrorCodeKey.String("unavailable")
	// data_loss
	RPCConnectRPCErrorCodeDataLoss = RPCConnectRPCErrorCodeKey.String("data_loss")
	// unauthenticated
	RPCConnectRPCErrorCodeUnauthenticated = RPCConnectRPCErrorCodeKey.String("unauthenticated")
)
