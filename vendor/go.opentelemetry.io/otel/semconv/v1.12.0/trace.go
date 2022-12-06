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

package semconv // import "go.opentelemetry.io/otel/semconv/v1.12.0"

import "go.opentelemetry.io/otel/attribute"

// Span attributes used by AWS Lambda (in addition to general `faas` attributes).
const (
	// The full invoked ARN as provided on the `Context` passed to the function
	// (`Lambda-Runtime-Invoked-Function-ARN` header on the `/runtime/invocation/next`
	// applicable).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:lambda:us-east-1:123456:function:myfunction:myalias'
	// Note: This may be different from `faas.id` if an alias is involved.
	AWSLambdaInvokedARNKey = attribute.Key("aws.lambda.invoked_arn")
)

// This document defines attributes for CloudEvents. CloudEvents is a specification on how to define event data in a standard way. These attributes can be attached to spans when performing operations with CloudEvents, regardless of the protocol being used.
const (
	// The [event_id](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec
	// .md#id) uniquely identifies the event.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: '123e4567-e89b-12d3-a456-426614174000', '0001'
	CloudeventsEventIDKey = attribute.Key("cloudevents.event_id")
	// The [source](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.m
	// d#source-1) identifies the context in which an event happened.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'https://github.com/cloudevents', '/cloudevents/spec/pull/123', 'my-
	// service'
	CloudeventsEventSourceKey = attribute.Key("cloudevents.event_source")
	// The [version of the CloudEvents specification](https://github.com/cloudevents/s
	// pec/blob/v1.0.2/cloudevents/spec.md#specversion) which the event uses.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: '1.0'
	CloudeventsEventSpecVersionKey = attribute.Key("cloudevents.event_spec_version")
	// The [event_type](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/sp
	// ec.md#type) contains a value describing the type of event related to the
	// originating occurrence.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'com.github.pull_request.opened', 'com.example.object.deleted.v2'
	CloudeventsEventTypeKey = attribute.Key("cloudevents.event_type")
	// The [subject](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.
	// md#subject) of the event in the context of the event producer (identified by
	// source).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'mynewfile.jpg'
	CloudeventsEventSubjectKey = attribute.Key("cloudevents.event_subject")
)

// This document defines semantic conventions for the OpenTracing Shim
const (
	// Parent-child Reference type
	//
	// Type: Enum
	// Required: No
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

// This document defines the attributes used to perform database client calls.
const (
	// An identifier for the database management system (DBMS) product being used. See
	// below for a list of well-known identifiers.
	//
	// Type: Enum
	// Required: Always
	// Stability: stable
	DBSystemKey = attribute.Key("db.system")
	// The connection string used to connect to the database. It is recommended to
	// remove embedded credentials.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Server=(localdb)\\v11.0;Integrated Security=true;'
	DBConnectionStringKey = attribute.Key("db.connection_string")
	// Username for accessing the database.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'readonly_user', 'reporting_user'
	DBUserKey = attribute.Key("db.user")
	// The fully-qualified class name of the [Java Database Connectivity
	// (JDBC)](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/) driver
	// used to connect.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'org.postgresql.Driver',
	// 'com.microsoft.sqlserver.jdbc.SQLServerDriver'
	DBJDBCDriverClassnameKey = attribute.Key("db.jdbc.driver_classname")
	// This attribute is used to report the name of the database being accessed. For
	// commands that switch the database, this should be set to the target database
	// (even if the command fails).
	//
	// Type: string
	// Required: Required, if applicable.
	// Stability: stable
	// Examples: 'customers', 'main'
	// Note: In some SQL databases, the database name to be used is called "schema
	// name". In case there are multiple layers that could be considered for database
	// name (e.g. Oracle instance name and schema name), the database name to be used
	// is the more specific layer (e.g. Oracle schema name).
	DBNameKey = attribute.Key("db.name")
	// The database statement being executed.
	//
	// Type: string
	// Required: Required if applicable and not explicitly disabled via
	// instrumentation configuration.
	// Stability: stable
	// Examples: 'SELECT * FROM wuser_table', 'SET mykey "WuValue"'
	// Note: The value may be sanitized to exclude sensitive information.
	DBStatementKey = attribute.Key("db.statement")
	// The name of the operation being executed, e.g. the [MongoDB command
	// name](https://docs.mongodb.com/manual/reference/command/#database-operations)
	// such as `findAndModify`, or the SQL keyword.
	//
	// Type: string
	// Required: Required, if `db.statement` is not applicable.
	// Stability: stable
	// Examples: 'findAndModify', 'HMSET', 'SELECT'
	// Note: When setting this to an SQL keyword, it is not recommended to attempt any
	// client-side parsing of `db.statement` just to get this property, but it should
	// be set if the operation name is provided by the library being instrumented. If
	// the SQL statement has an ambiguous operation, or performs more than one
	// operation, this value may be omitted.
	DBOperationKey = attribute.Key("db.operation")
)

var (
	// Some other SQL database. Fallback only. See notes
	DBSystemOtherSQL = DBSystemKey.String("other_sql")
	// Microsoft SQL Server
	DBSystemMSSQL = DBSystemKey.String("mssql")
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
)

// Connection-level attributes for Microsoft SQL Server
const (
	// The Microsoft SQL Server [instance name](https://docs.microsoft.com/en-
	// us/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15)
	// connecting to. This name is used to determine the port of a named instance.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'MSSQLSERVER'
	// Note: If setting a `db.mssql.instance_name`, `net.peer.port` is no longer
	// required (but still recommended if non-standard).
	DBMSSQLInstanceNameKey = attribute.Key("db.mssql.instance_name")
)

// Call-level attributes for Cassandra
const (
	// The fetch size used for paging, i.e. how many rows will be returned at once.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 5000
	DBCassandraPageSizeKey = attribute.Key("db.cassandra.page_size")
	// The consistency level of the query. Based on consistency values from
	// [CQL](https://docs.datastax.com/en/cassandra-
	// oss/3.0/cassandra/dml/dmlConfigConsistency.html).
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	DBCassandraConsistencyLevelKey = attribute.Key("db.cassandra.consistency_level")
	// The name of the primary table that the operation is acting upon, including the
	// keyspace name (if applicable).
	//
	// Type: string
	// Required: Recommended if available.
	// Stability: stable
	// Examples: 'mytable'
	// Note: This mirrors the db.sql.table attribute but references cassandra rather
	// than sql. It is not recommended to attempt any client-side parsing of
	// `db.statement` just to get this property, but it should be set if it is
	// provided by the library being instrumented. If the operation is acting upon an
	// anonymous table, or more than one table, this value MUST NOT be set.
	DBCassandraTableKey = attribute.Key("db.cassandra.table")
	// Whether or not the query is idempotent.
	//
	// Type: boolean
	// Required: No
	// Stability: stable
	DBCassandraIdempotenceKey = attribute.Key("db.cassandra.idempotence")
	// The number of times a query was speculatively executed. Not set or `0` if the
	// query was not executed speculatively.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 0, 2
	DBCassandraSpeculativeExecutionCountKey = attribute.Key("db.cassandra.speculative_execution_count")
	// The ID of the coordinating node for a query.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'be13faa2-8574-4d71-926d-27f16cf8a7af'
	DBCassandraCoordinatorIDKey = attribute.Key("db.cassandra.coordinator.id")
	// The data center of the coordinating node for a query.
	//
	// Type: string
	// Required: No
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

// Call-level attributes for Redis
const (
	// The index of the database being accessed as used in the [`SELECT`
	// command](https://redis.io/commands/select), provided as an integer. To be used
	// instead of the generic `db.name` attribute.
	//
	// Type: int
	// Required: Required, if other than the default database (`0`).
	// Stability: stable
	// Examples: 0, 1, 15
	DBRedisDBIndexKey = attribute.Key("db.redis.database_index")
)

// Call-level attributes for MongoDB
const (
	// The collection being accessed within the database stated in `db.name`.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'customers', 'products'
	DBMongoDBCollectionKey = attribute.Key("db.mongodb.collection")
)

// Call-level attributes for SQL databases
const (
	// The name of the primary table that the operation is acting upon, including the
	// database name (if applicable).
	//
	// Type: string
	// Required: Recommended if available.
	// Stability: stable
	// Examples: 'public.users', 'customers'
	// Note: It is not recommended to attempt any client-side parsing of
	// `db.statement` just to get this property, but it should be set if it is
	// provided by the library being instrumented. If the operation is acting upon an
	// anonymous table, or more than one table, this value MUST NOT be set.
	DBSQLTableKey = attribute.Key("db.sql.table")
)

// This document defines the attributes used to report a single exception associated with a span.
const (
	// The type of the exception (its fully-qualified class name, if applicable). The
	// dynamic type of the exception should be preferred over the static type in
	// languages that support it.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'java.net.ConnectException', 'OSError'
	ExceptionTypeKey = attribute.Key("exception.type")
	// The exception message.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Division by zero', "Can't convert 'int' object to str implicitly"
	ExceptionMessageKey = attribute.Key("exception.message")
	// A stacktrace as a string in the natural representation for the language
	// runtime. The representation is to be determined and documented by each language
	// SIG.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Exception in thread "main" java.lang.RuntimeException: Test
	// exception\\n at '
	//  'com.example.GenerateTrace.methodB(GenerateTrace.java:13)\\n at '
	//  'com.example.GenerateTrace.methodA(GenerateTrace.java:9)\\n at '
	//  'com.example.GenerateTrace.main(GenerateTrace.java:5)'
	ExceptionStacktraceKey = attribute.Key("exception.stacktrace")
	// SHOULD be set to true if the exception event is recorded at a point where it is
	// known that the exception is escaping the scope of the span.
	//
	// Type: boolean
	// Required: No
	// Stability: stable
	// Note: An exception is considered to have escaped (or left) the scope of a span,
	// if that span is ended while the exception is still logically "in flight".
	// This may be actually "in flight" in some languages (e.g. if the exception
	// is passed to a Context manager's `__exit__` method in Python) but will
	// usually be caught at the point of recording the exception in most languages.

	// It is usually not possible to determine at the point where an exception is
	// thrown
	// whether it will escape the scope of a span.
	// However, it is trivial to know that an exception
	// will escape, if one checks for an active exception just before ending the span,
	// as done in the [example above](#recording-an-exception).

	// It follows that an exception may still escape the scope of the span
	// even if the `exception.escaped` attribute was not set or set to false,
	// since the event might have been recorded at a time where it was not
	// clear whether the exception will escape.
	ExceptionEscapedKey = attribute.Key("exception.escaped")
)

// This semantic convention describes an instance of a function that runs without provisioning or managing of servers (also known as serverless functions or Function as a Service (FaaS)) with spans.
const (
	// Type of the trigger which caused this function execution.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Note: For the server/consumer span on the incoming side,
	// `faas.trigger` MUST be set.

	// Clients invoking FaaS instances usually cannot set `faas.trigger`,
	// since they would typically need to look in the payload to determine
	// the event type. If clients set it, it should be the same as the
	// trigger that corresponding incoming would have (i.e., this has
	// nothing to do with the underlying transport used to make the API
	// call to invoke the lambda, which is often HTTP).
	FaaSTriggerKey = attribute.Key("faas.trigger")
	// The execution ID of the current function execution.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'af9d5aa4-a685-4c5f-a22b-444f80b3cc28'
	FaaSExecutionKey = attribute.Key("faas.execution")
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

// Semantic Convention for FaaS triggered as a response to some data source operation such as a database or filesystem read/write.
const (
	// The name of the source on which the triggering operation was performed. For
	// example, in Cloud Storage or S3 corresponds to the bucket name, and in Cosmos
	// DB to the database name.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'myBucketName', 'myDBName'
	FaaSDocumentCollectionKey = attribute.Key("faas.document.collection")
	// Describes the type of the operation that was performed on the data.
	//
	// Type: Enum
	// Required: Always
	// Stability: stable
	FaaSDocumentOperationKey = attribute.Key("faas.document.operation")
	// A string containing the time when the data was accessed in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format expressed
	// in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: '2020-01-23T13:47:06Z'
	FaaSDocumentTimeKey = attribute.Key("faas.document.time")
	// The document name/table subjected to the operation. For example, in Cloud
	// Storage or S3 is the name of the file, and in Cosmos DB the table name.
	//
	// Type: string
	// Required: No
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

// Semantic Convention for FaaS scheduled to be executed regularly.
const (
	// A string containing the function invocation time in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format expressed
	// in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: '2020-01-23T13:47:06Z'
	FaaSTimeKey = attribute.Key("faas.time")
	// A string containing the schedule period as [Cron Expression](https://docs.oracl
	// e.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '0/5 * * * ? *'
	FaaSCronKey = attribute.Key("faas.cron")
)

// Contains additional attributes for incoming FaaS spans.
const (
	// A boolean that is true if the serverless function is executed for the first
	// time (aka cold-start).
	//
	// Type: boolean
	// Required: No
	// Stability: stable
	FaaSColdstartKey = attribute.Key("faas.coldstart")
)

// Contains additional attributes for outgoing FaaS spans.
const (
	// The name of the invoked function.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'my-function'
	// Note: SHOULD be equal to the `faas.name` resource attribute of the invoked
	// function.
	FaaSInvokedNameKey = attribute.Key("faas.invoked_name")
	// The cloud provider of the invoked function.
	//
	// Type: Enum
	// Required: Always
	// Stability: stable
	// Note: SHOULD be equal to the `cloud.provider` resource attribute of the invoked
	// function.
	FaaSInvokedProviderKey = attribute.Key("faas.invoked_provider")
	// The cloud region of the invoked function.
	//
	// Type: string
	// Required: For some cloud providers, like AWS or GCP, the region in which a
	// function is hosted is essential to uniquely identify the function and also part
	// of its endpoint. Since it's part of the endpoint being called, the region is
	// always known to clients. In these cases, `faas.invoked_region` MUST be set
	// accordingly. If the region is unknown to the client or not required for
	// identifying the invoked function, setting `faas.invoked_region` is optional.
	// Stability: stable
	// Examples: 'eu-central-1'
	// Note: SHOULD be equal to the `cloud.region` resource attribute of the invoked
	// function.
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

// These attributes may be used for any network related operation.
const (
	// Transport protocol used. See note below.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	NetTransportKey = attribute.Key("net.transport")
	// Remote address of the peer (dotted decimal for IPv4 or
	// [RFC5952](https://tools.ietf.org/html/rfc5952) for IPv6)
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '127.0.0.1'
	NetPeerIPKey = attribute.Key("net.peer.ip")
	// Remote port number.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 80, 8080, 443
	NetPeerPortKey = attribute.Key("net.peer.port")
	// Remote hostname or similar, see note below.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'example.com'
	// Note: `net.peer.name` SHOULD NOT be set if capturing it would require an extra
	// DNS lookup.
	NetPeerNameKey = attribute.Key("net.peer.name")
	// Like `net.peer.ip` but for the host IP. Useful in case of a multi-IP host.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '192.168.0.1'
	NetHostIPKey = attribute.Key("net.host.ip")
	// Like `net.peer.port` but for the host port.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 35555
	NetHostPortKey = attribute.Key("net.host.port")
	// Local hostname or similar, see note below.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'localhost'
	NetHostNameKey = attribute.Key("net.host.name")
	// The internet connection type currently being used by the host.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Examples: 'wifi'
	NetHostConnectionTypeKey = attribute.Key("net.host.connection.type")
	// This describes more details regarding the connection.type. It may be the type
	// of cell technology connection, but it could be used for describing details
	// about a wifi connection.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Examples: 'LTE'
	NetHostConnectionSubtypeKey = attribute.Key("net.host.connection.subtype")
	// The name of the mobile carrier.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'sprint'
	NetHostCarrierNameKey = attribute.Key("net.host.carrier.name")
	// The mobile carrier country code.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '310'
	NetHostCarrierMccKey = attribute.Key("net.host.carrier.mcc")
	// The mobile carrier network code.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '001'
	NetHostCarrierMncKey = attribute.Key("net.host.carrier.mnc")
	// The ISO 3166-1 alpha-2 2-character country code associated with the mobile
	// carrier network.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'DE'
	NetHostCarrierIccKey = attribute.Key("net.host.carrier.icc")
)

var (
	// ip_tcp
	NetTransportTCP = NetTransportKey.String("ip_tcp")
	// ip_udp
	NetTransportUDP = NetTransportKey.String("ip_udp")
	// Another IP-based protocol
	NetTransportIP = NetTransportKey.String("ip")
	// Unix Domain socket. See below
	NetTransportUnix = NetTransportKey.String("unix")
	// Named or anonymous pipe. See note below
	NetTransportPipe = NetTransportKey.String("pipe")
	// In-process communication
	NetTransportInProc = NetTransportKey.String("inproc")
	// Something else (non IP-based)
	NetTransportOther = NetTransportKey.String("other")
)

var (
	// wifi
	NetHostConnectionTypeWifi = NetHostConnectionTypeKey.String("wifi")
	// wired
	NetHostConnectionTypeWired = NetHostConnectionTypeKey.String("wired")
	// cell
	NetHostConnectionTypeCell = NetHostConnectionTypeKey.String("cell")
	// unavailable
	NetHostConnectionTypeUnavailable = NetHostConnectionTypeKey.String("unavailable")
	// unknown
	NetHostConnectionTypeUnknown = NetHostConnectionTypeKey.String("unknown")
)

var (
	// GPRS
	NetHostConnectionSubtypeGprs = NetHostConnectionSubtypeKey.String("gprs")
	// EDGE
	NetHostConnectionSubtypeEdge = NetHostConnectionSubtypeKey.String("edge")
	// UMTS
	NetHostConnectionSubtypeUmts = NetHostConnectionSubtypeKey.String("umts")
	// CDMA
	NetHostConnectionSubtypeCdma = NetHostConnectionSubtypeKey.String("cdma")
	// EVDO Rel. 0
	NetHostConnectionSubtypeEvdo0 = NetHostConnectionSubtypeKey.String("evdo_0")
	// EVDO Rev. A
	NetHostConnectionSubtypeEvdoA = NetHostConnectionSubtypeKey.String("evdo_a")
	// CDMA2000 1XRTT
	NetHostConnectionSubtypeCdma20001xrtt = NetHostConnectionSubtypeKey.String("cdma2000_1xrtt")
	// HSDPA
	NetHostConnectionSubtypeHsdpa = NetHostConnectionSubtypeKey.String("hsdpa")
	// HSUPA
	NetHostConnectionSubtypeHsupa = NetHostConnectionSubtypeKey.String("hsupa")
	// HSPA
	NetHostConnectionSubtypeHspa = NetHostConnectionSubtypeKey.String("hspa")
	// IDEN
	NetHostConnectionSubtypeIden = NetHostConnectionSubtypeKey.String("iden")
	// EVDO Rev. B
	NetHostConnectionSubtypeEvdoB = NetHostConnectionSubtypeKey.String("evdo_b")
	// LTE
	NetHostConnectionSubtypeLte = NetHostConnectionSubtypeKey.String("lte")
	// EHRPD
	NetHostConnectionSubtypeEhrpd = NetHostConnectionSubtypeKey.String("ehrpd")
	// HSPAP
	NetHostConnectionSubtypeHspap = NetHostConnectionSubtypeKey.String("hspap")
	// GSM
	NetHostConnectionSubtypeGsm = NetHostConnectionSubtypeKey.String("gsm")
	// TD-SCDMA
	NetHostConnectionSubtypeTdScdma = NetHostConnectionSubtypeKey.String("td_scdma")
	// IWLAN
	NetHostConnectionSubtypeIwlan = NetHostConnectionSubtypeKey.String("iwlan")
	// 5G NR (New Radio)
	NetHostConnectionSubtypeNr = NetHostConnectionSubtypeKey.String("nr")
	// 5G NRNSA (New Radio Non-Standalone)
	NetHostConnectionSubtypeNrnsa = NetHostConnectionSubtypeKey.String("nrnsa")
	// LTE CA
	NetHostConnectionSubtypeLteCa = NetHostConnectionSubtypeKey.String("lte_ca")
)

// Operations that access some remote service.
const (
	// The [`service.name`](../../resource/semantic_conventions/README.md#service) of
	// the remote service. SHOULD be equal to the actual `service.name` resource
	// attribute of the remote service if any.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'AuthTokenCache'
	PeerServiceKey = attribute.Key("peer.service")
)

// These attributes may be used for any operation with an authenticated and/or authorized enduser.
const (
	// Username or client_id extracted from the access token or
	// [Authorization](https://tools.ietf.org/html/rfc7235#section-4.2) header in the
	// inbound request from outside the system.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'username'
	EnduserIDKey = attribute.Key("enduser.id")
	// Actual/assumed role the client is making the request under extracted from token
	// or application security context.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'admin'
	EnduserRoleKey = attribute.Key("enduser.role")
	// Scopes or granted authorities the client currently possesses extracted from
	// token or application security context. The value would come from the scope
	// associated with an [OAuth 2.0 Access
	// Token](https://tools.ietf.org/html/rfc6749#section-3.3) or an attribute value
	// in a [SAML 2.0 Assertion](http://docs.oasis-
	// open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'read:message, write:files'
	EnduserScopeKey = attribute.Key("enduser.scope")
)

// These attributes may be used for any operation to store information about a thread that started a span.
const (
	// Current "managed" thread ID (as opposed to OS thread ID).
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 42
	ThreadIDKey = attribute.Key("thread.id")
	// Current thread name.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'main'
	ThreadNameKey = attribute.Key("thread.name")
)

// These attributes allow to report this unit of code and therefore to provide more context about the span.
const (
	// The method or function name, or equivalent (usually rightmost part of the code
	// unit's name).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'serveRequest'
	CodeFunctionKey = attribute.Key("code.function")
	// The "namespace" within which `code.function` is defined. Usually the qualified
	// class or module name, such that `code.namespace` + some separator +
	// `code.function` form a unique identifier for the code unit.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'com.example.MyHTTPService'
	CodeNamespaceKey = attribute.Key("code.namespace")
	// The source code file name that identifies the code unit as uniquely as possible
	// (preferably an absolute file path).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '/usr/local/MyApplication/content_root/app/index.php'
	CodeFilepathKey = attribute.Key("code.filepath")
	// The line number in `code.filepath` best representing the operation. It SHOULD
	// point within the code unit named in `code.function`.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 42
	CodeLineNumberKey = attribute.Key("code.lineno")
)

// This document defines semantic conventions for HTTP client and server Spans.
const (
	// HTTP request method.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'GET', 'POST', 'HEAD'
	HTTPMethodKey = attribute.Key("http.method")
	// Full HTTP request URL in the form `scheme://host[:port]/path?query[#fragment]`.
	// Usually the fragment is not transmitted over HTTP, but if it is known, it
	// should be included nevertheless.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'https://www.foo.bar/search?q=OpenTelemetry#SemConv'
	// Note: `http.url` MUST NOT contain credentials passed via URL in form of
	// `https://username:password@www.example.com/`. In such case the attribute's
	// value should be `https://www.example.com/`.
	HTTPURLKey = attribute.Key("http.url")
	// The full request target as passed in a HTTP request line or equivalent.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '/path/12314/?q=ddds#123'
	HTTPTargetKey = attribute.Key("http.target")
	// The value of the [HTTP host
	// header](https://tools.ietf.org/html/rfc7230#section-5.4). An empty Host header
	// should also be reported, see note.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'www.example.org'
	// Note: When the header is present but empty the attribute SHOULD be set to the
	// empty string. Note that this is a valid situation that is expected in certain
	// cases, according the aforementioned [section of RFC
	// 7230](https://tools.ietf.org/html/rfc7230#section-5.4). When the header is not
	// set the attribute MUST NOT be set.
	HTTPHostKey = attribute.Key("http.host")
	// The URI scheme identifying the used protocol.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'http', 'https'
	HTTPSchemeKey = attribute.Key("http.scheme")
	// [HTTP response status code](https://tools.ietf.org/html/rfc7231#section-6).
	//
	// Type: int
	// Required: If and only if one was received/sent.
	// Stability: stable
	// Examples: 200
	HTTPStatusCodeKey = attribute.Key("http.status_code")
	// Kind of HTTP protocol used.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Note: If `net.transport` is not specified, it can be assumed to be `IP.TCP`
	// except if `http.flavor` is `QUIC`, in which case `IP.UDP` is assumed.
	HTTPFlavorKey = attribute.Key("http.flavor")
	// Value of the [HTTP User-
	// Agent](https://tools.ietf.org/html/rfc7231#section-5.5.3) header sent by the
	// client.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'CERN-LineMode/2.15 libwww/2.17b3'
	HTTPUserAgentKey = attribute.Key("http.user_agent")
	// The size of the request payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as the
	// [Content-Length](https://tools.ietf.org/html/rfc7230#section-3.3.2) header. For
	// requests using transport encoding, this should be the compressed size.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 3495
	HTTPRequestContentLengthKey = attribute.Key("http.request_content_length")
	// The size of the uncompressed request payload body after transport decoding. Not
	// set if transport encoding not used.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 5493
	HTTPRequestContentLengthUncompressedKey = attribute.Key("http.request_content_length_uncompressed")
	// The size of the response payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as the
	// [Content-Length](https://tools.ietf.org/html/rfc7230#section-3.3.2) header. For
	// requests using transport encoding, this should be the compressed size.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 3495
	HTTPResponseContentLengthKey = attribute.Key("http.response_content_length")
	// The size of the uncompressed response payload body after transport decoding.
	// Not set if transport encoding not used.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 5493
	HTTPResponseContentLengthUncompressedKey = attribute.Key("http.response_content_length_uncompressed")
	// The ordinal number of request re-sending attempt.
	//
	// Type: int
	// Required: If and only if a request was retried.
	// Stability: stable
	// Examples: 3
	HTTPRetryCountKey = attribute.Key("http.retry_count")
)

var (
	// HTTP/1.0
	HTTPFlavorHTTP10 = HTTPFlavorKey.String("1.0")
	// HTTP/1.1
	HTTPFlavorHTTP11 = HTTPFlavorKey.String("1.1")
	// HTTP/2
	HTTPFlavorHTTP20 = HTTPFlavorKey.String("2.0")
	// HTTP/3
	HTTPFlavorHTTP30 = HTTPFlavorKey.String("3.0")
	// SPDY protocol
	HTTPFlavorSPDY = HTTPFlavorKey.String("SPDY")
	// QUIC protocol
	HTTPFlavorQUIC = HTTPFlavorKey.String("QUIC")
)

// Semantic Convention for HTTP Server
const (
	// The primary server name of the matched virtual host. This should be obtained
	// via configuration. If no such configuration can be obtained, this attribute
	// MUST NOT be set ( `net.host.name` should be used instead).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'example.com'
	// Note: `http.url` is usually not readily available on the server side but would
	// have to be assembled in a cumbersome and sometimes lossy process from other
	// information (see e.g. open-telemetry/opentelemetry-python/pull/148). It is thus
	// preferred to supply the raw data that is available.
	HTTPServerNameKey = attribute.Key("http.server_name")
	// The matched route (path template).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '/users/:userID?'
	HTTPRouteKey = attribute.Key("http.route")
	// The IP address of the original client behind all proxies, if known (e.g. from
	// [X-Forwarded-For](https://developer.mozilla.org/en-
	// US/docs/Web/HTTP/Headers/X-Forwarded-For)).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '83.164.160.102'
	// Note: This is not necessarily the same as `net.peer.ip`, which would
	// identify the network-level peer, which may be a proxy.

	// This attribute should be set when a source of information different
	// from the one used for `net.peer.ip`, is available even if that other
	// source just confirms the same value as `net.peer.ip`.
	// Rationale: For `net.peer.ip`, one typically does not know if it
	// comes from a proxy, reverse proxy, or the actual client. Setting
	// `http.client_ip` when it's the same as `net.peer.ip` means that
	// one is at least somewhat confident that the address is not that of
	// the closest proxy.
	HTTPClientIPKey = attribute.Key("http.client_ip")
)

// Attributes that exist for multiple DynamoDB request types.
const (
	// The keys in the `RequestItems` object field.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'Users', 'Cats'
	AWSDynamoDBTableNamesKey = attribute.Key("aws.dynamodb.table_names")
	// The JSON-serialized value of each item in the `ConsumedCapacity` response
	// field.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '{ "CapacityUnits": number, "GlobalSecondaryIndexes": { "string" : {
	// "CapacityUnits": number, "ReadCapacityUnits": number, "WriteCapacityUnits":
	// number } }, "LocalSecondaryIndexes": { "string" : { "CapacityUnits": number,
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number } },
	// "ReadCapacityUnits": number, "Table": { "CapacityUnits": number,
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number }, "TableName":
	// "string", "WriteCapacityUnits": number }'
	AWSDynamoDBConsumedCapacityKey = attribute.Key("aws.dynamodb.consumed_capacity")
	// The JSON-serialized value of the `ItemCollectionMetrics` response field.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '{ "string" : [ { "ItemCollectionKey": { "string" : { "B": blob,
	// "BOOL": boolean, "BS": [ blob ], "L": [ "AttributeValue" ], "M": { "string" :
	// "AttributeValue" }, "N": "string", "NS": [ "string" ], "NULL": boolean, "S":
	// "string", "SS": [ "string" ] } }, "SizeEstimateRangeGB": [ number ] } ] }'
	AWSDynamoDBItemCollectionMetricsKey = attribute.Key("aws.dynamodb.item_collection_metrics")
	// The value of the `ProvisionedThroughput.ReadCapacityUnits` request parameter.
	//
	// Type: double
	// Required: No
	// Stability: stable
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedReadCapacityKey = attribute.Key("aws.dynamodb.provisioned_read_capacity")
	// The value of the `ProvisionedThroughput.WriteCapacityUnits` request parameter.
	//
	// Type: double
	// Required: No
	// Stability: stable
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedWriteCapacityKey = attribute.Key("aws.dynamodb.provisioned_write_capacity")
	// The value of the `ConsistentRead` request parameter.
	//
	// Type: boolean
	// Required: No
	// Stability: stable
	AWSDynamoDBConsistentReadKey = attribute.Key("aws.dynamodb.consistent_read")
	// The value of the `ProjectionExpression` request parameter.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Title', 'Title, Price, Color', 'Title, Description, RelatedItems,
	// ProductReviews'
	AWSDynamoDBProjectionKey = attribute.Key("aws.dynamodb.projection")
	// The value of the `Limit` request parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 10
	AWSDynamoDBLimitKey = attribute.Key("aws.dynamodb.limit")
	// The value of the `AttributesToGet` request parameter.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'lives', 'id'
	AWSDynamoDBAttributesToGetKey = attribute.Key("aws.dynamodb.attributes_to_get")
	// The value of the `IndexName` request parameter.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'name_to_group'
	AWSDynamoDBIndexNameKey = attribute.Key("aws.dynamodb.index_name")
	// The value of the `Select` request parameter.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'ALL_ATTRIBUTES', 'COUNT'
	AWSDynamoDBSelectKey = attribute.Key("aws.dynamodb.select")
)

// DynamoDB.CreateTable
const (
	// The JSON-serialized value of each item of the `GlobalSecondaryIndexes` request
	// field
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '{ "IndexName": "string", "KeySchema": [ { "AttributeName": "string",
	// "KeyType": "string" } ], "Projection": { "NonKeyAttributes": [ "string" ],
	// "ProjectionType": "string" }, "ProvisionedThroughput": { "ReadCapacityUnits":
	// number, "WriteCapacityUnits": number } }'
	AWSDynamoDBGlobalSecondaryIndexesKey = attribute.Key("aws.dynamodb.global_secondary_indexes")
	// The JSON-serialized value of each item of the `LocalSecondaryIndexes` request
	// field.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '{ "IndexARN": "string", "IndexName": "string", "IndexSizeBytes":
	// number, "ItemCount": number, "KeySchema": [ { "AttributeName": "string",
	// "KeyType": "string" } ], "Projection": { "NonKeyAttributes": [ "string" ],
	// "ProjectionType": "string" } }'
	AWSDynamoDBLocalSecondaryIndexesKey = attribute.Key("aws.dynamodb.local_secondary_indexes")
)

// DynamoDB.ListTables
const (
	// The value of the `ExclusiveStartTableName` request parameter.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Users', 'CatsTable'
	AWSDynamoDBExclusiveStartTableKey = attribute.Key("aws.dynamodb.exclusive_start_table")
	// The the number of items in the `TableNames` response parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 20
	AWSDynamoDBTableCountKey = attribute.Key("aws.dynamodb.table_count")
)

// DynamoDB.Query
const (
	// The value of the `ScanIndexForward` request parameter.
	//
	// Type: boolean
	// Required: No
	// Stability: stable
	AWSDynamoDBScanForwardKey = attribute.Key("aws.dynamodb.scan_forward")
)

// DynamoDB.Scan
const (
	// The value of the `Segment` request parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 10
	AWSDynamoDBSegmentKey = attribute.Key("aws.dynamodb.segment")
	// The value of the `TotalSegments` request parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 100
	AWSDynamoDBTotalSegmentsKey = attribute.Key("aws.dynamodb.total_segments")
	// The value of the `Count` response parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 10
	AWSDynamoDBCountKey = attribute.Key("aws.dynamodb.count")
	// The value of the `ScannedCount` response parameter.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 50
	AWSDynamoDBScannedCountKey = attribute.Key("aws.dynamodb.scanned_count")
)

// DynamoDB.UpdateTable
const (
	// The JSON-serialized value of each item in the `AttributeDefinitions` request
	// field.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '{ "AttributeName": "string", "AttributeType": "string" }'
	AWSDynamoDBAttributeDefinitionsKey = attribute.Key("aws.dynamodb.attribute_definitions")
	// The JSON-serialized value of each item in the the `GlobalSecondaryIndexUpdates`
	// request field.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '{ "Create": { "IndexName": "string", "KeySchema": [ {
	// "AttributeName": "string", "KeyType": "string" } ], "Projection": {
	// "NonKeyAttributes": [ "string" ], "ProjectionType": "string" },
	// "ProvisionedThroughput": { "ReadCapacityUnits": number, "WriteCapacityUnits":
	// number } }'
	AWSDynamoDBGlobalSecondaryIndexUpdatesKey = attribute.Key("aws.dynamodb.global_secondary_index_updates")
)

// This document defines the attributes used in messaging systems.
const (
	// A string identifying the messaging system.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'kafka', 'rabbitmq', 'rocketmq', 'activemq', 'AmazonSQS'
	MessagingSystemKey = attribute.Key("messaging.system")
	// The message destination name. This might be equal to the span name but is
	// required nevertheless.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'MyQueue', 'MyTopic'
	MessagingDestinationKey = attribute.Key("messaging.destination")
	// The kind of message destination
	//
	// Type: Enum
	// Required: Required only if the message destination is either a `queue` or
	// `topic`.
	// Stability: stable
	MessagingDestinationKindKey = attribute.Key("messaging.destination_kind")
	// A boolean that is true if the message destination is temporary.
	//
	// Type: boolean
	// Required: If missing, it is assumed to be false.
	// Stability: stable
	MessagingTempDestinationKey = attribute.Key("messaging.temp_destination")
	// The name of the transport protocol.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'AMQP', 'MQTT'
	MessagingProtocolKey = attribute.Key("messaging.protocol")
	// The version of the transport protocol.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '0.9.1'
	MessagingProtocolVersionKey = attribute.Key("messaging.protocol_version")
	// Connection string.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'tibjmsnaming://localhost:7222',
	// 'https://queue.amazonaws.com/80398EXAMPLE/MyQueue'
	MessagingURLKey = attribute.Key("messaging.url")
	// A value used by the messaging system as an identifier for the message,
	// represented as a string.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '452a7c7c7c7048c2f887f61572b18fc2'
	MessagingMessageIDKey = attribute.Key("messaging.message_id")
	// The [conversation ID](#conversations) identifying the conversation to which the
	// message belongs, represented as a string. Sometimes called "Correlation ID".
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'MyConversationID'
	MessagingConversationIDKey = attribute.Key("messaging.conversation_id")
	// The (uncompressed) size of the message payload in bytes. Also use this
	// attribute if it is unknown whether the compressed or uncompressed payload size
	// is reported.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 2738
	MessagingMessagePayloadSizeBytesKey = attribute.Key("messaging.message_payload_size_bytes")
	// The compressed size of the message payload in bytes.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 2048
	MessagingMessagePayloadCompressedSizeBytesKey = attribute.Key("messaging.message_payload_compressed_size_bytes")
)

var (
	// A message sent to a queue
	MessagingDestinationKindQueue = MessagingDestinationKindKey.String("queue")
	// A message sent to a topic
	MessagingDestinationKindTopic = MessagingDestinationKindKey.String("topic")
)

// Semantic convention for a consumer of messages received from a messaging system
const (
	// A string identifying the kind of message consumption as defined in the
	// [Operation names](#operation-names) section above. If the operation is "send",
	// this attribute MUST NOT be set, since the operation can be inferred from the
	// span kind in that case.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	MessagingOperationKey = attribute.Key("messaging.operation")
	// The identifier for the consumer receiving a message. For Kafka, set it to
	// `{messaging.kafka.consumer_group} - {messaging.kafka.client_id}`, if both are
	// present, or only `messaging.kafka.consumer_group`. For brokers, such as
	// RabbitMQ and Artemis, set it to the `client_id` of the client consuming the
	// message.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'mygroup - client-6'
	MessagingConsumerIDKey = attribute.Key("messaging.consumer_id")
)

var (
	// receive
	MessagingOperationReceive = MessagingOperationKey.String("receive")
	// process
	MessagingOperationProcess = MessagingOperationKey.String("process")
)

// Attributes for RabbitMQ
const (
	// RabbitMQ message routing key.
	//
	// Type: string
	// Required: Unless it is empty.
	// Stability: stable
	// Examples: 'myKey'
	MessagingRabbitmqRoutingKeyKey = attribute.Key("messaging.rabbitmq.routing_key")
)

// Attributes for Apache Kafka
const (
	// Message keys in Kafka are used for grouping alike messages to ensure they're
	// processed on the same partition. They differ from `messaging.message_id` in
	// that they're not unique. If the key is `null`, the attribute MUST NOT be set.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'myKey'
	// Note: If the key type is not string, it's string representation has to be
	// supplied for the attribute. If the key has no unambiguous, canonical string
	// form, don't include its value.
	MessagingKafkaMessageKeyKey = attribute.Key("messaging.kafka.message_key")
	// Name of the Kafka Consumer Group that is handling the message. Only applies to
	// consumers, not producers.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'my-group'
	MessagingKafkaConsumerGroupKey = attribute.Key("messaging.kafka.consumer_group")
	// Client ID for the Consumer or Producer that is handling the message.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'client-5'
	MessagingKafkaClientIDKey = attribute.Key("messaging.kafka.client_id")
	// Partition the message is sent to.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 2
	MessagingKafkaPartitionKey = attribute.Key("messaging.kafka.partition")
	// A boolean that is true if the message is a tombstone.
	//
	// Type: boolean
	// Required: If missing, it is assumed to be false.
	// Stability: stable
	MessagingKafkaTombstoneKey = attribute.Key("messaging.kafka.tombstone")
)

// Attributes for Apache RocketMQ
const (
	// Namespace of RocketMQ resources, resources in different namespaces are
	// individual.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'myNamespace'
	MessagingRocketmqNamespaceKey = attribute.Key("messaging.rocketmq.namespace")
	// Name of the RocketMQ producer/consumer group that is handling the message. The
	// client type is identified by the SpanKind.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'myConsumerGroup'
	MessagingRocketmqClientGroupKey = attribute.Key("messaging.rocketmq.client_group")
	// The unique identifier for each client.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'myhost@8742@s8083jm'
	MessagingRocketmqClientIDKey = attribute.Key("messaging.rocketmq.client_id")
	// Type of message.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	MessagingRocketmqMessageTypeKey = attribute.Key("messaging.rocketmq.message_type")
	// The secondary classifier of message besides topic.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'tagA'
	MessagingRocketmqMessageTagKey = attribute.Key("messaging.rocketmq.message_tag")
	// Key(s) of message, another way to mark message besides message id.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'keyA', 'keyB'
	MessagingRocketmqMessageKeysKey = attribute.Key("messaging.rocketmq.message_keys")
	// Model of message consumption. This only applies to consumer spans.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	MessagingRocketmqConsumptionModelKey = attribute.Key("messaging.rocketmq.consumption_model")
)

var (
	// Normal message
	MessagingRocketmqMessageTypeNormal = MessagingRocketmqMessageTypeKey.String("normal")
	// FIFO message
	MessagingRocketmqMessageTypeFifo = MessagingRocketmqMessageTypeKey.String("fifo")
	// Delay message
	MessagingRocketmqMessageTypeDelay = MessagingRocketmqMessageTypeKey.String("delay")
	// Transaction message
	MessagingRocketmqMessageTypeTransaction = MessagingRocketmqMessageTypeKey.String("transaction")
)

var (
	// Clustering consumption model
	MessagingRocketmqConsumptionModelClustering = MessagingRocketmqConsumptionModelKey.String("clustering")
	// Broadcasting consumption model
	MessagingRocketmqConsumptionModelBroadcasting = MessagingRocketmqConsumptionModelKey.String("broadcasting")
)

// This document defines semantic conventions for remote procedure calls.
const (
	// A string identifying the remoting system. See below for a list of well-known
	// identifiers.
	//
	// Type: Enum
	// Required: Always
	// Stability: stable
	RPCSystemKey = attribute.Key("rpc.system")
	// The full (logical) name of the service being called, including its package
	// name, if applicable.
	//
	// Type: string
	// Required: No, but recommended
	// Stability: stable
	// Examples: 'myservice.EchoService'
	// Note: This is the logical name of the service from the RPC interface
	// perspective, which can be different from the name of any implementing class.
	// The `code.namespace` attribute may be used to store the latter (despite the
	// attribute name, it may include a class name; e.g., class with method actually
	// executing the call on the server side, RPC client stub class on the client
	// side).
	RPCServiceKey = attribute.Key("rpc.service")
	// The name of the (logical) method being called, must be equal to the $method
	// part in the span name.
	//
	// Type: string
	// Required: No, but recommended
	// Stability: stable
	// Examples: 'exampleMethod'
	// Note: This is the logical name of the method from the RPC interface
	// perspective, which can be different from the name of any implementing
	// method/function. The `code.function` attribute may be used to store the latter
	// (e.g., method actually executing the call on the server side, RPC client stub
	// method on the client side).
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
)

// Tech-specific attributes for gRPC.
const (
	// The [numeric status
	// code](https://github.com/grpc/grpc/blob/v1.33.2/doc/statuscodes.md) of the gRPC
	// request.
	//
	// Type: Enum
	// Required: Always
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
	// Protocol version as in `jsonrpc` property of request/response. Since JSON-RPC
	// 1.0 does not specify this, the value can be omitted.
	//
	// Type: string
	// Required: If missing, it is assumed to be "1.0".
	// Stability: stable
	// Examples: '2.0', '1.0'
	RPCJsonrpcVersionKey = attribute.Key("rpc.jsonrpc.version")
	// `id` property of request or response. Since protocol allows id to be int,
	// string, `null` or missing (for notifications), value is expected to be cast to
	// string for simplicity. Use empty string in case of `null` value. Omit entirely
	// if this is a notification.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '10', 'request-7', ''
	RPCJsonrpcRequestIDKey = attribute.Key("rpc.jsonrpc.request_id")
	// `error.code` property of response if it is an error response.
	//
	// Type: int
	// Required: If missing, response is assumed to be successful.
	// Stability: stable
	// Examples: -32700, 100
	RPCJsonrpcErrorCodeKey = attribute.Key("rpc.jsonrpc.error_code")
	// `error.message` property of response if it is an error response.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Parse error', 'User already exists'
	RPCJsonrpcErrorMessageKey = attribute.Key("rpc.jsonrpc.error_message")
)

// RPC received/sent message.
const (
	// Whether this is a received or sent message.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	MessageTypeKey = attribute.Key("message.type")
	// MUST be calculated as two different counters starting from `1` one for sent
	// messages and one for received message.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Note: This way we guarantee that the values will be consistent between
	// different implementations.
	MessageIDKey = attribute.Key("message.id")
	// Compressed size of the message in bytes.
	//
	// Type: int
	// Required: No
	// Stability: stable
	MessageCompressedSizeKey = attribute.Key("message.compressed_size")
	// Uncompressed size of the message in bytes.
	//
	// Type: int
	// Required: No
	// Stability: stable
	MessageUncompressedSizeKey = attribute.Key("message.uncompressed_size")
)

var (
	// sent
	MessageTypeSent = MessageTypeKey.String("SENT")
	// received
	MessageTypeReceived = MessageTypeKey.String("RECEIVED")
)
