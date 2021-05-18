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

package semconv // import "go.opentelemetry.io/otel/semconv"

import "go.opentelemetry.io/otel/attribute"

// Semantic conventions for attribute keys used for network related
// operations.
const (
	// Transport protocol used.
	NetTransportKey = attribute.Key("net.transport")

	// Remote address of the peer.
	NetPeerIPKey = attribute.Key("net.peer.ip")

	// Remote port number.
	NetPeerPortKey = attribute.Key("net.peer.port")

	// Remote hostname or similar.
	NetPeerNameKey = attribute.Key("net.peer.name")

	// Local host IP. Useful in case of a multi-IP host.
	NetHostIPKey = attribute.Key("net.host.ip")

	// Local host port.
	NetHostPortKey = attribute.Key("net.host.port")

	// Local hostname or similar.
	NetHostNameKey = attribute.Key("net.host.name")
)

// Semantic conventions for common transport protocol attributes.
var (
	NetTransportTCP    = NetTransportKey.String("IP.TCP")
	NetTransportUDP    = NetTransportKey.String("IP.UDP")
	NetTransportIP     = NetTransportKey.String("IP")
	NetTransportUnix   = NetTransportKey.String("Unix")
	NetTransportPipe   = NetTransportKey.String("pipe")
	NetTransportInProc = NetTransportKey.String("inproc")
	NetTransportOther  = NetTransportKey.String("other")
)

// General attribute keys for spans.
const (
	// Service name of the remote service. Should equal the actual
	// `service.name` resource attribute of the remote service, if any.
	PeerServiceKey = attribute.Key("peer.service")
)

// Semantic conventions for attribute keys used to identify an authorized
// user.
const (
	// Username or the client identifier extracted from the access token or
	// authorization header in the inbound request from outside the system.
	EnduserIDKey = attribute.Key("enduser.id")

	// Actual or assumed role the client is making the request with.
	EnduserRoleKey = attribute.Key("enduser.role")

	// Scopes or granted authorities the client currently possesses.
	EnduserScopeKey = attribute.Key("enduser.scope")
)

// Semantic conventions for attribute keys for HTTP.
const (
	// HTTP request method.
	HTTPMethodKey = attribute.Key("http.method")

	// Full HTTP request URL in the form:
	// scheme://host[:port]/path?query[#fragment].
	HTTPURLKey = attribute.Key("http.url")

	// The full request target as passed in a HTTP request line or
	// equivalent, e.g. "/path/12314/?q=ddds#123".
	HTTPTargetKey = attribute.Key("http.target")

	// The value of the HTTP host header.
	HTTPHostKey = attribute.Key("http.host")

	// The URI scheme identifying the used protocol.
	HTTPSchemeKey = attribute.Key("http.scheme")

	// HTTP response status code.
	HTTPStatusCodeKey = attribute.Key("http.status_code")

	// Kind of HTTP protocol used.
	HTTPFlavorKey = attribute.Key("http.flavor")

	// Value of the HTTP User-Agent header sent by the client.
	HTTPUserAgentKey = attribute.Key("http.user_agent")

	// The primary server name of the matched virtual host.
	HTTPServerNameKey = attribute.Key("http.server_name")

	// The matched route served (path template). For example,
	// "/users/:userID?".
	HTTPRouteKey = attribute.Key("http.route")

	// The IP address of the original client behind all proxies, if known
	// (e.g. from X-Forwarded-For).
	HTTPClientIPKey = attribute.Key("http.client_ip")

	// The size of the request payload body in bytes.
	HTTPRequestContentLengthKey = attribute.Key("http.request_content_length")

	// The size of the uncompressed request payload body after transport decoding.
	// Not set if transport encoding not used.
	HTTPRequestContentLengthUncompressedKey = attribute.Key("http.request_content_length_uncompressed")

	// The size of the response payload body in bytes.
	HTTPResponseContentLengthKey = attribute.Key("http.response_content_length")

	// The size of the uncompressed response payload body after transport decoding.
	// Not set if transport encoding not used.
	HTTPResponseContentLengthUncompressedKey = attribute.Key("http.response_content_length_uncompressed")
)

// Semantic conventions for common HTTP attributes.
var (
	// Semantic conventions for HTTP(S) URI schemes.
	HTTPSchemeHTTP  = HTTPSchemeKey.String("http")
	HTTPSchemeHTTPS = HTTPSchemeKey.String("https")

	// Semantic conventions for HTTP protocols.
	HTTPFlavor1_0  = HTTPFlavorKey.String("1.0")
	HTTPFlavor1_1  = HTTPFlavorKey.String("1.1")
	HTTPFlavor2    = HTTPFlavorKey.String("2")
	HTTPFlavorSPDY = HTTPFlavorKey.String("SPDY")
	HTTPFlavorQUIC = HTTPFlavorKey.String("QUIC")
)

// Semantic conventions for attribute keys for database connections.
const (
	// Identifier for the database system (DBMS) being used.
	DBSystemKey = attribute.Key("db.system")

	// Database Connection String with embedded credentials removed.
	DBConnectionStringKey = attribute.Key("db.connection_string")

	// Username for accessing database.
	DBUserKey = attribute.Key("db.user")
)

// Semantic conventions for common database system attributes.
var (
	DBSystemDB2       = DBSystemKey.String("db2")        // IBM DB2
	DBSystemDerby     = DBSystemKey.String("derby")      // Apache Derby
	DBSystemHive      = DBSystemKey.String("hive")       // Apache Hive
	DBSystemMariaDB   = DBSystemKey.String("mariadb")    // MariaDB
	DBSystemMSSql     = DBSystemKey.String("mssql")      // Microsoft SQL Server
	DBSystemMySQL     = DBSystemKey.String("mysql")      // MySQL
	DBSystemOracle    = DBSystemKey.String("oracle")     // Oracle Database
	DBSystemPostgres  = DBSystemKey.String("postgresql") // PostgreSQL
	DBSystemSqlite    = DBSystemKey.String("sqlite")     // SQLite
	DBSystemTeradata  = DBSystemKey.String("teradata")   // Teradata
	DBSystemOtherSQL  = DBSystemKey.String("other_sql")  // Some other Sql database. Fallback only
	DBSystemCassandra = DBSystemKey.String("cassandra")  // Cassandra
	DBSystemCosmosDB  = DBSystemKey.String("cosmosdb")   // Microsoft Azure CosmosDB
	DBSystemCouchbase = DBSystemKey.String("couchbase")  // Couchbase
	DBSystemCouchDB   = DBSystemKey.String("couchdb")    // CouchDB
	DBSystemDynamoDB  = DBSystemKey.String("dynamodb")   // Amazon DynamoDB
	DBSystemHBase     = DBSystemKey.String("hbase")      // HBase
	DBSystemMongodb   = DBSystemKey.String("mongodb")    // MongoDB
	DBSystemNeo4j     = DBSystemKey.String("neo4j")      // Neo4j
	DBSystemRedis     = DBSystemKey.String("redis")      // Redis
)

// Semantic conventions for attribute keys for database calls.
const (
	// Database instance name.
	DBNameKey = attribute.Key("db.name")

	// A database statement for the given database type.
	DBStatementKey = attribute.Key("db.statement")

	// A database operation for the given database type.
	DBOperationKey = attribute.Key("db.operation")
)

// Database technology-specific attributes
const (
	// Name of the Cassandra keyspace accessed. Use instead of `db.name`.
	DBCassandraKeyspaceKey = attribute.Key("db.cassandra.keyspace")

	// HBase namespace accessed. Use instead of `db.name`.
	DBHBaseNamespaceKey = attribute.Key("db.hbase.namespace")

	// Index of Redis database accessed. Use instead of `db.name`.
	DBRedisDBIndexKey = attribute.Key("db.redis.database_index")

	// Collection being accessed within the database in `db.name`.
	DBMongoDBCollectionKey = attribute.Key("db.mongodb.collection")
)

// Semantic conventions for attribute keys for RPC.
const (
	// A string identifying the remoting system.
	RPCSystemKey = attribute.Key("rpc.system")

	// The full name of the service being called.
	RPCServiceKey = attribute.Key("rpc.service")

	// The name of the method being called.
	RPCMethodKey = attribute.Key("rpc.method")

	// Name of message transmitted or received.
	RPCNameKey = attribute.Key("name")

	// Type of message transmitted or received.
	RPCMessageTypeKey = attribute.Key("message.type")

	// Identifier of message transmitted or received.
	RPCMessageIDKey = attribute.Key("message.id")

	// The compressed size of the message transmitted or received in bytes.
	RPCMessageCompressedSizeKey = attribute.Key("message.compressed_size")

	// The uncompressed size of the message transmitted or received in
	// bytes.
	RPCMessageUncompressedSizeKey = attribute.Key("message.uncompressed_size")
)

// Semantic conventions for common RPC attributes.
var (
	// Semantic convention for gRPC as the remoting system.
	RPCSystemGRPC = RPCSystemKey.String("grpc")

	// Semantic convention for a message named message.
	RPCNameMessage = RPCNameKey.String("message")

	// Semantic conventions for RPC message types.
	RPCMessageTypeSent     = RPCMessageTypeKey.String("SENT")
	RPCMessageTypeReceived = RPCMessageTypeKey.String("RECEIVED")
)

// Semantic conventions for attribute keys for messaging systems.
const (
	// A unique identifier describing the messaging system. For example,
	// kafka, rabbitmq or activemq.
	MessagingSystemKey = attribute.Key("messaging.system")

	// The message destination name, e.g. MyQueue or MyTopic.
	MessagingDestinationKey = attribute.Key("messaging.destination")

	// The kind of message destination.
	MessagingDestinationKindKey = attribute.Key("messaging.destination_kind")

	// Describes if the destination is temporary or not.
	MessagingTempDestinationKey = attribute.Key("messaging.temp_destination")

	// The name of the transport protocol.
	MessagingProtocolKey = attribute.Key("messaging.protocol")

	// The version of the transport protocol.
	MessagingProtocolVersionKey = attribute.Key("messaging.protocol_version")

	// Messaging service URL.
	MessagingURLKey = attribute.Key("messaging.url")

	// Identifier used by the messaging system for a message.
	MessagingMessageIDKey = attribute.Key("messaging.message_id")

	// Identifier used by the messaging system for a conversation.
	MessagingConversationIDKey = attribute.Key("messaging.conversation_id")

	// The (uncompressed) size of the message payload in bytes.
	MessagingMessagePayloadSizeBytesKey = attribute.Key("messaging.message_payload_size_bytes")

	// The compressed size of the message payload in bytes.
	MessagingMessagePayloadCompressedSizeBytesKey = attribute.Key("messaging.message_payload_compressed_size_bytes")

	// Identifies which part and kind of message consumption is being
	// preformed.
	MessagingOperationKey = attribute.Key("messaging.operation")

	// RabbitMQ specific attribute describing the destination routing key.
	MessagingRabbitMQRoutingKeyKey = attribute.Key("messaging.rabbitmq.routing_key")
)

// Semantic conventions for common messaging system attributes.
var (
	// Semantic conventions for message destinations.
	MessagingDestinationKindKeyQueue = MessagingDestinationKindKey.String("queue")
	MessagingDestinationKindKeyTopic = MessagingDestinationKindKey.String("topic")

	// Semantic convention for message destinations that are temporary.
	MessagingTempDestination = MessagingTempDestinationKey.Bool(true)

	// Semantic convention for the operation parts of message consumption.
	// This does not include a "send" attribute as that is explicitly not
	// allowed in the OpenTelemetry specification.
	MessagingOperationReceive = MessagingOperationKey.String("receive")
	MessagingOperationProcess = MessagingOperationKey.String("process")
)

// Semantic conventions for attribute keys for FaaS systems.
const (

	// Type of the trigger on which the function is executed.
	FaaSTriggerKey = attribute.Key("faas.trigger")

	// String containing the execution identifier of the function.
	FaaSExecutionKey = attribute.Key("faas.execution")

	// A boolean indicating that the serverless function is executed
	// for the first time (aka cold start).
	FaaSColdstartKey = attribute.Key("faas.coldstart")

	// The name of the source on which the operation was performed.
	// For example, in Cloud Storage or S3 corresponds to the bucket name,
	// and in Cosmos DB to the database name.
	FaaSDocumentCollectionKey = attribute.Key("faas.document.collection")

	// The type of the operation that was performed on the data.
	FaaSDocumentOperationKey = attribute.Key("faas.document.operation")

	// A string containing the time when the data was accessed.
	FaaSDocumentTimeKey = attribute.Key("faas.document.time")

	// The document name/table subjected to the operation.
	FaaSDocumentNameKey = attribute.Key("faas.document.name")

	// The function invocation time.
	FaaSTimeKey = attribute.Key("faas.time")

	// The schedule period as Cron Expression.
	FaaSCronKey = attribute.Key("faas.cron")
)

// Semantic conventions for common FaaS system attributes.
var (
	// Semantic conventions for the types of triggers.
	FaasTriggerDatasource = FaaSTriggerKey.String("datasource")
	FaasTriggerHTTP       = FaaSTriggerKey.String("http")
	FaasTriggerPubSub     = FaaSTriggerKey.String("pubsub")
	FaasTriggerTimer      = FaaSTriggerKey.String("timer")
	FaasTriggerOther      = FaaSTriggerKey.String("other")

	// Semantic conventions for the types of operations performed.
	FaaSDocumentOperationInsert = FaaSDocumentOperationKey.String("insert")
	FaaSDocumentOperationEdit   = FaaSDocumentOperationKey.String("edit")
	FaaSDocumentOperationDelete = FaaSDocumentOperationKey.String("delete")
)

// Semantic conventions for source code attributes.
const (
	// The method or function name, or equivalent (usually rightmost part of
	// the code unit's name).
	CodeFunctionKey = attribute.Key("code.function")

	// The "namespace" within which `code.function` is defined. Usually the
	// qualified class or module name, such that
	// `code.namespace` + some separator + `code.function` form a unique
	// identifier for the code unit.
	CodeNamespaceKey = attribute.Key("code.namespace")

	// The source code file name that identifies the code unit as uniquely as
	// possible (preferably an absolute file path).
	CodeFilepathKey = attribute.Key("code.filepath")

	// The line number in `code.filepath` best representing the operation.
	// It SHOULD point within the code unit named in `code.function`.
	CodeLineNumberKey = attribute.Key("code.lineno")
)
