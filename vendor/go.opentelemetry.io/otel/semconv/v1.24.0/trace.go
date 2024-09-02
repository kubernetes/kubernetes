// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.24.0"

import "go.opentelemetry.io/otel/attribute"

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
	// Stability: experimental
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
	// Stability: experimental
	// Examples: 'username'
	EnduserIDKey = attribute.Key("enduser.id")

	// EnduserRoleKey is the attribute Key conforming to the "enduser.role"
	// semantic conventions. It represents the actual/assumed role the client
	// is making the request under extracted from token or application security
	// context.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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
	// Stability: experimental
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

// These attributes allow to report this unit of code and therefore to provide
// more context about the span.
const (
	// CodeColumnKey is the attribute Key conforming to the "code.column"
	// semantic conventions. It represents the column number in `code.filepath`
	// best representing the operation. It SHOULD point within the code unit
	// named in `code.function`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 16
	CodeColumnKey = attribute.Key("code.column")

	// CodeFilepathKey is the attribute Key conforming to the "code.filepath"
	// semantic conventions. It represents the source code file name that
	// identifies the code unit as uniquely as possible (preferably an absolute
	// file path).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/usr/local/MyApplication/content_root/app/index.php'
	CodeFilepathKey = attribute.Key("code.filepath")

	// CodeFunctionKey is the attribute Key conforming to the "code.function"
	// semantic conventions. It represents the method or function name, or
	// equivalent (usually rightmost part of the code unit's name).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'serveRequest'
	CodeFunctionKey = attribute.Key("code.function")

	// CodeLineNumberKey is the attribute Key conforming to the "code.lineno"
	// semantic conventions. It represents the line number in `code.filepath`
	// best representing the operation. It SHOULD point within the code unit
	// named in `code.function`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 42
	CodeLineNumberKey = attribute.Key("code.lineno")

	// CodeNamespaceKey is the attribute Key conforming to the "code.namespace"
	// semantic conventions. It represents the "namespace" within which
	// `code.function` is defined. Usually the qualified class or module name,
	// such that `code.namespace` + some separator + `code.function` form a
	// unique identifier for the code unit.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'com.example.MyHTTPService'
	CodeNamespaceKey = attribute.Key("code.namespace")

	// CodeStacktraceKey is the attribute Key conforming to the
	// "code.stacktrace" semantic conventions. It represents a stacktrace as a
	// string in the natural representation for the language runtime. The
	// representation is to be determined and documented by each language SIG.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'at
	// com.example.GenerateTrace.methodB(GenerateTrace.java:13)\\n at '
	//  'com.example.GenerateTrace.methodA(GenerateTrace.java:9)\\n at '
	//  'com.example.GenerateTrace.main(GenerateTrace.java:5)'
	CodeStacktraceKey = attribute.Key("code.stacktrace")
)

// CodeColumn returns an attribute KeyValue conforming to the "code.column"
// semantic conventions. It represents the column number in `code.filepath`
// best representing the operation. It SHOULD point within the code unit named
// in `code.function`.
func CodeColumn(val int) attribute.KeyValue {
	return CodeColumnKey.Int(val)
}

// CodeFilepath returns an attribute KeyValue conforming to the
// "code.filepath" semantic conventions. It represents the source code file
// name that identifies the code unit as uniquely as possible (preferably an
// absolute file path).
func CodeFilepath(val string) attribute.KeyValue {
	return CodeFilepathKey.String(val)
}

// CodeFunction returns an attribute KeyValue conforming to the
// "code.function" semantic conventions. It represents the method or function
// name, or equivalent (usually rightmost part of the code unit's name).
func CodeFunction(val string) attribute.KeyValue {
	return CodeFunctionKey.String(val)
}

// CodeLineNumber returns an attribute KeyValue conforming to the "code.lineno"
// semantic conventions. It represents the line number in `code.filepath` best
// representing the operation. It SHOULD point within the code unit named in
// `code.function`.
func CodeLineNumber(val int) attribute.KeyValue {
	return CodeLineNumberKey.Int(val)
}

// CodeNamespace returns an attribute KeyValue conforming to the
// "code.namespace" semantic conventions. It represents the "namespace" within
// which `code.function` is defined. Usually the qualified class or module
// name, such that `code.namespace` + some separator + `code.function` form a
// unique identifier for the code unit.
func CodeNamespace(val string) attribute.KeyValue {
	return CodeNamespaceKey.String(val)
}

// CodeStacktrace returns an attribute KeyValue conforming to the
// "code.stacktrace" semantic conventions. It represents a stacktrace as a
// string in the natural representation for the language runtime. The
// representation is to be determined and documented by each language SIG.
func CodeStacktrace(val string) attribute.KeyValue {
	return CodeStacktraceKey.String(val)
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
	// Stability: experimental
	// Examples: 42
	ThreadIDKey = attribute.Key("thread.id")

	// ThreadNameKey is the attribute Key conforming to the "thread.name"
	// semantic conventions. It represents the current thread name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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
	// Stability: experimental
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
	// Stability: experimental
	// Examples: '123e4567-e89b-12d3-a456-426614174000', '0001'
	CloudeventsEventIDKey = attribute.Key("cloudevents.event_id")

	// CloudeventsEventSourceKey is the attribute Key conforming to the
	// "cloudevents.event_source" semantic conventions. It represents the
	// [source](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1)
	// identifies the context in which an event happened.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
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
	// Stability: experimental
	// Examples: '1.0'
	CloudeventsEventSpecVersionKey = attribute.Key("cloudevents.event_spec_version")

	// CloudeventsEventSubjectKey is the attribute Key conforming to the
	// "cloudevents.event_subject" semantic conventions. It represents the
	// [subject](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject)
	// of the event in the context of the event producer (identified by
	// source).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'mynewfile.jpg'
	CloudeventsEventSubjectKey = attribute.Key("cloudevents.event_subject")

	// CloudeventsEventTypeKey is the attribute Key conforming to the
	// "cloudevents.event_type" semantic conventions. It represents the
	// [event_type](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type)
	// contains a value describing the type of event related to the originating
	// occurrence.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'com.github.pull_request.opened',
	// 'com.example.object.deleted.v2'
	CloudeventsEventTypeKey = attribute.Key("cloudevents.event_type")
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

// CloudeventsEventSubject returns an attribute KeyValue conforming to the
// "cloudevents.event_subject" semantic conventions. It represents the
// [subject](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject)
// of the event in the context of the event producer (identified by source).
func CloudeventsEventSubject(val string) attribute.KeyValue {
	return CloudeventsEventSubjectKey.String(val)
}

// CloudeventsEventType returns an attribute KeyValue conforming to the
// "cloudevents.event_type" semantic conventions. It represents the
// [event_type](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type)
// contains a value describing the type of event related to the originating
// occurrence.
func CloudeventsEventType(val string) attribute.KeyValue {
	return CloudeventsEventTypeKey.String(val)
}

// Semantic conventions for the OpenTracing Shim
const (
	// OpentracingRefTypeKey is the attribute Key conforming to the
	// "opentracing.ref_type" semantic conventions. It represents the
	// parent-child Reference type
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Note: The causal relationship between a child Span and a parent Span.
	OpentracingRefTypeKey = attribute.Key("opentracing.ref_type")
)

var (
	// The parent Span depends on the child Span in some capacity
	OpentracingRefTypeChildOf = OpentracingRefTypeKey.String("child_of")
	// The parent Span doesn't depend in any way on the result of the child Span
	OpentracingRefTypeFollowsFrom = OpentracingRefTypeKey.String("follows_from")
)

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
	// Stability: experimental
	OTelStatusCodeKey = attribute.Key("otel.status_code")

	// OTelStatusDescriptionKey is the attribute Key conforming to the
	// "otel.status_description" semantic conventions. It represents the
	// description of the Status if it has a value, otherwise not set.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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
	// FaaSInvocationIDKey is the attribute Key conforming to the
	// "faas.invocation_id" semantic conventions. It represents the invocation
	// ID of the current function invocation.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'af9d5aa4-a685-4c5f-a22b-444f80b3cc28'
	FaaSInvocationIDKey = attribute.Key("faas.invocation_id")
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
	// Stability: experimental
	// Examples: 'myBucketName', 'myDBName'
	FaaSDocumentCollectionKey = attribute.Key("faas.document.collection")

	// FaaSDocumentNameKey is the attribute Key conforming to the
	// "faas.document.name" semantic conventions. It represents the document
	// name/table subjected to the operation. For example, in Cloud Storage or
	// S3 is the name of the file, and in Cosmos DB the table name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myFile.txt', 'myTableName'
	FaaSDocumentNameKey = attribute.Key("faas.document.name")

	// FaaSDocumentOperationKey is the attribute Key conforming to the
	// "faas.document.operation" semantic conventions. It represents the
	// describes the type of the operation that was performed on the data.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: experimental
	FaaSDocumentOperationKey = attribute.Key("faas.document.operation")

	// FaaSDocumentTimeKey is the attribute Key conforming to the
	// "faas.document.time" semantic conventions. It represents a string
	// containing the time when the data was accessed in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
	// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2020-01-23T13:47:06Z'
	FaaSDocumentTimeKey = attribute.Key("faas.document.time")
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

// FaaSDocumentName returns an attribute KeyValue conforming to the
// "faas.document.name" semantic conventions. It represents the document
// name/table subjected to the operation. For example, in Cloud Storage or S3
// is the name of the file, and in Cosmos DB the table name.
func FaaSDocumentName(val string) attribute.KeyValue {
	return FaaSDocumentNameKey.String(val)
}

// FaaSDocumentTime returns an attribute KeyValue conforming to the
// "faas.document.time" semantic conventions. It represents a string containing
// the time when the data was accessed in the [ISO
// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
func FaaSDocumentTime(val string) attribute.KeyValue {
	return FaaSDocumentTimeKey.String(val)
}

// Semantic Convention for FaaS scheduled to be executed regularly.
const (
	// FaaSCronKey is the attribute Key conforming to the "faas.cron" semantic
	// conventions. It represents a string containing the schedule period as
	// [Cron
	// Expression](https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '0/5 * * * ? *'
	FaaSCronKey = attribute.Key("faas.cron")

	// FaaSTimeKey is the attribute Key conforming to the "faas.time" semantic
	// conventions. It represents a string containing the function invocation
	// time in the [ISO
	// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
	// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2020-01-23T13:47:06Z'
	FaaSTimeKey = attribute.Key("faas.time")
)

// FaaSCron returns an attribute KeyValue conforming to the "faas.cron"
// semantic conventions. It represents a string containing the schedule period
// as [Cron
// Expression](https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm).
func FaaSCron(val string) attribute.KeyValue {
	return FaaSCronKey.String(val)
}

// FaaSTime returns an attribute KeyValue conforming to the "faas.time"
// semantic conventions. It represents a string containing the function
// invocation time in the [ISO
// 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format
// expressed in [UTC](https://www.w3.org/TR/NOTE-datetime).
func FaaSTime(val string) attribute.KeyValue {
	return FaaSTimeKey.String(val)
}

// Contains additional attributes for incoming FaaS spans.
const (
	// FaaSColdstartKey is the attribute Key conforming to the "faas.coldstart"
	// semantic conventions. It represents a boolean that is true if the
	// serverless function is executed for the first time (aka cold-start).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	FaaSColdstartKey = attribute.Key("faas.coldstart")
)

// FaaSColdstart returns an attribute KeyValue conforming to the
// "faas.coldstart" semantic conventions. It represents a boolean that is true
// if the serverless function is executed for the first time (aka cold-start).
func FaaSColdstart(val bool) attribute.KeyValue {
	return FaaSColdstartKey.Bool(val)
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
	// Stability: experimental
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
	// AWSDynamoDBAttributesToGetKey is the attribute Key conforming to the
	// "aws.dynamodb.attributes_to_get" semantic conventions. It represents the
	// value of the `AttributesToGet` request parameter.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'lives', 'id'
	AWSDynamoDBAttributesToGetKey = attribute.Key("aws.dynamodb.attributes_to_get")

	// AWSDynamoDBConsistentReadKey is the attribute Key conforming to the
	// "aws.dynamodb.consistent_read" semantic conventions. It represents the
	// value of the `ConsistentRead` request parameter.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	AWSDynamoDBConsistentReadKey = attribute.Key("aws.dynamodb.consistent_read")

	// AWSDynamoDBConsumedCapacityKey is the attribute Key conforming to the
	// "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
	// JSON-serialized value of each item in the `ConsumedCapacity` response
	// field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '{ "CapacityUnits": number, "GlobalSecondaryIndexes": {
	// "string" : { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number } }, "LocalSecondaryIndexes": { "string" :
	// { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number } }, "ReadCapacityUnits": number, "Table":
	// { "CapacityUnits": number, "ReadCapacityUnits": number,
	// "WriteCapacityUnits": number }, "TableName": "string",
	// "WriteCapacityUnits": number }'
	AWSDynamoDBConsumedCapacityKey = attribute.Key("aws.dynamodb.consumed_capacity")

	// AWSDynamoDBIndexNameKey is the attribute Key conforming to the
	// "aws.dynamodb.index_name" semantic conventions. It represents the value
	// of the `IndexName` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'name_to_group'
	AWSDynamoDBIndexNameKey = attribute.Key("aws.dynamodb.index_name")

	// AWSDynamoDBItemCollectionMetricsKey is the attribute Key conforming to
	// the "aws.dynamodb.item_collection_metrics" semantic conventions. It
	// represents the JSON-serialized value of the `ItemCollectionMetrics`
	// response field.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '{ "string" : [ { "ItemCollectionKey": { "string" : { "B":
	// blob, "BOOL": boolean, "BS": [ blob ], "L": [ "AttributeValue" ], "M": {
	// "string" : "AttributeValue" }, "N": "string", "NS": [ "string" ],
	// "NULL": boolean, "S": "string", "SS": [ "string" ] } },
	// "SizeEstimateRangeGB": [ number ] } ] }'
	AWSDynamoDBItemCollectionMetricsKey = attribute.Key("aws.dynamodb.item_collection_metrics")

	// AWSDynamoDBLimitKey is the attribute Key conforming to the
	// "aws.dynamodb.limit" semantic conventions. It represents the value of
	// the `Limit` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 10
	AWSDynamoDBLimitKey = attribute.Key("aws.dynamodb.limit")

	// AWSDynamoDBProjectionKey is the attribute Key conforming to the
	// "aws.dynamodb.projection" semantic conventions. It represents the value
	// of the `ProjectionExpression` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'Title', 'Title, Price, Color', 'Title, Description,
	// RelatedItems, ProductReviews'
	AWSDynamoDBProjectionKey = attribute.Key("aws.dynamodb.projection")

	// AWSDynamoDBProvisionedReadCapacityKey is the attribute Key conforming to
	// the "aws.dynamodb.provisioned_read_capacity" semantic conventions. It
	// represents the value of the `ProvisionedThroughput.ReadCapacityUnits`
	// request parameter.
	//
	// Type: double
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedReadCapacityKey = attribute.Key("aws.dynamodb.provisioned_read_capacity")

	// AWSDynamoDBProvisionedWriteCapacityKey is the attribute Key conforming
	// to the "aws.dynamodb.provisioned_write_capacity" semantic conventions.
	// It represents the value of the
	// `ProvisionedThroughput.WriteCapacityUnits` request parameter.
	//
	// Type: double
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedWriteCapacityKey = attribute.Key("aws.dynamodb.provisioned_write_capacity")

	// AWSDynamoDBSelectKey is the attribute Key conforming to the
	// "aws.dynamodb.select" semantic conventions. It represents the value of
	// the `Select` request parameter.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'ALL_ATTRIBUTES', 'COUNT'
	AWSDynamoDBSelectKey = attribute.Key("aws.dynamodb.select")

	// AWSDynamoDBTableNamesKey is the attribute Key conforming to the
	// "aws.dynamodb.table_names" semantic conventions. It represents the keys
	// in the `RequestItems` object field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'Users', 'Cats'
	AWSDynamoDBTableNamesKey = attribute.Key("aws.dynamodb.table_names")
)

// AWSDynamoDBAttributesToGet returns an attribute KeyValue conforming to
// the "aws.dynamodb.attributes_to_get" semantic conventions. It represents the
// value of the `AttributesToGet` request parameter.
func AWSDynamoDBAttributesToGet(val ...string) attribute.KeyValue {
	return AWSDynamoDBAttributesToGetKey.StringSlice(val)
}

// AWSDynamoDBConsistentRead returns an attribute KeyValue conforming to the
// "aws.dynamodb.consistent_read" semantic conventions. It represents the value
// of the `ConsistentRead` request parameter.
func AWSDynamoDBConsistentRead(val bool) attribute.KeyValue {
	return AWSDynamoDBConsistentReadKey.Bool(val)
}

// AWSDynamoDBConsumedCapacity returns an attribute KeyValue conforming to
// the "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
// JSON-serialized value of each item in the `ConsumedCapacity` response field.
func AWSDynamoDBConsumedCapacity(val ...string) attribute.KeyValue {
	return AWSDynamoDBConsumedCapacityKey.StringSlice(val)
}

// AWSDynamoDBIndexName returns an attribute KeyValue conforming to the
// "aws.dynamodb.index_name" semantic conventions. It represents the value of
// the `IndexName` request parameter.
func AWSDynamoDBIndexName(val string) attribute.KeyValue {
	return AWSDynamoDBIndexNameKey.String(val)
}

// AWSDynamoDBItemCollectionMetrics returns an attribute KeyValue conforming
// to the "aws.dynamodb.item_collection_metrics" semantic conventions. It
// represents the JSON-serialized value of the `ItemCollectionMetrics` response
// field.
func AWSDynamoDBItemCollectionMetrics(val string) attribute.KeyValue {
	return AWSDynamoDBItemCollectionMetricsKey.String(val)
}

// AWSDynamoDBLimit returns an attribute KeyValue conforming to the
// "aws.dynamodb.limit" semantic conventions. It represents the value of the
// `Limit` request parameter.
func AWSDynamoDBLimit(val int) attribute.KeyValue {
	return AWSDynamoDBLimitKey.Int(val)
}

// AWSDynamoDBProjection returns an attribute KeyValue conforming to the
// "aws.dynamodb.projection" semantic conventions. It represents the value of
// the `ProjectionExpression` request parameter.
func AWSDynamoDBProjection(val string) attribute.KeyValue {
	return AWSDynamoDBProjectionKey.String(val)
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

// AWSDynamoDBSelect returns an attribute KeyValue conforming to the
// "aws.dynamodb.select" semantic conventions. It represents the value of the
// `Select` request parameter.
func AWSDynamoDBSelect(val string) attribute.KeyValue {
	return AWSDynamoDBSelectKey.String(val)
}

// AWSDynamoDBTableNames returns an attribute KeyValue conforming to the
// "aws.dynamodb.table_names" semantic conventions. It represents the keys in
// the `RequestItems` object field.
func AWSDynamoDBTableNames(val ...string) attribute.KeyValue {
	return AWSDynamoDBTableNamesKey.StringSlice(val)
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
	// Stability: experimental
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
	// Stability: experimental
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
	// Stability: experimental
	// Examples: 'Users', 'CatsTable'
	AWSDynamoDBExclusiveStartTableKey = attribute.Key("aws.dynamodb.exclusive_start_table")

	// AWSDynamoDBTableCountKey is the attribute Key conforming to the
	// "aws.dynamodb.table_count" semantic conventions. It represents the the
	// number of items in the `TableNames` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
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
	// Stability: experimental
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
	// AWSDynamoDBCountKey is the attribute Key conforming to the
	// "aws.dynamodb.count" semantic conventions. It represents the value of
	// the `Count` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 10
	AWSDynamoDBCountKey = attribute.Key("aws.dynamodb.count")

	// AWSDynamoDBScannedCountKey is the attribute Key conforming to the
	// "aws.dynamodb.scanned_count" semantic conventions. It represents the
	// value of the `ScannedCount` response parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 50
	AWSDynamoDBScannedCountKey = attribute.Key("aws.dynamodb.scanned_count")

	// AWSDynamoDBSegmentKey is the attribute Key conforming to the
	// "aws.dynamodb.segment" semantic conventions. It represents the value of
	// the `Segment` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 10
	AWSDynamoDBSegmentKey = attribute.Key("aws.dynamodb.segment")

	// AWSDynamoDBTotalSegmentsKey is the attribute Key conforming to the
	// "aws.dynamodb.total_segments" semantic conventions. It represents the
	// value of the `TotalSegments` request parameter.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 100
	AWSDynamoDBTotalSegmentsKey = attribute.Key("aws.dynamodb.total_segments")
)

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

// DynamoDB.UpdateTable
const (
	// AWSDynamoDBAttributeDefinitionsKey is the attribute Key conforming to
	// the "aws.dynamodb.attribute_definitions" semantic conventions. It
	// represents the JSON-serialized value of each item in the
	// `AttributeDefinitions` request field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '{ "AttributeName": "string", "AttributeType": "string" }'
	AWSDynamoDBAttributeDefinitionsKey = attribute.Key("aws.dynamodb.attribute_definitions")

	// AWSDynamoDBGlobalSecondaryIndexUpdatesKey is the attribute Key
	// conforming to the "aws.dynamodb.global_secondary_index_updates" semantic
	// conventions. It represents the JSON-serialized value of each item in the
	// the `GlobalSecondaryIndexUpdates` request field.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
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
	// Stability: experimental
	// Examples: 'some-bucket-name'
	// Note: The `bucket` attribute is applicable to all S3 operations that
	// reference a bucket, i.e. that require the bucket name as a mandatory
	// parameter.
	// This applies to almost all S3 operations except `list-buckets`.
	AWSS3BucketKey = attribute.Key("aws.s3.bucket")

	// AWSS3CopySourceKey is the attribute Key conforming to the
	// "aws.s3.copy_source" semantic conventions. It represents the source
	// object (in the form `bucket`/`key`) for the copy operation.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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

	// AWSS3DeleteKey is the attribute Key conforming to the "aws.s3.delete"
	// semantic conventions. It represents the delete request container that
	// specifies the objects to be deleted.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// 'Objects=[{Key=string,VersionID=string},{Key=string,VersionID=string}],Quiet=boolean'
	// Note: The `delete` attribute is only applicable to the
	// [delete-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html)
	// operation.
	// The `delete` attribute corresponds to the `--delete` parameter of the
	// [delete-objects operation within the S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html).
	AWSS3DeleteKey = attribute.Key("aws.s3.delete")

	// AWSS3KeyKey is the attribute Key conforming to the "aws.s3.key" semantic
	// conventions. It represents the S3 object key the request refers to.
	// Corresponds to the `--key` parameter of the [S3
	// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
	// operations.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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

	// AWSS3PartNumberKey is the attribute Key conforming to the
	// "aws.s3.part_number" semantic conventions. It represents the part number
	// of the part being uploaded in a multipart-upload operation. This is a
	// positive integer between 1 and 10,000.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
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

	// AWSS3UploadIDKey is the attribute Key conforming to the
	// "aws.s3.upload_id" semantic conventions. It represents the upload ID
	// that identifies the multipart upload.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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
)

// AWSS3Bucket returns an attribute KeyValue conforming to the
// "aws.s3.bucket" semantic conventions. It represents the S3 bucket name the
// request refers to. Corresponds to the `--bucket` parameter of the [S3
// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
// operations.
func AWSS3Bucket(val string) attribute.KeyValue {
	return AWSS3BucketKey.String(val)
}

// AWSS3CopySource returns an attribute KeyValue conforming to the
// "aws.s3.copy_source" semantic conventions. It represents the source object
// (in the form `bucket`/`key`) for the copy operation.
func AWSS3CopySource(val string) attribute.KeyValue {
	return AWSS3CopySourceKey.String(val)
}

// AWSS3Delete returns an attribute KeyValue conforming to the
// "aws.s3.delete" semantic conventions. It represents the delete request
// container that specifies the objects to be deleted.
func AWSS3Delete(val string) attribute.KeyValue {
	return AWSS3DeleteKey.String(val)
}

// AWSS3Key returns an attribute KeyValue conforming to the "aws.s3.key"
// semantic conventions. It represents the S3 object key the request refers to.
// Corresponds to the `--key` parameter of the [S3
// API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html)
// operations.
func AWSS3Key(val string) attribute.KeyValue {
	return AWSS3KeyKey.String(val)
}

// AWSS3PartNumber returns an attribute KeyValue conforming to the
// "aws.s3.part_number" semantic conventions. It represents the part number of
// the part being uploaded in a multipart-upload operation. This is a positive
// integer between 1 and 10,000.
func AWSS3PartNumber(val int) attribute.KeyValue {
	return AWSS3PartNumberKey.Int(val)
}

// AWSS3UploadID returns an attribute KeyValue conforming to the
// "aws.s3.upload_id" semantic conventions. It represents the upload ID that
// identifies the multipart upload.
func AWSS3UploadID(val string) attribute.KeyValue {
	return AWSS3UploadIDKey.String(val)
}

// Semantic conventions to apply when instrumenting the GraphQL implementation.
// They map GraphQL operations to attributes on a Span.
const (
	// GraphqlDocumentKey is the attribute Key conforming to the
	// "graphql.document" semantic conventions. It represents the GraphQL
	// document being executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'query findBookByID { bookByID(id: ?) { name } }'
	// Note: The value may be sanitized to exclude sensitive information.
	GraphqlDocumentKey = attribute.Key("graphql.document")

	// GraphqlOperationNameKey is the attribute Key conforming to the
	// "graphql.operation.name" semantic conventions. It represents the name of
	// the operation being executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'findBookByID'
	GraphqlOperationNameKey = attribute.Key("graphql.operation.name")

	// GraphqlOperationTypeKey is the attribute Key conforming to the
	// "graphql.operation.type" semantic conventions. It represents the type of
	// the operation being executed.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'query', 'mutation', 'subscription'
	GraphqlOperationTypeKey = attribute.Key("graphql.operation.type")
)

var (
	// GraphQL query
	GraphqlOperationTypeQuery = GraphqlOperationTypeKey.String("query")
	// GraphQL mutation
	GraphqlOperationTypeMutation = GraphqlOperationTypeKey.String("mutation")
	// GraphQL subscription
	GraphqlOperationTypeSubscription = GraphqlOperationTypeKey.String("subscription")
)

// GraphqlDocument returns an attribute KeyValue conforming to the
// "graphql.document" semantic conventions. It represents the GraphQL document
// being executed.
func GraphqlDocument(val string) attribute.KeyValue {
	return GraphqlDocumentKey.String(val)
}

// GraphqlOperationName returns an attribute KeyValue conforming to the
// "graphql.operation.name" semantic conventions. It represents the name of the
// operation being executed.
func GraphqlOperationName(val string) attribute.KeyValue {
	return GraphqlOperationNameKey.String(val)
}
