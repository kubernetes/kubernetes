// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.24.0"

import "go.opentelemetry.io/otel/attribute"

// Describes FaaS attributes.
const (
	// FaaSInvokedNameKey is the attribute Key conforming to the
	// "faas.invoked_name" semantic conventions. It represents the name of the
	// invoked function.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
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
	// Stability: experimental
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
	// Stability: experimental
	// Examples: 'eu-central-1'
	// Note: SHOULD be equal to the `cloud.region` resource attribute of the
	// invoked function.
	FaaSInvokedRegionKey = attribute.Key("faas.invoked_region")

	// FaaSTriggerKey is the attribute Key conforming to the "faas.trigger"
	// semantic conventions. It represents the type of the trigger which caused
	// this function invocation.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	FaaSTriggerKey = attribute.Key("faas.trigger")
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

// Attributes for Events represented using Log Records.
const (
	// EventNameKey is the attribute Key conforming to the "event.name"
	// semantic conventions. It represents the identifies the class / type of
	// event.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'browser.mouse.click', 'device.app.lifecycle'
	// Note: Event names are subject to the same rules as [attribute
	// names](https://github.com/open-telemetry/opentelemetry-specification/tree/v1.26.0/specification/common/attribute-naming.md).
	// Notably, event names are namespaced to avoid collisions and provide a
	// clean separation of semantics for events in separate domains like
	// browser, mobile, and kubernetes.
	EventNameKey = attribute.Key("event.name")
)

// EventName returns an attribute KeyValue conforming to the "event.name"
// semantic conventions. It represents the identifies the class / type of
// event.
func EventName(val string) attribute.KeyValue {
	return EventNameKey.String(val)
}

// The attributes described in this section are rather generic. They may be
// used in any Log Record they apply to.
const (
	// LogRecordUIDKey is the attribute Key conforming to the "log.record.uid"
	// semantic conventions. It represents a unique identifier for the Log
	// Record.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '01ARZ3NDEKTSV4RRFFQ69G5FAV'
	// Note: If an id is provided, other log records with the same id will be
	// considered duplicates and can be removed safely. This means, that two
	// distinguishable log records MUST have different values.
	// The id MAY be an [Universally Unique Lexicographically Sortable
	// Identifier (ULID)](https://github.com/ulid/spec), but other identifiers
	// (e.g. UUID) may be used as needed.
	LogRecordUIDKey = attribute.Key("log.record.uid")
)

// LogRecordUID returns an attribute KeyValue conforming to the
// "log.record.uid" semantic conventions. It represents a unique identifier for
// the Log Record.
func LogRecordUID(val string) attribute.KeyValue {
	return LogRecordUIDKey.String(val)
}

// Describes Log attributes
const (
	// LogIostreamKey is the attribute Key conforming to the "log.iostream"
	// semantic conventions. It represents the stream associated with the log.
	// See below for a list of well-known values.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	LogIostreamKey = attribute.Key("log.iostream")
)

var (
	// Logs from stdout stream
	LogIostreamStdout = LogIostreamKey.String("stdout")
	// Events from stderr stream
	LogIostreamStderr = LogIostreamKey.String("stderr")
)

// A file to which log was emitted.
const (
	// LogFileNameKey is the attribute Key conforming to the "log.file.name"
	// semantic conventions. It represents the basename of the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: experimental
	// Examples: 'audit.log'
	LogFileNameKey = attribute.Key("log.file.name")

	// LogFileNameResolvedKey is the attribute Key conforming to the
	// "log.file.name_resolved" semantic conventions. It represents the
	// basename of the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'uuid.log'
	LogFileNameResolvedKey = attribute.Key("log.file.name_resolved")

	// LogFilePathKey is the attribute Key conforming to the "log.file.path"
	// semantic conventions. It represents the full path to the file.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/var/log/mysql/audit.log'
	LogFilePathKey = attribute.Key("log.file.path")

	// LogFilePathResolvedKey is the attribute Key conforming to the
	// "log.file.path_resolved" semantic conventions. It represents the full
	// path to the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/var/lib/docker/uuid.log'
	LogFilePathResolvedKey = attribute.Key("log.file.path_resolved")
)

// LogFileName returns an attribute KeyValue conforming to the
// "log.file.name" semantic conventions. It represents the basename of the
// file.
func LogFileName(val string) attribute.KeyValue {
	return LogFileNameKey.String(val)
}

// LogFileNameResolved returns an attribute KeyValue conforming to the
// "log.file.name_resolved" semantic conventions. It represents the basename of
// the file, with symlinks resolved.
func LogFileNameResolved(val string) attribute.KeyValue {
	return LogFileNameResolvedKey.String(val)
}

// LogFilePath returns an attribute KeyValue conforming to the
// "log.file.path" semantic conventions. It represents the full path to the
// file.
func LogFilePath(val string) attribute.KeyValue {
	return LogFilePathKey.String(val)
}

// LogFilePathResolved returns an attribute KeyValue conforming to the
// "log.file.path_resolved" semantic conventions. It represents the full path
// to the file, with symlinks resolved.
func LogFilePathResolved(val string) attribute.KeyValue {
	return LogFilePathResolvedKey.String(val)
}

// Describes Database attributes
const (
	// PoolNameKey is the attribute Key conforming to the "pool.name" semantic
	// conventions. It represents the name of the connection pool; unique
	// within the instrumented application. In case the connection pool
	// implementation doesn't provide a name, then the
	// [db.connection_string](/docs/database/database-spans.md#connection-level-attributes)
	// should be used
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'myDataSource'
	PoolNameKey = attribute.Key("pool.name")

	// StateKey is the attribute Key conforming to the "state" semantic
	// conventions. It represents the state of a connection in the pool
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'idle'
	StateKey = attribute.Key("state")
)

var (
	// idle
	StateIdle = StateKey.String("idle")
	// used
	StateUsed = StateKey.String("used")
)

// PoolName returns an attribute KeyValue conforming to the "pool.name"
// semantic conventions. It represents the name of the connection pool; unique
// within the instrumented application. In case the connection pool
// implementation doesn't provide a name, then the
// [db.connection_string](/docs/database/database-spans.md#connection-level-attributes)
// should be used
func PoolName(val string) attribute.KeyValue {
	return PoolNameKey.String(val)
}

// ASP.NET Core attributes
const (
	// AspnetcoreDiagnosticsHandlerTypeKey is the attribute Key conforming to
	// the "aspnetcore.diagnostics.handler.type" semantic conventions. It
	// represents the full type name of the
	// [`IExceptionHandler`](https://learn.microsoft.com/dotnet/api/microsoft.aspnetcore.diagnostics.iexceptionhandler)
	// implementation that handled the exception.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (if and only if the exception
	// was handled by this handler.)
	// Stability: experimental
	// Examples: 'Contoso.MyHandler'
	AspnetcoreDiagnosticsHandlerTypeKey = attribute.Key("aspnetcore.diagnostics.handler.type")

	// AspnetcoreRateLimitingPolicyKey is the attribute Key conforming to the
	// "aspnetcore.rate_limiting.policy" semantic conventions. It represents
	// the rate limiting policy name.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (if the matched endpoint for the
	// request had a rate-limiting policy.)
	// Stability: experimental
	// Examples: 'fixed', 'sliding', 'token'
	AspnetcoreRateLimitingPolicyKey = attribute.Key("aspnetcore.rate_limiting.policy")

	// AspnetcoreRateLimitingResultKey is the attribute Key conforming to the
	// "aspnetcore.rate_limiting.result" semantic conventions. It represents
	// the rate-limiting result, shows whether the lease was acquired or
	// contains a rejection reason
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'acquired', 'request_canceled'
	AspnetcoreRateLimitingResultKey = attribute.Key("aspnetcore.rate_limiting.result")

	// AspnetcoreRequestIsUnhandledKey is the attribute Key conforming to the
	// "aspnetcore.request.is_unhandled" semantic conventions. It represents
	// the flag indicating if request was handled by the application pipeline.
	//
	// Type: boolean
	// RequirementLevel: ConditionallyRequired (if and only if the request was
	// not handled.)
	// Stability: experimental
	// Examples: True
	AspnetcoreRequestIsUnhandledKey = attribute.Key("aspnetcore.request.is_unhandled")

	// AspnetcoreRoutingIsFallbackKey is the attribute Key conforming to the
	// "aspnetcore.routing.is_fallback" semantic conventions. It represents a
	// value that indicates whether the matched route is a fallback route.
	//
	// Type: boolean
	// RequirementLevel: ConditionallyRequired (If and only if a route was
	// successfully matched.)
	// Stability: experimental
	// Examples: True
	AspnetcoreRoutingIsFallbackKey = attribute.Key("aspnetcore.routing.is_fallback")
)

var (
	// Lease was acquired
	AspnetcoreRateLimitingResultAcquired = AspnetcoreRateLimitingResultKey.String("acquired")
	// Lease request was rejected by the endpoint limiter
	AspnetcoreRateLimitingResultEndpointLimiter = AspnetcoreRateLimitingResultKey.String("endpoint_limiter")
	// Lease request was rejected by the global limiter
	AspnetcoreRateLimitingResultGlobalLimiter = AspnetcoreRateLimitingResultKey.String("global_limiter")
	// Lease request was canceled
	AspnetcoreRateLimitingResultRequestCanceled = AspnetcoreRateLimitingResultKey.String("request_canceled")
)

// AspnetcoreDiagnosticsHandlerType returns an attribute KeyValue conforming
// to the "aspnetcore.diagnostics.handler.type" semantic conventions. It
// represents the full type name of the
// [`IExceptionHandler`](https://learn.microsoft.com/dotnet/api/microsoft.aspnetcore.diagnostics.iexceptionhandler)
// implementation that handled the exception.
func AspnetcoreDiagnosticsHandlerType(val string) attribute.KeyValue {
	return AspnetcoreDiagnosticsHandlerTypeKey.String(val)
}

// AspnetcoreRateLimitingPolicy returns an attribute KeyValue conforming to
// the "aspnetcore.rate_limiting.policy" semantic conventions. It represents
// the rate limiting policy name.
func AspnetcoreRateLimitingPolicy(val string) attribute.KeyValue {
	return AspnetcoreRateLimitingPolicyKey.String(val)
}

// AspnetcoreRequestIsUnhandled returns an attribute KeyValue conforming to
// the "aspnetcore.request.is_unhandled" semantic conventions. It represents
// the flag indicating if request was handled by the application pipeline.
func AspnetcoreRequestIsUnhandled(val bool) attribute.KeyValue {
	return AspnetcoreRequestIsUnhandledKey.Bool(val)
}

// AspnetcoreRoutingIsFallback returns an attribute KeyValue conforming to
// the "aspnetcore.routing.is_fallback" semantic conventions. It represents a
// value that indicates whether the matched route is a fallback route.
func AspnetcoreRoutingIsFallback(val bool) attribute.KeyValue {
	return AspnetcoreRoutingIsFallbackKey.Bool(val)
}

// SignalR attributes
const (
	// SignalrConnectionStatusKey is the attribute Key conforming to the
	// "signalr.connection.status" semantic conventions. It represents the
	// signalR HTTP connection closure status.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'app_shutdown', 'timeout'
	SignalrConnectionStatusKey = attribute.Key("signalr.connection.status")

	// SignalrTransportKey is the attribute Key conforming to the
	// "signalr.transport" semantic conventions. It represents the [SignalR
	// transport
	// type](https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/TransportProtocols.md)
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'web_sockets', 'long_polling'
	SignalrTransportKey = attribute.Key("signalr.transport")
)

var (
	// The connection was closed normally
	SignalrConnectionStatusNormalClosure = SignalrConnectionStatusKey.String("normal_closure")
	// The connection was closed due to a timeout
	SignalrConnectionStatusTimeout = SignalrConnectionStatusKey.String("timeout")
	// The connection was closed because the app is shutting down
	SignalrConnectionStatusAppShutdown = SignalrConnectionStatusKey.String("app_shutdown")
)

var (
	// ServerSentEvents protocol
	SignalrTransportServerSentEvents = SignalrTransportKey.String("server_sent_events")
	// LongPolling protocol
	SignalrTransportLongPolling = SignalrTransportKey.String("long_polling")
	// WebSockets protocol
	SignalrTransportWebSockets = SignalrTransportKey.String("web_sockets")
)

// Describes JVM buffer metric attributes.
const (
	// JvmBufferPoolNameKey is the attribute Key conforming to the
	// "jvm.buffer.pool.name" semantic conventions. It represents the name of
	// the buffer pool.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: experimental
	// Examples: 'mapped', 'direct'
	// Note: Pool names are generally obtained via
	// [BufferPoolMXBean#getName()](https://docs.oracle.com/en/java/javase/11/docs/api/java.management/java/lang/management/BufferPoolMXBean.html#getName()).
	JvmBufferPoolNameKey = attribute.Key("jvm.buffer.pool.name")
)

// JvmBufferPoolName returns an attribute KeyValue conforming to the
// "jvm.buffer.pool.name" semantic conventions. It represents the name of the
// buffer pool.
func JvmBufferPoolName(val string) attribute.KeyValue {
	return JvmBufferPoolNameKey.String(val)
}

// Describes JVM memory metric attributes.
const (
	// JvmMemoryPoolNameKey is the attribute Key conforming to the
	// "jvm.memory.pool.name" semantic conventions. It represents the name of
	// the memory pool.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'G1 Old Gen', 'G1 Eden space', 'G1 Survivor Space'
	// Note: Pool names are generally obtained via
	// [MemoryPoolMXBean#getName()](https://docs.oracle.com/en/java/javase/11/docs/api/java.management/java/lang/management/MemoryPoolMXBean.html#getName()).
	JvmMemoryPoolNameKey = attribute.Key("jvm.memory.pool.name")

	// JvmMemoryTypeKey is the attribute Key conforming to the
	// "jvm.memory.type" semantic conventions. It represents the type of
	// memory.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'heap', 'non_heap'
	JvmMemoryTypeKey = attribute.Key("jvm.memory.type")
)

var (
	// Heap memory
	JvmMemoryTypeHeap = JvmMemoryTypeKey.String("heap")
	// Non-heap memory
	JvmMemoryTypeNonHeap = JvmMemoryTypeKey.String("non_heap")
)

// JvmMemoryPoolName returns an attribute KeyValue conforming to the
// "jvm.memory.pool.name" semantic conventions. It represents the name of the
// memory pool.
func JvmMemoryPoolName(val string) attribute.KeyValue {
	return JvmMemoryPoolNameKey.String(val)
}

// Describes System metric attributes
const (
	// SystemDeviceKey is the attribute Key conforming to the "system.device"
	// semantic conventions. It represents the device identifier
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '(identifier)'
	SystemDeviceKey = attribute.Key("system.device")
)

// SystemDevice returns an attribute KeyValue conforming to the
// "system.device" semantic conventions. It represents the device identifier
func SystemDevice(val string) attribute.KeyValue {
	return SystemDeviceKey.String(val)
}

// Describes System CPU metric attributes
const (
	// SystemCPULogicalNumberKey is the attribute Key conforming to the
	// "system.cpu.logical_number" semantic conventions. It represents the
	// logical CPU number [0..n-1]
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1
	SystemCPULogicalNumberKey = attribute.Key("system.cpu.logical_number")

	// SystemCPUStateKey is the attribute Key conforming to the
	// "system.cpu.state" semantic conventions. It represents the state of the
	// CPU
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'idle', 'interrupt'
	SystemCPUStateKey = attribute.Key("system.cpu.state")
)

var (
	// user
	SystemCPUStateUser = SystemCPUStateKey.String("user")
	// system
	SystemCPUStateSystem = SystemCPUStateKey.String("system")
	// nice
	SystemCPUStateNice = SystemCPUStateKey.String("nice")
	// idle
	SystemCPUStateIdle = SystemCPUStateKey.String("idle")
	// iowait
	SystemCPUStateIowait = SystemCPUStateKey.String("iowait")
	// interrupt
	SystemCPUStateInterrupt = SystemCPUStateKey.String("interrupt")
	// steal
	SystemCPUStateSteal = SystemCPUStateKey.String("steal")
)

// SystemCPULogicalNumber returns an attribute KeyValue conforming to the
// "system.cpu.logical_number" semantic conventions. It represents the logical
// CPU number [0..n-1]
func SystemCPULogicalNumber(val int) attribute.KeyValue {
	return SystemCPULogicalNumberKey.Int(val)
}

// Describes System Memory metric attributes
const (
	// SystemMemoryStateKey is the attribute Key conforming to the
	// "system.memory.state" semantic conventions. It represents the memory
	// state
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'free', 'cached'
	SystemMemoryStateKey = attribute.Key("system.memory.state")
)

var (
	// used
	SystemMemoryStateUsed = SystemMemoryStateKey.String("used")
	// free
	SystemMemoryStateFree = SystemMemoryStateKey.String("free")
	// shared
	SystemMemoryStateShared = SystemMemoryStateKey.String("shared")
	// buffers
	SystemMemoryStateBuffers = SystemMemoryStateKey.String("buffers")
	// cached
	SystemMemoryStateCached = SystemMemoryStateKey.String("cached")
)

// Describes System Memory Paging metric attributes
const (
	// SystemPagingDirectionKey is the attribute Key conforming to the
	// "system.paging.direction" semantic conventions. It represents the paging
	// access direction
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'in'
	SystemPagingDirectionKey = attribute.Key("system.paging.direction")

	// SystemPagingStateKey is the attribute Key conforming to the
	// "system.paging.state" semantic conventions. It represents the memory
	// paging state
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'free'
	SystemPagingStateKey = attribute.Key("system.paging.state")

	// SystemPagingTypeKey is the attribute Key conforming to the
	// "system.paging.type" semantic conventions. It represents the memory
	// paging type
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'minor'
	SystemPagingTypeKey = attribute.Key("system.paging.type")
)

var (
	// in
	SystemPagingDirectionIn = SystemPagingDirectionKey.String("in")
	// out
	SystemPagingDirectionOut = SystemPagingDirectionKey.String("out")
)

var (
	// used
	SystemPagingStateUsed = SystemPagingStateKey.String("used")
	// free
	SystemPagingStateFree = SystemPagingStateKey.String("free")
)

var (
	// major
	SystemPagingTypeMajor = SystemPagingTypeKey.String("major")
	// minor
	SystemPagingTypeMinor = SystemPagingTypeKey.String("minor")
)

// Describes Filesystem metric attributes
const (
	// SystemFilesystemModeKey is the attribute Key conforming to the
	// "system.filesystem.mode" semantic conventions. It represents the
	// filesystem mode
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'rw, ro'
	SystemFilesystemModeKey = attribute.Key("system.filesystem.mode")

	// SystemFilesystemMountpointKey is the attribute Key conforming to the
	// "system.filesystem.mountpoint" semantic conventions. It represents the
	// filesystem mount path
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/mnt/data'
	SystemFilesystemMountpointKey = attribute.Key("system.filesystem.mountpoint")

	// SystemFilesystemStateKey is the attribute Key conforming to the
	// "system.filesystem.state" semantic conventions. It represents the
	// filesystem state
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'used'
	SystemFilesystemStateKey = attribute.Key("system.filesystem.state")

	// SystemFilesystemTypeKey is the attribute Key conforming to the
	// "system.filesystem.type" semantic conventions. It represents the
	// filesystem type
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'ext4'
	SystemFilesystemTypeKey = attribute.Key("system.filesystem.type")
)

var (
	// used
	SystemFilesystemStateUsed = SystemFilesystemStateKey.String("used")
	// free
	SystemFilesystemStateFree = SystemFilesystemStateKey.String("free")
	// reserved
	SystemFilesystemStateReserved = SystemFilesystemStateKey.String("reserved")
)

var (
	// fat32
	SystemFilesystemTypeFat32 = SystemFilesystemTypeKey.String("fat32")
	// exfat
	SystemFilesystemTypeExfat = SystemFilesystemTypeKey.String("exfat")
	// ntfs
	SystemFilesystemTypeNtfs = SystemFilesystemTypeKey.String("ntfs")
	// refs
	SystemFilesystemTypeRefs = SystemFilesystemTypeKey.String("refs")
	// hfsplus
	SystemFilesystemTypeHfsplus = SystemFilesystemTypeKey.String("hfsplus")
	// ext4
	SystemFilesystemTypeExt4 = SystemFilesystemTypeKey.String("ext4")
)

// SystemFilesystemMode returns an attribute KeyValue conforming to the
// "system.filesystem.mode" semantic conventions. It represents the filesystem
// mode
func SystemFilesystemMode(val string) attribute.KeyValue {
	return SystemFilesystemModeKey.String(val)
}

// SystemFilesystemMountpoint returns an attribute KeyValue conforming to
// the "system.filesystem.mountpoint" semantic conventions. It represents the
// filesystem mount path
func SystemFilesystemMountpoint(val string) attribute.KeyValue {
	return SystemFilesystemMountpointKey.String(val)
}

// Describes Network metric attributes
const (
	// SystemNetworkStateKey is the attribute Key conforming to the
	// "system.network.state" semantic conventions. It represents a stateless
	// protocol MUST NOT set this attribute
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'close_wait'
	SystemNetworkStateKey = attribute.Key("system.network.state")
)

var (
	// close
	SystemNetworkStateClose = SystemNetworkStateKey.String("close")
	// close_wait
	SystemNetworkStateCloseWait = SystemNetworkStateKey.String("close_wait")
	// closing
	SystemNetworkStateClosing = SystemNetworkStateKey.String("closing")
	// delete
	SystemNetworkStateDelete = SystemNetworkStateKey.String("delete")
	// established
	SystemNetworkStateEstablished = SystemNetworkStateKey.String("established")
	// fin_wait_1
	SystemNetworkStateFinWait1 = SystemNetworkStateKey.String("fin_wait_1")
	// fin_wait_2
	SystemNetworkStateFinWait2 = SystemNetworkStateKey.String("fin_wait_2")
	// last_ack
	SystemNetworkStateLastAck = SystemNetworkStateKey.String("last_ack")
	// listen
	SystemNetworkStateListen = SystemNetworkStateKey.String("listen")
	// syn_recv
	SystemNetworkStateSynRecv = SystemNetworkStateKey.String("syn_recv")
	// syn_sent
	SystemNetworkStateSynSent = SystemNetworkStateKey.String("syn_sent")
	// time_wait
	SystemNetworkStateTimeWait = SystemNetworkStateKey.String("time_wait")
)

// Describes System Process metric attributes
const (
	// SystemProcessesStatusKey is the attribute Key conforming to the
	// "system.processes.status" semantic conventions. It represents the
	// process state, e.g., [Linux Process State
	// Codes](https://man7.org/linux/man-pages/man1/ps.1.html#PROCESS_STATE_CODES)
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'running'
	SystemProcessesStatusKey = attribute.Key("system.processes.status")
)

var (
	// running
	SystemProcessesStatusRunning = SystemProcessesStatusKey.String("running")
	// sleeping
	SystemProcessesStatusSleeping = SystemProcessesStatusKey.String("sleeping")
	// stopped
	SystemProcessesStatusStopped = SystemProcessesStatusKey.String("stopped")
	// defunct
	SystemProcessesStatusDefunct = SystemProcessesStatusKey.String("defunct")
)

// These attributes may be used to describe the client in a connection-based
// network interaction where there is one side that initiates the connection
// (the client is the side that initiates the connection). This covers all TCP
// network interactions since TCP is connection-based and one side initiates
// the connection (an exception is made for peer-to-peer communication over TCP
// where the "user-facing" surface of the protocol / API doesn't expose a clear
// notion of client and server). This also covers UDP network interactions
// where one side initiates the interaction, e.g. QUIC (HTTP/3) and DNS.
const (
	// ClientAddressKey is the attribute Key conforming to the "client.address"
	// semantic conventions. It represents the client address - domain name if
	// available without reverse DNS lookup; otherwise, IP address or Unix
	// domain socket name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'client.example.com', '10.1.2.80', '/tmp/my.sock'
	// Note: When observed from the server side, and when communicating through
	// an intermediary, `client.address` SHOULD represent the client address
	// behind any intermediaries,  for example proxies, if it's available.
	ClientAddressKey = attribute.Key("client.address")

	// ClientPortKey is the attribute Key conforming to the "client.port"
	// semantic conventions. It represents the client port number.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 65123
	// Note: When observed from the server side, and when communicating through
	// an intermediary, `client.port` SHOULD represent the client port behind
	// any intermediaries,  for example proxies, if it's available.
	ClientPortKey = attribute.Key("client.port")
)

// ClientAddress returns an attribute KeyValue conforming to the
// "client.address" semantic conventions. It represents the client address -
// domain name if available without reverse DNS lookup; otherwise, IP address
// or Unix domain socket name.
func ClientAddress(val string) attribute.KeyValue {
	return ClientAddressKey.String(val)
}

// ClientPort returns an attribute KeyValue conforming to the "client.port"
// semantic conventions. It represents the client port number.
func ClientPort(val int) attribute.KeyValue {
	return ClientPortKey.Int(val)
}

// The attributes used to describe telemetry in the context of databases.
const (
	// DBCassandraConsistencyLevelKey is the attribute Key conforming to the
	// "db.cassandra.consistency_level" semantic conventions. It represents the
	// consistency level of the query. Based on consistency values from
	// [CQL](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html).
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	DBCassandraConsistencyLevelKey = attribute.Key("db.cassandra.consistency_level")

	// DBCassandraCoordinatorDCKey is the attribute Key conforming to the
	// "db.cassandra.coordinator.dc" semantic conventions. It represents the
	// data center of the coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'us-west-2'
	DBCassandraCoordinatorDCKey = attribute.Key("db.cassandra.coordinator.dc")

	// DBCassandraCoordinatorIDKey is the attribute Key conforming to the
	// "db.cassandra.coordinator.id" semantic conventions. It represents the ID
	// of the coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'be13faa2-8574-4d71-926d-27f16cf8a7af'
	DBCassandraCoordinatorIDKey = attribute.Key("db.cassandra.coordinator.id")

	// DBCassandraIdempotenceKey is the attribute Key conforming to the
	// "db.cassandra.idempotence" semantic conventions. It represents the
	// whether or not the query is idempotent.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	DBCassandraIdempotenceKey = attribute.Key("db.cassandra.idempotence")

	// DBCassandraPageSizeKey is the attribute Key conforming to the
	// "db.cassandra.page_size" semantic conventions. It represents the fetch
	// size used for paging, i.e. how many rows will be returned at once.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 5000
	DBCassandraPageSizeKey = attribute.Key("db.cassandra.page_size")

	// DBCassandraSpeculativeExecutionCountKey is the attribute Key conforming
	// to the "db.cassandra.speculative_execution_count" semantic conventions.
	// It represents the number of times a query was speculatively executed.
	// Not set or `0` if the query was not executed speculatively.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 0, 2
	DBCassandraSpeculativeExecutionCountKey = attribute.Key("db.cassandra.speculative_execution_count")

	// DBCassandraTableKey is the attribute Key conforming to the
	// "db.cassandra.table" semantic conventions. It represents the name of the
	// primary Cassandra table that the operation is acting upon, including the
	// keyspace name (if applicable).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'mytable'
	// Note: This mirrors the db.sql.table attribute but references cassandra
	// rather than sql. It is not recommended to attempt any client-side
	// parsing of `db.statement` just to get this property, but it should be
	// set if it is provided by the library being instrumented. If the
	// operation is acting upon an anonymous table, or more than one table,
	// this value MUST NOT be set.
	DBCassandraTableKey = attribute.Key("db.cassandra.table")

	// DBConnectionStringKey is the attribute Key conforming to the
	// "db.connection_string" semantic conventions. It represents the
	// connection string used to connect to the database. It is recommended to
	// remove embedded credentials.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'Server=(localdb)\\v11.0;Integrated Security=true;'
	DBConnectionStringKey = attribute.Key("db.connection_string")

	// DBCosmosDBClientIDKey is the attribute Key conforming to the
	// "db.cosmosdb.client_id" semantic conventions. It represents the unique
	// Cosmos client instance id.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '3ba4827d-4422-483f-b59f-85b74211c11d'
	DBCosmosDBClientIDKey = attribute.Key("db.cosmosdb.client_id")

	// DBCosmosDBConnectionModeKey is the attribute Key conforming to the
	// "db.cosmosdb.connection_mode" semantic conventions. It represents the
	// cosmos client connection mode.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	DBCosmosDBConnectionModeKey = attribute.Key("db.cosmosdb.connection_mode")

	// DBCosmosDBContainerKey is the attribute Key conforming to the
	// "db.cosmosdb.container" semantic conventions. It represents the cosmos
	// DB container name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'anystring'
	DBCosmosDBContainerKey = attribute.Key("db.cosmosdb.container")

	// DBCosmosDBOperationTypeKey is the attribute Key conforming to the
	// "db.cosmosdb.operation_type" semantic conventions. It represents the
	// cosmosDB Operation Type.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	DBCosmosDBOperationTypeKey = attribute.Key("db.cosmosdb.operation_type")

	// DBCosmosDBRequestChargeKey is the attribute Key conforming to the
	// "db.cosmosdb.request_charge" semantic conventions. It represents the rU
	// consumed for that operation
	//
	// Type: double
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 46.18, 1.0
	DBCosmosDBRequestChargeKey = attribute.Key("db.cosmosdb.request_charge")

	// DBCosmosDBRequestContentLengthKey is the attribute Key conforming to the
	// "db.cosmosdb.request_content_length" semantic conventions. It represents
	// the request payload size in bytes
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	DBCosmosDBRequestContentLengthKey = attribute.Key("db.cosmosdb.request_content_length")

	// DBCosmosDBStatusCodeKey is the attribute Key conforming to the
	// "db.cosmosdb.status_code" semantic conventions. It represents the cosmos
	// DB status code.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 200, 201
	DBCosmosDBStatusCodeKey = attribute.Key("db.cosmosdb.status_code")

	// DBCosmosDBSubStatusCodeKey is the attribute Key conforming to the
	// "db.cosmosdb.sub_status_code" semantic conventions. It represents the
	// cosmos DB sub status code.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1000, 1002
	DBCosmosDBSubStatusCodeKey = attribute.Key("db.cosmosdb.sub_status_code")

	// DBElasticsearchClusterNameKey is the attribute Key conforming to the
	// "db.elasticsearch.cluster.name" semantic conventions. It represents the
	// represents the identifier of an Elasticsearch cluster.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'e9106fc68e3044f0b1475b04bf4ffd5f'
	DBElasticsearchClusterNameKey = attribute.Key("db.elasticsearch.cluster.name")

	// DBElasticsearchNodeNameKey is the attribute Key conforming to the
	// "db.elasticsearch.node.name" semantic conventions. It represents the
	// represents the human-readable identifier of the node/instance to which a
	// request was routed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'instance-0000000001'
	DBElasticsearchNodeNameKey = attribute.Key("db.elasticsearch.node.name")

	// DBInstanceIDKey is the attribute Key conforming to the "db.instance.id"
	// semantic conventions. It represents an identifier (address, unique name,
	// or any other identifier) of the database instance that is executing
	// queries or mutations on the current connection. This is useful in cases
	// where the database is running in a clustered environment and the
	// instrumentation is able to record the node executing the query. The
	// client may obtain this value in databases like MySQL using queries like
	// `select @@hostname`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'mysql-e26b99z.example.com'
	DBInstanceIDKey = attribute.Key("db.instance.id")

	// DBJDBCDriverClassnameKey is the attribute Key conforming to the
	// "db.jdbc.driver_classname" semantic conventions. It represents the
	// fully-qualified class name of the [Java Database Connectivity
	// (JDBC)](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/)
	// driver used to connect.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'org.postgresql.Driver',
	// 'com.microsoft.sqlserver.jdbc.SQLServerDriver'
	DBJDBCDriverClassnameKey = attribute.Key("db.jdbc.driver_classname")

	// DBMongoDBCollectionKey is the attribute Key conforming to the
	// "db.mongodb.collection" semantic conventions. It represents the MongoDB
	// collection being accessed within the database stated in `db.name`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'customers', 'products'
	DBMongoDBCollectionKey = attribute.Key("db.mongodb.collection")

	// DBMSSQLInstanceNameKey is the attribute Key conforming to the
	// "db.mssql.instance_name" semantic conventions. It represents the
	// Microsoft SQL Server [instance
	// name](https://docs.microsoft.com/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15)
	// connecting to. This name is used to determine the port of a named
	// instance.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MSSQLSERVER'
	// Note: If setting a `db.mssql.instance_name`, `server.port` is no longer
	// required (but still recommended if non-standard).
	DBMSSQLInstanceNameKey = attribute.Key("db.mssql.instance_name")

	// DBNameKey is the attribute Key conforming to the "db.name" semantic
	// conventions. It represents the this attribute is used to report the name
	// of the database being accessed. For commands that switch the database,
	// this should be set to the target database (even if the command fails).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'customers', 'main'
	// Note: In some SQL databases, the database name to be used is called
	// "schema name". In case there are multiple layers that could be
	// considered for database name (e.g. Oracle instance name and schema
	// name), the database name to be used is the more specific layer (e.g.
	// Oracle schema name).
	DBNameKey = attribute.Key("db.name")

	// DBOperationKey is the attribute Key conforming to the "db.operation"
	// semantic conventions. It represents the name of the operation being
	// executed, e.g. the [MongoDB command
	// name](https://docs.mongodb.com/manual/reference/command/#database-operations)
	// such as `findAndModify`, or the SQL keyword.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'findAndModify', 'HMSET', 'SELECT'
	// Note: When setting this to an SQL keyword, it is not recommended to
	// attempt any client-side parsing of `db.statement` just to get this
	// property, but it should be set if the operation name is provided by the
	// library being instrumented. If the SQL statement has an ambiguous
	// operation, or performs more than one operation, this value may be
	// omitted.
	DBOperationKey = attribute.Key("db.operation")

	// DBRedisDBIndexKey is the attribute Key conforming to the
	// "db.redis.database_index" semantic conventions. It represents the index
	// of the database being accessed as used in the [`SELECT`
	// command](https://redis.io/commands/select), provided as an integer. To
	// be used instead of the generic `db.name` attribute.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 0, 1, 15
	DBRedisDBIndexKey = attribute.Key("db.redis.database_index")

	// DBSQLTableKey is the attribute Key conforming to the "db.sql.table"
	// semantic conventions. It represents the name of the primary table that
	// the operation is acting upon, including the database name (if
	// applicable).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'public.users', 'customers'
	// Note: It is not recommended to attempt any client-side parsing of
	// `db.statement` just to get this property, but it should be set if it is
	// provided by the library being instrumented. If the operation is acting
	// upon an anonymous table, or more than one table, this value MUST NOT be
	// set.
	DBSQLTableKey = attribute.Key("db.sql.table")

	// DBStatementKey is the attribute Key conforming to the "db.statement"
	// semantic conventions. It represents the database statement being
	// executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'SELECT * FROM wuser_table', 'SET mykey "WuValue"'
	DBStatementKey = attribute.Key("db.statement")

	// DBSystemKey is the attribute Key conforming to the "db.system" semantic
	// conventions. It represents an identifier for the database management
	// system (DBMS) product being used. See below for a list of well-known
	// identifiers.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	DBSystemKey = attribute.Key("db.system")

	// DBUserKey is the attribute Key conforming to the "db.user" semantic
	// conventions. It represents the username for accessing the database.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'readonly_user', 'reporting_user'
	DBUserKey = attribute.Key("db.user")
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

var (
	// Gateway (HTTP) connections mode
	DBCosmosDBConnectionModeGateway = DBCosmosDBConnectionModeKey.String("gateway")
	// Direct connection
	DBCosmosDBConnectionModeDirect = DBCosmosDBConnectionModeKey.String("direct")
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

// DBCassandraCoordinatorDC returns an attribute KeyValue conforming to the
// "db.cassandra.coordinator.dc" semantic conventions. It represents the data
// center of the coordinating node for a query.
func DBCassandraCoordinatorDC(val string) attribute.KeyValue {
	return DBCassandraCoordinatorDCKey.String(val)
}

// DBCassandraCoordinatorID returns an attribute KeyValue conforming to the
// "db.cassandra.coordinator.id" semantic conventions. It represents the ID of
// the coordinating node for a query.
func DBCassandraCoordinatorID(val string) attribute.KeyValue {
	return DBCassandraCoordinatorIDKey.String(val)
}

// DBCassandraIdempotence returns an attribute KeyValue conforming to the
// "db.cassandra.idempotence" semantic conventions. It represents the whether
// or not the query is idempotent.
func DBCassandraIdempotence(val bool) attribute.KeyValue {
	return DBCassandraIdempotenceKey.Bool(val)
}

// DBCassandraPageSize returns an attribute KeyValue conforming to the
// "db.cassandra.page_size" semantic conventions. It represents the fetch size
// used for paging, i.e. how many rows will be returned at once.
func DBCassandraPageSize(val int) attribute.KeyValue {
	return DBCassandraPageSizeKey.Int(val)
}

// DBCassandraSpeculativeExecutionCount returns an attribute KeyValue
// conforming to the "db.cassandra.speculative_execution_count" semantic
// conventions. It represents the number of times a query was speculatively
// executed. Not set or `0` if the query was not executed speculatively.
func DBCassandraSpeculativeExecutionCount(val int) attribute.KeyValue {
	return DBCassandraSpeculativeExecutionCountKey.Int(val)
}

// DBCassandraTable returns an attribute KeyValue conforming to the
// "db.cassandra.table" semantic conventions. It represents the name of the
// primary Cassandra table that the operation is acting upon, including the
// keyspace name (if applicable).
func DBCassandraTable(val string) attribute.KeyValue {
	return DBCassandraTableKey.String(val)
}

// DBConnectionString returns an attribute KeyValue conforming to the
// "db.connection_string" semantic conventions. It represents the connection
// string used to connect to the database. It is recommended to remove embedded
// credentials.
func DBConnectionString(val string) attribute.KeyValue {
	return DBConnectionStringKey.String(val)
}

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

// DBCosmosDBRequestCharge returns an attribute KeyValue conforming to the
// "db.cosmosdb.request_charge" semantic conventions. It represents the rU
// consumed for that operation
func DBCosmosDBRequestCharge(val float64) attribute.KeyValue {
	return DBCosmosDBRequestChargeKey.Float64(val)
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

// DBElasticsearchClusterName returns an attribute KeyValue conforming to
// the "db.elasticsearch.cluster.name" semantic conventions. It represents the
// represents the identifier of an Elasticsearch cluster.
func DBElasticsearchClusterName(val string) attribute.KeyValue {
	return DBElasticsearchClusterNameKey.String(val)
}

// DBElasticsearchNodeName returns an attribute KeyValue conforming to the
// "db.elasticsearch.node.name" semantic conventions. It represents the
// represents the human-readable identifier of the node/instance to which a
// request was routed.
func DBElasticsearchNodeName(val string) attribute.KeyValue {
	return DBElasticsearchNodeNameKey.String(val)
}

// DBInstanceID returns an attribute KeyValue conforming to the
// "db.instance.id" semantic conventions. It represents an identifier (address,
// unique name, or any other identifier) of the database instance that is
// executing queries or mutations on the current connection. This is useful in
// cases where the database is running in a clustered environment and the
// instrumentation is able to record the node executing the query. The client
// may obtain this value in databases like MySQL using queries like `select
// @@hostname`.
func DBInstanceID(val string) attribute.KeyValue {
	return DBInstanceIDKey.String(val)
}

// DBJDBCDriverClassname returns an attribute KeyValue conforming to the
// "db.jdbc.driver_classname" semantic conventions. It represents the
// fully-qualified class name of the [Java Database Connectivity
// (JDBC)](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/) driver
// used to connect.
func DBJDBCDriverClassname(val string) attribute.KeyValue {
	return DBJDBCDriverClassnameKey.String(val)
}

// DBMongoDBCollection returns an attribute KeyValue conforming to the
// "db.mongodb.collection" semantic conventions. It represents the MongoDB
// collection being accessed within the database stated in `db.name`.
func DBMongoDBCollection(val string) attribute.KeyValue {
	return DBMongoDBCollectionKey.String(val)
}

// DBMSSQLInstanceName returns an attribute KeyValue conforming to the
// "db.mssql.instance_name" semantic conventions. It represents the Microsoft
// SQL Server [instance
// name](https://docs.microsoft.com/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15)
// connecting to. This name is used to determine the port of a named instance.
func DBMSSQLInstanceName(val string) attribute.KeyValue {
	return DBMSSQLInstanceNameKey.String(val)
}

// DBName returns an attribute KeyValue conforming to the "db.name" semantic
// conventions. It represents the this attribute is used to report the name of
// the database being accessed. For commands that switch the database, this
// should be set to the target database (even if the command fails).
func DBName(val string) attribute.KeyValue {
	return DBNameKey.String(val)
}

// DBOperation returns an attribute KeyValue conforming to the
// "db.operation" semantic conventions. It represents the name of the operation
// being executed, e.g. the [MongoDB command
// name](https://docs.mongodb.com/manual/reference/command/#database-operations)
// such as `findAndModify`, or the SQL keyword.
func DBOperation(val string) attribute.KeyValue {
	return DBOperationKey.String(val)
}

// DBRedisDBIndex returns an attribute KeyValue conforming to the
// "db.redis.database_index" semantic conventions. It represents the index of
// the database being accessed as used in the [`SELECT`
// command](https://redis.io/commands/select), provided as an integer. To be
// used instead of the generic `db.name` attribute.
func DBRedisDBIndex(val int) attribute.KeyValue {
	return DBRedisDBIndexKey.Int(val)
}

// DBSQLTable returns an attribute KeyValue conforming to the "db.sql.table"
// semantic conventions. It represents the name of the primary table that the
// operation is acting upon, including the database name (if applicable).
func DBSQLTable(val string) attribute.KeyValue {
	return DBSQLTableKey.String(val)
}

// DBStatement returns an attribute KeyValue conforming to the
// "db.statement" semantic conventions. It represents the database statement
// being executed.
func DBStatement(val string) attribute.KeyValue {
	return DBStatementKey.String(val)
}

// DBUser returns an attribute KeyValue conforming to the "db.user" semantic
// conventions. It represents the username for accessing the database.
func DBUser(val string) attribute.KeyValue {
	return DBUserKey.String(val)
}

// Describes deprecated HTTP attributes.
const (
	// HTTPFlavorKey is the attribute Key conforming to the "http.flavor"
	// semantic conventions.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: deprecated
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorKey = attribute.Key("http.flavor")

	// HTTPMethodKey is the attribute Key conforming to the "http.method"
	// semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'GET', 'POST', 'HEAD'
	// Deprecated: use `http.request.method` instead.
	HTTPMethodKey = attribute.Key("http.method")

	// HTTPRequestContentLengthKey is the attribute Key conforming to the
	// "http.request_content_length" semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 3495
	// Deprecated: use `http.request.header.content-length` instead.
	HTTPRequestContentLengthKey = attribute.Key("http.request_content_length")

	// HTTPResponseContentLengthKey is the attribute Key conforming to the
	// "http.response_content_length" semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 3495
	// Deprecated: use `http.response.header.content-length` instead.
	HTTPResponseContentLengthKey = attribute.Key("http.response_content_length")

	// HTTPSchemeKey is the attribute Key conforming to the "http.scheme"
	// semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'http', 'https'
	// Deprecated: use `url.scheme` instead.
	HTTPSchemeKey = attribute.Key("http.scheme")

	// HTTPStatusCodeKey is the attribute Key conforming to the
	// "http.status_code" semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 200
	// Deprecated: use `http.response.status_code` instead.
	HTTPStatusCodeKey = attribute.Key("http.status_code")

	// HTTPTargetKey is the attribute Key conforming to the "http.target"
	// semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/search?q=OpenTelemetry#SemConv'
	// Deprecated: use `url.path` and `url.query` instead.
	HTTPTargetKey = attribute.Key("http.target")

	// HTTPURLKey is the attribute Key conforming to the "http.url" semantic
	// conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'https://www.foo.bar/search?q=OpenTelemetry#SemConv'
	// Deprecated: use `url.full` instead.
	HTTPURLKey = attribute.Key("http.url")

	// HTTPUserAgentKey is the attribute Key conforming to the
	// "http.user_agent" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'CERN-LineMode/2.15 libwww/2.17b3', 'Mozilla/5.0 (iPhone; CPU
	// iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko)
	// Version/14.1.2 Mobile/15E148 Safari/604.1'
	// Deprecated: use `user_agent.original` instead.
	HTTPUserAgentKey = attribute.Key("http.user_agent")
)

var (
	// HTTP/1.0
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorHTTP10 = HTTPFlavorKey.String("1.0")
	// HTTP/1.1
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorHTTP11 = HTTPFlavorKey.String("1.1")
	// HTTP/2
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorHTTP20 = HTTPFlavorKey.String("2.0")
	// HTTP/3
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorHTTP30 = HTTPFlavorKey.String("3.0")
	// SPDY protocol
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorSPDY = HTTPFlavorKey.String("SPDY")
	// QUIC protocol
	//
	// Deprecated: use `network.protocol.name` instead.
	HTTPFlavorQUIC = HTTPFlavorKey.String("QUIC")
)

// HTTPMethod returns an attribute KeyValue conforming to the "http.method"
// semantic conventions.
//
// Deprecated: use `http.request.method` instead.
func HTTPMethod(val string) attribute.KeyValue {
	return HTTPMethodKey.String(val)
}

// HTTPRequestContentLength returns an attribute KeyValue conforming to the
// "http.request_content_length" semantic conventions.
//
// Deprecated: use `http.request.header.content-length` instead.
func HTTPRequestContentLength(val int) attribute.KeyValue {
	return HTTPRequestContentLengthKey.Int(val)
}

// HTTPResponseContentLength returns an attribute KeyValue conforming to the
// "http.response_content_length" semantic conventions.
//
// Deprecated: use `http.response.header.content-length` instead.
func HTTPResponseContentLength(val int) attribute.KeyValue {
	return HTTPResponseContentLengthKey.Int(val)
}

// HTTPScheme returns an attribute KeyValue conforming to the "http.scheme"
// semantic conventions.
//
// Deprecated: use `url.scheme` instead.
func HTTPScheme(val string) attribute.KeyValue {
	return HTTPSchemeKey.String(val)
}

// HTTPStatusCode returns an attribute KeyValue conforming to the
// "http.status_code" semantic conventions.
//
// Deprecated: use `http.response.status_code` instead.
func HTTPStatusCode(val int) attribute.KeyValue {
	return HTTPStatusCodeKey.Int(val)
}

// HTTPTarget returns an attribute KeyValue conforming to the "http.target"
// semantic conventions.
//
// Deprecated: use `url.path` and `url.query` instead.
func HTTPTarget(val string) attribute.KeyValue {
	return HTTPTargetKey.String(val)
}

// HTTPURL returns an attribute KeyValue conforming to the "http.url"
// semantic conventions.
//
// Deprecated: use `url.full` instead.
func HTTPURL(val string) attribute.KeyValue {
	return HTTPURLKey.String(val)
}

// HTTPUserAgent returns an attribute KeyValue conforming to the
// "http.user_agent" semantic conventions.
//
// Deprecated: use `user_agent.original` instead.
func HTTPUserAgent(val string) attribute.KeyValue {
	return HTTPUserAgentKey.String(val)
}

// These attributes may be used for any network related operation.
const (
	// NetHostNameKey is the attribute Key conforming to the "net.host.name"
	// semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'example.com'
	// Deprecated: use `server.address`.
	NetHostNameKey = attribute.Key("net.host.name")

	// NetHostPortKey is the attribute Key conforming to the "net.host.port"
	// semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	// Deprecated: use `server.port`.
	NetHostPortKey = attribute.Key("net.host.port")

	// NetPeerNameKey is the attribute Key conforming to the "net.peer.name"
	// semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'example.com'
	// Deprecated: use `server.address` on client spans and `client.address` on
	// server spans.
	NetPeerNameKey = attribute.Key("net.peer.name")

	// NetPeerPortKey is the attribute Key conforming to the "net.peer.port"
	// semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	// Deprecated: use `server.port` on client spans and `client.port` on
	// server spans.
	NetPeerPortKey = attribute.Key("net.peer.port")

	// NetProtocolNameKey is the attribute Key conforming to the
	// "net.protocol.name" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'amqp', 'http', 'mqtt'
	// Deprecated: use `network.protocol.name`.
	NetProtocolNameKey = attribute.Key("net.protocol.name")

	// NetProtocolVersionKey is the attribute Key conforming to the
	// "net.protocol.version" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '3.1.1'
	// Deprecated: use `network.protocol.version`.
	NetProtocolVersionKey = attribute.Key("net.protocol.version")

	// NetSockFamilyKey is the attribute Key conforming to the
	// "net.sock.family" semantic conventions.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: deprecated
	// Deprecated: use `network.transport` and `network.type`.
	NetSockFamilyKey = attribute.Key("net.sock.family")

	// NetSockHostAddrKey is the attribute Key conforming to the
	// "net.sock.host.addr" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/var/my.sock'
	// Deprecated: use `network.local.address`.
	NetSockHostAddrKey = attribute.Key("net.sock.host.addr")

	// NetSockHostPortKey is the attribute Key conforming to the
	// "net.sock.host.port" semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	// Deprecated: use `network.local.port`.
	NetSockHostPortKey = attribute.Key("net.sock.host.port")

	// NetSockPeerAddrKey is the attribute Key conforming to the
	// "net.sock.peer.addr" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '192.168.0.1'
	// Deprecated: use `network.peer.address`.
	NetSockPeerAddrKey = attribute.Key("net.sock.peer.addr")

	// NetSockPeerNameKey is the attribute Key conforming to the
	// "net.sock.peer.name" semantic conventions.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/var/my.sock'
	// Deprecated: no replacement at this time.
	NetSockPeerNameKey = attribute.Key("net.sock.peer.name")

	// NetSockPeerPortKey is the attribute Key conforming to the
	// "net.sock.peer.port" semantic conventions.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 65531
	// Deprecated: use `network.peer.port`.
	NetSockPeerPortKey = attribute.Key("net.sock.peer.port")

	// NetTransportKey is the attribute Key conforming to the "net.transport"
	// semantic conventions.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: deprecated
	// Deprecated: use `network.transport`.
	NetTransportKey = attribute.Key("net.transport")
)

var (
	// IPv4 address
	//
	// Deprecated: use `network.transport` and `network.type`.
	NetSockFamilyInet = NetSockFamilyKey.String("inet")
	// IPv6 address
	//
	// Deprecated: use `network.transport` and `network.type`.
	NetSockFamilyInet6 = NetSockFamilyKey.String("inet6")
	// Unix domain socket path
	//
	// Deprecated: use `network.transport` and `network.type`.
	NetSockFamilyUnix = NetSockFamilyKey.String("unix")
)

var (
	// ip_tcp
	//
	// Deprecated: use `network.transport`.
	NetTransportTCP = NetTransportKey.String("ip_tcp")
	// ip_udp
	//
	// Deprecated: use `network.transport`.
	NetTransportUDP = NetTransportKey.String("ip_udp")
	// Named or anonymous pipe
	//
	// Deprecated: use `network.transport`.
	NetTransportPipe = NetTransportKey.String("pipe")
	// In-process communication
	//
	// Deprecated: use `network.transport`.
	NetTransportInProc = NetTransportKey.String("inproc")
	// Something else (non IP-based)
	//
	// Deprecated: use `network.transport`.
	NetTransportOther = NetTransportKey.String("other")
)

// NetHostName returns an attribute KeyValue conforming to the
// "net.host.name" semantic conventions.
//
// Deprecated: use `server.address`.
func NetHostName(val string) attribute.KeyValue {
	return NetHostNameKey.String(val)
}

// NetHostPort returns an attribute KeyValue conforming to the
// "net.host.port" semantic conventions.
//
// Deprecated: use `server.port`.
func NetHostPort(val int) attribute.KeyValue {
	return NetHostPortKey.Int(val)
}

// NetPeerName returns an attribute KeyValue conforming to the
// "net.peer.name" semantic conventions.
//
// Deprecated: use `server.address` on client spans and `client.address` on
// server spans.
func NetPeerName(val string) attribute.KeyValue {
	return NetPeerNameKey.String(val)
}

// NetPeerPort returns an attribute KeyValue conforming to the
// "net.peer.port" semantic conventions.
//
// Deprecated: use `server.port` on client spans and `client.port` on server
// spans.
func NetPeerPort(val int) attribute.KeyValue {
	return NetPeerPortKey.Int(val)
}

// NetProtocolName returns an attribute KeyValue conforming to the
// "net.protocol.name" semantic conventions.
//
// Deprecated: use `network.protocol.name`.
func NetProtocolName(val string) attribute.KeyValue {
	return NetProtocolNameKey.String(val)
}

// NetProtocolVersion returns an attribute KeyValue conforming to the
// "net.protocol.version" semantic conventions.
//
// Deprecated: use `network.protocol.version`.
func NetProtocolVersion(val string) attribute.KeyValue {
	return NetProtocolVersionKey.String(val)
}

// NetSockHostAddr returns an attribute KeyValue conforming to the
// "net.sock.host.addr" semantic conventions.
//
// Deprecated: use `network.local.address`.
func NetSockHostAddr(val string) attribute.KeyValue {
	return NetSockHostAddrKey.String(val)
}

// NetSockHostPort returns an attribute KeyValue conforming to the
// "net.sock.host.port" semantic conventions.
//
// Deprecated: use `network.local.port`.
func NetSockHostPort(val int) attribute.KeyValue {
	return NetSockHostPortKey.Int(val)
}

// NetSockPeerAddr returns an attribute KeyValue conforming to the
// "net.sock.peer.addr" semantic conventions.
//
// Deprecated: use `network.peer.address`.
func NetSockPeerAddr(val string) attribute.KeyValue {
	return NetSockPeerAddrKey.String(val)
}

// NetSockPeerName returns an attribute KeyValue conforming to the
// "net.sock.peer.name" semantic conventions.
//
// Deprecated: no replacement at this time.
func NetSockPeerName(val string) attribute.KeyValue {
	return NetSockPeerNameKey.String(val)
}

// NetSockPeerPort returns an attribute KeyValue conforming to the
// "net.sock.peer.port" semantic conventions.
//
// Deprecated: use `network.peer.port`.
func NetSockPeerPort(val int) attribute.KeyValue {
	return NetSockPeerPortKey.Int(val)
}

// These attributes may be used to describe the receiver of a network
// exchange/packet. These should be used when there is no client/server
// relationship between the two sides, or when that relationship is unknown.
// This covers low-level network interactions (e.g. packet tracing) where you
// don't know if there was a connection or which side initiated it. This also
// covers unidirectional UDP flows and peer-to-peer communication where the
// "user-facing" surface of the protocol / API doesn't expose a clear notion of
// client and server.
const (
	// DestinationAddressKey is the attribute Key conforming to the
	// "destination.address" semantic conventions. It represents the
	// destination address - domain name if available without reverse DNS
	// lookup; otherwise, IP address or Unix domain socket name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'destination.example.com', '10.1.2.80', '/tmp/my.sock'
	// Note: When observed from the source side, and when communicating through
	// an intermediary, `destination.address` SHOULD represent the destination
	// address behind any intermediaries, for example proxies, if it's
	// available.
	DestinationAddressKey = attribute.Key("destination.address")

	// DestinationPortKey is the attribute Key conforming to the
	// "destination.port" semantic conventions. It represents the destination
	// port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 3389, 2888
	DestinationPortKey = attribute.Key("destination.port")
)

// DestinationAddress returns an attribute KeyValue conforming to the
// "destination.address" semantic conventions. It represents the destination
// address - domain name if available without reverse DNS lookup; otherwise, IP
// address or Unix domain socket name.
func DestinationAddress(val string) attribute.KeyValue {
	return DestinationAddressKey.String(val)
}

// DestinationPort returns an attribute KeyValue conforming to the
// "destination.port" semantic conventions. It represents the destination port
// number
func DestinationPort(val int) attribute.KeyValue {
	return DestinationPortKey.Int(val)
}

// These attributes may be used for any disk related operation.
const (
	// DiskIoDirectionKey is the attribute Key conforming to the
	// "disk.io.direction" semantic conventions. It represents the disk IO
	// operation direction.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'read'
	DiskIoDirectionKey = attribute.Key("disk.io.direction")
)

var (
	// read
	DiskIoDirectionRead = DiskIoDirectionKey.String("read")
	// write
	DiskIoDirectionWrite = DiskIoDirectionKey.String("write")
)

// The shared attributes used to report an error.
const (
	// ErrorTypeKey is the attribute Key conforming to the "error.type"
	// semantic conventions. It represents the describes a class of error the
	// operation ended with.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'timeout', 'java.net.UnknownHostException',
	// 'server_certificate_invalid', '500'
	// Note: The `error.type` SHOULD be predictable and SHOULD have low
	// cardinality.
	// Instrumentations SHOULD document the list of errors they report.
	//
	// The cardinality of `error.type` within one instrumentation library
	// SHOULD be low.
	// Telemetry consumers that aggregate data from multiple instrumentation
	// libraries and applications
	// should be prepared for `error.type` to have high cardinality at query
	// time when no
	// additional filters are applied.
	//
	// If the operation has completed successfully, instrumentations SHOULD NOT
	// set `error.type`.
	//
	// If a specific domain defines its own set of error identifiers (such as
	// HTTP or gRPC status codes),
	// it's RECOMMENDED to:
	//
	// * Use a domain-specific attribute
	// * Set `error.type` to capture all errors, regardless of whether they are
	// defined within the domain-specific set or not.
	ErrorTypeKey = attribute.Key("error.type")
)

var (
	// A fallback error value to be used when the instrumentation doesn't define a custom value
	ErrorTypeOther = ErrorTypeKey.String("_OTHER")
)

// The shared attributes used to report a single exception associated with a
// span or log.
const (
	// ExceptionEscapedKey is the attribute Key conforming to the
	// "exception.escaped" semantic conventions. It represents the sHOULD be
	// set to true if the exception event is recorded at a point where it is
	// known that the exception is escaping the scope of the span.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	// Note: An exception is considered to have escaped (or left) the scope of
	// a span,
	// if that span is ended while the exception is still logically "in
	// flight".
	// This may be actually "in flight" in some languages (e.g. if the
	// exception
	// is passed to a Context manager's `__exit__` method in Python) but will
	// usually be caught at the point of recording the exception in most
	// languages.
	//
	// It is usually not possible to determine at the point where an exception
	// is thrown
	// whether it will escape the scope of a span.
	// However, it is trivial to know that an exception
	// will escape, if one checks for an active exception just before ending
	// the span,
	// as done in the [example for recording span
	// exceptions](#recording-an-exception).
	//
	// It follows that an exception may still escape the scope of the span
	// even if the `exception.escaped` attribute was not set or set to false,
	// since the event might have been recorded at a time where it was not
	// clear whether the exception will escape.
	ExceptionEscapedKey = attribute.Key("exception.escaped")

	// ExceptionMessageKey is the attribute Key conforming to the
	// "exception.message" semantic conventions. It represents the exception
	// message.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
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
	// Stability: experimental
	// Examples: 'Exception in thread "main" java.lang.RuntimeException: Test
	// exception\\n at '
	//  'com.example.GenerateTrace.methodB(GenerateTrace.java:13)\\n at '
	//  'com.example.GenerateTrace.methodA(GenerateTrace.java:9)\\n at '
	//  'com.example.GenerateTrace.main(GenerateTrace.java:5)'
	ExceptionStacktraceKey = attribute.Key("exception.stacktrace")

	// ExceptionTypeKey is the attribute Key conforming to the "exception.type"
	// semantic conventions. It represents the type of the exception (its
	// fully-qualified class name, if applicable). The dynamic type of the
	// exception should be preferred over the static type in languages that
	// support it.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'java.net.ConnectException', 'OSError'
	ExceptionTypeKey = attribute.Key("exception.type")
)

// ExceptionEscaped returns an attribute KeyValue conforming to the
// "exception.escaped" semantic conventions. It represents the sHOULD be set to
// true if the exception event is recorded at a point where it is known that
// the exception is escaping the scope of the span.
func ExceptionEscaped(val bool) attribute.KeyValue {
	return ExceptionEscapedKey.Bool(val)
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

// ExceptionType returns an attribute KeyValue conforming to the
// "exception.type" semantic conventions. It represents the type of the
// exception (its fully-qualified class name, if applicable). The dynamic type
// of the exception should be preferred over the static type in languages that
// support it.
func ExceptionType(val string) attribute.KeyValue {
	return ExceptionTypeKey.String(val)
}

// Semantic convention attributes in the HTTP namespace.
const (
	// HTTPRequestBodySizeKey is the attribute Key conforming to the
	// "http.request.body.size" semantic conventions. It represents the size of
	// the request payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as
	// the
	// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
	// header. For requests using transport encoding, this should be the
	// compressed size.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 3495
	HTTPRequestBodySizeKey = attribute.Key("http.request.body.size")

	// HTTPRequestMethodKey is the attribute Key conforming to the
	// "http.request.method" semantic conventions. It represents the hTTP
	// request method.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'GET', 'POST', 'HEAD'
	// Note: HTTP request method value SHOULD be "known" to the
	// instrumentation.
	// By default, this convention defines "known" methods as the ones listed
	// in [RFC9110](https://www.rfc-editor.org/rfc/rfc9110.html#name-methods)
	// and the PATCH method defined in
	// [RFC5789](https://www.rfc-editor.org/rfc/rfc5789.html).
	//
	// If the HTTP request method is not known to instrumentation, it MUST set
	// the `http.request.method` attribute to `_OTHER`.
	//
	// If the HTTP instrumentation could end up converting valid HTTP request
	// methods to `_OTHER`, then it MUST provide a way to override
	// the list of known HTTP methods. If this override is done via environment
	// variable, then the environment variable MUST be named
	// OTEL_INSTRUMENTATION_HTTP_KNOWN_METHODS and support a comma-separated
	// list of case-sensitive known HTTP methods
	// (this list MUST be a full override of the default known method, it is
	// not a list of known methods in addition to the defaults).
	//
	// HTTP method names are case-sensitive and `http.request.method` attribute
	// value MUST match a known HTTP method name exactly.
	// Instrumentations for specific web frameworks that consider HTTP methods
	// to be case insensitive, SHOULD populate a canonical equivalent.
	// Tracing instrumentations that do so, MUST also set
	// `http.request.method_original` to the original value.
	HTTPRequestMethodKey = attribute.Key("http.request.method")

	// HTTPRequestMethodOriginalKey is the attribute Key conforming to the
	// "http.request.method_original" semantic conventions. It represents the
	// original HTTP method sent by the client in the request line.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'GeT', 'ACL', 'foo'
	HTTPRequestMethodOriginalKey = attribute.Key("http.request.method_original")

	// HTTPRequestResendCountKey is the attribute Key conforming to the
	// "http.request.resend_count" semantic conventions. It represents the
	// ordinal number of request resending attempt (for any reason, including
	// redirects).
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3
	// Note: The resend count SHOULD be updated each time an HTTP request gets
	// resent by the client, regardless of what was the cause of the resending
	// (e.g. redirection, authorization failure, 503 Server Unavailable,
	// network issues, or any other).
	HTTPRequestResendCountKey = attribute.Key("http.request.resend_count")

	// HTTPResponseBodySizeKey is the attribute Key conforming to the
	// "http.response.body.size" semantic conventions. It represents the size
	// of the response payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as
	// the
	// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
	// header. For requests using transport encoding, this should be the
	// compressed size.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 3495
	HTTPResponseBodySizeKey = attribute.Key("http.response.body.size")

	// HTTPResponseStatusCodeKey is the attribute Key conforming to the
	// "http.response.status_code" semantic conventions. It represents the
	// [HTTP response status
	// code](https://tools.ietf.org/html/rfc7231#section-6).
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 200
	HTTPResponseStatusCodeKey = attribute.Key("http.response.status_code")

	// HTTPRouteKey is the attribute Key conforming to the "http.route"
	// semantic conventions. It represents the matched route, that is, the path
	// template in the format used by the respective server framework.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/users/:userID?', '{controller}/{action}/{id?}'
	// Note: MUST NOT be populated when this is not supported by the HTTP
	// server framework as the route attribute should have low-cardinality and
	// the URI path can NOT substitute it.
	// SHOULD include the [application
	// root](/docs/http/http-spans.md#http-server-definitions) if there is one.
	HTTPRouteKey = attribute.Key("http.route")
)

var (
	// CONNECT method
	HTTPRequestMethodConnect = HTTPRequestMethodKey.String("CONNECT")
	// DELETE method
	HTTPRequestMethodDelete = HTTPRequestMethodKey.String("DELETE")
	// GET method
	HTTPRequestMethodGet = HTTPRequestMethodKey.String("GET")
	// HEAD method
	HTTPRequestMethodHead = HTTPRequestMethodKey.String("HEAD")
	// OPTIONS method
	HTTPRequestMethodOptions = HTTPRequestMethodKey.String("OPTIONS")
	// PATCH method
	HTTPRequestMethodPatch = HTTPRequestMethodKey.String("PATCH")
	// POST method
	HTTPRequestMethodPost = HTTPRequestMethodKey.String("POST")
	// PUT method
	HTTPRequestMethodPut = HTTPRequestMethodKey.String("PUT")
	// TRACE method
	HTTPRequestMethodTrace = HTTPRequestMethodKey.String("TRACE")
	// Any HTTP method that the instrumentation has no prior knowledge of
	HTTPRequestMethodOther = HTTPRequestMethodKey.String("_OTHER")
)

// HTTPRequestBodySize returns an attribute KeyValue conforming to the
// "http.request.body.size" semantic conventions. It represents the size of the
// request payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPRequestBodySize(val int) attribute.KeyValue {
	return HTTPRequestBodySizeKey.Int(val)
}

// HTTPRequestMethodOriginal returns an attribute KeyValue conforming to the
// "http.request.method_original" semantic conventions. It represents the
// original HTTP method sent by the client in the request line.
func HTTPRequestMethodOriginal(val string) attribute.KeyValue {
	return HTTPRequestMethodOriginalKey.String(val)
}

// HTTPRequestResendCount returns an attribute KeyValue conforming to the
// "http.request.resend_count" semantic conventions. It represents the ordinal
// number of request resending attempt (for any reason, including redirects).
func HTTPRequestResendCount(val int) attribute.KeyValue {
	return HTTPRequestResendCountKey.Int(val)
}

// HTTPResponseBodySize returns an attribute KeyValue conforming to the
// "http.response.body.size" semantic conventions. It represents the size of
// the response payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPResponseBodySize(val int) attribute.KeyValue {
	return HTTPResponseBodySizeKey.Int(val)
}

// HTTPResponseStatusCode returns an attribute KeyValue conforming to the
// "http.response.status_code" semantic conventions. It represents the [HTTP
// response status code](https://tools.ietf.org/html/rfc7231#section-6).
func HTTPResponseStatusCode(val int) attribute.KeyValue {
	return HTTPResponseStatusCodeKey.Int(val)
}

// HTTPRoute returns an attribute KeyValue conforming to the "http.route"
// semantic conventions. It represents the matched route, that is, the path
// template in the format used by the respective server framework.
func HTTPRoute(val string) attribute.KeyValue {
	return HTTPRouteKey.String(val)
}

// Attributes describing telemetry around messaging systems and messaging
// activities.
const (
	// MessagingBatchMessageCountKey is the attribute Key conforming to the
	// "messaging.batch.message_count" semantic conventions. It represents the
	// number of messages sent, received, or processed in the scope of the
	// batching operation.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
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
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'client-5', 'myhost@8742@s8083jm'
	MessagingClientIDKey = attribute.Key("messaging.client_id")

	// MessagingDestinationAnonymousKey is the attribute Key conforming to the
	// "messaging.destination.anonymous" semantic conventions. It represents a
	// boolean that is true if the message destination is anonymous (could be
	// unnamed or have auto-generated name).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingDestinationAnonymousKey = attribute.Key("messaging.destination.anonymous")

	// MessagingDestinationNameKey is the attribute Key conforming to the
	// "messaging.destination.name" semantic conventions. It represents the
	// message destination name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MyQueue', 'MyTopic'
	// Note: Destination name SHOULD uniquely identify a specific queue, topic
	// or other entity within the broker. If
	// the broker doesn't have such notion, the destination name SHOULD
	// uniquely identify the broker.
	MessagingDestinationNameKey = attribute.Key("messaging.destination.name")

	// MessagingDestinationTemplateKey is the attribute Key conforming to the
	// "messaging.destination.template" semantic conventions. It represents the
	// low cardinality representation of the messaging destination name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/customers/{customerID}'
	// Note: Destination names could be constructed from templates. An example
	// would be a destination name involving a user name or product id.
	// Although the destination name in this case is of high cardinality, the
	// underlying template is of low cardinality and can be effectively used
	// for grouping and aggregation.
	MessagingDestinationTemplateKey = attribute.Key("messaging.destination.template")

	// MessagingDestinationTemporaryKey is the attribute Key conforming to the
	// "messaging.destination.temporary" semantic conventions. It represents a
	// boolean that is true if the message destination is temporary and might
	// not exist anymore after messages are processed.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingDestinationTemporaryKey = attribute.Key("messaging.destination.temporary")

	// MessagingDestinationPublishAnonymousKey is the attribute Key conforming
	// to the "messaging.destination_publish.anonymous" semantic conventions.
	// It represents a boolean that is true if the publish message destination
	// is anonymous (could be unnamed or have auto-generated name).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingDestinationPublishAnonymousKey = attribute.Key("messaging.destination_publish.anonymous")

	// MessagingDestinationPublishNameKey is the attribute Key conforming to
	// the "messaging.destination_publish.name" semantic conventions. It
	// represents the name of the original destination the message was
	// published to
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MyQueue', 'MyTopic'
	// Note: The name SHOULD uniquely identify a specific queue, topic, or
	// other entity within the broker. If
	// the broker doesn't have such notion, the original destination name
	// SHOULD uniquely identify the broker.
	MessagingDestinationPublishNameKey = attribute.Key("messaging.destination_publish.name")

	// MessagingGCPPubsubMessageOrderingKeyKey is the attribute Key conforming
	// to the "messaging.gcp_pubsub.message.ordering_key" semantic conventions.
	// It represents the ordering key for a given message. If the attribute is
	// not present, the message does not have an ordering key.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'ordering_key'
	MessagingGCPPubsubMessageOrderingKeyKey = attribute.Key("messaging.gcp_pubsub.message.ordering_key")

	// MessagingKafkaConsumerGroupKey is the attribute Key conforming to the
	// "messaging.kafka.consumer.group" semantic conventions. It represents the
	// name of the Kafka Consumer Group that is handling the message. Only
	// applies to consumers, not producers.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'my-group'
	MessagingKafkaConsumerGroupKey = attribute.Key("messaging.kafka.consumer.group")

	// MessagingKafkaDestinationPartitionKey is the attribute Key conforming to
	// the "messaging.kafka.destination.partition" semantic conventions. It
	// represents the partition the message is sent to.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 2
	MessagingKafkaDestinationPartitionKey = attribute.Key("messaging.kafka.destination.partition")

	// MessagingKafkaMessageKeyKey is the attribute Key conforming to the
	// "messaging.kafka.message.key" semantic conventions. It represents the
	// message keys in Kafka are used for grouping alike messages to ensure
	// they're processed on the same partition. They differ from
	// `messaging.message.id` in that they're not unique. If the key is `null`,
	// the attribute MUST NOT be set.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myKey'
	// Note: If the key type is not string, it's string representation has to
	// be supplied for the attribute. If the key has no unambiguous, canonical
	// string form, don't include its value.
	MessagingKafkaMessageKeyKey = attribute.Key("messaging.kafka.message.key")

	// MessagingKafkaMessageOffsetKey is the attribute Key conforming to the
	// "messaging.kafka.message.offset" semantic conventions. It represents the
	// offset of a record in the corresponding Kafka partition.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 42
	MessagingKafkaMessageOffsetKey = attribute.Key("messaging.kafka.message.offset")

	// MessagingKafkaMessageTombstoneKey is the attribute Key conforming to the
	// "messaging.kafka.message.tombstone" semantic conventions. It represents
	// a boolean that is true if the message is a tombstone.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingKafkaMessageTombstoneKey = attribute.Key("messaging.kafka.message.tombstone")

	// MessagingMessageBodySizeKey is the attribute Key conforming to the
	// "messaging.message.body.size" semantic conventions. It represents the
	// size of the message body in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1439
	// Note: This can refer to both the compressed or uncompressed body size.
	// If both sizes are known, the uncompressed
	// body size should be used.
	MessagingMessageBodySizeKey = attribute.Key("messaging.message.body.size")

	// MessagingMessageConversationIDKey is the attribute Key conforming to the
	// "messaging.message.conversation_id" semantic conventions. It represents
	// the conversation ID identifying the conversation to which the message
	// belongs, represented as a string. Sometimes called "Correlation ID".
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MyConversationID'
	MessagingMessageConversationIDKey = attribute.Key("messaging.message.conversation_id")

	// MessagingMessageEnvelopeSizeKey is the attribute Key conforming to the
	// "messaging.message.envelope.size" semantic conventions. It represents
	// the size of the message body and metadata in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 2738
	// Note: This can refer to both the compressed or uncompressed size. If
	// both sizes are known, the uncompressed
	// size should be used.
	MessagingMessageEnvelopeSizeKey = attribute.Key("messaging.message.envelope.size")

	// MessagingMessageIDKey is the attribute Key conforming to the
	// "messaging.message.id" semantic conventions. It represents a value used
	// by the messaging system as an identifier for the message, represented as
	// a string.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '452a7c7c7c7048c2f887f61572b18fc2'
	MessagingMessageIDKey = attribute.Key("messaging.message.id")

	// MessagingOperationKey is the attribute Key conforming to the
	// "messaging.operation" semantic conventions. It represents a string
	// identifying the kind of messaging operation.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Note: If a custom value is used, it MUST be of low cardinality.
	MessagingOperationKey = attribute.Key("messaging.operation")

	// MessagingRabbitmqDestinationRoutingKeyKey is the attribute Key
	// conforming to the "messaging.rabbitmq.destination.routing_key" semantic
	// conventions. It represents the rabbitMQ message routing key.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myKey'
	MessagingRabbitmqDestinationRoutingKeyKey = attribute.Key("messaging.rabbitmq.destination.routing_key")

	// MessagingRocketmqClientGroupKey is the attribute Key conforming to the
	// "messaging.rocketmq.client_group" semantic conventions. It represents
	// the name of the RocketMQ producer/consumer group that is handling the
	// message. The client type is identified by the SpanKind.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myConsumerGroup'
	MessagingRocketmqClientGroupKey = attribute.Key("messaging.rocketmq.client_group")

	// MessagingRocketmqConsumptionModelKey is the attribute Key conforming to
	// the "messaging.rocketmq.consumption_model" semantic conventions. It
	// represents the model of message consumption. This only applies to
	// consumer spans.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingRocketmqConsumptionModelKey = attribute.Key("messaging.rocketmq.consumption_model")

	// MessagingRocketmqMessageDelayTimeLevelKey is the attribute Key
	// conforming to the "messaging.rocketmq.message.delay_time_level" semantic
	// conventions. It represents the delay time level for delay message, which
	// determines the message delay time.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 3
	MessagingRocketmqMessageDelayTimeLevelKey = attribute.Key("messaging.rocketmq.message.delay_time_level")

	// MessagingRocketmqMessageDeliveryTimestampKey is the attribute Key
	// conforming to the "messaging.rocketmq.message.delivery_timestamp"
	// semantic conventions. It represents the timestamp in milliseconds that
	// the delay message is expected to be delivered to consumer.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 1665987217045
	MessagingRocketmqMessageDeliveryTimestampKey = attribute.Key("messaging.rocketmq.message.delivery_timestamp")

	// MessagingRocketmqMessageGroupKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.group" semantic conventions. It represents
	// the it is essential for FIFO message. Messages that belong to the same
	// message group are always processed one by one within the same consumer
	// group.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myMessageGroup'
	MessagingRocketmqMessageGroupKey = attribute.Key("messaging.rocketmq.message.group")

	// MessagingRocketmqMessageKeysKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.keys" semantic conventions. It represents
	// the key(s) of message, another way to mark message besides message id.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'keyA', 'keyB'
	MessagingRocketmqMessageKeysKey = attribute.Key("messaging.rocketmq.message.keys")

	// MessagingRocketmqMessageTagKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.tag" semantic conventions. It represents the
	// secondary classifier of message besides topic.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'tagA'
	MessagingRocketmqMessageTagKey = attribute.Key("messaging.rocketmq.message.tag")

	// MessagingRocketmqMessageTypeKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.type" semantic conventions. It represents
	// the type of message.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingRocketmqMessageTypeKey = attribute.Key("messaging.rocketmq.message.type")

	// MessagingRocketmqNamespaceKey is the attribute Key conforming to the
	// "messaging.rocketmq.namespace" semantic conventions. It represents the
	// namespace of RocketMQ resources, resources in different namespaces are
	// individual.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myNamespace'
	MessagingRocketmqNamespaceKey = attribute.Key("messaging.rocketmq.namespace")

	// MessagingSystemKey is the attribute Key conforming to the
	// "messaging.system" semantic conventions. It represents an identifier for
	// the messaging system being used. See below for a list of well-known
	// identifiers.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	MessagingSystemKey = attribute.Key("messaging.system")
)

var (
	// One or more messages are provided for publishing to an intermediary. If a single message is published, the context of the "Publish" span can be used as the creation context and no "Create" span needs to be created
	MessagingOperationPublish = MessagingOperationKey.String("publish")
	// A message is created. "Create" spans always refer to a single message and are used to provide a unique creation context for messages in batch publishing scenarios
	MessagingOperationCreate = MessagingOperationKey.String("create")
	// One or more messages are requested by a consumer. This operation refers to pull-based scenarios, where consumers explicitly call methods of messaging SDKs to receive messages
	MessagingOperationReceive = MessagingOperationKey.String("receive")
	// One or more messages are passed to a consumer. This operation refers to push-based scenarios, where consumer register callbacks which get called by messaging SDKs
	MessagingOperationDeliver = MessagingOperationKey.String("deliver")
)

var (
	// Clustering consumption model
	MessagingRocketmqConsumptionModelClustering = MessagingRocketmqConsumptionModelKey.String("clustering")
	// Broadcasting consumption model
	MessagingRocketmqConsumptionModelBroadcasting = MessagingRocketmqConsumptionModelKey.String("broadcasting")
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
	// Apache ActiveMQ
	MessagingSystemActivemq = MessagingSystemKey.String("activemq")
	// Amazon Simple Queue Service (SQS)
	MessagingSystemAWSSqs = MessagingSystemKey.String("aws_sqs")
	// Azure Event Grid
	MessagingSystemAzureEventgrid = MessagingSystemKey.String("azure_eventgrid")
	// Azure Event Hubs
	MessagingSystemAzureEventhubs = MessagingSystemKey.String("azure_eventhubs")
	// Azure Service Bus
	MessagingSystemAzureServicebus = MessagingSystemKey.String("azure_servicebus")
	// Google Cloud Pub/Sub
	MessagingSystemGCPPubsub = MessagingSystemKey.String("gcp_pubsub")
	// Java Message Service
	MessagingSystemJms = MessagingSystemKey.String("jms")
	// Apache Kafka
	MessagingSystemKafka = MessagingSystemKey.String("kafka")
	// RabbitMQ
	MessagingSystemRabbitmq = MessagingSystemKey.String("rabbitmq")
	// Apache RocketMQ
	MessagingSystemRocketmq = MessagingSystemKey.String("rocketmq")
)

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

// MessagingDestinationAnonymous returns an attribute KeyValue conforming to
// the "messaging.destination.anonymous" semantic conventions. It represents a
// boolean that is true if the message destination is anonymous (could be
// unnamed or have auto-generated name).
func MessagingDestinationAnonymous(val bool) attribute.KeyValue {
	return MessagingDestinationAnonymousKey.Bool(val)
}

// MessagingDestinationName returns an attribute KeyValue conforming to the
// "messaging.destination.name" semantic conventions. It represents the message
// destination name
func MessagingDestinationName(val string) attribute.KeyValue {
	return MessagingDestinationNameKey.String(val)
}

// MessagingDestinationTemplate returns an attribute KeyValue conforming to
// the "messaging.destination.template" semantic conventions. It represents the
// low cardinality representation of the messaging destination name
func MessagingDestinationTemplate(val string) attribute.KeyValue {
	return MessagingDestinationTemplateKey.String(val)
}

// MessagingDestinationTemporary returns an attribute KeyValue conforming to
// the "messaging.destination.temporary" semantic conventions. It represents a
// boolean that is true if the message destination is temporary and might not
// exist anymore after messages are processed.
func MessagingDestinationTemporary(val bool) attribute.KeyValue {
	return MessagingDestinationTemporaryKey.Bool(val)
}

// MessagingDestinationPublishAnonymous returns an attribute KeyValue
// conforming to the "messaging.destination_publish.anonymous" semantic
// conventions. It represents a boolean that is true if the publish message
// destination is anonymous (could be unnamed or have auto-generated name).
func MessagingDestinationPublishAnonymous(val bool) attribute.KeyValue {
	return MessagingDestinationPublishAnonymousKey.Bool(val)
}

// MessagingDestinationPublishName returns an attribute KeyValue conforming
// to the "messaging.destination_publish.name" semantic conventions. It
// represents the name of the original destination the message was published to
func MessagingDestinationPublishName(val string) attribute.KeyValue {
	return MessagingDestinationPublishNameKey.String(val)
}

// MessagingGCPPubsubMessageOrderingKey returns an attribute KeyValue
// conforming to the "messaging.gcp_pubsub.message.ordering_key" semantic
// conventions. It represents the ordering key for a given message. If the
// attribute is not present, the message does not have an ordering key.
func MessagingGCPPubsubMessageOrderingKey(val string) attribute.KeyValue {
	return MessagingGCPPubsubMessageOrderingKeyKey.String(val)
}

// MessagingKafkaConsumerGroup returns an attribute KeyValue conforming to
// the "messaging.kafka.consumer.group" semantic conventions. It represents the
// name of the Kafka Consumer Group that is handling the message. Only applies
// to consumers, not producers.
func MessagingKafkaConsumerGroup(val string) attribute.KeyValue {
	return MessagingKafkaConsumerGroupKey.String(val)
}

// MessagingKafkaDestinationPartition returns an attribute KeyValue
// conforming to the "messaging.kafka.destination.partition" semantic
// conventions. It represents the partition the message is sent to.
func MessagingKafkaDestinationPartition(val int) attribute.KeyValue {
	return MessagingKafkaDestinationPartitionKey.Int(val)
}

// MessagingKafkaMessageKey returns an attribute KeyValue conforming to the
// "messaging.kafka.message.key" semantic conventions. It represents the
// message keys in Kafka are used for grouping alike messages to ensure they're
// processed on the same partition. They differ from `messaging.message.id` in
// that they're not unique. If the key is `null`, the attribute MUST NOT be
// set.
func MessagingKafkaMessageKey(val string) attribute.KeyValue {
	return MessagingKafkaMessageKeyKey.String(val)
}

// MessagingKafkaMessageOffset returns an attribute KeyValue conforming to
// the "messaging.kafka.message.offset" semantic conventions. It represents the
// offset of a record in the corresponding Kafka partition.
func MessagingKafkaMessageOffset(val int) attribute.KeyValue {
	return MessagingKafkaMessageOffsetKey.Int(val)
}

// MessagingKafkaMessageTombstone returns an attribute KeyValue conforming
// to the "messaging.kafka.message.tombstone" semantic conventions. It
// represents a boolean that is true if the message is a tombstone.
func MessagingKafkaMessageTombstone(val bool) attribute.KeyValue {
	return MessagingKafkaMessageTombstoneKey.Bool(val)
}

// MessagingMessageBodySize returns an attribute KeyValue conforming to the
// "messaging.message.body.size" semantic conventions. It represents the size
// of the message body in bytes.
func MessagingMessageBodySize(val int) attribute.KeyValue {
	return MessagingMessageBodySizeKey.Int(val)
}

// MessagingMessageConversationID returns an attribute KeyValue conforming
// to the "messaging.message.conversation_id" semantic conventions. It
// represents the conversation ID identifying the conversation to which the
// message belongs, represented as a string. Sometimes called "Correlation ID".
func MessagingMessageConversationID(val string) attribute.KeyValue {
	return MessagingMessageConversationIDKey.String(val)
}

// MessagingMessageEnvelopeSize returns an attribute KeyValue conforming to
// the "messaging.message.envelope.size" semantic conventions. It represents
// the size of the message body and metadata in bytes.
func MessagingMessageEnvelopeSize(val int) attribute.KeyValue {
	return MessagingMessageEnvelopeSizeKey.Int(val)
}

// MessagingMessageID returns an attribute KeyValue conforming to the
// "messaging.message.id" semantic conventions. It represents a value used by
// the messaging system as an identifier for the message, represented as a
// string.
func MessagingMessageID(val string) attribute.KeyValue {
	return MessagingMessageIDKey.String(val)
}

// MessagingRabbitmqDestinationRoutingKey returns an attribute KeyValue
// conforming to the "messaging.rabbitmq.destination.routing_key" semantic
// conventions. It represents the rabbitMQ message routing key.
func MessagingRabbitmqDestinationRoutingKey(val string) attribute.KeyValue {
	return MessagingRabbitmqDestinationRoutingKeyKey.String(val)
}

// MessagingRocketmqClientGroup returns an attribute KeyValue conforming to
// the "messaging.rocketmq.client_group" semantic conventions. It represents
// the name of the RocketMQ producer/consumer group that is handling the
// message. The client type is identified by the SpanKind.
func MessagingRocketmqClientGroup(val string) attribute.KeyValue {
	return MessagingRocketmqClientGroupKey.String(val)
}

// MessagingRocketmqMessageDelayTimeLevel returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delay_time_level" semantic
// conventions. It represents the delay time level for delay message, which
// determines the message delay time.
func MessagingRocketmqMessageDelayTimeLevel(val int) attribute.KeyValue {
	return MessagingRocketmqMessageDelayTimeLevelKey.Int(val)
}

// MessagingRocketmqMessageDeliveryTimestamp returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delivery_timestamp" semantic
// conventions. It represents the timestamp in milliseconds that the delay
// message is expected to be delivered to consumer.
func MessagingRocketmqMessageDeliveryTimestamp(val int) attribute.KeyValue {
	return MessagingRocketmqMessageDeliveryTimestampKey.Int(val)
}

// MessagingRocketmqMessageGroup returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.group" semantic conventions. It represents
// the it is essential for FIFO message. Messages that belong to the same
// message group are always processed one by one within the same consumer
// group.
func MessagingRocketmqMessageGroup(val string) attribute.KeyValue {
	return MessagingRocketmqMessageGroupKey.String(val)
}

// MessagingRocketmqMessageKeys returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.keys" semantic conventions. It represents
// the key(s) of message, another way to mark message besides message id.
func MessagingRocketmqMessageKeys(val ...string) attribute.KeyValue {
	return MessagingRocketmqMessageKeysKey.StringSlice(val)
}

// MessagingRocketmqMessageTag returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.tag" semantic conventions. It represents the
// secondary classifier of message besides topic.
func MessagingRocketmqMessageTag(val string) attribute.KeyValue {
	return MessagingRocketmqMessageTagKey.String(val)
}

// MessagingRocketmqNamespace returns an attribute KeyValue conforming to
// the "messaging.rocketmq.namespace" semantic conventions. It represents the
// namespace of RocketMQ resources, resources in different namespaces are
// individual.
func MessagingRocketmqNamespace(val string) attribute.KeyValue {
	return MessagingRocketmqNamespaceKey.String(val)
}

// These attributes may be used for any network related operation.
const (
	// NetworkCarrierIccKey is the attribute Key conforming to the
	// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
	// alpha-2 2-character country code associated with the mobile carrier
	// network.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'DE'
	NetworkCarrierIccKey = attribute.Key("network.carrier.icc")

	// NetworkCarrierMccKey is the attribute Key conforming to the
	// "network.carrier.mcc" semantic conventions. It represents the mobile
	// carrier country code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '310'
	NetworkCarrierMccKey = attribute.Key("network.carrier.mcc")

	// NetworkCarrierMncKey is the attribute Key conforming to the
	// "network.carrier.mnc" semantic conventions. It represents the mobile
	// carrier network code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '001'
	NetworkCarrierMncKey = attribute.Key("network.carrier.mnc")

	// NetworkCarrierNameKey is the attribute Key conforming to the
	// "network.carrier.name" semantic conventions. It represents the name of
	// the mobile carrier.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'sprint'
	NetworkCarrierNameKey = attribute.Key("network.carrier.name")

	// NetworkConnectionSubtypeKey is the attribute Key conforming to the
	// "network.connection.subtype" semantic conventions. It represents the
	// this describes more details regarding the connection.type. It may be the
	// type of cell technology connection, but it could be used for describing
	// details about a wifi connection.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'LTE'
	NetworkConnectionSubtypeKey = attribute.Key("network.connection.subtype")

	// NetworkConnectionTypeKey is the attribute Key conforming to the
	// "network.connection.type" semantic conventions. It represents the
	// internet connection type.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'wifi'
	NetworkConnectionTypeKey = attribute.Key("network.connection.type")

	// NetworkIoDirectionKey is the attribute Key conforming to the
	// "network.io.direction" semantic conventions. It represents the network
	// IO operation direction.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'transmit'
	NetworkIoDirectionKey = attribute.Key("network.io.direction")

	// NetworkLocalAddressKey is the attribute Key conforming to the
	// "network.local.address" semantic conventions. It represents the local
	// address of the network connection - IP address or Unix domain socket
	// name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '10.1.2.80', '/tmp/my.sock'
	NetworkLocalAddressKey = attribute.Key("network.local.address")

	// NetworkLocalPortKey is the attribute Key conforming to the
	// "network.local.port" semantic conventions. It represents the local port
	// number of the network connection.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 65123
	NetworkLocalPortKey = attribute.Key("network.local.port")

	// NetworkPeerAddressKey is the attribute Key conforming to the
	// "network.peer.address" semantic conventions. It represents the peer
	// address of the network connection - IP address or Unix domain socket
	// name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '10.1.2.80', '/tmp/my.sock'
	NetworkPeerAddressKey = attribute.Key("network.peer.address")

	// NetworkPeerPortKey is the attribute Key conforming to the
	// "network.peer.port" semantic conventions. It represents the peer port
	// number of the network connection.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 65123
	NetworkPeerPortKey = attribute.Key("network.peer.port")

	// NetworkProtocolNameKey is the attribute Key conforming to the
	// "network.protocol.name" semantic conventions. It represents the [OSI
	// application layer](https://osi-model.com/application-layer/) or non-OSI
	// equivalent.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'amqp', 'http', 'mqtt'
	// Note: The value SHOULD be normalized to lowercase.
	NetworkProtocolNameKey = attribute.Key("network.protocol.name")

	// NetworkProtocolVersionKey is the attribute Key conforming to the
	// "network.protocol.version" semantic conventions. It represents the
	// version of the protocol specified in `network.protocol.name`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '3.1.1'
	// Note: `network.protocol.version` refers to the version of the protocol
	// used and might be different from the protocol client's version. If the
	// HTTP client has a version of `0.27.2`, but sends HTTP version `1.1`,
	// this attribute should be set to `1.1`.
	NetworkProtocolVersionKey = attribute.Key("network.protocol.version")

	// NetworkTransportKey is the attribute Key conforming to the
	// "network.transport" semantic conventions. It represents the [OSI
	// transport layer](https://osi-model.com/transport-layer/) or
	// [inter-process communication
	// method](https://wikipedia.org/wiki/Inter-process_communication).
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'tcp', 'udp'
	// Note: The value SHOULD be normalized to lowercase.
	//
	// Consider always setting the transport when setting a port number, since
	// a port number is ambiguous without knowing the transport. For example
	// different processes could be listening on TCP port 12345 and UDP port
	// 12345.
	NetworkTransportKey = attribute.Key("network.transport")

	// NetworkTypeKey is the attribute Key conforming to the "network.type"
	// semantic conventions. It represents the [OSI network
	// layer](https://osi-model.com/network-layer/) or non-OSI equivalent.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'ipv4', 'ipv6'
	// Note: The value SHOULD be normalized to lowercase.
	NetworkTypeKey = attribute.Key("network.type")
)

var (
	// GPRS
	NetworkConnectionSubtypeGprs = NetworkConnectionSubtypeKey.String("gprs")
	// EDGE
	NetworkConnectionSubtypeEdge = NetworkConnectionSubtypeKey.String("edge")
	// UMTS
	NetworkConnectionSubtypeUmts = NetworkConnectionSubtypeKey.String("umts")
	// CDMA
	NetworkConnectionSubtypeCdma = NetworkConnectionSubtypeKey.String("cdma")
	// EVDO Rel. 0
	NetworkConnectionSubtypeEvdo0 = NetworkConnectionSubtypeKey.String("evdo_0")
	// EVDO Rev. A
	NetworkConnectionSubtypeEvdoA = NetworkConnectionSubtypeKey.String("evdo_a")
	// CDMA2000 1XRTT
	NetworkConnectionSubtypeCdma20001xrtt = NetworkConnectionSubtypeKey.String("cdma2000_1xrtt")
	// HSDPA
	NetworkConnectionSubtypeHsdpa = NetworkConnectionSubtypeKey.String("hsdpa")
	// HSUPA
	NetworkConnectionSubtypeHsupa = NetworkConnectionSubtypeKey.String("hsupa")
	// HSPA
	NetworkConnectionSubtypeHspa = NetworkConnectionSubtypeKey.String("hspa")
	// IDEN
	NetworkConnectionSubtypeIden = NetworkConnectionSubtypeKey.String("iden")
	// EVDO Rev. B
	NetworkConnectionSubtypeEvdoB = NetworkConnectionSubtypeKey.String("evdo_b")
	// LTE
	NetworkConnectionSubtypeLte = NetworkConnectionSubtypeKey.String("lte")
	// EHRPD
	NetworkConnectionSubtypeEhrpd = NetworkConnectionSubtypeKey.String("ehrpd")
	// HSPAP
	NetworkConnectionSubtypeHspap = NetworkConnectionSubtypeKey.String("hspap")
	// GSM
	NetworkConnectionSubtypeGsm = NetworkConnectionSubtypeKey.String("gsm")
	// TD-SCDMA
	NetworkConnectionSubtypeTdScdma = NetworkConnectionSubtypeKey.String("td_scdma")
	// IWLAN
	NetworkConnectionSubtypeIwlan = NetworkConnectionSubtypeKey.String("iwlan")
	// 5G NR (New Radio)
	NetworkConnectionSubtypeNr = NetworkConnectionSubtypeKey.String("nr")
	// 5G NRNSA (New Radio Non-Standalone)
	NetworkConnectionSubtypeNrnsa = NetworkConnectionSubtypeKey.String("nrnsa")
	// LTE CA
	NetworkConnectionSubtypeLteCa = NetworkConnectionSubtypeKey.String("lte_ca")
)

var (
	// wifi
	NetworkConnectionTypeWifi = NetworkConnectionTypeKey.String("wifi")
	// wired
	NetworkConnectionTypeWired = NetworkConnectionTypeKey.String("wired")
	// cell
	NetworkConnectionTypeCell = NetworkConnectionTypeKey.String("cell")
	// unavailable
	NetworkConnectionTypeUnavailable = NetworkConnectionTypeKey.String("unavailable")
	// unknown
	NetworkConnectionTypeUnknown = NetworkConnectionTypeKey.String("unknown")
)

var (
	// transmit
	NetworkIoDirectionTransmit = NetworkIoDirectionKey.String("transmit")
	// receive
	NetworkIoDirectionReceive = NetworkIoDirectionKey.String("receive")
)

var (
	// TCP
	NetworkTransportTCP = NetworkTransportKey.String("tcp")
	// UDP
	NetworkTransportUDP = NetworkTransportKey.String("udp")
	// Named or anonymous pipe
	NetworkTransportPipe = NetworkTransportKey.String("pipe")
	// Unix domain socket
	NetworkTransportUnix = NetworkTransportKey.String("unix")
)

var (
	// IPv4
	NetworkTypeIpv4 = NetworkTypeKey.String("ipv4")
	// IPv6
	NetworkTypeIpv6 = NetworkTypeKey.String("ipv6")
)

// NetworkCarrierIcc returns an attribute KeyValue conforming to the
// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
// alpha-2 2-character country code associated with the mobile carrier network.
func NetworkCarrierIcc(val string) attribute.KeyValue {
	return NetworkCarrierIccKey.String(val)
}

// NetworkCarrierMcc returns an attribute KeyValue conforming to the
// "network.carrier.mcc" semantic conventions. It represents the mobile carrier
// country code.
func NetworkCarrierMcc(val string) attribute.KeyValue {
	return NetworkCarrierMccKey.String(val)
}

// NetworkCarrierMnc returns an attribute KeyValue conforming to the
// "network.carrier.mnc" semantic conventions. It represents the mobile carrier
// network code.
func NetworkCarrierMnc(val string) attribute.KeyValue {
	return NetworkCarrierMncKey.String(val)
}

// NetworkCarrierName returns an attribute KeyValue conforming to the
// "network.carrier.name" semantic conventions. It represents the name of the
// mobile carrier.
func NetworkCarrierName(val string) attribute.KeyValue {
	return NetworkCarrierNameKey.String(val)
}

// NetworkLocalAddress returns an attribute KeyValue conforming to the
// "network.local.address" semantic conventions. It represents the local
// address of the network connection - IP address or Unix domain socket name.
func NetworkLocalAddress(val string) attribute.KeyValue {
	return NetworkLocalAddressKey.String(val)
}

// NetworkLocalPort returns an attribute KeyValue conforming to the
// "network.local.port" semantic conventions. It represents the local port
// number of the network connection.
func NetworkLocalPort(val int) attribute.KeyValue {
	return NetworkLocalPortKey.Int(val)
}

// NetworkPeerAddress returns an attribute KeyValue conforming to the
// "network.peer.address" semantic conventions. It represents the peer address
// of the network connection - IP address or Unix domain socket name.
func NetworkPeerAddress(val string) attribute.KeyValue {
	return NetworkPeerAddressKey.String(val)
}

// NetworkPeerPort returns an attribute KeyValue conforming to the
// "network.peer.port" semantic conventions. It represents the peer port number
// of the network connection.
func NetworkPeerPort(val int) attribute.KeyValue {
	return NetworkPeerPortKey.Int(val)
}

// NetworkProtocolName returns an attribute KeyValue conforming to the
// "network.protocol.name" semantic conventions. It represents the [OSI
// application layer](https://osi-model.com/application-layer/) or non-OSI
// equivalent.
func NetworkProtocolName(val string) attribute.KeyValue {
	return NetworkProtocolNameKey.String(val)
}

// NetworkProtocolVersion returns an attribute KeyValue conforming to the
// "network.protocol.version" semantic conventions. It represents the version
// of the protocol specified in `network.protocol.name`.
func NetworkProtocolVersion(val string) attribute.KeyValue {
	return NetworkProtocolVersionKey.String(val)
}

// Attributes for remote procedure calls.
const (
	// RPCConnectRPCErrorCodeKey is the attribute Key conforming to the
	// "rpc.connect_rpc.error_code" semantic conventions. It represents the
	// [error codes](https://connect.build/docs/protocol/#error-codes) of the
	// Connect request. Error codes are always string values.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	RPCConnectRPCErrorCodeKey = attribute.Key("rpc.connect_rpc.error_code")

	// RPCGRPCStatusCodeKey is the attribute Key conforming to the
	// "rpc.grpc.status_code" semantic conventions. It represents the [numeric
	// status
	// code](https://github.com/grpc/grpc/blob/v1.33.2/doc/statuscodes.md) of
	// the gRPC request.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	RPCGRPCStatusCodeKey = attribute.Key("rpc.grpc.status_code")

	// RPCJsonrpcErrorCodeKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_code" semantic conventions. It represents the
	// `error.code` property of response if it is an error response.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: -32700, 100
	RPCJsonrpcErrorCodeKey = attribute.Key("rpc.jsonrpc.error_code")

	// RPCJsonrpcErrorMessageKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_message" semantic conventions. It represents the
	// `error.message` property of response if it is an error response.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'Parse error', 'User already exists'
	RPCJsonrpcErrorMessageKey = attribute.Key("rpc.jsonrpc.error_message")

	// RPCJsonrpcRequestIDKey is the attribute Key conforming to the
	// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id`
	// property of request or response. Since protocol allows id to be int,
	// string, `null` or missing (for notifications), value is expected to be
	// cast to string for simplicity. Use empty string in case of `null` value.
	// Omit entirely if this is a notification.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '10', 'request-7', ''
	RPCJsonrpcRequestIDKey = attribute.Key("rpc.jsonrpc.request_id")

	// RPCJsonrpcVersionKey is the attribute Key conforming to the
	// "rpc.jsonrpc.version" semantic conventions. It represents the protocol
	// version as in `jsonrpc` property of request/response. Since JSON-RPC 1.0
	// doesn't specify this, the value can be omitted.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2.0', '1.0'
	RPCJsonrpcVersionKey = attribute.Key("rpc.jsonrpc.version")

	// RPCMethodKey is the attribute Key conforming to the "rpc.method"
	// semantic conventions. It represents the name of the (logical) method
	// being called, must be equal to the $method part in the span name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'exampleMethod'
	// Note: This is the logical name of the method from the RPC interface
	// perspective, which can be different from the name of any implementing
	// method/function. The `code.function` attribute may be used to store the
	// latter (e.g., method actually executing the call on the server side, RPC
	// client stub method on the client side).
	RPCMethodKey = attribute.Key("rpc.method")

	// RPCServiceKey is the attribute Key conforming to the "rpc.service"
	// semantic conventions. It represents the full (logical) name of the
	// service being called, including its package name, if applicable.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'myservice.EchoService'
	// Note: This is the logical name of the service from the RPC interface
	// perspective, which can be different from the name of any implementing
	// class. The `code.namespace` attribute may be used to store the latter
	// (despite the attribute name, it may include a class name; e.g., class
	// with method actually executing the call on the server side, RPC client
	// stub class on the client side).
	RPCServiceKey = attribute.Key("rpc.service")

	// RPCSystemKey is the attribute Key conforming to the "rpc.system"
	// semantic conventions. It represents a string identifying the remoting
	// system. See below for a list of well-known identifiers.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	RPCSystemKey = attribute.Key("rpc.system")
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

// RPCJsonrpcRequestID returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id`
// property of request or response. Since protocol allows id to be int, string,
// `null` or missing (for notifications), value is expected to be cast to
// string for simplicity. Use empty string in case of `null` value. Omit
// entirely if this is a notification.
func RPCJsonrpcRequestID(val string) attribute.KeyValue {
	return RPCJsonrpcRequestIDKey.String(val)
}

// RPCJsonrpcVersion returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.version" semantic conventions. It represents the protocol
// version as in `jsonrpc` property of request/response. Since JSON-RPC 1.0
// doesn't specify this, the value can be omitted.
func RPCJsonrpcVersion(val string) attribute.KeyValue {
	return RPCJsonrpcVersionKey.String(val)
}

// RPCMethod returns an attribute KeyValue conforming to the "rpc.method"
// semantic conventions. It represents the name of the (logical) method being
// called, must be equal to the $method part in the span name.
func RPCMethod(val string) attribute.KeyValue {
	return RPCMethodKey.String(val)
}

// RPCService returns an attribute KeyValue conforming to the "rpc.service"
// semantic conventions. It represents the full (logical) name of the service
// being called, including its package name, if applicable.
func RPCService(val string) attribute.KeyValue {
	return RPCServiceKey.String(val)
}

// These attributes may be used to describe the server in a connection-based
// network interaction where there is one side that initiates the connection
// (the client is the side that initiates the connection). This covers all TCP
// network interactions since TCP is connection-based and one side initiates
// the connection (an exception is made for peer-to-peer communication over TCP
// where the "user-facing" surface of the protocol / API doesn't expose a clear
// notion of client and server). This also covers UDP network interactions
// where one side initiates the interaction, e.g. QUIC (HTTP/3) and DNS.
const (
	// ServerAddressKey is the attribute Key conforming to the "server.address"
	// semantic conventions. It represents the server domain name if available
	// without reverse DNS lookup; otherwise, IP address or Unix domain socket
	// name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'example.com', '10.1.2.80', '/tmp/my.sock'
	// Note: When observed from the client side, and when communicating through
	// an intermediary, `server.address` SHOULD represent the server address
	// behind any intermediaries, for example proxies, if it's available.
	ServerAddressKey = attribute.Key("server.address")

	// ServerPortKey is the attribute Key conforming to the "server.port"
	// semantic conventions. It represents the server port number.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 80, 8080, 443
	// Note: When observed from the client side, and when communicating through
	// an intermediary, `server.port` SHOULD represent the server port behind
	// any intermediaries, for example proxies, if it's available.
	ServerPortKey = attribute.Key("server.port")
)

// ServerAddress returns an attribute KeyValue conforming to the
// "server.address" semantic conventions. It represents the server domain name
// if available without reverse DNS lookup; otherwise, IP address or Unix
// domain socket name.
func ServerAddress(val string) attribute.KeyValue {
	return ServerAddressKey.String(val)
}

// ServerPort returns an attribute KeyValue conforming to the "server.port"
// semantic conventions. It represents the server port number.
func ServerPort(val int) attribute.KeyValue {
	return ServerPortKey.Int(val)
}

// These attributes may be used to describe the sender of a network
// exchange/packet. These should be used when there is no client/server
// relationship between the two sides, or when that relationship is unknown.
// This covers low-level network interactions (e.g. packet tracing) where you
// don't know if there was a connection or which side initiated it. This also
// covers unidirectional UDP flows and peer-to-peer communication where the
// "user-facing" surface of the protocol / API doesn't expose a clear notion of
// client and server.
const (
	// SourceAddressKey is the attribute Key conforming to the "source.address"
	// semantic conventions. It represents the source address - domain name if
	// available without reverse DNS lookup; otherwise, IP address or Unix
	// domain socket name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'source.example.com', '10.1.2.80', '/tmp/my.sock'
	// Note: When observed from the destination side, and when communicating
	// through an intermediary, `source.address` SHOULD represent the source
	// address behind any intermediaries, for example proxies, if it's
	// available.
	SourceAddressKey = attribute.Key("source.address")

	// SourcePortKey is the attribute Key conforming to the "source.port"
	// semantic conventions. It represents the source port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 3389, 2888
	SourcePortKey = attribute.Key("source.port")
)

// SourceAddress returns an attribute KeyValue conforming to the
// "source.address" semantic conventions. It represents the source address -
// domain name if available without reverse DNS lookup; otherwise, IP address
// or Unix domain socket name.
func SourceAddress(val string) attribute.KeyValue {
	return SourceAddressKey.String(val)
}

// SourcePort returns an attribute KeyValue conforming to the "source.port"
// semantic conventions. It represents the source port number
func SourcePort(val int) attribute.KeyValue {
	return SourcePortKey.Int(val)
}

// Semantic convention attributes in the TLS namespace.
const (
	// TLSCipherKey is the attribute Key conforming to the "tls.cipher"
	// semantic conventions. It represents the string indicating the
	// [cipher](https://datatracker.ietf.org/doc/html/rfc5246#appendix-A.5)
	// used during the current connection.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'TLS_RSA_WITH_3DES_EDE_CBC_SHA',
	// 'TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256'
	// Note: The values allowed for `tls.cipher` MUST be one of the
	// `Descriptions` of the [registered TLS Cipher
	// Suits](https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#table-tls-parameters-4).
	TLSCipherKey = attribute.Key("tls.cipher")

	// TLSClientCertificateKey is the attribute Key conforming to the
	// "tls.client.certificate" semantic conventions. It represents the
	// pEM-encoded stand-alone certificate offered by the client. This is
	// usually mutually-exclusive of `client.certificate_chain` since this
	// value also exists in that list.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MII...'
	TLSClientCertificateKey = attribute.Key("tls.client.certificate")

	// TLSClientCertificateChainKey is the attribute Key conforming to the
	// "tls.client.certificate_chain" semantic conventions. It represents the
	// array of PEM-encoded certificates that make up the certificate chain
	// offered by the client. This is usually mutually-exclusive of
	// `client.certificate` since that value should be the first certificate in
	// the chain.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MII...', 'MI...'
	TLSClientCertificateChainKey = attribute.Key("tls.client.certificate_chain")

	// TLSClientHashMd5Key is the attribute Key conforming to the
	// "tls.client.hash.md5" semantic conventions. It represents the
	// certificate fingerprint using the MD5 digest of DER-encoded version of
	// certificate offered by the client. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '0F76C7F2C55BFD7D8E8B8F4BFBF0C9EC'
	TLSClientHashMd5Key = attribute.Key("tls.client.hash.md5")

	// TLSClientHashSha1Key is the attribute Key conforming to the
	// "tls.client.hash.sha1" semantic conventions. It represents the
	// certificate fingerprint using the SHA1 digest of DER-encoded version of
	// certificate offered by the client. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '9E393D93138888D288266C2D915214D1D1CCEB2A'
	TLSClientHashSha1Key = attribute.Key("tls.client.hash.sha1")

	// TLSClientHashSha256Key is the attribute Key conforming to the
	// "tls.client.hash.sha256" semantic conventions. It represents the
	// certificate fingerprint using the SHA256 digest of DER-encoded version
	// of certificate offered by the client. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// '0687F666A054EF17A08E2F2162EAB4CBC0D265E1D7875BE74BF3C712CA92DAF0'
	TLSClientHashSha256Key = attribute.Key("tls.client.hash.sha256")

	// TLSClientIssuerKey is the attribute Key conforming to the
	// "tls.client.issuer" semantic conventions. It represents the
	// distinguished name of
	// [subject](https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6)
	// of the issuer of the x.509 certificate presented by the client.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'CN=Example Root CA, OU=Infrastructure Team, DC=example,
	// DC=com'
	TLSClientIssuerKey = attribute.Key("tls.client.issuer")

	// TLSClientJa3Key is the attribute Key conforming to the "tls.client.ja3"
	// semantic conventions. It represents a hash that identifies clients based
	// on how they perform an SSL/TLS handshake.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'd4e5b18d6b55c71272893221c96ba240'
	TLSClientJa3Key = attribute.Key("tls.client.ja3")

	// TLSClientNotAfterKey is the attribute Key conforming to the
	// "tls.client.not_after" semantic conventions. It represents the date/Time
	// indicating when client certificate is no longer considered valid.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2021-01-01T00:00:00.000Z'
	TLSClientNotAfterKey = attribute.Key("tls.client.not_after")

	// TLSClientNotBeforeKey is the attribute Key conforming to the
	// "tls.client.not_before" semantic conventions. It represents the
	// date/Time indicating when client certificate is first considered valid.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '1970-01-01T00:00:00.000Z'
	TLSClientNotBeforeKey = attribute.Key("tls.client.not_before")

	// TLSClientServerNameKey is the attribute Key conforming to the
	// "tls.client.server_name" semantic conventions. It represents the also
	// called an SNI, this tells the server which hostname to which the client
	// is attempting to connect to.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'opentelemetry.io'
	TLSClientServerNameKey = attribute.Key("tls.client.server_name")

	// TLSClientSubjectKey is the attribute Key conforming to the
	// "tls.client.subject" semantic conventions. It represents the
	// distinguished name of subject of the x.509 certificate presented by the
	// client.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'CN=myclient, OU=Documentation Team, DC=example, DC=com'
	TLSClientSubjectKey = attribute.Key("tls.client.subject")

	// TLSClientSupportedCiphersKey is the attribute Key conforming to the
	// "tls.client.supported_ciphers" semantic conventions. It represents the
	// array of ciphers offered by the client during the client hello.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
	// "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "..."'
	TLSClientSupportedCiphersKey = attribute.Key("tls.client.supported_ciphers")

	// TLSCurveKey is the attribute Key conforming to the "tls.curve" semantic
	// conventions. It represents the string indicating the curve used for the
	// given cipher, when applicable
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'secp256r1'
	TLSCurveKey = attribute.Key("tls.curve")

	// TLSEstablishedKey is the attribute Key conforming to the
	// "tls.established" semantic conventions. It represents the boolean flag
	// indicating if the TLS negotiation was successful and transitioned to an
	// encrypted tunnel.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: True
	TLSEstablishedKey = attribute.Key("tls.established")

	// TLSNextProtocolKey is the attribute Key conforming to the
	// "tls.next_protocol" semantic conventions. It represents the string
	// indicating the protocol being tunneled. Per the values in the [IANA
	// registry](https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids),
	// this string should be lower case.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'http/1.1'
	TLSNextProtocolKey = attribute.Key("tls.next_protocol")

	// TLSProtocolNameKey is the attribute Key conforming to the
	// "tls.protocol.name" semantic conventions. It represents the normalized
	// lowercase protocol name parsed from original string of the negotiated
	// [SSL/TLS protocol
	// version](https://www.openssl.org/docs/man1.1.1/man3/SSL_get_version.html#RETURN-VALUES)
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	TLSProtocolNameKey = attribute.Key("tls.protocol.name")

	// TLSProtocolVersionKey is the attribute Key conforming to the
	// "tls.protocol.version" semantic conventions. It represents the numeric
	// part of the version parsed from the original string of the negotiated
	// [SSL/TLS protocol
	// version](https://www.openssl.org/docs/man1.1.1/man3/SSL_get_version.html#RETURN-VALUES)
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '1.2', '3'
	TLSProtocolVersionKey = attribute.Key("tls.protocol.version")

	// TLSResumedKey is the attribute Key conforming to the "tls.resumed"
	// semantic conventions. It represents the boolean flag indicating if this
	// TLS connection was resumed from an existing TLS negotiation.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: True
	TLSResumedKey = attribute.Key("tls.resumed")

	// TLSServerCertificateKey is the attribute Key conforming to the
	// "tls.server.certificate" semantic conventions. It represents the
	// pEM-encoded stand-alone certificate offered by the server. This is
	// usually mutually-exclusive of `server.certificate_chain` since this
	// value also exists in that list.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MII...'
	TLSServerCertificateKey = attribute.Key("tls.server.certificate")

	// TLSServerCertificateChainKey is the attribute Key conforming to the
	// "tls.server.certificate_chain" semantic conventions. It represents the
	// array of PEM-encoded certificates that make up the certificate chain
	// offered by the server. This is usually mutually-exclusive of
	// `server.certificate` since that value should be the first certificate in
	// the chain.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'MII...', 'MI...'
	TLSServerCertificateChainKey = attribute.Key("tls.server.certificate_chain")

	// TLSServerHashMd5Key is the attribute Key conforming to the
	// "tls.server.hash.md5" semantic conventions. It represents the
	// certificate fingerprint using the MD5 digest of DER-encoded version of
	// certificate offered by the server. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '0F76C7F2C55BFD7D8E8B8F4BFBF0C9EC'
	TLSServerHashMd5Key = attribute.Key("tls.server.hash.md5")

	// TLSServerHashSha1Key is the attribute Key conforming to the
	// "tls.server.hash.sha1" semantic conventions. It represents the
	// certificate fingerprint using the SHA1 digest of DER-encoded version of
	// certificate offered by the server. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '9E393D93138888D288266C2D915214D1D1CCEB2A'
	TLSServerHashSha1Key = attribute.Key("tls.server.hash.sha1")

	// TLSServerHashSha256Key is the attribute Key conforming to the
	// "tls.server.hash.sha256" semantic conventions. It represents the
	// certificate fingerprint using the SHA256 digest of DER-encoded version
	// of certificate offered by the server. For consistency with other hash
	// values, this value should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// '0687F666A054EF17A08E2F2162EAB4CBC0D265E1D7875BE74BF3C712CA92DAF0'
	TLSServerHashSha256Key = attribute.Key("tls.server.hash.sha256")

	// TLSServerIssuerKey is the attribute Key conforming to the
	// "tls.server.issuer" semantic conventions. It represents the
	// distinguished name of
	// [subject](https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6)
	// of the issuer of the x.509 certificate presented by the client.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'CN=Example Root CA, OU=Infrastructure Team, DC=example,
	// DC=com'
	TLSServerIssuerKey = attribute.Key("tls.server.issuer")

	// TLSServerJa3sKey is the attribute Key conforming to the
	// "tls.server.ja3s" semantic conventions. It represents a hash that
	// identifies servers based on how they perform an SSL/TLS handshake.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'd4e5b18d6b55c71272893221c96ba240'
	TLSServerJa3sKey = attribute.Key("tls.server.ja3s")

	// TLSServerNotAfterKey is the attribute Key conforming to the
	// "tls.server.not_after" semantic conventions. It represents the date/Time
	// indicating when server certificate is no longer considered valid.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2021-01-01T00:00:00.000Z'
	TLSServerNotAfterKey = attribute.Key("tls.server.not_after")

	// TLSServerNotBeforeKey is the attribute Key conforming to the
	// "tls.server.not_before" semantic conventions. It represents the
	// date/Time indicating when server certificate is first considered valid.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '1970-01-01T00:00:00.000Z'
	TLSServerNotBeforeKey = attribute.Key("tls.server.not_before")

	// TLSServerSubjectKey is the attribute Key conforming to the
	// "tls.server.subject" semantic conventions. It represents the
	// distinguished name of subject of the x.509 certificate presented by the
	// server.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'CN=myserver, OU=Documentation Team, DC=example, DC=com'
	TLSServerSubjectKey = attribute.Key("tls.server.subject")
)

var (
	// ssl
	TLSProtocolNameSsl = TLSProtocolNameKey.String("ssl")
	// tls
	TLSProtocolNameTLS = TLSProtocolNameKey.String("tls")
)

// TLSCipher returns an attribute KeyValue conforming to the "tls.cipher"
// semantic conventions. It represents the string indicating the
// [cipher](https://datatracker.ietf.org/doc/html/rfc5246#appendix-A.5) used
// during the current connection.
func TLSCipher(val string) attribute.KeyValue {
	return TLSCipherKey.String(val)
}

// TLSClientCertificate returns an attribute KeyValue conforming to the
// "tls.client.certificate" semantic conventions. It represents the pEM-encoded
// stand-alone certificate offered by the client. This is usually
// mutually-exclusive of `client.certificate_chain` since this value also
// exists in that list.
func TLSClientCertificate(val string) attribute.KeyValue {
	return TLSClientCertificateKey.String(val)
}

// TLSClientCertificateChain returns an attribute KeyValue conforming to the
// "tls.client.certificate_chain" semantic conventions. It represents the array
// of PEM-encoded certificates that make up the certificate chain offered by
// the client. This is usually mutually-exclusive of `client.certificate` since
// that value should be the first certificate in the chain.
func TLSClientCertificateChain(val ...string) attribute.KeyValue {
	return TLSClientCertificateChainKey.StringSlice(val)
}

// TLSClientHashMd5 returns an attribute KeyValue conforming to the
// "tls.client.hash.md5" semantic conventions. It represents the certificate
// fingerprint using the MD5 digest of DER-encoded version of certificate
// offered by the client. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSClientHashMd5(val string) attribute.KeyValue {
	return TLSClientHashMd5Key.String(val)
}

// TLSClientHashSha1 returns an attribute KeyValue conforming to the
// "tls.client.hash.sha1" semantic conventions. It represents the certificate
// fingerprint using the SHA1 digest of DER-encoded version of certificate
// offered by the client. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSClientHashSha1(val string) attribute.KeyValue {
	return TLSClientHashSha1Key.String(val)
}

// TLSClientHashSha256 returns an attribute KeyValue conforming to the
// "tls.client.hash.sha256" semantic conventions. It represents the certificate
// fingerprint using the SHA256 digest of DER-encoded version of certificate
// offered by the client. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSClientHashSha256(val string) attribute.KeyValue {
	return TLSClientHashSha256Key.String(val)
}

// TLSClientIssuer returns an attribute KeyValue conforming to the
// "tls.client.issuer" semantic conventions. It represents the distinguished
// name of
// [subject](https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6) of
// the issuer of the x.509 certificate presented by the client.
func TLSClientIssuer(val string) attribute.KeyValue {
	return TLSClientIssuerKey.String(val)
}

// TLSClientJa3 returns an attribute KeyValue conforming to the
// "tls.client.ja3" semantic conventions. It represents a hash that identifies
// clients based on how they perform an SSL/TLS handshake.
func TLSClientJa3(val string) attribute.KeyValue {
	return TLSClientJa3Key.String(val)
}

// TLSClientNotAfter returns an attribute KeyValue conforming to the
// "tls.client.not_after" semantic conventions. It represents the date/Time
// indicating when client certificate is no longer considered valid.
func TLSClientNotAfter(val string) attribute.KeyValue {
	return TLSClientNotAfterKey.String(val)
}

// TLSClientNotBefore returns an attribute KeyValue conforming to the
// "tls.client.not_before" semantic conventions. It represents the date/Time
// indicating when client certificate is first considered valid.
func TLSClientNotBefore(val string) attribute.KeyValue {
	return TLSClientNotBeforeKey.String(val)
}

// TLSClientServerName returns an attribute KeyValue conforming to the
// "tls.client.server_name" semantic conventions. It represents the also called
// an SNI, this tells the server which hostname to which the client is
// attempting to connect to.
func TLSClientServerName(val string) attribute.KeyValue {
	return TLSClientServerNameKey.String(val)
}

// TLSClientSubject returns an attribute KeyValue conforming to the
// "tls.client.subject" semantic conventions. It represents the distinguished
// name of subject of the x.509 certificate presented by the client.
func TLSClientSubject(val string) attribute.KeyValue {
	return TLSClientSubjectKey.String(val)
}

// TLSClientSupportedCiphers returns an attribute KeyValue conforming to the
// "tls.client.supported_ciphers" semantic conventions. It represents the array
// of ciphers offered by the client during the client hello.
func TLSClientSupportedCiphers(val ...string) attribute.KeyValue {
	return TLSClientSupportedCiphersKey.StringSlice(val)
}

// TLSCurve returns an attribute KeyValue conforming to the "tls.curve"
// semantic conventions. It represents the string indicating the curve used for
// the given cipher, when applicable
func TLSCurve(val string) attribute.KeyValue {
	return TLSCurveKey.String(val)
}

// TLSEstablished returns an attribute KeyValue conforming to the
// "tls.established" semantic conventions. It represents the boolean flag
// indicating if the TLS negotiation was successful and transitioned to an
// encrypted tunnel.
func TLSEstablished(val bool) attribute.KeyValue {
	return TLSEstablishedKey.Bool(val)
}

// TLSNextProtocol returns an attribute KeyValue conforming to the
// "tls.next_protocol" semantic conventions. It represents the string
// indicating the protocol being tunneled. Per the values in the [IANA
// registry](https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids),
// this string should be lower case.
func TLSNextProtocol(val string) attribute.KeyValue {
	return TLSNextProtocolKey.String(val)
}

// TLSProtocolVersion returns an attribute KeyValue conforming to the
// "tls.protocol.version" semantic conventions. It represents the numeric part
// of the version parsed from the original string of the negotiated [SSL/TLS
// protocol
// version](https://www.openssl.org/docs/man1.1.1/man3/SSL_get_version.html#RETURN-VALUES)
func TLSProtocolVersion(val string) attribute.KeyValue {
	return TLSProtocolVersionKey.String(val)
}

// TLSResumed returns an attribute KeyValue conforming to the "tls.resumed"
// semantic conventions. It represents the boolean flag indicating if this TLS
// connection was resumed from an existing TLS negotiation.
func TLSResumed(val bool) attribute.KeyValue {
	return TLSResumedKey.Bool(val)
}

// TLSServerCertificate returns an attribute KeyValue conforming to the
// "tls.server.certificate" semantic conventions. It represents the pEM-encoded
// stand-alone certificate offered by the server. This is usually
// mutually-exclusive of `server.certificate_chain` since this value also
// exists in that list.
func TLSServerCertificate(val string) attribute.KeyValue {
	return TLSServerCertificateKey.String(val)
}

// TLSServerCertificateChain returns an attribute KeyValue conforming to the
// "tls.server.certificate_chain" semantic conventions. It represents the array
// of PEM-encoded certificates that make up the certificate chain offered by
// the server. This is usually mutually-exclusive of `server.certificate` since
// that value should be the first certificate in the chain.
func TLSServerCertificateChain(val ...string) attribute.KeyValue {
	return TLSServerCertificateChainKey.StringSlice(val)
}

// TLSServerHashMd5 returns an attribute KeyValue conforming to the
// "tls.server.hash.md5" semantic conventions. It represents the certificate
// fingerprint using the MD5 digest of DER-encoded version of certificate
// offered by the server. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSServerHashMd5(val string) attribute.KeyValue {
	return TLSServerHashMd5Key.String(val)
}

// TLSServerHashSha1 returns an attribute KeyValue conforming to the
// "tls.server.hash.sha1" semantic conventions. It represents the certificate
// fingerprint using the SHA1 digest of DER-encoded version of certificate
// offered by the server. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSServerHashSha1(val string) attribute.KeyValue {
	return TLSServerHashSha1Key.String(val)
}

// TLSServerHashSha256 returns an attribute KeyValue conforming to the
// "tls.server.hash.sha256" semantic conventions. It represents the certificate
// fingerprint using the SHA256 digest of DER-encoded version of certificate
// offered by the server. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSServerHashSha256(val string) attribute.KeyValue {
	return TLSServerHashSha256Key.String(val)
}

// TLSServerIssuer returns an attribute KeyValue conforming to the
// "tls.server.issuer" semantic conventions. It represents the distinguished
// name of
// [subject](https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6) of
// the issuer of the x.509 certificate presented by the client.
func TLSServerIssuer(val string) attribute.KeyValue {
	return TLSServerIssuerKey.String(val)
}

// TLSServerJa3s returns an attribute KeyValue conforming to the
// "tls.server.ja3s" semantic conventions. It represents a hash that identifies
// servers based on how they perform an SSL/TLS handshake.
func TLSServerJa3s(val string) attribute.KeyValue {
	return TLSServerJa3sKey.String(val)
}

// TLSServerNotAfter returns an attribute KeyValue conforming to the
// "tls.server.not_after" semantic conventions. It represents the date/Time
// indicating when server certificate is no longer considered valid.
func TLSServerNotAfter(val string) attribute.KeyValue {
	return TLSServerNotAfterKey.String(val)
}

// TLSServerNotBefore returns an attribute KeyValue conforming to the
// "tls.server.not_before" semantic conventions. It represents the date/Time
// indicating when server certificate is first considered valid.
func TLSServerNotBefore(val string) attribute.KeyValue {
	return TLSServerNotBeforeKey.String(val)
}

// TLSServerSubject returns an attribute KeyValue conforming to the
// "tls.server.subject" semantic conventions. It represents the distinguished
// name of subject of the x.509 certificate presented by the server.
func TLSServerSubject(val string) attribute.KeyValue {
	return TLSServerSubjectKey.String(val)
}

// Attributes describing URL.
const (
	// URLFragmentKey is the attribute Key conforming to the "url.fragment"
	// semantic conventions. It represents the [URI
	// fragment](https://www.rfc-editor.org/rfc/rfc3986#section-3.5) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'SemConv'
	URLFragmentKey = attribute.Key("url.fragment")

	// URLFullKey is the attribute Key conforming to the "url.full" semantic
	// conventions. It represents the absolute URL describing a network
	// resource according to [RFC3986](https://www.rfc-editor.org/rfc/rfc3986)
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'https://www.foo.bar/search?q=OpenTelemetry#SemConv',
	// '//localhost'
	// Note: For network calls, URL usually has
	// `scheme://host[:port][path][?query][#fragment]` format, where the
	// fragment is not transmitted over HTTP, but if it is known, it SHOULD be
	// included nevertheless.
	// `url.full` MUST NOT contain credentials passed via URL in form of
	// `https://username:password@www.example.com/`. In such case username and
	// password SHOULD be redacted and attribute's value SHOULD be
	// `https://REDACTED:REDACTED@www.example.com/`.
	// `url.full` SHOULD capture the absolute URL when it is available (or can
	// be reconstructed) and SHOULD NOT be validated or modified except for
	// sanitizing purposes.
	URLFullKey = attribute.Key("url.full")

	// URLPathKey is the attribute Key conforming to the "url.path" semantic
	// conventions. It represents the [URI
	// path](https://www.rfc-editor.org/rfc/rfc3986#section-3.3) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/search'
	URLPathKey = attribute.Key("url.path")

	// URLQueryKey is the attribute Key conforming to the "url.query" semantic
	// conventions. It represents the [URI
	// query](https://www.rfc-editor.org/rfc/rfc3986#section-3.4) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'q=OpenTelemetry'
	// Note: Sensitive content provided in query string SHOULD be scrubbed when
	// instrumentations can identify it.
	URLQueryKey = attribute.Key("url.query")

	// URLSchemeKey is the attribute Key conforming to the "url.scheme"
	// semantic conventions. It represents the [URI
	// scheme](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) component
	// identifying the used protocol.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'https', 'ftp', 'telnet'
	URLSchemeKey = attribute.Key("url.scheme")
)

// URLFragment returns an attribute KeyValue conforming to the
// "url.fragment" semantic conventions. It represents the [URI
// fragment](https://www.rfc-editor.org/rfc/rfc3986#section-3.5) component
func URLFragment(val string) attribute.KeyValue {
	return URLFragmentKey.String(val)
}

// URLFull returns an attribute KeyValue conforming to the "url.full"
// semantic conventions. It represents the absolute URL describing a network
// resource according to [RFC3986](https://www.rfc-editor.org/rfc/rfc3986)
func URLFull(val string) attribute.KeyValue {
	return URLFullKey.String(val)
}

// URLPath returns an attribute KeyValue conforming to the "url.path"
// semantic conventions. It represents the [URI
// path](https://www.rfc-editor.org/rfc/rfc3986#section-3.3) component
func URLPath(val string) attribute.KeyValue {
	return URLPathKey.String(val)
}

// URLQuery returns an attribute KeyValue conforming to the "url.query"
// semantic conventions. It represents the [URI
// query](https://www.rfc-editor.org/rfc/rfc3986#section-3.4) component
func URLQuery(val string) attribute.KeyValue {
	return URLQueryKey.String(val)
}

// URLScheme returns an attribute KeyValue conforming to the "url.scheme"
// semantic conventions. It represents the [URI
// scheme](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) component
// identifying the used protocol.
func URLScheme(val string) attribute.KeyValue {
	return URLSchemeKey.String(val)
}

// Describes user-agent attributes.
const (
	// UserAgentOriginalKey is the attribute Key conforming to the
	// "user_agent.original" semantic conventions. It represents the value of
	// the [HTTP
	// User-Agent](https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent)
	// header sent by the client.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'CERN-LineMode/2.15 libwww/2.17b3', 'Mozilla/5.0 (iPhone; CPU
	// iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko)
	// Version/14.1.2 Mobile/15E148 Safari/604.1'
	UserAgentOriginalKey = attribute.Key("user_agent.original")
)

// UserAgentOriginal returns an attribute KeyValue conforming to the
// "user_agent.original" semantic conventions. It represents the value of the
// [HTTP
// User-Agent](https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent)
// header sent by the client.
func UserAgentOriginal(val string) attribute.KeyValue {
	return UserAgentOriginalKey.String(val)
}

// Session is defined as the period of time encompassing all activities
// performed by the application and the actions executed by the end user.
// Consequently, a Session is represented as a collection of Logs, Events, and
// Spans emitted by the Client Application throughout the Session's duration.
// Each Session is assigned a unique identifier, which is included as an
// attribute in the Logs, Events, and Spans generated during the Session's
// lifecycle.
// When a session reaches end of life, typically due to user inactivity or
// session timeout, a new session identifier will be assigned. The previous
// session identifier may be provided by the instrumentation so that telemetry
// backends can link the two sessions.
const (
	// SessionIDKey is the attribute Key conforming to the "session.id"
	// semantic conventions. It represents a unique id to identify a session.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '00112233-4455-6677-8899-aabbccddeeff'
	SessionIDKey = attribute.Key("session.id")

	// SessionPreviousIDKey is the attribute Key conforming to the
	// "session.previous_id" semantic conventions. It represents the previous
	// `session.id` for this user, when known.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '00112233-4455-6677-8899-aabbccddeeff'
	SessionPreviousIDKey = attribute.Key("session.previous_id")
)

// SessionID returns an attribute KeyValue conforming to the "session.id"
// semantic conventions. It represents a unique id to identify a session.
func SessionID(val string) attribute.KeyValue {
	return SessionIDKey.String(val)
}

// SessionPreviousID returns an attribute KeyValue conforming to the
// "session.previous_id" semantic conventions. It represents the previous
// `session.id` for this user, when known.
func SessionPreviousID(val string) attribute.KeyValue {
	return SessionPreviousIDKey.String(val)
}
