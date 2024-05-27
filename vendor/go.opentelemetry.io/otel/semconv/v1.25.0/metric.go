// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.25.0"

const (

	// ContainerCPUTime is the metric conforming to the "container.cpu.time"
	// semantic conventions. It represents the total CPU time consumed.
	// Instrument: counter
	// Unit: s
	// Stability: Experimental
	ContainerCPUTimeName        = "container.cpu.time"
	ContainerCPUTimeUnit        = "s"
	ContainerCPUTimeDescription = "Total CPU time consumed"

	// ContainerMemoryUsage is the metric conforming to the
	// "container.memory.usage" semantic conventions. It represents the memory
	// usage of the container.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	ContainerMemoryUsageName        = "container.memory.usage"
	ContainerMemoryUsageUnit        = "By"
	ContainerMemoryUsageDescription = "Memory usage of the container."

	// ContainerDiskIo is the metric conforming to the "container.disk.io" semantic
	// conventions. It represents the disk bytes for the container.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	ContainerDiskIoName        = "container.disk.io"
	ContainerDiskIoUnit        = "By"
	ContainerDiskIoDescription = "Disk bytes for the container."

	// ContainerNetworkIo is the metric conforming to the "container.network.io"
	// semantic conventions. It represents the network bytes for the container.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	ContainerNetworkIoName        = "container.network.io"
	ContainerNetworkIoUnit        = "By"
	ContainerNetworkIoDescription = "Network bytes for the container."

	// DBClientConnectionsUsage is the metric conforming to the
	// "db.client.connections.usage" semantic conventions. It represents the number
	// of connections that are currently in state described by the `state`
	// attribute.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	DBClientConnectionsUsageName        = "db.client.connections.usage"
	DBClientConnectionsUsageUnit        = "{connection}"
	DBClientConnectionsUsageDescription = "The number of connections that are currently in state described by the `state` attribute"

	// DBClientConnectionsIdleMax is the metric conforming to the
	// "db.client.connections.idle.max" semantic conventions. It represents the
	// maximum number of idle open connections allowed.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	DBClientConnectionsIdleMaxName        = "db.client.connections.idle.max"
	DBClientConnectionsIdleMaxUnit        = "{connection}"
	DBClientConnectionsIdleMaxDescription = "The maximum number of idle open connections allowed"

	// DBClientConnectionsIdleMin is the metric conforming to the
	// "db.client.connections.idle.min" semantic conventions. It represents the
	// minimum number of idle open connections allowed.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	DBClientConnectionsIdleMinName        = "db.client.connections.idle.min"
	DBClientConnectionsIdleMinUnit        = "{connection}"
	DBClientConnectionsIdleMinDescription = "The minimum number of idle open connections allowed"

	// DBClientConnectionsMax is the metric conforming to the
	// "db.client.connections.max" semantic conventions. It represents the maximum
	// number of open connections allowed.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	DBClientConnectionsMaxName        = "db.client.connections.max"
	DBClientConnectionsMaxUnit        = "{connection}"
	DBClientConnectionsMaxDescription = "The maximum number of open connections allowed"

	// DBClientConnectionsPendingRequests is the metric conforming to the
	// "db.client.connections.pending_requests" semantic conventions. It represents
	// the number of pending requests for an open connection, cumulative for the
	// entire pool.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Experimental
	DBClientConnectionsPendingRequestsName        = "db.client.connections.pending_requests"
	DBClientConnectionsPendingRequestsUnit        = "{request}"
	DBClientConnectionsPendingRequestsDescription = "The number of pending requests for an open connection, cumulative for the entire pool"

	// DBClientConnectionsTimeouts is the metric conforming to the
	// "db.client.connections.timeouts" semantic conventions. It represents the
	// number of connection timeouts that have occurred trying to obtain a
	// connection from the pool.
	// Instrument: counter
	// Unit: {timeout}
	// Stability: Experimental
	DBClientConnectionsTimeoutsName        = "db.client.connections.timeouts"
	DBClientConnectionsTimeoutsUnit        = "{timeout}"
	DBClientConnectionsTimeoutsDescription = "The number of connection timeouts that have occurred trying to obtain a connection from the pool"

	// DBClientConnectionsCreateTime is the metric conforming to the
	// "db.client.connections.create_time" semantic conventions. It represents the
	// time it took to create a new connection.
	// Instrument: histogram
	// Unit: ms
	// Stability: Experimental
	DBClientConnectionsCreateTimeName        = "db.client.connections.create_time"
	DBClientConnectionsCreateTimeUnit        = "ms"
	DBClientConnectionsCreateTimeDescription = "The time it took to create a new connection"

	// DBClientConnectionsWaitTime is the metric conforming to the
	// "db.client.connections.wait_time" semantic conventions. It represents the
	// time it took to obtain an open connection from the pool.
	// Instrument: histogram
	// Unit: ms
	// Stability: Experimental
	DBClientConnectionsWaitTimeName        = "db.client.connections.wait_time"
	DBClientConnectionsWaitTimeUnit        = "ms"
	DBClientConnectionsWaitTimeDescription = "The time it took to obtain an open connection from the pool"

	// DBClientConnectionsUseTime is the metric conforming to the
	// "db.client.connections.use_time" semantic conventions. It represents the
	// time between borrowing a connection and returning it to the pool.
	// Instrument: histogram
	// Unit: ms
	// Stability: Experimental
	DBClientConnectionsUseTimeName        = "db.client.connections.use_time"
	DBClientConnectionsUseTimeUnit        = "ms"
	DBClientConnectionsUseTimeDescription = "The time between borrowing a connection and returning it to the pool"

	// DNSLookupDuration is the metric conforming to the "dns.lookup.duration"
	// semantic conventions. It represents the measures the time taken to perform a
	// DNS lookup.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	DNSLookupDurationName        = "dns.lookup.duration"
	DNSLookupDurationUnit        = "s"
	DNSLookupDurationDescription = "Measures the time taken to perform a DNS lookup."

	// AspnetcoreRoutingMatchAttempts is the metric conforming to the
	// "aspnetcore.routing.match_attempts" semantic conventions. It represents the
	// number of requests that were attempted to be matched to an endpoint.
	// Instrument: counter
	// Unit: {match_attempt}
	// Stability: Stable
	AspnetcoreRoutingMatchAttemptsName        = "aspnetcore.routing.match_attempts"
	AspnetcoreRoutingMatchAttemptsUnit        = "{match_attempt}"
	AspnetcoreRoutingMatchAttemptsDescription = "Number of requests that were attempted to be matched to an endpoint."

	// AspnetcoreDiagnosticsExceptions is the metric conforming to the
	// "aspnetcore.diagnostics.exceptions" semantic conventions. It represents the
	// number of exceptions caught by exception handling middleware.
	// Instrument: counter
	// Unit: {exception}
	// Stability: Stable
	AspnetcoreDiagnosticsExceptionsName        = "aspnetcore.diagnostics.exceptions"
	AspnetcoreDiagnosticsExceptionsUnit        = "{exception}"
	AspnetcoreDiagnosticsExceptionsDescription = "Number of exceptions caught by exception handling middleware."

	// AspnetcoreRateLimitingActiveRequestLeases is the metric conforming to the
	// "aspnetcore.rate_limiting.active_request_leases" semantic conventions. It
	// represents the number of requests that are currently active on the server
	// that hold a rate limiting lease.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Stable
	AspnetcoreRateLimitingActiveRequestLeasesName        = "aspnetcore.rate_limiting.active_request_leases"
	AspnetcoreRateLimitingActiveRequestLeasesUnit        = "{request}"
	AspnetcoreRateLimitingActiveRequestLeasesDescription = "Number of requests that are currently active on the server that hold a rate limiting lease."

	// AspnetcoreRateLimitingRequestLeaseDuration is the metric conforming to the
	// "aspnetcore.rate_limiting.request_lease.duration" semantic conventions. It
	// represents the duration of rate limiting lease held by requests on the
	// server.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	AspnetcoreRateLimitingRequestLeaseDurationName        = "aspnetcore.rate_limiting.request_lease.duration"
	AspnetcoreRateLimitingRequestLeaseDurationUnit        = "s"
	AspnetcoreRateLimitingRequestLeaseDurationDescription = "The duration of rate limiting lease held by requests on the server."

	// AspnetcoreRateLimitingRequestTimeInQueue is the metric conforming to the
	// "aspnetcore.rate_limiting.request.time_in_queue" semantic conventions. It
	// represents the time the request spent in a queue waiting to acquire a rate
	// limiting lease.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	AspnetcoreRateLimitingRequestTimeInQueueName        = "aspnetcore.rate_limiting.request.time_in_queue"
	AspnetcoreRateLimitingRequestTimeInQueueUnit        = "s"
	AspnetcoreRateLimitingRequestTimeInQueueDescription = "The time the request spent in a queue waiting to acquire a rate limiting lease."

	// AspnetcoreRateLimitingQueuedRequests is the metric conforming to the
	// "aspnetcore.rate_limiting.queued_requests" semantic conventions. It
	// represents the number of requests that are currently queued, waiting to
	// acquire a rate limiting lease.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Stable
	AspnetcoreRateLimitingQueuedRequestsName        = "aspnetcore.rate_limiting.queued_requests"
	AspnetcoreRateLimitingQueuedRequestsUnit        = "{request}"
	AspnetcoreRateLimitingQueuedRequestsDescription = "Number of requests that are currently queued, waiting to acquire a rate limiting lease."

	// AspnetcoreRateLimitingRequests is the metric conforming to the
	// "aspnetcore.rate_limiting.requests" semantic conventions. It represents the
	// number of requests that tried to acquire a rate limiting lease.
	// Instrument: counter
	// Unit: {request}
	// Stability: Stable
	AspnetcoreRateLimitingRequestsName        = "aspnetcore.rate_limiting.requests"
	AspnetcoreRateLimitingRequestsUnit        = "{request}"
	AspnetcoreRateLimitingRequestsDescription = "Number of requests that tried to acquire a rate limiting lease."

	// KestrelActiveConnections is the metric conforming to the
	// "kestrel.active_connections" semantic conventions. It represents the number
	// of connections that are currently active on the server.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Stable
	KestrelActiveConnectionsName        = "kestrel.active_connections"
	KestrelActiveConnectionsUnit        = "{connection}"
	KestrelActiveConnectionsDescription = "Number of connections that are currently active on the server."

	// KestrelConnectionDuration is the metric conforming to the
	// "kestrel.connection.duration" semantic conventions. It represents the
	// duration of connections on the server.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	KestrelConnectionDurationName        = "kestrel.connection.duration"
	KestrelConnectionDurationUnit        = "s"
	KestrelConnectionDurationDescription = "The duration of connections on the server."

	// KestrelRejectedConnections is the metric conforming to the
	// "kestrel.rejected_connections" semantic conventions. It represents the
	// number of connections rejected by the server.
	// Instrument: counter
	// Unit: {connection}
	// Stability: Stable
	KestrelRejectedConnectionsName        = "kestrel.rejected_connections"
	KestrelRejectedConnectionsUnit        = "{connection}"
	KestrelRejectedConnectionsDescription = "Number of connections rejected by the server."

	// KestrelQueuedConnections is the metric conforming to the
	// "kestrel.queued_connections" semantic conventions. It represents the number
	// of connections that are currently queued and are waiting to start.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Stable
	KestrelQueuedConnectionsName        = "kestrel.queued_connections"
	KestrelQueuedConnectionsUnit        = "{connection}"
	KestrelQueuedConnectionsDescription = "Number of connections that are currently queued and are waiting to start."

	// KestrelQueuedRequests is the metric conforming to the
	// "kestrel.queued_requests" semantic conventions. It represents the number of
	// HTTP requests on multiplexed connections (HTTP/2 and HTTP/3) that are
	// currently queued and are waiting to start.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Stable
	KestrelQueuedRequestsName        = "kestrel.queued_requests"
	KestrelQueuedRequestsUnit        = "{request}"
	KestrelQueuedRequestsDescription = "Number of HTTP requests on multiplexed connections (HTTP/2 and HTTP/3) that are currently queued and are waiting to start."

	// KestrelUpgradedConnections is the metric conforming to the
	// "kestrel.upgraded_connections" semantic conventions. It represents the
	// number of connections that are currently upgraded (WebSockets). .
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Stable
	KestrelUpgradedConnectionsName        = "kestrel.upgraded_connections"
	KestrelUpgradedConnectionsUnit        = "{connection}"
	KestrelUpgradedConnectionsDescription = "Number of connections that are currently upgraded (WebSockets). ."

	// KestrelTLSHandshakeDuration is the metric conforming to the
	// "kestrel.tls_handshake.duration" semantic conventions. It represents the
	// duration of TLS handshakes on the server.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	KestrelTLSHandshakeDurationName        = "kestrel.tls_handshake.duration"
	KestrelTLSHandshakeDurationUnit        = "s"
	KestrelTLSHandshakeDurationDescription = "The duration of TLS handshakes on the server."

	// KestrelActiveTLSHandshakes is the metric conforming to the
	// "kestrel.active_tls_handshakes" semantic conventions. It represents the
	// number of TLS handshakes that are currently in progress on the server.
	// Instrument: updowncounter
	// Unit: {handshake}
	// Stability: Stable
	KestrelActiveTLSHandshakesName        = "kestrel.active_tls_handshakes"
	KestrelActiveTLSHandshakesUnit        = "{handshake}"
	KestrelActiveTLSHandshakesDescription = "Number of TLS handshakes that are currently in progress on the server."

	// SignalrServerConnectionDuration is the metric conforming to the
	// "signalr.server.connection.duration" semantic conventions. It represents the
	// duration of connections on the server.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	SignalrServerConnectionDurationName        = "signalr.server.connection.duration"
	SignalrServerConnectionDurationUnit        = "s"
	SignalrServerConnectionDurationDescription = "The duration of connections on the server."

	// SignalrServerActiveConnections is the metric conforming to the
	// "signalr.server.active_connections" semantic conventions. It represents the
	// number of connections that are currently active on the server.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Stable
	SignalrServerActiveConnectionsName        = "signalr.server.active_connections"
	SignalrServerActiveConnectionsUnit        = "{connection}"
	SignalrServerActiveConnectionsDescription = "Number of connections that are currently active on the server."

	// FaaSInvokeDuration is the metric conforming to the "faas.invoke_duration"
	// semantic conventions. It represents the measures the duration of the
	// function's logic execution.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	FaaSInvokeDurationName        = "faas.invoke_duration"
	FaaSInvokeDurationUnit        = "s"
	FaaSInvokeDurationDescription = "Measures the duration of the function's logic execution"

	// FaaSInitDuration is the metric conforming to the "faas.init_duration"
	// semantic conventions. It represents the measures the duration of the
	// function's initialization, such as a cold start.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	FaaSInitDurationName        = "faas.init_duration"
	FaaSInitDurationUnit        = "s"
	FaaSInitDurationDescription = "Measures the duration of the function's initialization, such as a cold start"

	// FaaSColdstarts is the metric conforming to the "faas.coldstarts" semantic
	// conventions. It represents the number of invocation cold starts.
	// Instrument: counter
	// Unit: {coldstart}
	// Stability: Experimental
	FaaSColdstartsName        = "faas.coldstarts"
	FaaSColdstartsUnit        = "{coldstart}"
	FaaSColdstartsDescription = "Number of invocation cold starts"

	// FaaSErrors is the metric conforming to the "faas.errors" semantic
	// conventions. It represents the number of invocation errors.
	// Instrument: counter
	// Unit: {error}
	// Stability: Experimental
	FaaSErrorsName        = "faas.errors"
	FaaSErrorsUnit        = "{error}"
	FaaSErrorsDescription = "Number of invocation errors"

	// FaaSInvocations is the metric conforming to the "faas.invocations" semantic
	// conventions. It represents the number of successful invocations.
	// Instrument: counter
	// Unit: {invocation}
	// Stability: Experimental
	FaaSInvocationsName        = "faas.invocations"
	FaaSInvocationsUnit        = "{invocation}"
	FaaSInvocationsDescription = "Number of successful invocations"

	// FaaSTimeouts is the metric conforming to the "faas.timeouts" semantic
	// conventions. It represents the number of invocation timeouts.
	// Instrument: counter
	// Unit: {timeout}
	// Stability: Experimental
	FaaSTimeoutsName        = "faas.timeouts"
	FaaSTimeoutsUnit        = "{timeout}"
	FaaSTimeoutsDescription = "Number of invocation timeouts"

	// FaaSMemUsage is the metric conforming to the "faas.mem_usage" semantic
	// conventions. It represents the distribution of max memory usage per
	// invocation.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	FaaSMemUsageName        = "faas.mem_usage"
	FaaSMemUsageUnit        = "By"
	FaaSMemUsageDescription = "Distribution of max memory usage per invocation"

	// FaaSCPUUsage is the metric conforming to the "faas.cpu_usage" semantic
	// conventions. It represents the distribution of CPU usage per invocation.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	FaaSCPUUsageName        = "faas.cpu_usage"
	FaaSCPUUsageUnit        = "s"
	FaaSCPUUsageDescription = "Distribution of CPU usage per invocation"

	// FaaSNetIo is the metric conforming to the "faas.net_io" semantic
	// conventions. It represents the distribution of net I/O usage per invocation.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	FaaSNetIoName        = "faas.net_io"
	FaaSNetIoUnit        = "By"
	FaaSNetIoDescription = "Distribution of net I/O usage per invocation"

	// HTTPServerRequestDuration is the metric conforming to the
	// "http.server.request.duration" semantic conventions. It represents the
	// duration of HTTP server requests.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	HTTPServerRequestDurationName        = "http.server.request.duration"
	HTTPServerRequestDurationUnit        = "s"
	HTTPServerRequestDurationDescription = "Duration of HTTP server requests."

	// HTTPServerActiveRequests is the metric conforming to the
	// "http.server.active_requests" semantic conventions. It represents the number
	// of active HTTP server requests.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Experimental
	HTTPServerActiveRequestsName        = "http.server.active_requests"
	HTTPServerActiveRequestsUnit        = "{request}"
	HTTPServerActiveRequestsDescription = "Number of active HTTP server requests."

	// HTTPServerRequestBodySize is the metric conforming to the
	// "http.server.request.body.size" semantic conventions. It represents the size
	// of HTTP server request bodies.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	HTTPServerRequestBodySizeName        = "http.server.request.body.size"
	HTTPServerRequestBodySizeUnit        = "By"
	HTTPServerRequestBodySizeDescription = "Size of HTTP server request bodies."

	// HTTPServerResponseBodySize is the metric conforming to the
	// "http.server.response.body.size" semantic conventions. It represents the
	// size of HTTP server response bodies.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	HTTPServerResponseBodySizeName        = "http.server.response.body.size"
	HTTPServerResponseBodySizeUnit        = "By"
	HTTPServerResponseBodySizeDescription = "Size of HTTP server response bodies."

	// HTTPClientRequestDuration is the metric conforming to the
	// "http.client.request.duration" semantic conventions. It represents the
	// duration of HTTP client requests.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	HTTPClientRequestDurationName        = "http.client.request.duration"
	HTTPClientRequestDurationUnit        = "s"
	HTTPClientRequestDurationDescription = "Duration of HTTP client requests."

	// HTTPClientRequestBodySize is the metric conforming to the
	// "http.client.request.body.size" semantic conventions. It represents the size
	// of HTTP client request bodies.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	HTTPClientRequestBodySizeName        = "http.client.request.body.size"
	HTTPClientRequestBodySizeUnit        = "By"
	HTTPClientRequestBodySizeDescription = "Size of HTTP client request bodies."

	// HTTPClientResponseBodySize is the metric conforming to the
	// "http.client.response.body.size" semantic conventions. It represents the
	// size of HTTP client response bodies.
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	HTTPClientResponseBodySizeName        = "http.client.response.body.size"
	HTTPClientResponseBodySizeUnit        = "By"
	HTTPClientResponseBodySizeDescription = "Size of HTTP client response bodies."

	// HTTPClientOpenConnections is the metric conforming to the
	// "http.client.open_connections" semantic conventions. It represents the
	// number of outbound HTTP connections that are currently active or idle on the
	// client.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	HTTPClientOpenConnectionsName        = "http.client.open_connections"
	HTTPClientOpenConnectionsUnit        = "{connection}"
	HTTPClientOpenConnectionsDescription = "Number of outbound HTTP connections that are currently active or idle on the client."

	// HTTPClientConnectionDuration is the metric conforming to the
	// "http.client.connection.duration" semantic conventions. It represents the
	// duration of the successfully established outbound HTTP connections.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	HTTPClientConnectionDurationName        = "http.client.connection.duration"
	HTTPClientConnectionDurationUnit        = "s"
	HTTPClientConnectionDurationDescription = "The duration of the successfully established outbound HTTP connections."

	// HTTPClientActiveRequests is the metric conforming to the
	// "http.client.active_requests" semantic conventions. It represents the number
	// of active HTTP requests.
	// Instrument: updowncounter
	// Unit: {request}
	// Stability: Experimental
	HTTPClientActiveRequestsName        = "http.client.active_requests"
	HTTPClientActiveRequestsUnit        = "{request}"
	HTTPClientActiveRequestsDescription = "Number of active HTTP requests."

	// JvmMemoryInit is the metric conforming to the "jvm.memory.init" semantic
	// conventions. It represents the measure of initial memory requested.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	JvmMemoryInitName        = "jvm.memory.init"
	JvmMemoryInitUnit        = "By"
	JvmMemoryInitDescription = "Measure of initial memory requested."

	// JvmSystemCPUUtilization is the metric conforming to the
	// "jvm.system.cpu.utilization" semantic conventions. It represents the recent
	// CPU utilization for the whole system as reported by the JVM.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	JvmSystemCPUUtilizationName        = "jvm.system.cpu.utilization"
	JvmSystemCPUUtilizationUnit        = "1"
	JvmSystemCPUUtilizationDescription = "Recent CPU utilization for the whole system as reported by the JVM."

	// JvmSystemCPULoad1m is the metric conforming to the "jvm.system.cpu.load_1m"
	// semantic conventions. It represents the average CPU load of the whole system
	// for the last minute as reported by the JVM.
	// Instrument: gauge
	// Unit: {run_queue_item}
	// Stability: Experimental
	JvmSystemCPULoad1mName        = "jvm.system.cpu.load_1m"
	JvmSystemCPULoad1mUnit        = "{run_queue_item}"
	JvmSystemCPULoad1mDescription = "Average CPU load of the whole system for the last minute as reported by the JVM."

	// JvmBufferMemoryUsage is the metric conforming to the
	// "jvm.buffer.memory.usage" semantic conventions. It represents the measure of
	// memory used by buffers.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	JvmBufferMemoryUsageName        = "jvm.buffer.memory.usage"
	JvmBufferMemoryUsageUnit        = "By"
	JvmBufferMemoryUsageDescription = "Measure of memory used by buffers."

	// JvmBufferMemoryLimit is the metric conforming to the
	// "jvm.buffer.memory.limit" semantic conventions. It represents the measure of
	// total memory capacity of buffers.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	JvmBufferMemoryLimitName        = "jvm.buffer.memory.limit"
	JvmBufferMemoryLimitUnit        = "By"
	JvmBufferMemoryLimitDescription = "Measure of total memory capacity of buffers."

	// JvmBufferCount is the metric conforming to the "jvm.buffer.count" semantic
	// conventions. It represents the number of buffers in the pool.
	// Instrument: updowncounter
	// Unit: {buffer}
	// Stability: Experimental
	JvmBufferCountName        = "jvm.buffer.count"
	JvmBufferCountUnit        = "{buffer}"
	JvmBufferCountDescription = "Number of buffers in the pool."

	// JvmMemoryUsed is the metric conforming to the "jvm.memory.used" semantic
	// conventions. It represents the measure of memory used.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Stable
	JvmMemoryUsedName        = "jvm.memory.used"
	JvmMemoryUsedUnit        = "By"
	JvmMemoryUsedDescription = "Measure of memory used."

	// JvmMemoryCommitted is the metric conforming to the "jvm.memory.committed"
	// semantic conventions. It represents the measure of memory committed.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Stable
	JvmMemoryCommittedName        = "jvm.memory.committed"
	JvmMemoryCommittedUnit        = "By"
	JvmMemoryCommittedDescription = "Measure of memory committed."

	// JvmMemoryLimit is the metric conforming to the "jvm.memory.limit" semantic
	// conventions. It represents the measure of max obtainable memory.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Stable
	JvmMemoryLimitName        = "jvm.memory.limit"
	JvmMemoryLimitUnit        = "By"
	JvmMemoryLimitDescription = "Measure of max obtainable memory."

	// JvmMemoryUsedAfterLastGc is the metric conforming to the
	// "jvm.memory.used_after_last_gc" semantic conventions. It represents the
	// measure of memory used, as measured after the most recent garbage collection
	// event on this pool.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Stable
	JvmMemoryUsedAfterLastGcName        = "jvm.memory.used_after_last_gc"
	JvmMemoryUsedAfterLastGcUnit        = "By"
	JvmMemoryUsedAfterLastGcDescription = "Measure of memory used, as measured after the most recent garbage collection event on this pool."

	// JvmGcDuration is the metric conforming to the "jvm.gc.duration" semantic
	// conventions. It represents the duration of JVM garbage collection actions.
	// Instrument: histogram
	// Unit: s
	// Stability: Stable
	JvmGcDurationName        = "jvm.gc.duration"
	JvmGcDurationUnit        = "s"
	JvmGcDurationDescription = "Duration of JVM garbage collection actions."

	// JvmThreadCount is the metric conforming to the "jvm.thread.count" semantic
	// conventions. It represents the number of executing platform threads.
	// Instrument: updowncounter
	// Unit: {thread}
	// Stability: Stable
	JvmThreadCountName        = "jvm.thread.count"
	JvmThreadCountUnit        = "{thread}"
	JvmThreadCountDescription = "Number of executing platform threads."

	// JvmClassLoaded is the metric conforming to the "jvm.class.loaded" semantic
	// conventions. It represents the number of classes loaded since JVM start.
	// Instrument: counter
	// Unit: {class}
	// Stability: Stable
	JvmClassLoadedName        = "jvm.class.loaded"
	JvmClassLoadedUnit        = "{class}"
	JvmClassLoadedDescription = "Number of classes loaded since JVM start."

	// JvmClassUnloaded is the metric conforming to the "jvm.class.unloaded"
	// semantic conventions. It represents the number of classes unloaded since JVM
	// start.
	// Instrument: counter
	// Unit: {class}
	// Stability: Stable
	JvmClassUnloadedName        = "jvm.class.unloaded"
	JvmClassUnloadedUnit        = "{class}"
	JvmClassUnloadedDescription = "Number of classes unloaded since JVM start."

	// JvmClassCount is the metric conforming to the "jvm.class.count" semantic
	// conventions. It represents the number of classes currently loaded.
	// Instrument: updowncounter
	// Unit: {class}
	// Stability: Stable
	JvmClassCountName        = "jvm.class.count"
	JvmClassCountUnit        = "{class}"
	JvmClassCountDescription = "Number of classes currently loaded."

	// JvmCPUCount is the metric conforming to the "jvm.cpu.count" semantic
	// conventions. It represents the number of processors available to the Java
	// virtual machine.
	// Instrument: updowncounter
	// Unit: {cpu}
	// Stability: Stable
	JvmCPUCountName        = "jvm.cpu.count"
	JvmCPUCountUnit        = "{cpu}"
	JvmCPUCountDescription = "Number of processors available to the Java virtual machine."

	// JvmCPUTime is the metric conforming to the "jvm.cpu.time" semantic
	// conventions. It represents the cPU time used by the process as reported by
	// the JVM.
	// Instrument: counter
	// Unit: s
	// Stability: Stable
	JvmCPUTimeName        = "jvm.cpu.time"
	JvmCPUTimeUnit        = "s"
	JvmCPUTimeDescription = "CPU time used by the process as reported by the JVM."

	// JvmCPURecentUtilization is the metric conforming to the
	// "jvm.cpu.recent_utilization" semantic conventions. It represents the recent
	// CPU utilization for the process as reported by the JVM.
	// Instrument: gauge
	// Unit: 1
	// Stability: Stable
	JvmCPURecentUtilizationName        = "jvm.cpu.recent_utilization"
	JvmCPURecentUtilizationUnit        = "1"
	JvmCPURecentUtilizationDescription = "Recent CPU utilization for the process as reported by the JVM."

	// MessagingPublishDuration is the metric conforming to the
	// "messaging.publish.duration" semantic conventions. It represents the
	// measures the duration of publish operation.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	MessagingPublishDurationName        = "messaging.publish.duration"
	MessagingPublishDurationUnit        = "s"
	MessagingPublishDurationDescription = "Measures the duration of publish operation."

	// MessagingReceiveDuration is the metric conforming to the
	// "messaging.receive.duration" semantic conventions. It represents the
	// measures the duration of receive operation.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	MessagingReceiveDurationName        = "messaging.receive.duration"
	MessagingReceiveDurationUnit        = "s"
	MessagingReceiveDurationDescription = "Measures the duration of receive operation."

	// MessagingProcessDuration is the metric conforming to the
	// "messaging.process.duration" semantic conventions. It represents the
	// measures the duration of process operation.
	// Instrument: histogram
	// Unit: s
	// Stability: Experimental
	MessagingProcessDurationName        = "messaging.process.duration"
	MessagingProcessDurationUnit        = "s"
	MessagingProcessDurationDescription = "Measures the duration of process operation."

	// MessagingPublishMessages is the metric conforming to the
	// "messaging.publish.messages" semantic conventions. It represents the
	// measures the number of published messages.
	// Instrument: counter
	// Unit: {message}
	// Stability: Experimental
	MessagingPublishMessagesName        = "messaging.publish.messages"
	MessagingPublishMessagesUnit        = "{message}"
	MessagingPublishMessagesDescription = "Measures the number of published messages."

	// MessagingReceiveMessages is the metric conforming to the
	// "messaging.receive.messages" semantic conventions. It represents the
	// measures the number of received messages.
	// Instrument: counter
	// Unit: {message}
	// Stability: Experimental
	MessagingReceiveMessagesName        = "messaging.receive.messages"
	MessagingReceiveMessagesUnit        = "{message}"
	MessagingReceiveMessagesDescription = "Measures the number of received messages."

	// MessagingProcessMessages is the metric conforming to the
	// "messaging.process.messages" semantic conventions. It represents the
	// measures the number of processed messages.
	// Instrument: counter
	// Unit: {message}
	// Stability: Experimental
	MessagingProcessMessagesName        = "messaging.process.messages"
	MessagingProcessMessagesUnit        = "{message}"
	MessagingProcessMessagesDescription = "Measures the number of processed messages."

	// ProcessCPUTime is the metric conforming to the "process.cpu.time" semantic
	// conventions. It represents the total CPU seconds broken down by different
	// states.
	// Instrument: counter
	// Unit: s
	// Stability: Experimental
	ProcessCPUTimeName        = "process.cpu.time"
	ProcessCPUTimeUnit        = "s"
	ProcessCPUTimeDescription = "Total CPU seconds broken down by different states."

	// ProcessCPUUtilization is the metric conforming to the
	// "process.cpu.utilization" semantic conventions. It represents the difference
	// in process.cpu.time since the last measurement, divided by the elapsed time
	// and number of CPUs available to the process.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	ProcessCPUUtilizationName        = "process.cpu.utilization"
	ProcessCPUUtilizationUnit        = "1"
	ProcessCPUUtilizationDescription = "Difference in process.cpu.time since the last measurement, divided by the elapsed time and number of CPUs available to the process."

	// ProcessMemoryUsage is the metric conforming to the "process.memory.usage"
	// semantic conventions. It represents the amount of physical memory in use.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	ProcessMemoryUsageName        = "process.memory.usage"
	ProcessMemoryUsageUnit        = "By"
	ProcessMemoryUsageDescription = "The amount of physical memory in use."

	// ProcessMemoryVirtual is the metric conforming to the
	// "process.memory.virtual" semantic conventions. It represents the amount of
	// committed virtual memory.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	ProcessMemoryVirtualName        = "process.memory.virtual"
	ProcessMemoryVirtualUnit        = "By"
	ProcessMemoryVirtualDescription = "The amount of committed virtual memory."

	// ProcessDiskIo is the metric conforming to the "process.disk.io" semantic
	// conventions. It represents the disk bytes transferred.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	ProcessDiskIoName        = "process.disk.io"
	ProcessDiskIoUnit        = "By"
	ProcessDiskIoDescription = "Disk bytes transferred."

	// ProcessNetworkIo is the metric conforming to the "process.network.io"
	// semantic conventions. It represents the network bytes transferred.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	ProcessNetworkIoName        = "process.network.io"
	ProcessNetworkIoUnit        = "By"
	ProcessNetworkIoDescription = "Network bytes transferred."

	// ProcessThreadCount is the metric conforming to the "process.thread.count"
	// semantic conventions. It represents the process threads count.
	// Instrument: updowncounter
	// Unit: {thread}
	// Stability: Experimental
	ProcessThreadCountName        = "process.thread.count"
	ProcessThreadCountUnit        = "{thread}"
	ProcessThreadCountDescription = "Process threads count."

	// ProcessOpenFileDescriptorCount is the metric conforming to the
	// "process.open_file_descriptor.count" semantic conventions. It represents the
	// number of file descriptors in use by the process.
	// Instrument: updowncounter
	// Unit: {count}
	// Stability: Experimental
	ProcessOpenFileDescriptorCountName        = "process.open_file_descriptor.count"
	ProcessOpenFileDescriptorCountUnit        = "{count}"
	ProcessOpenFileDescriptorCountDescription = "Number of file descriptors in use by the process."

	// ProcessContextSwitches is the metric conforming to the
	// "process.context_switches" semantic conventions. It represents the number of
	// times the process has been context switched.
	// Instrument: counter
	// Unit: {count}
	// Stability: Experimental
	ProcessContextSwitchesName        = "process.context_switches"
	ProcessContextSwitchesUnit        = "{count}"
	ProcessContextSwitchesDescription = "Number of times the process has been context switched."

	// ProcessPagingFaults is the metric conforming to the "process.paging.faults"
	// semantic conventions. It represents the number of page faults the process
	// has made.
	// Instrument: counter
	// Unit: {fault}
	// Stability: Experimental
	ProcessPagingFaultsName        = "process.paging.faults"
	ProcessPagingFaultsUnit        = "{fault}"
	ProcessPagingFaultsDescription = "Number of page faults the process has made."

	// RPCServerDuration is the metric conforming to the "rpc.server.duration"
	// semantic conventions. It represents the measures the duration of inbound
	// RPC.
	// Instrument: histogram
	// Unit: ms
	// Stability: Experimental
	RPCServerDurationName        = "rpc.server.duration"
	RPCServerDurationUnit        = "ms"
	RPCServerDurationDescription = "Measures the duration of inbound RPC."

	// RPCServerRequestSize is the metric conforming to the
	// "rpc.server.request.size" semantic conventions. It represents the measures
	// the size of RPC request messages (uncompressed).
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	RPCServerRequestSizeName        = "rpc.server.request.size"
	RPCServerRequestSizeUnit        = "By"
	RPCServerRequestSizeDescription = "Measures the size of RPC request messages (uncompressed)."

	// RPCServerResponseSize is the metric conforming to the
	// "rpc.server.response.size" semantic conventions. It represents the measures
	// the size of RPC response messages (uncompressed).
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	RPCServerResponseSizeName        = "rpc.server.response.size"
	RPCServerResponseSizeUnit        = "By"
	RPCServerResponseSizeDescription = "Measures the size of RPC response messages (uncompressed)."

	// RPCServerRequestsPerRPC is the metric conforming to the
	// "rpc.server.requests_per_rpc" semantic conventions. It represents the
	// measures the number of messages received per RPC.
	// Instrument: histogram
	// Unit: {count}
	// Stability: Experimental
	RPCServerRequestsPerRPCName        = "rpc.server.requests_per_rpc"
	RPCServerRequestsPerRPCUnit        = "{count}"
	RPCServerRequestsPerRPCDescription = "Measures the number of messages received per RPC."

	// RPCServerResponsesPerRPC is the metric conforming to the
	// "rpc.server.responses_per_rpc" semantic conventions. It represents the
	// measures the number of messages sent per RPC.
	// Instrument: histogram
	// Unit: {count}
	// Stability: Experimental
	RPCServerResponsesPerRPCName        = "rpc.server.responses_per_rpc"
	RPCServerResponsesPerRPCUnit        = "{count}"
	RPCServerResponsesPerRPCDescription = "Measures the number of messages sent per RPC."

	// RPCClientDuration is the metric conforming to the "rpc.client.duration"
	// semantic conventions. It represents the measures the duration of outbound
	// RPC.
	// Instrument: histogram
	// Unit: ms
	// Stability: Experimental
	RPCClientDurationName        = "rpc.client.duration"
	RPCClientDurationUnit        = "ms"
	RPCClientDurationDescription = "Measures the duration of outbound RPC."

	// RPCClientRequestSize is the metric conforming to the
	// "rpc.client.request.size" semantic conventions. It represents the measures
	// the size of RPC request messages (uncompressed).
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	RPCClientRequestSizeName        = "rpc.client.request.size"
	RPCClientRequestSizeUnit        = "By"
	RPCClientRequestSizeDescription = "Measures the size of RPC request messages (uncompressed)."

	// RPCClientResponseSize is the metric conforming to the
	// "rpc.client.response.size" semantic conventions. It represents the measures
	// the size of RPC response messages (uncompressed).
	// Instrument: histogram
	// Unit: By
	// Stability: Experimental
	RPCClientResponseSizeName        = "rpc.client.response.size"
	RPCClientResponseSizeUnit        = "By"
	RPCClientResponseSizeDescription = "Measures the size of RPC response messages (uncompressed)."

	// RPCClientRequestsPerRPC is the metric conforming to the
	// "rpc.client.requests_per_rpc" semantic conventions. It represents the
	// measures the number of messages received per RPC.
	// Instrument: histogram
	// Unit: {count}
	// Stability: Experimental
	RPCClientRequestsPerRPCName        = "rpc.client.requests_per_rpc"
	RPCClientRequestsPerRPCUnit        = "{count}"
	RPCClientRequestsPerRPCDescription = "Measures the number of messages received per RPC."

	// RPCClientResponsesPerRPC is the metric conforming to the
	// "rpc.client.responses_per_rpc" semantic conventions. It represents the
	// measures the number of messages sent per RPC.
	// Instrument: histogram
	// Unit: {count}
	// Stability: Experimental
	RPCClientResponsesPerRPCName        = "rpc.client.responses_per_rpc"
	RPCClientResponsesPerRPCUnit        = "{count}"
	RPCClientResponsesPerRPCDescription = "Measures the number of messages sent per RPC."

	// SystemCPUTime is the metric conforming to the "system.cpu.time" semantic
	// conventions. It represents the seconds each logical CPU spent on each mode.
	// Instrument: counter
	// Unit: s
	// Stability: Experimental
	SystemCPUTimeName        = "system.cpu.time"
	SystemCPUTimeUnit        = "s"
	SystemCPUTimeDescription = "Seconds each logical CPU spent on each mode"

	// SystemCPUUtilization is the metric conforming to the
	// "system.cpu.utilization" semantic conventions. It represents the difference
	// in system.cpu.time since the last measurement, divided by the elapsed time
	// and number of logical CPUs.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	SystemCPUUtilizationName        = "system.cpu.utilization"
	SystemCPUUtilizationUnit        = "1"
	SystemCPUUtilizationDescription = "Difference in system.cpu.time since the last measurement, divided by the elapsed time and number of logical CPUs"

	// SystemCPUFrequency is the metric conforming to the "system.cpu.frequency"
	// semantic conventions. It represents the reports the current frequency of the
	// CPU in Hz.
	// Instrument: gauge
	// Unit: {Hz}
	// Stability: Experimental
	SystemCPUFrequencyName        = "system.cpu.frequency"
	SystemCPUFrequencyUnit        = "{Hz}"
	SystemCPUFrequencyDescription = "Reports the current frequency of the CPU in Hz"

	// SystemCPUPhysicalCount is the metric conforming to the
	// "system.cpu.physical.count" semantic conventions. It represents the reports
	// the number of actual physical processor cores on the hardware.
	// Instrument: updowncounter
	// Unit: {cpu}
	// Stability: Experimental
	SystemCPUPhysicalCountName        = "system.cpu.physical.count"
	SystemCPUPhysicalCountUnit        = "{cpu}"
	SystemCPUPhysicalCountDescription = "Reports the number of actual physical processor cores on the hardware"

	// SystemCPULogicalCount is the metric conforming to the
	// "system.cpu.logical.count" semantic conventions. It represents the reports
	// the number of logical (virtual) processor cores created by the operating
	// system to manage multitasking.
	// Instrument: updowncounter
	// Unit: {cpu}
	// Stability: Experimental
	SystemCPULogicalCountName        = "system.cpu.logical.count"
	SystemCPULogicalCountUnit        = "{cpu}"
	SystemCPULogicalCountDescription = "Reports the number of logical (virtual) processor cores created by the operating system to manage multitasking"

	// SystemMemoryUsage is the metric conforming to the "system.memory.usage"
	// semantic conventions. It represents the reports memory in use by state.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	SystemMemoryUsageName        = "system.memory.usage"
	SystemMemoryUsageUnit        = "By"
	SystemMemoryUsageDescription = "Reports memory in use by state."

	// SystemMemoryLimit is the metric conforming to the "system.memory.limit"
	// semantic conventions. It represents the total memory available in the
	// system.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	SystemMemoryLimitName        = "system.memory.limit"
	SystemMemoryLimitUnit        = "By"
	SystemMemoryLimitDescription = "Total memory available in the system."

	// SystemMemoryUtilization is the metric conforming to the
	// "system.memory.utilization" semantic conventions.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemMemoryUtilizationName = "system.memory.utilization"
	SystemMemoryUtilizationUnit = "1"

	// SystemPagingUsage is the metric conforming to the "system.paging.usage"
	// semantic conventions. It represents the unix swap or windows pagefile usage.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	SystemPagingUsageName        = "system.paging.usage"
	SystemPagingUsageUnit        = "By"
	SystemPagingUsageDescription = "Unix swap or windows pagefile usage"

	// SystemPagingUtilization is the metric conforming to the
	// "system.paging.utilization" semantic conventions.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemPagingUtilizationName = "system.paging.utilization"
	SystemPagingUtilizationUnit = "1"

	// SystemPagingFaults is the metric conforming to the "system.paging.faults"
	// semantic conventions.
	// Instrument: counter
	// Unit: {fault}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemPagingFaultsName = "system.paging.faults"
	SystemPagingFaultsUnit = "{fault}"

	// SystemPagingOperations is the metric conforming to the
	// "system.paging.operations" semantic conventions.
	// Instrument: counter
	// Unit: {operation}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemPagingOperationsName = "system.paging.operations"
	SystemPagingOperationsUnit = "{operation}"

	// SystemDiskIo is the metric conforming to the "system.disk.io" semantic
	// conventions.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemDiskIoName = "system.disk.io"
	SystemDiskIoUnit = "By"

	// SystemDiskOperations is the metric conforming to the
	// "system.disk.operations" semantic conventions.
	// Instrument: counter
	// Unit: {operation}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemDiskOperationsName = "system.disk.operations"
	SystemDiskOperationsUnit = "{operation}"

	// SystemDiskIoTime is the metric conforming to the "system.disk.io_time"
	// semantic conventions. It represents the time disk spent activated.
	// Instrument: counter
	// Unit: s
	// Stability: Experimental
	SystemDiskIoTimeName        = "system.disk.io_time"
	SystemDiskIoTimeUnit        = "s"
	SystemDiskIoTimeDescription = "Time disk spent activated"

	// SystemDiskOperationTime is the metric conforming to the
	// "system.disk.operation_time" semantic conventions. It represents the sum of
	// the time each operation took to complete.
	// Instrument: counter
	// Unit: s
	// Stability: Experimental
	SystemDiskOperationTimeName        = "system.disk.operation_time"
	SystemDiskOperationTimeUnit        = "s"
	SystemDiskOperationTimeDescription = "Sum of the time each operation took to complete"

	// SystemDiskMerged is the metric conforming to the "system.disk.merged"
	// semantic conventions.
	// Instrument: counter
	// Unit: {operation}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemDiskMergedName = "system.disk.merged"
	SystemDiskMergedUnit = "{operation}"

	// SystemFilesystemUsage is the metric conforming to the
	// "system.filesystem.usage" semantic conventions.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemFilesystemUsageName = "system.filesystem.usage"
	SystemFilesystemUsageUnit = "By"

	// SystemFilesystemUtilization is the metric conforming to the
	// "system.filesystem.utilization" semantic conventions.
	// Instrument: gauge
	// Unit: 1
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemFilesystemUtilizationName = "system.filesystem.utilization"
	SystemFilesystemUtilizationUnit = "1"

	// SystemNetworkDropped is the metric conforming to the
	// "system.network.dropped" semantic conventions. It represents the count of
	// packets that are dropped or discarded even though there was no error.
	// Instrument: counter
	// Unit: {packet}
	// Stability: Experimental
	SystemNetworkDroppedName        = "system.network.dropped"
	SystemNetworkDroppedUnit        = "{packet}"
	SystemNetworkDroppedDescription = "Count of packets that are dropped or discarded even though there was no error"

	// SystemNetworkPackets is the metric conforming to the
	// "system.network.packets" semantic conventions.
	// Instrument: counter
	// Unit: {packet}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemNetworkPacketsName = "system.network.packets"
	SystemNetworkPacketsUnit = "{packet}"

	// SystemNetworkErrors is the metric conforming to the "system.network.errors"
	// semantic conventions. It represents the count of network errors detected.
	// Instrument: counter
	// Unit: {error}
	// Stability: Experimental
	SystemNetworkErrorsName        = "system.network.errors"
	SystemNetworkErrorsUnit        = "{error}"
	SystemNetworkErrorsDescription = "Count of network errors detected"

	// SystemNetworkIo is the metric conforming to the "system.network.io" semantic
	// conventions.
	// Instrument: counter
	// Unit: By
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemNetworkIoName = "system.network.io"
	SystemNetworkIoUnit = "By"

	// SystemNetworkConnections is the metric conforming to the
	// "system.network.connections" semantic conventions.
	// Instrument: updowncounter
	// Unit: {connection}
	// Stability: Experimental
	// NOTE: The description (brief) for this metric is not defined in the semantic-conventions repository.
	SystemNetworkConnectionsName = "system.network.connections"
	SystemNetworkConnectionsUnit = "{connection}"

	// SystemProcessCount is the metric conforming to the "system.process.count"
	// semantic conventions. It represents the total number of processes in each
	// state.
	// Instrument: updowncounter
	// Unit: {process}
	// Stability: Experimental
	SystemProcessCountName        = "system.process.count"
	SystemProcessCountUnit        = "{process}"
	SystemProcessCountDescription = "Total number of processes in each state"

	// SystemProcessCreated is the metric conforming to the
	// "system.process.created" semantic conventions. It represents the total
	// number of processes created over uptime of the host.
	// Instrument: counter
	// Unit: {process}
	// Stability: Experimental
	SystemProcessCreatedName        = "system.process.created"
	SystemProcessCreatedUnit        = "{process}"
	SystemProcessCreatedDescription = "Total number of processes created over uptime of the host"

	// SystemLinuxMemoryAvailable is the metric conforming to the
	// "system.linux.memory.available" semantic conventions. It represents an
	// estimate of how much memory is available for starting new applications,
	// without causing swapping.
	// Instrument: updowncounter
	// Unit: By
	// Stability: Experimental
	SystemLinuxMemoryAvailableName        = "system.linux.memory.available"
	SystemLinuxMemoryAvailableUnit        = "By"
	SystemLinuxMemoryAvailableDescription = "An estimate of how much memory is available for starting new applications, without causing swapping"
)
