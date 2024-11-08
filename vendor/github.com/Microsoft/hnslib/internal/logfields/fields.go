package logfields

const (
	// Identifiers

	Name      = "name"
	Namespace = "namespace"
	Operation = "operation"

	ID          = "id"
	SandboxID   = "sid"
	ContainerID = "cid"
	ExecID      = "eid"
	ProcessID   = "pid"
	TaskID      = "tid"
	UVMID       = "uvm-id"

	// networking and IO

	File  = "file"
	Path  = "path"
	Bytes = "bytes"
	Pipe  = "pipe"

	// Common Misc

	Attempt = "attemptNo"
	JSON    = "json"

	// Time

	StartTime = "startTime"
	EndTime   = "endTime"
	Duration  = "duration"
	Timeout   = "timeout"

	// Keys/values

	Field         = "field"
	Key           = "key"
	OCIAnnotation = "oci-annotation"
	Value         = "value"
	Options       = "options"

	// Golang type's

	ExpectedType = "expected-type"
	Bool         = "bool"
	Int32        = "int32"
	Uint32       = "uint32"
	Uint64       = "uint64"

	// runhcs

	VMShimOperation = "vmshim-op"

	// logging and tracing

	TraceID      = "traceID"
	SpanID       = "spanID"
	ParentSpanID = "parentSpanID"
)
