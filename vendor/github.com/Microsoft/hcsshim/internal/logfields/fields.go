package logfields

const (
	// Identifiers

	ContainerID = "cid"
	UVMID       = "uvm-id"
	ProcessID   = "pid"

	// Common Misc

	// Timeout represents an operation timeout.
	Timeout = "timeout"
	JSON    = "json"

	// Keys/values

	Field         = "field"
	OCIAnnotation = "oci-annotation"
	Value         = "value"

	// Golang type's

	ExpectedType = "expected-type"
	Bool         = "bool"
	Uint32       = "uint32"
	Uint64       = "uint64"

	// runhcs

	VMShimOperation = "vmshim-op"
)
