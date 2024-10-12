package guestrequest

// These are constants for v2 schema modify requests.

type RequestType string
type ResourceType string

// RequestType const.
const (
	RequestTypeAdd    RequestType = "Add"
	RequestTypeRemove RequestType = "Remove"
	RequestTypePreAdd RequestType = "PreAdd" // For networking
	RequestTypeUpdate RequestType = "Update"
)

type SignalValueWCOW string

const (
	SignalValueWCOWCtrlC        SignalValueWCOW = "CtrlC"
	SignalValueWCOWCtrlBreak    SignalValueWCOW = "CtrlBreak"
	SignalValueWCOWCtrlClose    SignalValueWCOW = "CtrlClose"
	SignalValueWCOWCtrlLogOff   SignalValueWCOW = "CtrlLogOff"
	SignalValueWCOWCtrlShutdown SignalValueWCOW = "CtrlShutdown"
)

// ModificationRequest is for modify commands passed to the guest.
type ModificationRequest struct {
	RequestType  RequestType  `json:"RequestType,omitempty"`
	ResourceType ResourceType `json:"ResourceType,omitempty"`
	Settings     interface{}  `json:"Settings,omitempty"`
}

type NetworkModifyRequest struct {
	AdapterId   string      `json:"AdapterId,omitempty"` //nolint:stylecheck
	RequestType RequestType `json:"RequestType,omitempty"`
	Settings    interface{} `json:"Settings,omitempty"`
}

type RS4NetworkModifyRequest struct {
	AdapterInstanceId string      `json:"AdapterInstanceId,omitempty"` //nolint:stylecheck
	RequestType       RequestType `json:"RequestType,omitempty"`
	Settings          interface{} `json:"Settings,omitempty"`
}

var (
	// V5 GUIDs for SCSI controllers
	// These GUIDs are created with namespace GUID "d422512d-2bf2-4752-809d-7b82b5fcb1b4"
	// and index as names. For example, first GUID is created like this:
	// guid.NewV5("d422512d-2bf2-4752-809d-7b82b5fcb1b4", []byte("0"))
	ScsiControllerGuids = []string{
		"df6d0690-79e5-55b6-a5ec-c1e2f77f580a",
		"0110f83b-de10-5172-a266-78bca56bf50a",
		"b5d2d8d4-3a75-51bf-945b-3444dc6b8579",
		"305891a9-b251-5dfe-91a2-c25d9212275b",
	}
)

// constants for v2 schema ProcessModifyRequest

// Operation type for [hcsschema.ProcessModifyRequest].
type ProcessModifyOperation string

const (
	ModifyProcessConsoleSize ProcessModifyOperation = "ConsoleSize"
	CloseProcessHandle       ProcessModifyOperation = "CloseHandle"
)

// Standard IO handle(s) to close for [hcsschema.CloseHandle] in [hcsschema.ProcessModifyRequest].
type STDIOHandle string

const (
	STDInHandle  STDIOHandle = "StdIn"
	STDOutHandle STDIOHandle = "StdOut"
	STDErrHandle STDIOHandle = "StdErr"
	AllHandles   STDIOHandle = "All"
)
