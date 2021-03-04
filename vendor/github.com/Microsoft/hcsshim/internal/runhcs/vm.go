package runhcs

import (
	"encoding/json"

	"github.com/Microsoft/go-winio"
)

// VMRequestOp is an operation that can be issued to a VM shim.
type VMRequestOp string

const (
	// OpCreateContainer is a create container request.
	OpCreateContainer VMRequestOp = "create"
	// OpSyncNamespace is a `cni.NamespaceTypeGuest` sync request with the UVM.
	OpSyncNamespace VMRequestOp = "sync"
	// OpUnmountContainer is a container unmount request.
	OpUnmountContainer VMRequestOp = "unmount"
	// OpUnmountContainerDiskOnly is a container unmount disk request.
	OpUnmountContainerDiskOnly VMRequestOp = "unmount-disk"
)

// VMRequest is an operation request that is issued to a VM shim.
type VMRequest struct {
	ID string
	Op VMRequestOp
}

// IssueVMRequest issues a request to a shim at the given pipe.
func IssueVMRequest(pipepath string, req *VMRequest) error {
	pipe, err := winio.DialPipe(pipepath, nil)
	if err != nil {
		return err
	}
	defer pipe.Close()
	if err := json.NewEncoder(pipe).Encode(req); err != nil {
		return err
	}
	if err := GetErrorFromPipe(pipe, nil); err != nil {
		return err
	}
	return nil
}
