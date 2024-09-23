//go:build windows

// Shim for the Host Compute Service (HCS) to manage Windows Server
// containers and Hyper-V containers.

package hcsshim

import (
	"golang.org/x/sys/windows"

	"github.com/Microsoft/hcsshim/internal/hcserror"
)

//go:generate go run github.com/Microsoft/go-winio/tools/mkwinsyscall -output zsyscall_windows.go hcsshim.go

//sys SetCurrentThreadCompartmentId(compartmentId uint32) (hr error) = iphlpapi.SetCurrentThreadCompartmentId

const (
	// Specific user-visible exit codes
	WaitErrExecFailed = 32767

	ERROR_GEN_FAILURE          = windows.ERROR_GEN_FAILURE
	ERROR_SHUTDOWN_IN_PROGRESS = windows.ERROR_SHUTDOWN_IN_PROGRESS
	WSAEINVAL                  = windows.WSAEINVAL

	// Timeout on wait calls
	TimeoutInfinite = 0xFFFFFFFF
)

type HcsError = hcserror.HcsError
