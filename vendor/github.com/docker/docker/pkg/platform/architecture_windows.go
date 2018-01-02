package platform

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	modkernel32       = windows.NewLazySystemDLL("kernel32.dll")
	procGetSystemInfo = modkernel32.NewProc("GetSystemInfo")
)

// see http://msdn.microsoft.com/en-us/library/windows/desktop/ms724958(v=vs.85).aspx
type systeminfo struct {
	wProcessorArchitecture      uint16
	wReserved                   uint16
	dwPageSize                  uint32
	lpMinimumApplicationAddress uintptr
	lpMaximumApplicationAddress uintptr
	dwActiveProcessorMask       uintptr
	dwNumberOfProcessors        uint32
	dwProcessorType             uint32
	dwAllocationGranularity     uint32
	wProcessorLevel             uint16
	wProcessorRevision          uint16
}

// Constants
const (
	ProcessorArchitecture64   = 9 // PROCESSOR_ARCHITECTURE_AMD64
	ProcessorArchitectureIA64 = 6 // PROCESSOR_ARCHITECTURE_IA64
	ProcessorArchitecture32   = 0 // PROCESSOR_ARCHITECTURE_INTEL
	ProcessorArchitectureArm  = 5 // PROCESSOR_ARCHITECTURE_ARM
)

// runtimeArchitecture gets the name of the current architecture (x86, x86_64, …)
func runtimeArchitecture() (string, error) {
	var sysinfo systeminfo
	syscall.Syscall(procGetSystemInfo.Addr(), 1, uintptr(unsafe.Pointer(&sysinfo)), 0, 0)
	switch sysinfo.wProcessorArchitecture {
	case ProcessorArchitecture64, ProcessorArchitectureIA64:
		return "x86_64", nil
	case ProcessorArchitecture32:
		return "i686", nil
	case ProcessorArchitectureArm:
		return "arm", nil
	default:
		return "", fmt.Errorf("Unknown processor architecture")
	}
}

// NumProcs returns the number of processors on the system
func NumProcs() uint32 {
	var sysinfo systeminfo
	syscall.Syscall(procGetSystemInfo.Addr(), 1, uintptr(unsafe.Pointer(&sysinfo)), 0, 0)
	return sysinfo.dwNumberOfProcessors
}
