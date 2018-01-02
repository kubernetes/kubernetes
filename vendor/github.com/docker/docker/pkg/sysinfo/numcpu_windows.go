// +build windows

package sysinfo

import (
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32               = windows.NewLazySystemDLL("kernel32.dll")
	getCurrentProcess      = kernel32.NewProc("GetCurrentProcess")
	getProcessAffinityMask = kernel32.NewProc("GetProcessAffinityMask")
)

func numCPU() int {
	// Gets the affinity mask for a process
	var mask, sysmask uintptr
	currentProcess, _, _ := getCurrentProcess.Call()
	ret, _, _ := getProcessAffinityMask.Call(currentProcess, uintptr(unsafe.Pointer(&mask)), uintptr(unsafe.Pointer(&sysmask)))
	if ret == 0 {
		return 0
	}
	// For every available thread a bit is set in the mask.
	ncpu := int(popcnt(uint64(mask)))
	return ncpu
}

// NumCPU returns the number of CPUs which are currently online
func NumCPU() int {
	if ncpu := numCPU(); ncpu > 0 {
		return ncpu
	}
	return runtime.NumCPU()
}
