// +build linux

package sysinfo

import (
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"
)

// numCPU queries the system for the count of threads available
// for use to this process.
//
// Issues two syscalls.
// Returns 0 on errors. Use |runtime.NumCPU| in that case.
func numCPU() int {
	// Gets the affinity mask for a process: The very one invoking this function.
	pid, _, _ := unix.RawSyscall(unix.SYS_GETPID, 0, 0, 0)

	var mask [1024 / 64]uintptr
	_, _, err := unix.RawSyscall(unix.SYS_SCHED_GETAFFINITY, pid, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0])))
	if err != 0 {
		return 0
	}

	// For every available thread a bit is set in the mask.
	ncpu := 0
	for _, e := range mask {
		if e == 0 {
			continue
		}
		ncpu += int(popcnt(uint64(e)))
	}
	return ncpu
}

// NumCPU returns the number of CPUs which are currently online
func NumCPU() int {
	if ncpu := numCPU(); ncpu > 0 {
		return ncpu
	}
	return runtime.NumCPU()
}
