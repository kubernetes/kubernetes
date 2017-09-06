package system

import (
	"syscall"
	"time"
	"unsafe"
)

// Used by chtimes
var maxTime time.Time

func init() {
	// chtimes initialization
	if unsafe.Sizeof(syscall.Timespec{}.Nsec) == 8 {
		// This is a 64 bit timespec
		// os.Chtimes limits time to the following
		maxTime = time.Unix(0, 1<<63-1)
	} else {
		// This is a 32 bit timespec
		maxTime = time.Unix(1<<31-1, 0)
	}
}
