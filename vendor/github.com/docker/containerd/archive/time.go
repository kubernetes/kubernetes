package archive

import (
	"syscall"
	"time"
	"unsafe"
)

var (
	minTime = time.Unix(0, 0)
	maxTime time.Time
)

func init() {
	if unsafe.Sizeof(syscall.Timespec{}.Nsec) == 8 {
		// This is a 64 bit timespec
		// os.Chtimes limits time to the following
		maxTime = time.Unix(0, 1<<63-1)
	} else {
		// This is a 32 bit timespec
		maxTime = time.Unix(1<<31-1, 0)
	}
}

func boundTime(t time.Time) time.Time {
	if t.Before(minTime) || t.After(maxTime) {
		return minTime
	}

	return t
}

func latestTime(t1, t2 time.Time) time.Time {
	if t1.Before(t2) {
		return t2
	}
	return t1
}
