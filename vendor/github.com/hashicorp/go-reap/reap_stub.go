// +build windows solaris

package reap

import (
	"sync"
)

// IsSupported returns true if child process reaping is supported on this
// platform. This version always returns false.
func IsSupported() bool {
	return false
}

// ReapChildren is not supported so this always returns right away.
func ReapChildren(pids PidCh, errors ErrorCh, done chan struct{}, reapLock *sync.RWMutex) {
}
