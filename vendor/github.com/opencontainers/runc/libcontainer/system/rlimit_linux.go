//go:build go1.23

package system

import (
	"syscall"
)

// ClearRlimitNofileCache clears go runtime's nofile rlimit cache. The argument
// is process RLIMIT_NOFILE values. Relies on go.dev/cl/588076.
func ClearRlimitNofileCache(lim *syscall.Rlimit) {
	// Ignore the return values since we only need to clean the cache,
	// the limit is going to be set via unix.Prlimit elsewhere.
	_ = syscall.Setrlimit(syscall.RLIMIT_NOFILE, lim)
}
