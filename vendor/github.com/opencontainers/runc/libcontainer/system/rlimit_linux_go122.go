//go:build go1.19 && !go1.23

// TODO: remove this file once go 1.22 is no longer supported.

package system

import (
	"sync/atomic"
	"syscall"
	_ "unsafe" // Needed for go:linkname to work.
)

//go:linkname syscallOrigRlimitNofile syscall.origRlimitNofile
var syscallOrigRlimitNofile atomic.Pointer[syscall.Rlimit]

// ClearRlimitNofileCache clears go runtime's nofile rlimit cache.
// The argument is process RLIMIT_NOFILE values.
func ClearRlimitNofileCache(_ *syscall.Rlimit) {
	// As reported in issue #4195, the new version of go runtime(since 1.19)
	// will cache rlimit-nofile. Before executing execve, the rlimit-nofile
	// of the process will be restored with the cache. In runc, this will
	// cause the rlimit-nofile setting by the parent process for the container
	// to become invalid. It can be solved by clearing this cache. But
	// unfortunately, go stdlib doesn't provide such function, so we need to
	// link to the private var `origRlimitNofile` in package syscall to hack.
	syscallOrigRlimitNofile.Store(nil)
}
