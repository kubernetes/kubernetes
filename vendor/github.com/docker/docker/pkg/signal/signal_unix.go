// +build !windows

package signal

import (
	"syscall"
)

// Signals used in api/client (no windows equivalent, use
// invalid signals so they don't get handled)
const SIGCHLD = syscall.SIGCHLD
const SIGWINCH = syscall.SIGWINCH
