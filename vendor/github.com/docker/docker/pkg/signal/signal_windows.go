// +build windows

package signal

import (
	"syscall"
)

// Signals used in api/client (no windows equivalent, use
// invalid signals so they don't get handled)
const SIGCHLD = syscall.Signal(0xff)
const SIGWINCH = syscall.Signal(0xff)
