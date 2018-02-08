// +build !appengine,!plan9

package request

import (
	"net"
	"os"
	"syscall"
)

func isErrConnectionReset(err error) bool {
	if opErr, ok := err.(*net.OpError); ok {
		if sysErr, ok := opErr.Err.(*os.SyscallError); ok {
			return sysErr.Err == syscall.ECONNRESET
		}
	}

	return false
}
