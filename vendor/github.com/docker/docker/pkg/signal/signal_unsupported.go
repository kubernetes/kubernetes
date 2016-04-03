// +build !linux,!darwin,!freebsd

package signal

import (
	"syscall"
)

var SignalMap = map[string]syscall.Signal{}
