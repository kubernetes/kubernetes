//go:build darwin
// +build darwin

package fsnotify

import "golang.org/x/sys/unix"

// note: this constant is not defined on BSD
const openMode = unix.O_EVTONLY | unix.O_CLOEXEC
