//go:build !go1.19

package system

import "syscall"

func ClearRlimitNofileCache(_ *syscall.Rlimit) {}
