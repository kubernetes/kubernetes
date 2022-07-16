// +build linux

package seccomp

import "syscall"

// System values passed through on linux
const (
	CloneNewIPC  = syscall.CLONE_NEWIPC
	CloneNewNet  = syscall.CLONE_NEWNET
	CloneNewNS   = syscall.CLONE_NEWNS
	CloneNewPID  = syscall.CLONE_NEWPID
	CloneNewUser = syscall.CLONE_NEWUSER
	CloneNewUTS  = syscall.CLONE_NEWUTS
)
