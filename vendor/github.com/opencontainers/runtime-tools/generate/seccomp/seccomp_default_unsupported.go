// +build !linux

package seccomp

// These are copied from linux/amd64 syscall values, as a reference for other
// platforms to have access to
const (
	CloneNewIPC    = 0x8000000
	CloneNewNet    = 0x40000000
	CloneNewNS     = 0x20000
	CloneNewPID    = 0x20000000
	CloneNewUser   = 0x10000000
	CloneNewUTS    = 0x4000000
	CloneNewCgroup = 0x02000000
)
