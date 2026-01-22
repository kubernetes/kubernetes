//go:build windows

package internal

import (
	"errors"

	"golang.org/x/sys/windows"
)

// Just a dummy.
var (
	ErrSyscallEACCES = errors.New("dummy")
	ErrUnixEACCES    = errors.New("dummy")
)

func SetRlimit()                                    {}
func Maxfiles() uint64                              { return 1<<64 - 1 }
func Mkfifo(path string, mode uint32) error         { return errors.New("no FIFOs on Windows") }
func Mknod(path string, mode uint32, dev int) error { return errors.New("no device nodes on Windows") }

func HasPrivilegesForSymlink() bool {
	var sid *windows.SID
	err := windows.AllocateAndInitializeSid(
		&windows.SECURITY_NT_AUTHORITY,
		2,
		windows.SECURITY_BUILTIN_DOMAIN_RID,
		windows.DOMAIN_ALIAS_RID_ADMINS,
		0, 0, 0, 0, 0, 0,
		&sid)
	if err != nil {
		return false
	}
	defer windows.FreeSid(sid)
	token := windows.Token(0)
	member, err := token.IsMember(sid)
	if err != nil {
		return false
	}
	return member || token.IsElevated()
}
