package aufs

import "syscall"

const MsRemount = syscall.MS_REMOUNT

func mount(source string, target string, fstype string, flags uintptr, data string) error {
	return syscall.Mount(source, target, fstype, flags, data)
}
