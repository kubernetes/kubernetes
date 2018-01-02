package aufs

import "golang.org/x/sys/unix"

func mount(source string, target string, fstype string, flags uintptr, data string) error {
	return unix.Mount(source, target, fstype, flags, data)
}
