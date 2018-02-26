// +build solaris,cgo

package mount

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

// #include <stdlib.h>
// #include <stdio.h>
// #include <sys/mount.h>
// int Mount(const char *spec, const char *dir, int mflag,
// char *fstype, char *dataptr, int datalen, char *optptr, int optlen) {
//     return mount(spec, dir, mflag, fstype, dataptr, datalen, optptr, optlen);
// }
import "C"

func mount(device, target, mType string, flag uintptr, data string) error {
	spec := C.CString(device)
	dir := C.CString(target)
	fstype := C.CString(mType)
	_, err := C.Mount(spec, dir, C.int(flag), fstype, nil, 0, nil, 0)
	C.free(unsafe.Pointer(spec))
	C.free(unsafe.Pointer(dir))
	C.free(unsafe.Pointer(fstype))
	return err
}

func unmount(target string, flag int) error {
	err := unix.Unmount(target, flag)
	return err
}
