package mount

/*
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/_iovec.h>
#include <sys/mount.h>
#include <sys/param.h>
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"

	"golang.org/x/sys/unix"
)

func allocateIOVecs(options []string) []C.struct_iovec {
	out := make([]C.struct_iovec, len(options))
	for i, option := range options {
		out[i].iov_base = unsafe.Pointer(C.CString(option))
		out[i].iov_len = C.size_t(len(option) + 1)
	}
	return out
}

func mount(device, target, mType string, flag uintptr, data string) error {
	isNullFS := false

	xs := strings.Split(data, ",")
	for _, x := range xs {
		if x == "bind" {
			isNullFS = true
		}
	}

	options := []string{"fspath", target}
	if isNullFS {
		options = append(options, "fstype", "nullfs", "target", device)
	} else {
		options = append(options, "fstype", mType, "from", device)
	}
	rawOptions := allocateIOVecs(options)
	for _, rawOption := range rawOptions {
		defer C.free(rawOption.iov_base)
	}

	if errno := C.nmount(&rawOptions[0], C.uint(len(options)), C.int(flag)); errno != 0 {
		reason := C.GoString(C.strerror(*C.__error()))
		return fmt.Errorf("Failed to call nmount: %s", reason)
	}
	return nil
}

func unmount(target string, flag int) error {
	return unix.Unmount(target, flag)
}
