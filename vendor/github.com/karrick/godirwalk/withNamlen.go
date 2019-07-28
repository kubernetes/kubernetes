// +build darwin dragonfly freebsd netbsd openbsd

package godirwalk

import (
	"reflect"
	"syscall"
	"unsafe"
)

func nameFromDirent(de *syscall.Dirent) []byte {
	// Because this GOOS' syscall.Dirent provides a Namlen field that says how
	// long the name is, this function does not need to search for the NULL
	// byte.
	ml := int(de.Namlen)

	// Convert syscall.Dirent.Name, which is array of int8, to []byte, by
	// overwriting Cap, Len, and Data slice header fields to values from
	// syscall.Dirent fields. Setting the Cap, Len, and Data field values for
	// the slice header modifies what the slice header points to, and in this
	// case, the name buffer.
	var name []byte
	sh := (*reflect.SliceHeader)(unsafe.Pointer(&name))
	sh.Cap = ml
	sh.Len = ml
	sh.Data = uintptr(unsafe.Pointer(&de.Name[0]))

	return name
}
