// +build nacl linux js solaris

package godirwalk

import (
	"bytes"
	"reflect"
	"syscall"
	"unsafe"
)

// nameOffset is a compile time constant
const nameOffset = int(unsafe.Offsetof(syscall.Dirent{}.Name))

func nameFromDirent(de *syscall.Dirent) (name []byte) {
	// Because this GOOS' syscall.Dirent does not provide a field that specifies
	// the name length, this function must first calculate the max possible name
	// length, and then search for the NULL byte.
	ml := int(de.Reclen) - nameOffset

	// Convert syscall.Dirent.Name, which is array of int8, to []byte, by
	// overwriting Cap, Len, and Data slice header fields to the max possible
	// name length computed above, and finding the terminating NULL byte.
	sh := (*reflect.SliceHeader)(unsafe.Pointer(&name))
	sh.Cap = ml
	sh.Len = ml
	sh.Data = uintptr(unsafe.Pointer(&de.Name[0]))

	if index := bytes.IndexByte(name, 0); index >= 0 {
		// Found NULL byte; set slice's cap and len accordingly.
		sh.Cap = index
		sh.Len = index
		return
	}

	// NOTE: This branch is not expected, but included for defensive
	// programming, and provides a hard stop on the name based on the structure
	// field array size.
	sh.Cap = len(de.Name)
	sh.Len = sh.Cap
	return
}
