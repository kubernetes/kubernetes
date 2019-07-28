// +build nacl linux solaris

package godirwalk

import (
	"bytes"
	"reflect"
	"syscall"
	"unsafe"
)

func nameFromDirent(de *syscall.Dirent) []byte {
	// Because this GOOS' syscall.Dirent does not provide a field that specifies
	// the name length, this function must first calculate the max possible name
	// length, and then search for the NULL byte.
	ml := int(uint64(de.Reclen) - uint64(unsafe.Offsetof(syscall.Dirent{}.Name)))

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

	if index := bytes.IndexByte(name, 0); index >= 0 {
		// Found NULL byte; set slice's cap and len accordingly.
		sh.Cap = index
		sh.Len = index
	}

	return name
}
