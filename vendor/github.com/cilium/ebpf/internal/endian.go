package internal

import (
	"encoding/binary"
	"unsafe"
)

// NativeEndian is set to either binary.BigEndian or binary.LittleEndian,
// depending on the host's endianness.
var NativeEndian binary.ByteOrder

// Clang is set to either "el" or "eb" depending on the host's endianness.
var ClangEndian string

func init() {
	if isBigEndian() {
		NativeEndian = binary.BigEndian
		ClangEndian = "eb"
	} else {
		NativeEndian = binary.LittleEndian
		ClangEndian = "el"
	}
}

func isBigEndian() (ret bool) {
	i := int(0x1)
	bs := (*[int(unsafe.Sizeof(i))]byte)(unsafe.Pointer(&i))
	return bs[0] == 0
}
