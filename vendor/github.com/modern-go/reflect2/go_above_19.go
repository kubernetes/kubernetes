//+build go1.9

package reflect2

import (
	"unsafe"
)

//go:linkname resolveTypeOff reflect.resolveTypeOff
func resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer

//go:linkname makemap reflect.makemap
func makemap(rtype unsafe.Pointer, cap int) (m unsafe.Pointer)

func makeMapWithSize(rtype unsafe.Pointer, cap int) unsafe.Pointer {
	return makemap(rtype, cap)
}
