//+build !go1.9

package reflect2

import (
	"unsafe"
)

//go:linkname makemap reflect.makemap
func makemap(rtype unsafe.Pointer) (m unsafe.Pointer)

func makeMapWithSize(rtype unsafe.Pointer, cap int) unsafe.Pointer {
	return makemap(rtype)
}
