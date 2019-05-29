//+build !go1.7

package reflect2

import "unsafe"

func resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer {
	return nil
}
