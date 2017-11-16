// This file will only be included to the build if neither
// easyjson_nounsafe nor appengine build tag is set. See README notes
// for more details.

//+build !easyjson_nounsafe
//+build !appengine

package jlexer

import (
	"reflect"
	"unsafe"
)

// bytesToStr creates a string pointing at the slice to avoid copying.
//
// Warning: the string returned by the function should be used with care, as the whole input data
// chunk may be either blocked from being freed by GC because of a single string or the buffer.Data
// may be garbage-collected even when the string exists.
func bytesToStr(data []byte) string {
	h := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	shdr := reflect.StringHeader{Data: h.Data, Len: h.Len}
	return *(*string)(unsafe.Pointer(&shdr))
}
