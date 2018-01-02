package models // import "github.com/influxdata/influxdb/models"

import (
	"reflect"
	"strconv"
	"unsafe"
)

// parseIntBytes is a zero-alloc wrapper around strconv.ParseInt.
func parseIntBytes(b []byte, base int, bitSize int) (i int64, err error) {
	s := unsafeBytesToString(b)
	return strconv.ParseInt(s, base, bitSize)
}

// parseFloatBytes is a zero-alloc wrapper around strconv.ParseFloat.
func parseFloatBytes(b []byte, bitSize int) (float64, error) {
	s := unsafeBytesToString(b)
	return strconv.ParseFloat(s, bitSize)
}

// parseBoolBytes is a zero-alloc wrapper around strconv.ParseBool.
func parseBoolBytes(b []byte) (bool, error) {
	return strconv.ParseBool(unsafeBytesToString(b))
}

// unsafeBytesToString converts a []byte to a string without a heap allocation.
//
// It is unsafe, and is intended to prepare input to short-lived functions
// that require strings.
func unsafeBytesToString(in []byte) string {
	src := *(*reflect.SliceHeader)(unsafe.Pointer(&in))
	dst := reflect.StringHeader{
		Data: src.Data,
		Len:  src.Len,
	}
	s := *(*string)(unsafe.Pointer(&dst))
	return s
}
