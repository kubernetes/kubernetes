package reflect2

import (
	"reflect"
	"unsafe"
)

// DefaultTypeOfKind return the non aliased default type for the kind
func DefaultTypeOfKind(kind reflect.Kind) Type {
	return kindTypes[kind]
}

var kindTypes = map[reflect.Kind]Type{
	reflect.Bool:          TypeOf(true),
	reflect.Uint8:         TypeOf(uint8(0)),
	reflect.Int8:          TypeOf(int8(0)),
	reflect.Uint16:        TypeOf(uint16(0)),
	reflect.Int16:         TypeOf(int16(0)),
	reflect.Uint32:        TypeOf(uint32(0)),
	reflect.Int32:         TypeOf(int32(0)),
	reflect.Uint64:        TypeOf(uint64(0)),
	reflect.Int64:         TypeOf(int64(0)),
	reflect.Uint:          TypeOf(uint(0)),
	reflect.Int:           TypeOf(int(0)),
	reflect.Float32:       TypeOf(float32(0)),
	reflect.Float64:       TypeOf(float64(0)),
	reflect.Uintptr:       TypeOf(uintptr(0)),
	reflect.String:        TypeOf(""),
	reflect.UnsafePointer: TypeOf(unsafe.Pointer(nil)),
}
