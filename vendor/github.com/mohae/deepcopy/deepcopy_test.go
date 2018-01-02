package deepcopy

import (
	"fmt"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

// just basic is this working stuff
func TestSimple(t *testing.T) {
	Strings := []string{"a", "b", "c"}
	cpyS := Copy(Strings).([]string)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyS)).Data {
		t.Error("[]string: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyBools
	}
	if len(cpyS) != len(Strings) {
		t.Errorf("[]string: len was %d; want %d", len(cpyS), len(Strings))
		goto CopyBools
	}
	for i, v := range Strings {
		if v != cpyS[i] {
			t.Errorf("[]string: got %v at index %d of the copy; want %v", cpyS[i], i, v)
		}
	}

CopyBools:
	Bools := []bool{true, true, false, false}
	cpyB := Copy(Bools).([]bool)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyB)).Data {
		t.Error("[]bool: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyBytes
	}
	if len(cpyB) != len(Bools) {
		t.Errorf("[]bool: len was %d; want %d", len(cpyB), len(Bools))
		goto CopyBytes
	}
	for i, v := range Bools {
		if v != cpyB[i] {
			t.Errorf("[]bool: got %v at index %d of the copy; want %v", cpyB[i], i, v)
		}
	}

CopyBytes:
	Bytes := []byte("hello")
	cpyBt := Copy(Bytes).([]byte)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyBt)).Data {
		t.Error("[]byte: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyInts
	}
	if len(cpyBt) != len(Bytes) {
		t.Errorf("[]byte: len was %d; want %d", len(cpyBt), len(Bytes))
		goto CopyInts
	}
	for i, v := range Bytes {
		if v != cpyBt[i] {
			t.Errorf("[]byte: got %v at index %d of the copy; want %v", cpyBt[i], i, v)
		}
	}

CopyInts:
	Ints := []int{42}
	cpyI := Copy(Ints).([]int)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyI)).Data {
		t.Error("[]int: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyUints
	}
	if len(cpyI) != len(Ints) {
		t.Errorf("[]int: len was %d; want %d", len(cpyI), len(Ints))
		goto CopyUints
	}
	for i, v := range Ints {
		if v != cpyI[i] {
			t.Errorf("[]int: got %v at index %d of the copy; want %v", cpyI[i], i, v)
		}
	}

CopyUints:
	Uints := []uint{1, 2, 3, 4, 5}
	cpyU := Copy(Uints).([]uint)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyU)).Data {
		t.Error("[]: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyFloat32s
	}
	if len(cpyU) != len(Uints) {
		t.Errorf("[]uint: len was %d; want %d", len(cpyU), len(Uints))
		goto CopyFloat32s
	}
	for i, v := range Uints {
		if v != cpyU[i] {
			t.Errorf("[]uint: got %v at index %d of the copy; want %v", cpyU[i], i, v)
		}
	}

CopyFloat32s:
	Float32s := []float32{3.14}
	cpyF := Copy(Float32s).([]float32)
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyF)).Data {
		t.Error("[]float32: expected SliceHeader data pointers to point to different locations, they didn't")
		goto CopyInterfaces
	}
	if len(cpyF) != len(Float32s) {
		t.Errorf("[]float32: len was %d; want %d", len(cpyF), len(Float32s))
		goto CopyInterfaces
	}
	for i, v := range Float32s {
		if v != cpyF[i] {
			t.Errorf("[]float32: got %v at index %d of the copy; want %v", cpyF[i], i, v)
		}
	}

CopyInterfaces:
	Interfaces := []interface{}{"a", 42, true, 4.32}
	cpyIf := Copy(Interfaces).([]interface{})
	if (*reflect.SliceHeader)(unsafe.Pointer(&Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyIf)).Data {
		t.Error("[]interfaces: expected SliceHeader data pointers to point to different locations, they didn't")
		return
	}
	if len(cpyIf) != len(Interfaces) {
		t.Errorf("[]interface{}: len was %d; want %d", len(cpyIf), len(Interfaces))
		return
	}
	for i, v := range Interfaces {
		if v != cpyIf[i] {
			t.Errorf("[]interface{}: got %v at index %d of the copy; want %v", cpyIf[i], i, v)
		}
	}
}

type Basics struct {
	String      string
	Strings     []string
	StringArr   [4]string
	Bool        bool
	Bools       []bool
	Byte        byte
	Bytes       []byte
	Int         int
	Ints        []int
	Int8        int8
	Int8s       []int8
	Int16       int16
	Int16s      []int16
	Int32       int32
	Int32s      []int32
	Int64       int64
	Int64s      []int64
	Uint        uint
	Uints       []uint
	Uint8       uint8
	Uint8s      []uint8
	Uint16      uint16
	Uint16s     []uint16
	Uint32      uint32
	Uint32s     []uint32
	Uint64      uint64
	Uint64s     []uint64
	Float32     float32
	Float32s    []float32
	Float64     float64
	Float64s    []float64
	Complex64   complex64
	Complex64s  []complex64
	Complex128  complex128
	Complex128s []complex128
	Interface   interface{}
	Interfaces  []interface{}
}

// These tests test that all supported basic types are copied correctly.  This
// is done by copying a struct with fields of most of the basic types as []T.
func TestMostTypes(t *testing.T) {
	test := Basics{
		String:      "kimchi",
		Strings:     []string{"uni", "ika"},
		StringArr:   [4]string{"malort", "barenjager", "fernet", "salmiakki"},
		Bool:        true,
		Bools:       []bool{true, false, true},
		Byte:        'z',
		Bytes:       []byte("abc"),
		Int:         42,
		Ints:        []int{0, 1, 3, 4},
		Int8:        8,
		Int8s:       []int8{8, 9, 10},
		Int16:       16,
		Int16s:      []int16{16, 17, 18, 19},
		Int32:       32,
		Int32s:      []int32{32, 33},
		Int64:       64,
		Int64s:      []int64{64},
		Uint:        420,
		Uints:       []uint{11, 12, 13},
		Uint8:       81,
		Uint8s:      []uint8{81, 82},
		Uint16:      160,
		Uint16s:     []uint16{160, 161, 162, 163, 164},
		Uint32:      320,
		Uint32s:     []uint32{320, 321},
		Uint64:      640,
		Uint64s:     []uint64{6400, 6401, 6402, 6403},
		Float32:     32.32,
		Float32s:    []float32{32.32, 33},
		Float64:     64.1,
		Float64s:    []float64{64, 65, 66},
		Complex64:   complex64(-64 + 12i),
		Complex64s:  []complex64{complex64(-65 + 11i), complex64(66 + 10i)},
		Complex128:  complex128(-128 + 12i),
		Complex128s: []complex128{complex128(-128 + 11i), complex128(129 + 10i)},
		Interfaces:  []interface{}{42, true, "pan-galactic"},
	}

	cpy := Copy(test).(Basics)

	// see if they point to the same location
	if fmt.Sprintf("%p", &cpy) == fmt.Sprintf("%p", &test) {
		t.Error("address of copy was the same as original; they should be different")
		return
	}

	// Go through each field and check to see it got copied properly
	if cpy.String != test.String {
		t.Errorf("String: got %v; want %v", cpy.String, test.String)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Strings)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Strings)).Data {
		t.Error("Strings: address of copy was the same as original; they should be different")
		goto StringArr
	}

	if len(cpy.Strings) != len(test.Strings) {
		t.Errorf("Strings: len was %d; want %d", len(cpy.Strings), len(test.Strings))
		goto StringArr
	}
	for i, v := range test.Strings {
		if v != cpy.Strings[i] {
			t.Errorf("Strings: got %v at index %d of the copy; want %v", cpy.Strings[i], i, v)
		}
	}

StringArr:
	if unsafe.Pointer(&test.StringArr) == unsafe.Pointer(&cpy.StringArr) {
		t.Error("StringArr: address of copy was the same as original; they should be different")
		goto Bools
	}
	for i, v := range test.StringArr {
		if v != cpy.StringArr[i] {
			t.Errorf("StringArr: got %v at index %d of the copy; want %v", cpy.StringArr[i], i, v)
		}
	}

Bools:
	if cpy.Bool != test.Bool {
		t.Errorf("Bool: got %v; want %v", cpy.Bool, test.Bool)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Bools)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Bools)).Data {
		t.Error("Bools: address of copy was the same as original; they should be different")
		goto Bytes
	}
	if len(cpy.Bools) != len(test.Bools) {
		t.Errorf("Bools: len was %d; want %d", len(cpy.Bools), len(test.Bools))
		goto Bytes
	}
	for i, v := range test.Bools {
		if v != cpy.Bools[i] {
			t.Errorf("Bools: got %v at index %d of the copy; want %v", cpy.Bools[i], i, v)
		}
	}

Bytes:
	if cpy.Byte != test.Byte {
		t.Errorf("Byte: got %v; want %v", cpy.Byte, test.Byte)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Bytes)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Bytes)).Data {
		t.Error("Bytes: address of copy was the same as original; they should be different")
		goto Ints
	}
	if len(cpy.Bytes) != len(test.Bytes) {
		t.Errorf("Bytes: len was %d; want %d", len(cpy.Bytes), len(test.Bytes))
		goto Ints
	}
	for i, v := range test.Bytes {
		if v != cpy.Bytes[i] {
			t.Errorf("Bytes: got %v at index %d of the copy; want %v", cpy.Bytes[i], i, v)
		}
	}

Ints:
	if cpy.Int != test.Int {
		t.Errorf("Int: got %v; want %v", cpy.Int, test.Int)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Ints)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Ints)).Data {
		t.Error("Ints: address of copy was the same as original; they should be different")
		goto Int8s
	}
	if len(cpy.Ints) != len(test.Ints) {
		t.Errorf("Ints: len was %d; want %d", len(cpy.Ints), len(test.Ints))
		goto Int8s
	}
	for i, v := range test.Ints {
		if v != cpy.Ints[i] {
			t.Errorf("Ints: got %v at index %d of the copy; want %v", cpy.Ints[i], i, v)
		}
	}

Int8s:
	if cpy.Int8 != test.Int8 {
		t.Errorf("Int8: got %v; want %v", cpy.Int8, test.Int8)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Int8s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Int8s)).Data {
		t.Error("Int8s: address of copy was the same as original; they should be different")
		goto Int16s
	}
	if len(cpy.Int8s) != len(test.Int8s) {
		t.Errorf("Int8s: len was %d; want %d", len(cpy.Int8s), len(test.Int8s))
		goto Int16s
	}
	for i, v := range test.Int8s {
		if v != cpy.Int8s[i] {
			t.Errorf("Int8s: got %v at index %d of the copy; want %v", cpy.Int8s[i], i, v)
		}
	}

Int16s:
	if cpy.Int16 != test.Int16 {
		t.Errorf("Int16: got %v; want %v", cpy.Int16, test.Int16)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Int16s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Int16s)).Data {
		t.Error("Int16s: address of copy was the same as original; they should be different")
		goto Int32s
	}
	if len(cpy.Int16s) != len(test.Int16s) {
		t.Errorf("Int16s: len was %d; want %d", len(cpy.Int16s), len(test.Int16s))
		goto Int32s
	}
	for i, v := range test.Int16s {
		if v != cpy.Int16s[i] {
			t.Errorf("Int16s: got %v at index %d of the copy; want %v", cpy.Int16s[i], i, v)
		}
	}

Int32s:
	if cpy.Int32 != test.Int32 {
		t.Errorf("Int32: got %v; want %v", cpy.Int32, test.Int32)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Int32s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Int32s)).Data {
		t.Error("Int32s: address of copy was the same as original; they should be different")
		goto Int64s
	}
	if len(cpy.Int32s) != len(test.Int32s) {
		t.Errorf("Int32s: len was %d; want %d", len(cpy.Int32s), len(test.Int32s))
		goto Int64s
	}
	for i, v := range test.Int32s {
		if v != cpy.Int32s[i] {
			t.Errorf("Int32s: got %v at index %d of the copy; want %v", cpy.Int32s[i], i, v)
		}
	}

Int64s:
	if cpy.Int64 != test.Int64 {
		t.Errorf("Int64: got %v; want %v", cpy.Int64, test.Int64)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Int64s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Int64s)).Data {
		t.Error("Int64s: address of copy was the same as original; they should be different")
		goto Uints
	}
	if len(cpy.Int64s) != len(test.Int64s) {
		t.Errorf("Int64s: len was %d; want %d", len(cpy.Int64s), len(test.Int64s))
		goto Uints
	}
	for i, v := range test.Int64s {
		if v != cpy.Int64s[i] {
			t.Errorf("Int64s: got %v at index %d of the copy; want %v", cpy.Int64s[i], i, v)
		}
	}

Uints:
	if cpy.Uint != test.Uint {
		t.Errorf("Uint: got %v; want %v", cpy.Uint, test.Uint)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Uints)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Uints)).Data {
		t.Error("Uints: address of copy was the same as original; they should be different")
		goto Uint8s
	}
	if len(cpy.Uints) != len(test.Uints) {
		t.Errorf("Uints: len was %d; want %d", len(cpy.Uints), len(test.Uints))
		goto Uint8s
	}
	for i, v := range test.Uints {
		if v != cpy.Uints[i] {
			t.Errorf("Uints: got %v at index %d of the copy; want %v", cpy.Uints[i], i, v)
		}
	}

Uint8s:
	if cpy.Uint8 != test.Uint8 {
		t.Errorf("Uint8: got %v; want %v", cpy.Uint8, test.Uint8)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Uint8s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Uint8s)).Data {
		t.Error("Uint8s: address of copy was the same as original; they should be different")
		goto Uint16s
	}
	if len(cpy.Uint8s) != len(test.Uint8s) {
		t.Errorf("Uint8s: len was %d; want %d", len(cpy.Uint8s), len(test.Uint8s))
		goto Uint16s
	}
	for i, v := range test.Uint8s {
		if v != cpy.Uint8s[i] {
			t.Errorf("Uint8s: got %v at index %d of the copy; want %v", cpy.Uint8s[i], i, v)
		}
	}

Uint16s:
	if cpy.Uint16 != test.Uint16 {
		t.Errorf("Uint16: got %v; want %v", cpy.Uint16, test.Uint16)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Uint16s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Uint16s)).Data {
		t.Error("Uint16s: address of copy was the same as original; they should be different")
		goto Uint32s
	}
	if len(cpy.Uint16s) != len(test.Uint16s) {
		t.Errorf("Uint16s: len was %d; want %d", len(cpy.Uint16s), len(test.Uint16s))
		goto Uint32s
	}
	for i, v := range test.Uint16s {
		if v != cpy.Uint16s[i] {
			t.Errorf("Uint16s: got %v at index %d of the copy; want %v", cpy.Uint16s[i], i, v)
		}
	}

Uint32s:
	if cpy.Uint32 != test.Uint32 {
		t.Errorf("Uint32: got %v; want %v", cpy.Uint32, test.Uint32)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Uint32s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Uint32s)).Data {
		t.Error("Uint32s: address of copy was the same as original; they should be different")
		goto Uint64s
	}
	if len(cpy.Uint32s) != len(test.Uint32s) {
		t.Errorf("Uint32s: len was %d; want %d", len(cpy.Uint32s), len(test.Uint32s))
		goto Uint64s
	}
	for i, v := range test.Uint32s {
		if v != cpy.Uint32s[i] {
			t.Errorf("Uint32s: got %v at index %d of the copy; want %v", cpy.Uint32s[i], i, v)
		}
	}

Uint64s:
	if cpy.Uint64 != test.Uint64 {
		t.Errorf("Uint64: got %v; want %v", cpy.Uint64, test.Uint64)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Uint64s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Uint64s)).Data {
		t.Error("Uint64s: address of copy was the same as original; they should be different")
		goto Float32s
	}
	if len(cpy.Uint64s) != len(test.Uint64s) {
		t.Errorf("Uint64s: len was %d; want %d", len(cpy.Uint64s), len(test.Uint64s))
		goto Float32s
	}
	for i, v := range test.Uint64s {
		if v != cpy.Uint64s[i] {
			t.Errorf("Uint64s: got %v at index %d of the copy; want %v", cpy.Uint64s[i], i, v)
		}
	}

Float32s:
	if cpy.Float32 != test.Float32 {
		t.Errorf("Float32: got %v; want %v", cpy.Float32, test.Float32)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Float32s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Float32s)).Data {
		t.Error("Float32s: address of copy was the same as original; they should be different")
		goto Float64s
	}
	if len(cpy.Float32s) != len(test.Float32s) {
		t.Errorf("Float32s: len was %d; want %d", len(cpy.Float32s), len(test.Float32s))
		goto Float64s
	}
	for i, v := range test.Float32s {
		if v != cpy.Float32s[i] {
			t.Errorf("Float32s: got %v at index %d of the copy; want %v", cpy.Float32s[i], i, v)
		}
	}

Float64s:
	if cpy.Float64 != test.Float64 {
		t.Errorf("Float64: got %v; want %v", cpy.Float64, test.Float64)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Float64s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Float64s)).Data {
		t.Error("Float64s: address of copy was the same as original; they should be different")
		goto Complex64s
	}
	if len(cpy.Float64s) != len(test.Float64s) {
		t.Errorf("Float64s: len was %d; want %d", len(cpy.Float64s), len(test.Float64s))
		goto Complex64s
	}
	for i, v := range test.Float64s {
		if v != cpy.Float64s[i] {
			t.Errorf("Float64s: got %v at index %d of the copy; want %v", cpy.Float64s[i], i, v)
		}
	}

Complex64s:
	if cpy.Complex64 != test.Complex64 {
		t.Errorf("Complex64: got %v; want %v", cpy.Complex64, test.Complex64)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Complex64s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Complex64s)).Data {
		t.Error("Complex64s: address of copy was the same as original; they should be different")
		goto Complex128s
	}
	if len(cpy.Complex64s) != len(test.Complex64s) {
		t.Errorf("Complex64s: len was %d; want %d", len(cpy.Complex64s), len(test.Complex64s))
		goto Complex128s
	}
	for i, v := range test.Complex64s {
		if v != cpy.Complex64s[i] {
			t.Errorf("Complex64s: got %v at index %d of the copy; want %v", cpy.Complex64s[i], i, v)
		}
	}

Complex128s:
	if cpy.Complex128 != test.Complex128 {
		t.Errorf("Complex128s: got %v; want %v", cpy.Complex128s, test.Complex128s)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Complex128s)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Complex128s)).Data {
		t.Error("Complex128s: address of copy was the same as original; they should be different")
		goto Interfaces
	}
	if len(cpy.Complex128s) != len(test.Complex128s) {
		t.Errorf("Complex128s: len was %d; want %d", len(cpy.Complex128s), len(test.Complex128s))
		goto Interfaces
	}
	for i, v := range test.Complex128s {
		if v != cpy.Complex128s[i] {
			t.Errorf("Complex128s: got %v at index %d of the copy; want %v", cpy.Complex128s[i], i, v)
		}
	}

Interfaces:
	if cpy.Interface != test.Interface {
		t.Errorf("Interface: got %v; want %v", cpy.Interface, test.Interface)
	}

	if (*reflect.SliceHeader)(unsafe.Pointer(&test.Interfaces)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.Interfaces)).Data {
		t.Error("Interfaces: address of copy was the same as original; they should be different")
		return
	}
	if len(cpy.Interfaces) != len(test.Interfaces) {
		t.Errorf("Interfaces: len was %d; want %d", len(cpy.Interfaces), len(test.Interfaces))
		return
	}
	for i, v := range test.Interfaces {
		if v != cpy.Interfaces[i] {
			t.Errorf("Interfaces: got %v at index %d of the copy; want %v", cpy.Interfaces[i], i, v)
		}
	}
}

// not meant to be exhaustive
func TestComplexSlices(t *testing.T) {
	orig3Int := [][][]int{[][]int{[]int{1, 2, 3}, []int{11, 22, 33}}, [][]int{[]int{7, 8, 9}, []int{66, 77, 88, 99}}}
	cpyI := Copy(orig3Int).([][][]int)
	if (*reflect.SliceHeader)(unsafe.Pointer(&orig3Int)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyI)).Data {
		t.Error("[][][]int: address of copy was the same as original; they should be different")
		return
	}
	if len(orig3Int) != len(cpyI) {
		t.Errorf("[][][]int: len of copy was %d; want %d", len(cpyI), len(orig3Int))
		goto sliceMap
	}
	for i, v := range orig3Int {
		if len(v) != len(cpyI[i]) {
			t.Errorf("[][][]int: len of element %d was %d; want %d", i, len(cpyI[i]), len(v))
			continue
		}
		for j, vv := range v {
			if len(vv) != len(cpyI[i][j]) {
				t.Errorf("[][][]int: len of element %d:%d was %d, want %d", i, j, len(cpyI[i][j]), len(vv))
				continue
			}
			for k, vvv := range vv {
				if vvv != cpyI[i][j][k] {
					t.Errorf("[][][]int: element %d:%d:%d was %d, want %d", i, j, k, cpyI[i][j][k], vvv)
				}
			}
		}

	}

sliceMap:
	slMap := []map[int]string{map[int]string{0: "a", 1: "b"}, map[int]string{10: "k", 11: "l", 12: "m"}}
	cpyM := Copy(slMap).([]map[int]string)
	if (*reflect.SliceHeader)(unsafe.Pointer(&slMap)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpyM)).Data {
		t.Error("[]map[int]string: address of copy was the same as original; they should be different")
	}
	if len(slMap) != len(cpyM) {
		t.Errorf("[]map[int]string: len of copy was %d; want %d", len(cpyM), len(slMap))
		goto done
	}
	for i, v := range slMap {
		if len(v) != len(cpyM[i]) {
			t.Errorf("[]map[int]string: len of element %d was %d; want %d", i, len(cpyM[i]), len(v))
			continue
		}
		for k, vv := range v {
			val, ok := cpyM[i][k]
			if !ok {
				t.Errorf("[]map[int]string: element %d was expected to have a value at key %d, it didn't", i, k)
				continue
			}
			if val != vv {
				t.Errorf("[]map[int]string: element %d, key %d: got %s, want %s", i, k, val, vv)
			}
		}
	}
done:
}

type A struct {
	Int    int
	String string
	UintSl []uint
	NilSl  []string
	Map    map[string]int
	MapB   map[string]*B
	SliceB []B
	B
	T time.Time
}

type B struct {
	Vals []string
}

var AStruct = A{
	Int:    42,
	String: "Konichiwa",
	UintSl: []uint{0, 1, 2, 3},
	Map:    map[string]int{"a": 1, "b": 2},
	MapB: map[string]*B{
		"hi":  &B{Vals: []string{"hello", "bonjour"}},
		"bye": &B{Vals: []string{"good-bye", "au revoir"}},
	},
	SliceB: []B{
		B{Vals: []string{"Ciao", "Aloha"}},
	},
	B: B{Vals: []string{"42"}},
	T: time.Now(),
}

func TestStructA(t *testing.T) {
	cpy := Copy(AStruct).(A)
	if &cpy == &AStruct {
		t.Error("expected copy to have a different address than the original; it was the same")
		return
	}
	if cpy.Int != AStruct.Int {
		t.Errorf("A.Int: got %v, want %v", cpy.Int, AStruct.Int)
	}
	if cpy.String != AStruct.String {
		t.Errorf("A.String: got %v; want %v", cpy.String, AStruct.String)
	}
	if (*reflect.SliceHeader)(unsafe.Pointer(&cpy.UintSl)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&AStruct.UintSl)).Data {
		t.Error("A.Uintsl: expected the copies address to be different; it wasn't")
		goto NilSl
	}
	if len(cpy.UintSl) != len(AStruct.UintSl) {
		t.Errorf("A.UintSl: got len of %d, want %d", len(cpy.UintSl), len(AStruct.UintSl))
		goto NilSl
	}
	for i, v := range AStruct.UintSl {
		if cpy.UintSl[i] != v {
			t.Errorf("A.UintSl %d: got %d, want %d", i, cpy.UintSl[i], v)
		}
	}

NilSl:
	if cpy.NilSl != nil {
		t.Error("A.NilSl: expected slice to be nil, it wasn't")
	}

	if *(*uintptr)(unsafe.Pointer(&cpy.Map)) == *(*uintptr)(unsafe.Pointer(&AStruct.Map)) {
		t.Error("A.Map: expected the copy's address to be different; it wasn't")
		goto AMapB
	}
	if len(cpy.Map) != len(AStruct.Map) {
		t.Errorf("A.Map: got len of %d, want %d", len(cpy.Map), len(AStruct.Map))
		goto AMapB
	}
	for k, v := range AStruct.Map {
		val, ok := cpy.Map[k]
		if !ok {
			t.Errorf("A.Map: expected the key %s to exist in the copy, it didn't", k)
			continue
		}
		if val != v {
			t.Errorf("A.Map[%s]: got %d, want %d", k, val, v)
		}
	}

AMapB:
	if *(*uintptr)(unsafe.Pointer(&cpy.MapB)) == *(*uintptr)(unsafe.Pointer(&AStruct.MapB)) {
		t.Error("A.MapB: expected the copy's address to be different; it wasn't")
		goto ASliceB
	}
	if len(cpy.MapB) != len(AStruct.MapB) {
		t.Errorf("A.MapB: got len of %d, want %d", len(cpy.MapB), len(AStruct.MapB))
		goto ASliceB
	}
	for k, v := range AStruct.MapB {
		val, ok := cpy.MapB[k]
		if !ok {
			t.Errorf("A.MapB: expected the key %s to exist in the copy, it didn't", k)
			continue
		}
		if unsafe.Pointer(val) == unsafe.Pointer(v) {
			t.Errorf("A.MapB[%s]: expected the addresses of the values to be different; they weren't", k)
			continue
		}
		// the slice headers should point to different data
		if (*reflect.SliceHeader)(unsafe.Pointer(&v.Vals)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&val.Vals)).Data {
			t.Errorf("%s: expected B's SliceHeaders to point to different Data locations; they did not.", k)
			continue
		}
		for i, vv := range v.Vals {
			if vv != val.Vals[i] {
				t.Errorf("A.MapB[%s].Vals[%d]: got %s want %s", k, i, vv, val.Vals[i])
			}
		}
	}

ASliceB:
	if (*reflect.SliceHeader)(unsafe.Pointer(&AStruct.SliceB)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.SliceB)).Data {
		t.Error("A.SliceB: expected the copy's address to be different; it wasn't")
		goto B
	}

	if len(AStruct.SliceB) != len(cpy.SliceB) {
		t.Errorf("A.SliceB: got length of %d; want %d", len(cpy.SliceB), len(AStruct.SliceB))
		goto B
	}

	for i := range AStruct.SliceB {
		if unsafe.Pointer(&AStruct.SliceB[i]) == unsafe.Pointer(&cpy.SliceB[i]) {
			t.Errorf("A.SliceB[%d]: expected them to have different addresses, they didn't", i)
			continue
		}
		if (*reflect.SliceHeader)(unsafe.Pointer(&AStruct.SliceB[i].Vals)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.SliceB[i].Vals)).Data {
			t.Errorf("A.SliceB[%d]: expected B.Vals SliceHeader.Data to point to different locations; they did not", i)
			continue
		}
		if len(AStruct.SliceB[i].Vals) != len(cpy.SliceB[i].Vals) {
			t.Errorf("A.SliceB[%d]: expected B's vals to have the same length, they didn't", i)
			continue
		}
		for j, val := range AStruct.SliceB[i].Vals {
			if val != cpy.SliceB[i].Vals[j] {
				t.Errorf("A.SliceB[%d].Vals[%d]: got %v; want %v", i, j, cpy.SliceB[i].Vals[j], val)
			}
		}
	}
B:
	if unsafe.Pointer(&AStruct.B) == unsafe.Pointer(&cpy.B) {
		t.Error("A.B: expected them to have different addresses, they didn't")
		goto T
	}
	if (*reflect.SliceHeader)(unsafe.Pointer(&AStruct.B.Vals)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&cpy.B.Vals)).Data {
		t.Error("A.B.Vals: expected the SliceHeaders.Data to point to different locations; they didn't")
		goto T
	}
	if len(AStruct.B.Vals) != len(cpy.B.Vals) {
		t.Error("A.B.Vals: expected their lengths to be the same, they weren't")
		goto T
	}
	for i, v := range AStruct.B.Vals {
		if v != cpy.B.Vals[i] {
			t.Errorf("A.B.Vals[%d]: got %s want %s", i, cpy.B.Vals[i], v)
		}
	}
T:
	if fmt.Sprintf("%p", &AStruct.T) == fmt.Sprintf("%p", &cpy.T) {
		t.Error("A.T: expected them to have different addresses, they didn't")
		return
	}
	if AStruct.T != cpy.T {
		t.Errorf("A.T: got %v, want %v", cpy.T, AStruct.T)
	}
}

type Unexported struct {
	A  string
	B  int
	aa string
	bb int
	cc []int
	dd map[string]string
}

func TestUnexportedFields(t *testing.T) {
	u := &Unexported{
		A:  "A",
		B:  42,
		aa: "aa",
		bb: 42,
		cc: []int{1, 2, 3},
		dd: map[string]string{"hello": "bonjour"},
	}
	cpy := Copy(u).(*Unexported)
	if cpy == u {
		t.Error("expected addresses to be different, they weren't")
		return
	}
	if u.A != cpy.A {
		t.Errorf("Unexported.A: got %s want %s", cpy.A, u.A)
	}
	if u.B != cpy.B {
		t.Errorf("Unexported.A: got %d want %d", cpy.B, u.B)
	}
	if cpy.aa != "" {
		t.Errorf("Unexported.aa: unexported field should not be set, it was set to %s", cpy.aa)
	}
	if cpy.bb != 0 {
		t.Errorf("Unexported.bb: unexported field should not be set, it was set to %d", cpy.bb)
	}
	if cpy.cc != nil {
		t.Errorf("Unexported.cc: unexported field should not be set, it was set to %#v", cpy.cc)
	}
	if cpy.dd != nil {
		t.Errorf("Unexported.dd: unexported field should not be set, it was set to %#v", cpy.dd)
	}
}

// Note: this test will fail until https://github.com/golang/go/issues/15716 is
// fixed and the version it is part of gets released.
type T struct {
	time.Time
}

func TestTimeCopy(t *testing.T) {
	tests := []struct {
		Y    int
		M    time.Month
		D    int
		h    int
		m    int
		s    int
		nsec int
		TZ   string
	}{
		{2016, time.July, 4, 23, 11, 33, 3000, "America/New_York"},
		{2015, time.October, 31, 9, 44, 23, 45935, "UTC"},
		{2014, time.May, 5, 22, 01, 50, 219300, "Europe/Prague"},
	}

	for i, test := range tests {
		l, err := time.LoadLocation(test.TZ)
		if err != nil {
			t.Errorf("%d: unexpected error: %s", i, err)
			continue
		}
		var x T
		x.Time = time.Date(test.Y, test.M, test.D, test.h, test.m, test.s, test.nsec, l)
		c := Copy(x).(T)
		if fmt.Sprintf("%p", &c) == fmt.Sprintf("%p", &x) {
			t.Errorf("%d: expected the copy to have a different address than the original value; they were the same: %p %p", i, &c, &x)
			continue
		}
		if x.UnixNano() != c.UnixNano() {
			t.Errorf("%d: nanotime: got %v; want %v", i, c.UnixNano(), x.UnixNano())
			continue
		}
		if x.Location() != c.Location() {
			t.Errorf("%d: location: got %q; want %q", i, c.Location(), x.Location())
		}
	}
}

func TestPointerToStruct(t *testing.T) {
	type Foo struct {
		Bar int
	}

	f := &Foo{Bar: 42}
	cpy := Copy(f)
	if f == cpy {
		t.Errorf("expected copy to point to a different location: orig: %p; copy: %p", f, cpy)
	}
	if !reflect.DeepEqual(f, cpy) {
		t.Errorf("expected the copy to be equal to the original (except for memory location); it wasn't: got %#v; want %#v", f, cpy)
	}
}

func TestIssue9(t *testing.T) {
	// simple pointer copy
	x := 42
	testA := map[string]*int{
		"a": nil,
		"b": &x,
	}
	copyA := Copy(testA).(map[string]*int)
	if unsafe.Pointer(&testA) == unsafe.Pointer(&copyA) {
		t.Fatalf("expected the map pointers to be different: testA: %v\tcopyA: %v", unsafe.Pointer(&testA), unsafe.Pointer(&copyA))
	}
	if !reflect.DeepEqual(testA, copyA) {
		t.Errorf("got %#v; want %#v", copyA, testA)
	}
	if testA["b"] == copyA["b"] {
		t.Errorf("entries for 'b' pointed to the same address: %v; expected them to point to different addresses", testA["b"])
	}

	// map copy
	type Foo struct {
		Alpha string
	}

	type Bar struct {
		Beta  string
		Gamma int
		Delta *Foo
	}

	type Biz struct {
		Epsilon map[int]*Bar
	}

	testB := Biz{
		Epsilon: map[int]*Bar{
			0: &Bar{},
			1: &Bar{
				Beta:  "don't panic",
				Gamma: 42,
				Delta: nil,
			},
			2: &Bar{
				Beta:  "sudo make me a sandwich.",
				Gamma: 11,
				Delta: &Foo{
					Alpha: "okay.",
				},
			},
		},
	}

	copyB := Copy(testB).(Biz)
	if !reflect.DeepEqual(testB, copyB) {
		t.Errorf("got %#v; want %#v", copyB, testB)
		return
	}

	// check that the maps point to different locations
	if unsafe.Pointer(&testB.Epsilon) == unsafe.Pointer(&copyB.Epsilon) {
		t.Fatalf("expected the map pointers to be different; they weren't: testB: %v\tcopyB: %v", unsafe.Pointer(&testB.Epsilon), unsafe.Pointer(&copyB.Epsilon))
	}

	for k, v := range testB.Epsilon {
		if v == nil && copyB.Epsilon[k] == nil {
			continue
		}
		if v == nil && copyB.Epsilon[k] != nil {
			t.Errorf("%d: expected copy of a nil entry to be nil; it wasn't: %#v", copyB.Epsilon[k])
			continue
		}
		if v == copyB.Epsilon[k] {
			t.Errorf("entries for '%d' pointed to the same address: %v; expected them to point to different addresses", v)
			continue
		}
		if v.Beta != copyB.Epsilon[k].Beta {
			t.Errorf("%d.Beta: got %q; want %q", copyB.Epsilon[k].Beta, v.Beta)
		}
		if v.Gamma != copyB.Epsilon[k].Gamma {
			t.Errorf("%d.Gamma: got %d; want %d", copyB.Epsilon[k].Gamma, v.Gamma)
		}
		if v.Delta == nil && copyB.Epsilon[k].Delta == nil {
			continue
		}
		if v.Delta == nil && copyB.Epsilon[k].Delta != nil {
			t.Errorf("%d.Delta: got %#v; want nil", copyB.Epsilon[k].Delta)
		}
		if v.Delta == copyB.Epsilon[k].Delta {
			t.Errorf("%d.Delta: expected the pointers to be different, they were the same: %v", k, v.Delta)
			continue
		}
		if v.Delta.Alpha != copyB.Epsilon[k].Delta.Alpha {
			t.Errorf("%d.Delta.Foo: got %q; want %q", v.Delta.Alpha, copyB.Epsilon[k].Delta.Alpha)
		}
	}

	// test that map keys are deep copied
	testC := map[*Foo][]string{
		&Foo{Alpha: "Henry Dorsett Case"}: []string{
			"Cutter",
		},
		&Foo{Alpha: "Molly Millions"}: []string{
			"Rose Kolodny",
			"Cat Mother",
			"Steppin' Razor",
		},
	}

	copyC := Copy(testC).(map[*Foo][]string)
	if unsafe.Pointer(&testC) == unsafe.Pointer(&copyC) {
		t.Fatalf("expected the map pointers to be different; they weren't: testB: %v\tcopyB: %v", unsafe.Pointer(&testB.Epsilon), unsafe.Pointer(&copyB.Epsilon))
	}

	// make sure the lengths are the same
	if len(testC) != len(copyC) {
		t.Fatalf("got len %d; want %d", len(copyC), len(testC))
	}

	// check that everything was deep copied: since the key is a pointer, we check to
	// see if the pointers are different but the values being pointed to are the same.
	for k, v := range testC {
		for kk, vv := range copyC {
			if *kk == *k {
				if kk == k {
					t.Errorf("key pointers should be different: orig: %p; copy: %p", k, kk)
				}
				// check that the slices are the same but different
				if !reflect.DeepEqual(v, vv) {
					t.Errorf("expected slice contents to be the same; they weren't: orig: %v; copy: %v", v, vv)
				}

				if (*reflect.SliceHeader)(unsafe.Pointer(&v)).Data == (*reflect.SliceHeader)(unsafe.Pointer(&vv)).Data {
					t.Error("expected the SliceHeaders.Data to point to different locations; they didn't: %v", (*reflect.SliceHeader)(unsafe.Pointer(&v)).Data)
				}
				break
			}
		}
	}

	type Bizz struct {
		*Foo
	}

	testD := map[Bizz]string{
		Bizz{&Foo{"Neuromancer"}}: "Rio",
		Bizz{&Foo{"Wintermute"}}:  "Berne",
	}
	copyD := Copy(testD).(map[Bizz]string)
	if len(copyD) != len(testD) {
		t.Fatalf("copy had %d elements; expected %d", len(copyD), len(testD))
	}

	for k, v := range testD {
		var found bool
		for kk, vv := range copyD {
			if reflect.DeepEqual(k, kk) {
				found = true
				// check that Foo points to different locations
				if unsafe.Pointer(k.Foo) == unsafe.Pointer(kk.Foo) {
					t.Errorf("Expected Foo to point to different locations; they didn't: orig: %p; copy %p", k.Foo, kk.Foo)
					break
				}
				if *k.Foo != *kk.Foo {
					t.Errorf("Expected copy of the key's Foo field to have the same value as the original, it wasn't: orig: %#v; copy: %#v", k.Foo, kk.Foo)
				}
				if v != vv {
					t.Errorf("Expected the values to be the same; the weren't: got %v; want %v", vv, v)
				}
			}
		}
		if !found {
			t.Errorf("expected key %v to exist in the copy; it didn't", k)
		}
	}
}
