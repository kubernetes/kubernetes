/*
 * Copyright (c) 2013 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/*
Test Summary:
NOTE: For each test, a nil pointer, a single pointer and double pointer to the
base test element are also tested to ensure proper indirection across all types.

- Max int8, int16, int32, int64, int
- Max uint8, uint16, uint32, uint64, uint
- Boolean true and false
- Standard complex64 and complex128
- Array containing standard ints
- Array containing type with custom formatter on pointer receiver only
- Array containing interfaces
- Slice containing standard float32 values
- Slice containing type with custom formatter on pointer receiver only
- Slice containing interfaces
- Nil slice
- Standard string
- Nil interface
- Sub-interface
- Map with string keys and int vals
- Map with custom formatter type on pointer receiver only keys and vals
- Map with interface keys and values
- Map with nil interface value
- Struct with primitives
- Struct that contains another struct
- Struct that contains custom type with Stringer pointer interface via both
  exported and unexported fields
- Struct that contains embedded struct and field to same struct
- Uintptr to 0 (null pointer)
- Uintptr address of real variable
- Unsafe.Pointer to 0 (null pointer)
- Unsafe.Pointer to address of real variable
- Nil channel
- Standard int channel
- Function with no params and no returns
- Function with param and no returns
- Function with multiple params and multiple returns
- Struct that is circular through self referencing
- Structs that are circular through cross referencing
- Structs that are indirectly circular
- Type that panics in its Stringer interface
- Type that has a custom Error interface
- %x passthrough with uint
- %#x passthrough with uint
- %f passthrough with precision
- %f passthrough with width and precision
- %d passthrough with width
- %q passthrough with string
*/

package spew_test

import (
	"bytes"
	"fmt"
	"github.com/davecgh/go-spew/spew"
	"testing"
	"unsafe"
)

// formatterTest is used to describe a test to be perfomed against NewFormatter.
type formatterTest struct {
	format string
	in     interface{}
	wants  []string
}

// formatterTests houses all of the tests to be performed against NewFormatter.
var formatterTests = make([]formatterTest, 0)

// addFormatterTest is a helper method to append the passed input and desired
// result to formatterTests.
func addFormatterTest(format string, in interface{}, wants ...string) {
	test := formatterTest{format, in, wants}
	formatterTests = append(formatterTests, test)
}

func addIntFormatterTests() {
	// Max int8.
	v := int8(127)
	nv := (*int8)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "int8"
	vs := "127"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Max int16.
	v2 := int16(32767)
	nv2 := (*int16)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "int16"
	v2s := "32767"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Max int32.
	v3 := int32(2147483647)
	nv3 := (*int32)(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "int32"
	v3s := "2147483647"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")

	// Max int64.
	v4 := int64(9223372036854775807)
	nv4 := (*int64)(nil)
	pv4 := &v4
	v4Addr := fmt.Sprintf("%p", pv4)
	pv4Addr := fmt.Sprintf("%p", &pv4)
	v4t := "int64"
	v4s := "9223372036854775807"
	addFormatterTest("%v", v4, v4s)
	addFormatterTest("%v", pv4, "<*>"+v4s)
	addFormatterTest("%v", &pv4, "<**>"+v4s)
	addFormatterTest("%v", nv4, "<nil>")
	addFormatterTest("%+v", v4, v4s)
	addFormatterTest("%+v", pv4, "<*>("+v4Addr+")"+v4s)
	addFormatterTest("%+v", &pv4, "<**>("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%#v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#v", pv4, "(*"+v4t+")"+v4s)
	addFormatterTest("%#v", &pv4, "(**"+v4t+")"+v4s)
	addFormatterTest("%#v", nv4, "(*"+v4t+")"+"<nil>")
	addFormatterTest("%#+v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#+v", pv4, "(*"+v4t+")("+v4Addr+")"+v4s)
	addFormatterTest("%#+v", &pv4, "(**"+v4t+")("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%#+v", nv4, "(*"+v4t+")"+"<nil>")

	// Max int.
	v5 := int(2147483647)
	nv5 := (*int)(nil)
	pv5 := &v5
	v5Addr := fmt.Sprintf("%p", pv5)
	pv5Addr := fmt.Sprintf("%p", &pv5)
	v5t := "int"
	v5s := "2147483647"
	addFormatterTest("%v", v5, v5s)
	addFormatterTest("%v", pv5, "<*>"+v5s)
	addFormatterTest("%v", &pv5, "<**>"+v5s)
	addFormatterTest("%v", nv5, "<nil>")
	addFormatterTest("%+v", v5, v5s)
	addFormatterTest("%+v", pv5, "<*>("+v5Addr+")"+v5s)
	addFormatterTest("%+v", &pv5, "<**>("+pv5Addr+"->"+v5Addr+")"+v5s)
	addFormatterTest("%+v", nv5, "<nil>")
	addFormatterTest("%#v", v5, "("+v5t+")"+v5s)
	addFormatterTest("%#v", pv5, "(*"+v5t+")"+v5s)
	addFormatterTest("%#v", &pv5, "(**"+v5t+")"+v5s)
	addFormatterTest("%#v", nv5, "(*"+v5t+")"+"<nil>")
	addFormatterTest("%#+v", v5, "("+v5t+")"+v5s)
	addFormatterTest("%#+v", pv5, "(*"+v5t+")("+v5Addr+")"+v5s)
	addFormatterTest("%#+v", &pv5, "(**"+v5t+")("+pv5Addr+"->"+v5Addr+")"+v5s)
	addFormatterTest("%#+v", nv5, "(*"+v5t+")"+"<nil>")
}

func addUintFormatterTests() {
	// Max uint8.
	v := uint8(255)
	nv := (*uint8)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "uint8"
	vs := "255"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Max uint16.
	v2 := uint16(65535)
	nv2 := (*uint16)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "uint16"
	v2s := "65535"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Max uint32.
	v3 := uint32(4294967295)
	nv3 := (*uint32)(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "uint32"
	v3s := "4294967295"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")

	// Max uint64.
	v4 := uint64(18446744073709551615)
	nv4 := (*uint64)(nil)
	pv4 := &v4
	v4Addr := fmt.Sprintf("%p", pv4)
	pv4Addr := fmt.Sprintf("%p", &pv4)
	v4t := "uint64"
	v4s := "18446744073709551615"
	addFormatterTest("%v", v4, v4s)
	addFormatterTest("%v", pv4, "<*>"+v4s)
	addFormatterTest("%v", &pv4, "<**>"+v4s)
	addFormatterTest("%v", nv4, "<nil>")
	addFormatterTest("%+v", v4, v4s)
	addFormatterTest("%+v", pv4, "<*>("+v4Addr+")"+v4s)
	addFormatterTest("%+v", &pv4, "<**>("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%#v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#v", pv4, "(*"+v4t+")"+v4s)
	addFormatterTest("%#v", &pv4, "(**"+v4t+")"+v4s)
	addFormatterTest("%#v", nv4, "(*"+v4t+")"+"<nil>")
	addFormatterTest("%#+v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#+v", pv4, "(*"+v4t+")("+v4Addr+")"+v4s)
	addFormatterTest("%#+v", &pv4, "(**"+v4t+")("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%#+v", nv4, "(*"+v4t+")"+"<nil>")

	// Max uint.
	v5 := uint(4294967295)
	nv5 := (*uint)(nil)
	pv5 := &v5
	v5Addr := fmt.Sprintf("%p", pv5)
	pv5Addr := fmt.Sprintf("%p", &pv5)
	v5t := "uint"
	v5s := "4294967295"
	addFormatterTest("%v", v5, v5s)
	addFormatterTest("%v", pv5, "<*>"+v5s)
	addFormatterTest("%v", &pv5, "<**>"+v5s)
	addFormatterTest("%v", nv5, "<nil>")
	addFormatterTest("%+v", v5, v5s)
	addFormatterTest("%+v", pv5, "<*>("+v5Addr+")"+v5s)
	addFormatterTest("%+v", &pv5, "<**>("+pv5Addr+"->"+v5Addr+")"+v5s)
	addFormatterTest("%+v", nv5, "<nil>")
	addFormatterTest("%#v", v5, "("+v5t+")"+v5s)
	addFormatterTest("%#v", pv5, "(*"+v5t+")"+v5s)
	addFormatterTest("%#v", &pv5, "(**"+v5t+")"+v5s)
	addFormatterTest("%#v", nv5, "(*"+v5t+")"+"<nil>")
	addFormatterTest("%#+v", v5, "("+v5t+")"+v5s)
	addFormatterTest("%#+v", pv5, "(*"+v5t+")("+v5Addr+")"+v5s)
	addFormatterTest("%#+v", &pv5, "(**"+v5t+")("+pv5Addr+"->"+v5Addr+")"+v5s)
	addFormatterTest("%#v", nv5, "(*"+v5t+")"+"<nil>")
}

func addBoolFormatterTests() {
	// Boolean true.
	v := bool(true)
	nv := (*bool)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "bool"
	vs := "true"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Boolean false.
	v2 := bool(false)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "bool"
	v2s := "false"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
}

func addFloatFormatterTests() {
	// Standard float32.
	v := float32(3.1415)
	nv := (*float32)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "float32"
	vs := "3.1415"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Standard float64.
	v2 := float64(3.1415926)
	nv2 := (*float64)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "float64"
	v2s := "3.1415926"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")
}

func addComplexFormatterTests() {
	// Standard complex64.
	v := complex(float32(6), -2)
	nv := (*complex64)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "complex64"
	vs := "(6-2i)"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Standard complex128.
	v2 := complex(float64(-6), 2)
	nv2 := (*complex128)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "complex128"
	v2s := "(-6+2i)"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")
}

func addArrayFormatterTests() {
	// Array containing standard ints.
	v := [3]int{1, 2, 3}
	nv := (*[3]int)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "[3]int"
	vs := "[1 2 3]"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Array containing type with custom formatter on pointer receiver only.
	v2 := [3]pstringer{"1", "2", "3"}
	nv2 := (*[3]pstringer)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "[3]spew_test.pstringer"
	v2s := "[stringer 1 stringer 2 stringer 3]"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Array containing interfaces.
	v3 := [3]interface{}{"one", int(2), uint(3)}
	nv3 := (*[3]interface{})(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "[3]interface {}"
	v3t2 := "string"
	v3t3 := "int"
	v3t4 := "uint"
	v3s := "[one 2 3]"
	v3s2 := "[(" + v3t2 + ")one (" + v3t3 + ")2 (" + v3t4 + ")3]"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s2)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s2)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", nv3, "(*"+v3t+")"+"<nil>")
}

func addSliceFormatterTests() {
	// Slice containing standard float32 values.
	v := []float32{3.14, 6.28, 12.56}
	nv := (*[]float32)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "[]float32"
	vs := "[3.14 6.28 12.56]"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Slice containing type with custom formatter on pointer receiver only.
	v2 := []pstringer{"1", "2", "3"}
	nv2 := (*[]pstringer)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "[]spew_test.pstringer"
	v2s := "[stringer 1 stringer 2 stringer 3]"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Slice containing interfaces.
	v3 := []interface{}{"one", int(2), uint(3), nil}
	nv3 := (*[]interface{})(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "[]interface {}"
	v3t2 := "string"
	v3t3 := "int"
	v3t4 := "uint"
	v3t5 := "interface {}"
	v3s := "[one 2 3 <nil>]"
	v3s2 := "[(" + v3t2 + ")one (" + v3t3 + ")2 (" + v3t4 + ")3 (" + v3t5 +
		")<nil>]"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s2)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s2)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", nv3, "(*"+v3t+")"+"<nil>")

	// Nil slice.
	var v4 []int
	nv4 := (*[]int)(nil)
	pv4 := &v4
	v4Addr := fmt.Sprintf("%p", pv4)
	pv4Addr := fmt.Sprintf("%p", &pv4)
	v4t := "[]int"
	v4s := "<nil>"
	addFormatterTest("%v", v4, v4s)
	addFormatterTest("%v", pv4, "<*>"+v4s)
	addFormatterTest("%v", &pv4, "<**>"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%+v", v4, v4s)
	addFormatterTest("%+v", pv4, "<*>("+v4Addr+")"+v4s)
	addFormatterTest("%+v", &pv4, "<**>("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%#v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#v", pv4, "(*"+v4t+")"+v4s)
	addFormatterTest("%#v", &pv4, "(**"+v4t+")"+v4s)
	addFormatterTest("%#v", nv4, "(*"+v4t+")"+"<nil>")
	addFormatterTest("%#+v", v4, "("+v4t+")"+v4s)
	addFormatterTest("%#+v", pv4, "(*"+v4t+")("+v4Addr+")"+v4s)
	addFormatterTest("%#+v", &pv4, "(**"+v4t+")("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%#+v", nv4, "(*"+v4t+")"+"<nil>")
}

func addStringFormatterTests() {
	// Standard string.
	v := "test"
	nv := (*string)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "string"
	vs := "test"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")
}

func addInterfaceFormatterTests() {
	// Nil interface.
	var v interface{}
	nv := (*interface{})(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "interface {}"
	vs := "<nil>"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Sub-interface.
	v2 := interface{}(uint16(65535))
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "uint16"
	v2s := "65535"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
}

func addMapFormatterTests() {
	// Map with string keys and int vals.
	v := map[string]int{"one": 1, "two": 2}
	nv := (*map[string]int)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "map[string]int"
	vs := "map[one:1 two:2]"
	vs2 := "map[two:2 one:1]"
	addFormatterTest("%v", v, vs, vs2)
	addFormatterTest("%v", pv, "<*>"+vs, "<*>"+vs2)
	addFormatterTest("%v", &pv, "<**>"+vs, "<**>"+vs2)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs, vs2)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs, "<*>("+vAddr+")"+vs2)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs,
		"<**>("+pvAddr+"->"+vAddr+")"+vs2)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs, "("+vt+")"+vs2)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs, "(*"+vt+")"+vs2)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs, "(**"+vt+")"+vs2)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs, "("+vt+")"+vs2)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs,
		"(*"+vt+")("+vAddr+")"+vs2)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs,
		"(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs2)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Map with custom formatter type on pointer receiver only keys and vals.
	v2 := map[pstringer]pstringer{"one": "1"}
	nv2 := (*map[pstringer]pstringer)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "map[spew_test.pstringer]spew_test.pstringer"
	v2s := "map[stringer one:stringer 1]"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Map with interface keys and values.
	v3 := map[interface{}]interface{}{"one": 1}
	nv3 := (*map[interface{}]interface{})(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "map[interface {}]interface {}"
	v3t1 := "string"
	v3t2 := "int"
	v3s := "map[one:1]"
	v3s2 := "map[(" + v3t1 + ")one:(" + v3t2 + ")1]"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s2)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s2)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s2)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s2)
	addFormatterTest("%#+v", nv3, "(*"+v3t+")"+"<nil>")

	// Map with nil interface value
	v4 := map[string]interface{}{"nil": nil}
	nv4 := (*map[string]interface{})(nil)
	pv4 := &v4
	v4Addr := fmt.Sprintf("%p", pv4)
	pv4Addr := fmt.Sprintf("%p", &pv4)
	v4t := "map[string]interface {}"
	v4t1 := "interface {}"
	v4s := "map[nil:<nil>]"
	v4s2 := "map[nil:(" + v4t1 + ")<nil>]"
	addFormatterTest("%v", v4, v4s)
	addFormatterTest("%v", pv4, "<*>"+v4s)
	addFormatterTest("%v", &pv4, "<**>"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%+v", v4, v4s)
	addFormatterTest("%+v", pv4, "<*>("+v4Addr+")"+v4s)
	addFormatterTest("%+v", &pv4, "<**>("+pv4Addr+"->"+v4Addr+")"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%#v", v4, "("+v4t+")"+v4s2)
	addFormatterTest("%#v", pv4, "(*"+v4t+")"+v4s2)
	addFormatterTest("%#v", &pv4, "(**"+v4t+")"+v4s2)
	addFormatterTest("%#v", nv4, "(*"+v4t+")"+"<nil>")
	addFormatterTest("%#+v", v4, "("+v4t+")"+v4s2)
	addFormatterTest("%#+v", pv4, "(*"+v4t+")("+v4Addr+")"+v4s2)
	addFormatterTest("%#+v", &pv4, "(**"+v4t+")("+pv4Addr+"->"+v4Addr+")"+v4s2)
	addFormatterTest("%#+v", nv4, "(*"+v4t+")"+"<nil>")
}

func addStructFormatterTests() {
	// Struct with primitives.
	type s1 struct {
		a int8
		b uint8
	}
	v := s1{127, 255}
	nv := (*s1)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "spew_test.s1"
	vt2 := "int8"
	vt3 := "uint8"
	vs := "{127 255}"
	vs2 := "{a:127 b:255}"
	vs3 := "{a:(" + vt2 + ")127 b:(" + vt3 + ")255}"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs2)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs2)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs2)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs3)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs3)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs3)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs3)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs3)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs3)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Struct that contains another struct.
	type s2 struct {
		s1 s1
		b  bool
	}
	v2 := s2{s1{127, 255}, true}
	nv2 := (*s2)(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "spew_test.s2"
	v2t2 := "spew_test.s1"
	v2t3 := "int8"
	v2t4 := "uint8"
	v2t5 := "bool"
	v2s := "{{127 255} true}"
	v2s2 := "{s1:{a:127 b:255} b:true}"
	v2s3 := "{s1:(" + v2t2 + "){a:(" + v2t3 + ")127 b:(" + v2t4 + ")255} b:(" +
		v2t5 + ")true}"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s2)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s2)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s2)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s3)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s3)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s3)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s3)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s3)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s3)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Struct that contains custom type with Stringer pointer interface via both
	// exported and unexported fields.
	type s3 struct {
		s pstringer
		S pstringer
	}
	v3 := s3{"test", "test2"}
	nv3 := (*s3)(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "spew_test.s3"
	v3t2 := "spew_test.pstringer"
	v3s := "{stringer test stringer test2}"
	v3s2 := "{s:stringer test S:stringer test2}"
	v3s3 := "{s:(" + v3t2 + ")stringer test S:(" + v3t2 + ")stringer test2}"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s2)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s2)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s2)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s3)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s3)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s3)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s3)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s3)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s3)
	addFormatterTest("%#+v", nv3, "(*"+v3t+")"+"<nil>")

	// Struct that contains embedded struct and field to same struct.
	e := embed{"embedstr"}
	v4 := embedwrap{embed: &e, e: &e}
	nv4 := (*embedwrap)(nil)
	pv4 := &v4
	eAddr := fmt.Sprintf("%p", &e)
	v4Addr := fmt.Sprintf("%p", pv4)
	pv4Addr := fmt.Sprintf("%p", &pv4)
	v4t := "spew_test.embedwrap"
	v4t2 := "spew_test.embed"
	v4t3 := "string"
	v4s := "{<*>{embedstr} <*>{embedstr}}"
	v4s2 := "{embed:<*>(" + eAddr + "){a:embedstr} e:<*>(" + eAddr +
		"){a:embedstr}}"
	v4s3 := "{embed:(*" + v4t2 + "){a:(" + v4t3 + ")embedstr} e:(*" + v4t2 +
		"){a:(" + v4t3 + ")embedstr}}"
	v4s4 := "{embed:(*" + v4t2 + ")(" + eAddr + "){a:(" + v4t3 +
		")embedstr} e:(*" + v4t2 + ")(" + eAddr + "){a:(" + v4t3 + ")embedstr}}"
	addFormatterTest("%v", v4, v4s)
	addFormatterTest("%v", pv4, "<*>"+v4s)
	addFormatterTest("%v", &pv4, "<**>"+v4s)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%+v", v4, v4s2)
	addFormatterTest("%+v", pv4, "<*>("+v4Addr+")"+v4s2)
	addFormatterTest("%+v", &pv4, "<**>("+pv4Addr+"->"+v4Addr+")"+v4s2)
	addFormatterTest("%+v", nv4, "<nil>")
	addFormatterTest("%#v", v4, "("+v4t+")"+v4s3)
	addFormatterTest("%#v", pv4, "(*"+v4t+")"+v4s3)
	addFormatterTest("%#v", &pv4, "(**"+v4t+")"+v4s3)
	addFormatterTest("%#v", nv4, "(*"+v4t+")"+"<nil>")
	addFormatterTest("%#+v", v4, "("+v4t+")"+v4s4)
	addFormatterTest("%#+v", pv4, "(*"+v4t+")("+v4Addr+")"+v4s4)
	addFormatterTest("%#+v", &pv4, "(**"+v4t+")("+pv4Addr+"->"+v4Addr+")"+v4s4)
	addFormatterTest("%#+v", nv4, "(*"+v4t+")"+"<nil>")
}

func addUintptrFormatterTests() {
	// Null pointer.
	v := uintptr(0)
	nv := (*uintptr)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "uintptr"
	vs := "<nil>"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Address of real variable.
	i := 1
	v2 := uintptr(unsafe.Pointer(&i))
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "uintptr"
	v2s := fmt.Sprintf("%p", &i)
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
}

func addUnsafePointerFormatterTests() {
	// Null pointer.
	v := unsafe.Pointer(uintptr(0))
	nv := (*unsafe.Pointer)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "unsafe.Pointer"
	vs := "<nil>"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Address of real variable.
	i := 1
	v2 := unsafe.Pointer(&i)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "unsafe.Pointer"
	v2s := fmt.Sprintf("%p", &i)
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
}

func addChanFormatterTests() {
	// Nil channel.
	var v chan int
	pv := &v
	nv := (*chan int)(nil)
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "chan int"
	vs := "<nil>"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Real channel.
	v2 := make(chan int)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "chan int"
	v2s := fmt.Sprintf("%p", v2)
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
}

func addFuncFormatterTests() {
	// Function with no params and no returns.
	v := addIntFormatterTests
	nv := (*func())(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "func()"
	vs := fmt.Sprintf("%p", v)
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")

	// Function with param and no returns.
	v2 := TestFormatter
	nv2 := (*func(*testing.T))(nil)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "func(*testing.T)"
	v2s := fmt.Sprintf("%p", v2)
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s)
	addFormatterTest("%v", &pv2, "<**>"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%+v", v2, v2s)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%+v", nv2, "<nil>")
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s)
	addFormatterTest("%#v", nv2, "(*"+v2t+")"+"<nil>")
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s)
	addFormatterTest("%#+v", nv2, "(*"+v2t+")"+"<nil>")

	// Function with multiple params and multiple returns.
	var v3 = func(i int, s string) (b bool, err error) {
		return true, nil
	}
	nv3 := (*func(int, string) (bool, error))(nil)
	pv3 := &v3
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "func(int, string) (bool, error)"
	v3s := fmt.Sprintf("%p", v3)
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s)
	addFormatterTest("%v", &pv3, "<**>"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%+v", v3, v3s)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%+v", nv3, "<nil>")
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s)
	addFormatterTest("%#v", nv3, "(*"+v3t+")"+"<nil>")
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s)
	addFormatterTest("%#+v", nv3, "(*"+v3t+")"+"<nil>")
}

func addCircularFormatterTests() {
	// Struct that is circular through self referencing.
	type circular struct {
		c *circular
	}
	v := circular{nil}
	v.c = &v
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "spew_test.circular"
	vs := "{<*>{<*><shown>}}"
	vs2 := "{<*><shown>}"
	vs3 := "{c:<*>(" + vAddr + "){c:<*>(" + vAddr + ")<shown>}}"
	vs4 := "{c:<*>(" + vAddr + ")<shown>}"
	vs5 := "{c:(*" + vt + "){c:(*" + vt + ")<shown>}}"
	vs6 := "{c:(*" + vt + ")<shown>}"
	vs7 := "{c:(*" + vt + ")(" + vAddr + "){c:(*" + vt + ")(" + vAddr +
		")<shown>}}"
	vs8 := "{c:(*" + vt + ")(" + vAddr + ")<shown>}"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs2)
	addFormatterTest("%v", &pv, "<**>"+vs2)
	addFormatterTest("%+v", v, vs3)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs4)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs4)
	addFormatterTest("%#v", v, "("+vt+")"+vs5)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs6)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs6)
	addFormatterTest("%#+v", v, "("+vt+")"+vs7)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs8)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs8)

	// Structs that are circular through cross referencing.
	v2 := xref1{nil}
	ts2 := xref2{&v2}
	v2.ps2 = &ts2
	pv2 := &v2
	ts2Addr := fmt.Sprintf("%p", &ts2)
	v2Addr := fmt.Sprintf("%p", pv2)
	pv2Addr := fmt.Sprintf("%p", &pv2)
	v2t := "spew_test.xref1"
	v2t2 := "spew_test.xref2"
	v2s := "{<*>{<*>{<*><shown>}}}"
	v2s2 := "{<*>{<*><shown>}}"
	v2s3 := "{ps2:<*>(" + ts2Addr + "){ps1:<*>(" + v2Addr + "){ps2:<*>(" +
		ts2Addr + ")<shown>}}}"
	v2s4 := "{ps2:<*>(" + ts2Addr + "){ps1:<*>(" + v2Addr + ")<shown>}}"
	v2s5 := "{ps2:(*" + v2t2 + "){ps1:(*" + v2t + "){ps2:(*" + v2t2 +
		")<shown>}}}"
	v2s6 := "{ps2:(*" + v2t2 + "){ps1:(*" + v2t + ")<shown>}}"
	v2s7 := "{ps2:(*" + v2t2 + ")(" + ts2Addr + "){ps1:(*" + v2t +
		")(" + v2Addr + "){ps2:(*" + v2t2 + ")(" + ts2Addr +
		")<shown>}}}"
	v2s8 := "{ps2:(*" + v2t2 + ")(" + ts2Addr + "){ps1:(*" + v2t +
		")(" + v2Addr + ")<shown>}}"
	addFormatterTest("%v", v2, v2s)
	addFormatterTest("%v", pv2, "<*>"+v2s2)
	addFormatterTest("%v", &pv2, "<**>"+v2s2)
	addFormatterTest("%+v", v2, v2s3)
	addFormatterTest("%+v", pv2, "<*>("+v2Addr+")"+v2s4)
	addFormatterTest("%+v", &pv2, "<**>("+pv2Addr+"->"+v2Addr+")"+v2s4)
	addFormatterTest("%#v", v2, "("+v2t+")"+v2s5)
	addFormatterTest("%#v", pv2, "(*"+v2t+")"+v2s6)
	addFormatterTest("%#v", &pv2, "(**"+v2t+")"+v2s6)
	addFormatterTest("%#+v", v2, "("+v2t+")"+v2s7)
	addFormatterTest("%#+v", pv2, "(*"+v2t+")("+v2Addr+")"+v2s8)
	addFormatterTest("%#+v", &pv2, "(**"+v2t+")("+pv2Addr+"->"+v2Addr+")"+v2s8)

	// Structs that are indirectly circular.
	v3 := indirCir1{nil}
	tic2 := indirCir2{nil}
	tic3 := indirCir3{&v3}
	tic2.ps3 = &tic3
	v3.ps2 = &tic2
	pv3 := &v3
	tic2Addr := fmt.Sprintf("%p", &tic2)
	tic3Addr := fmt.Sprintf("%p", &tic3)
	v3Addr := fmt.Sprintf("%p", pv3)
	pv3Addr := fmt.Sprintf("%p", &pv3)
	v3t := "spew_test.indirCir1"
	v3t2 := "spew_test.indirCir2"
	v3t3 := "spew_test.indirCir3"
	v3s := "{<*>{<*>{<*>{<*><shown>}}}}"
	v3s2 := "{<*>{<*>{<*><shown>}}}"
	v3s3 := "{ps2:<*>(" + tic2Addr + "){ps3:<*>(" + tic3Addr + "){ps1:<*>(" +
		v3Addr + "){ps2:<*>(" + tic2Addr + ")<shown>}}}}"
	v3s4 := "{ps2:<*>(" + tic2Addr + "){ps3:<*>(" + tic3Addr + "){ps1:<*>(" +
		v3Addr + ")<shown>}}}"
	v3s5 := "{ps2:(*" + v3t2 + "){ps3:(*" + v3t3 + "){ps1:(*" + v3t +
		"){ps2:(*" + v3t2 + ")<shown>}}}}"
	v3s6 := "{ps2:(*" + v3t2 + "){ps3:(*" + v3t3 + "){ps1:(*" + v3t +
		")<shown>}}}"
	v3s7 := "{ps2:(*" + v3t2 + ")(" + tic2Addr + "){ps3:(*" + v3t3 + ")(" +
		tic3Addr + "){ps1:(*" + v3t + ")(" + v3Addr + "){ps2:(*" + v3t2 +
		")(" + tic2Addr + ")<shown>}}}}"
	v3s8 := "{ps2:(*" + v3t2 + ")(" + tic2Addr + "){ps3:(*" + v3t3 + ")(" +
		tic3Addr + "){ps1:(*" + v3t + ")(" + v3Addr + ")<shown>}}}"
	addFormatterTest("%v", v3, v3s)
	addFormatterTest("%v", pv3, "<*>"+v3s2)
	addFormatterTest("%v", &pv3, "<**>"+v3s2)
	addFormatterTest("%+v", v3, v3s3)
	addFormatterTest("%+v", pv3, "<*>("+v3Addr+")"+v3s4)
	addFormatterTest("%+v", &pv3, "<**>("+pv3Addr+"->"+v3Addr+")"+v3s4)
	addFormatterTest("%#v", v3, "("+v3t+")"+v3s5)
	addFormatterTest("%#v", pv3, "(*"+v3t+")"+v3s6)
	addFormatterTest("%#v", &pv3, "(**"+v3t+")"+v3s6)
	addFormatterTest("%#+v", v3, "("+v3t+")"+v3s7)
	addFormatterTest("%#+v", pv3, "(*"+v3t+")("+v3Addr+")"+v3s8)
	addFormatterTest("%#+v", &pv3, "(**"+v3t+")("+pv3Addr+"->"+v3Addr+")"+v3s8)
}

func addPanicFormatterTests() {
	// Type that panics in its Stringer interface.
	v := panicer(127)
	nv := (*panicer)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "spew_test.panicer"
	vs := "(PANIC=test panic)127"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")
}

func addErrorFormatterTests() {
	// Type that has a custom Error interface.
	v := customError(127)
	nv := (*customError)(nil)
	pv := &v
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "spew_test.customError"
	vs := "error: 127"
	addFormatterTest("%v", v, vs)
	addFormatterTest("%v", pv, "<*>"+vs)
	addFormatterTest("%v", &pv, "<**>"+vs)
	addFormatterTest("%v", nv, "<nil>")
	addFormatterTest("%+v", v, vs)
	addFormatterTest("%+v", pv, "<*>("+vAddr+")"+vs)
	addFormatterTest("%+v", &pv, "<**>("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%+v", nv, "<nil>")
	addFormatterTest("%#v", v, "("+vt+")"+vs)
	addFormatterTest("%#v", pv, "(*"+vt+")"+vs)
	addFormatterTest("%#v", &pv, "(**"+vt+")"+vs)
	addFormatterTest("%#v", nv, "(*"+vt+")"+"<nil>")
	addFormatterTest("%#+v", v, "("+vt+")"+vs)
	addFormatterTest("%#+v", pv, "(*"+vt+")("+vAddr+")"+vs)
	addFormatterTest("%#+v", &pv, "(**"+vt+")("+pvAddr+"->"+vAddr+")"+vs)
	addFormatterTest("%#+v", nv, "(*"+vt+")"+"<nil>")
}

func addPassthroughFormatterTests() {
	// %x passthrough with uint.
	v := uint(4294967295)
	pv := &v
	vAddr := fmt.Sprintf("%x", pv)
	pvAddr := fmt.Sprintf("%x", &pv)
	vs := "ffffffff"
	addFormatterTest("%x", v, vs)
	addFormatterTest("%x", pv, vAddr)
	addFormatterTest("%x", &pv, pvAddr)

	// %#x passthrough with uint.
	v2 := int(2147483647)
	pv2 := &v2
	v2Addr := fmt.Sprintf("%#x", pv2)
	pv2Addr := fmt.Sprintf("%#x", &pv2)
	v2s := "0x7fffffff"
	addFormatterTest("%#x", v2, v2s)
	addFormatterTest("%#x", pv2, v2Addr)
	addFormatterTest("%#x", &pv2, pv2Addr)

	// %f passthrough with precision.
	addFormatterTest("%.2f", 3.1415, "3.14")
	addFormatterTest("%.3f", 3.1415, "3.142")
	addFormatterTest("%.4f", 3.1415, "3.1415")

	// %f passthrough with width and precision.
	addFormatterTest("%5.2f", 3.1415, " 3.14")
	addFormatterTest("%6.3f", 3.1415, " 3.142")
	addFormatterTest("%7.4f", 3.1415, " 3.1415")

	// %d passthrough with width.
	addFormatterTest("%3d", 127, "127")
	addFormatterTest("%4d", 127, " 127")
	addFormatterTest("%5d", 127, "  127")

	// %q passthrough with string.
	addFormatterTest("%q", "test", "\"test\"")
}

// TestFormatter executes all of the tests described by formatterTests.
func TestFormatter(t *testing.T) {
	// Setup tests.
	addIntFormatterTests()
	addUintFormatterTests()
	addBoolFormatterTests()
	addFloatFormatterTests()
	addComplexFormatterTests()
	addArrayFormatterTests()
	addSliceFormatterTests()
	addStringFormatterTests()
	addInterfaceFormatterTests()
	addMapFormatterTests()
	addStructFormatterTests()
	addUintptrFormatterTests()
	addUnsafePointerFormatterTests()
	addChanFormatterTests()
	addFuncFormatterTests()
	addCircularFormatterTests()
	addPanicFormatterTests()
	addErrorFormatterTests()
	addPassthroughFormatterTests()

	t.Logf("Running %d tests", len(formatterTests))
	for i, test := range formatterTests {
		buf := new(bytes.Buffer)
		spew.Fprintf(buf, test.format, test.in)
		s := buf.String()
		if testFailed(s, test.wants) {
			t.Errorf("Formatter #%d format: %s got: %s %s", i, test.format, s,
				stringizeWants(test.wants))
			continue
		}
	}
}

func TestPrintSortedKeys(t *testing.T) {
	cfg := spew.ConfigState{SortKeys: true}
	s := cfg.Sprint(map[int]string{1: "1", 3: "3", 2: "2"})
	expected := "map[1:1 2:2 3:3]"
	if s != expected {
		t.Errorf("Sorted keys mismatch:\n  %v %v", s, expected)
	}
}
