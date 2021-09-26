/*
 * Copyright (c) 2013-2016 Dave Collins <dave@davec.name>
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

package spew_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

// custom type to test Stinger interface on non-pointer receiver.
type stringer string

// String implements the Stringer interface for testing invocation of custom
// stringers on types with non-pointer receivers.
func (s stringer) String() string {
	return "stringer " + string(s)
}

// custom type to test Stinger interface on pointer receiver.
type pstringer string

// String implements the Stringer interface for testing invocation of custom
// stringers on types with only pointer receivers.
func (s *pstringer) String() string {
	return "stringer " + string(*s)
}

// xref1 and xref2 are cross referencing structs for testing circular reference
// detection.
type xref1 struct {
	ps2 *xref2
}
type xref2 struct {
	ps1 *xref1
}

// indirCir1, indirCir2, and indirCir3 are used to generate an indirect circular
// reference for testing detection.
type indirCir1 struct {
	ps2 *indirCir2
}
type indirCir2 struct {
	ps3 *indirCir3
}
type indirCir3 struct {
	ps1 *indirCir1
}

// embed is used to test embedded structures.
type embed struct {
	a string
}

// embedwrap is used to test embedded structures.
type embedwrap struct {
	*embed
	e *embed
}

// panicer is used to intentionally cause a panic for testing spew properly
// handles them
type panicer int

func (p panicer) String() string {
	panic("test panic")
}

// customError is used to test custom error interface invocation.
type customError int

func (e customError) Error() string {
	return fmt.Sprintf("error: %d", int(e))
}

// stringizeWants converts a slice of wanted test output into a format suitable
// for a test error message.
func stringizeWants(wants []string) string {
	s := ""
	for i, want := range wants {
		if i > 0 {
			s += fmt.Sprintf("want%d: %s", i+1, want)
		} else {
			s += "want: " + want
		}
	}
	return s
}

// testFailed returns whether or not a test failed by checking if the result
// of the test is in the slice of wanted strings.
func testFailed(result string, wants []string) bool {
	for _, want := range wants {
		if result == want {
			return false
		}
	}
	return true
}

type sortableStruct struct {
	x int
}

func (ss sortableStruct) String() string {
	return fmt.Sprintf("ss.%d", ss.x)
}

type unsortableStruct struct {
	x int
}

type sortTestCase struct {
	input    []reflect.Value
	expected []reflect.Value
}

func helpTestSortValues(tests []sortTestCase, cs *spew.ConfigState, t *testing.T) {
	getInterfaces := func(values []reflect.Value) []interface{} {
		interfaces := []interface{}{}
		for _, v := range values {
			interfaces = append(interfaces, v.Interface())
		}
		return interfaces
	}

	for _, test := range tests {
		spew.SortValues(test.input, cs)
		// reflect.DeepEqual cannot really make sense of reflect.Value,
		// probably because of all the pointer tricks. For instance,
		// v(2.0) != v(2.0) on a 32-bits system. Turn them into interface{}
		// instead.
		input := getInterfaces(test.input)
		expected := getInterfaces(test.expected)
		if !reflect.DeepEqual(input, expected) {
			t.Errorf("Sort mismatch:\n %v != %v", input, expected)
		}
	}
}

// TestSortValues ensures the sort functionality for relect.Value based sorting
// works as intended.
func TestSortValues(t *testing.T) {
	v := reflect.ValueOf

	a := v("a")
	b := v("b")
	c := v("c")
	embedA := v(embed{"a"})
	embedB := v(embed{"b"})
	embedC := v(embed{"c"})
	tests := []sortTestCase{
		// No values.
		{
			[]reflect.Value{},
			[]reflect.Value{},
		},
		// Bools.
		{
			[]reflect.Value{v(false), v(true), v(false)},
			[]reflect.Value{v(false), v(false), v(true)},
		},
		// Ints.
		{
			[]reflect.Value{v(2), v(1), v(3)},
			[]reflect.Value{v(1), v(2), v(3)},
		},
		// Uints.
		{
			[]reflect.Value{v(uint8(2)), v(uint8(1)), v(uint8(3))},
			[]reflect.Value{v(uint8(1)), v(uint8(2)), v(uint8(3))},
		},
		// Floats.
		{
			[]reflect.Value{v(2.0), v(1.0), v(3.0)},
			[]reflect.Value{v(1.0), v(2.0), v(3.0)},
		},
		// Strings.
		{
			[]reflect.Value{b, a, c},
			[]reflect.Value{a, b, c},
		},
		// Array
		{
			[]reflect.Value{v([3]int{3, 2, 1}), v([3]int{1, 3, 2}), v([3]int{1, 2, 3})},
			[]reflect.Value{v([3]int{1, 2, 3}), v([3]int{1, 3, 2}), v([3]int{3, 2, 1})},
		},
		// Uintptrs.
		{
			[]reflect.Value{v(uintptr(2)), v(uintptr(1)), v(uintptr(3))},
			[]reflect.Value{v(uintptr(1)), v(uintptr(2)), v(uintptr(3))},
		},
		// SortableStructs.
		{
			// Note: not sorted - DisableMethods is set.
			[]reflect.Value{v(sortableStruct{2}), v(sortableStruct{1}), v(sortableStruct{3})},
			[]reflect.Value{v(sortableStruct{2}), v(sortableStruct{1}), v(sortableStruct{3})},
		},
		// UnsortableStructs.
		{
			// Note: not sorted - SpewKeys is false.
			[]reflect.Value{v(unsortableStruct{2}), v(unsortableStruct{1}), v(unsortableStruct{3})},
			[]reflect.Value{v(unsortableStruct{2}), v(unsortableStruct{1}), v(unsortableStruct{3})},
		},
		// Invalid.
		{
			[]reflect.Value{embedB, embedA, embedC},
			[]reflect.Value{embedB, embedA, embedC},
		},
	}
	cs := spew.ConfigState{DisableMethods: true, SpewKeys: false}
	helpTestSortValues(tests, &cs, t)
}

// TestSortValuesWithMethods ensures the sort functionality for relect.Value
// based sorting works as intended when using string methods.
func TestSortValuesWithMethods(t *testing.T) {
	v := reflect.ValueOf

	a := v("a")
	b := v("b")
	c := v("c")
	tests := []sortTestCase{
		// Ints.
		{
			[]reflect.Value{v(2), v(1), v(3)},
			[]reflect.Value{v(1), v(2), v(3)},
		},
		// Strings.
		{
			[]reflect.Value{b, a, c},
			[]reflect.Value{a, b, c},
		},
		// SortableStructs.
		{
			[]reflect.Value{v(sortableStruct{2}), v(sortableStruct{1}), v(sortableStruct{3})},
			[]reflect.Value{v(sortableStruct{1}), v(sortableStruct{2}), v(sortableStruct{3})},
		},
		// UnsortableStructs.
		{
			// Note: not sorted - SpewKeys is false.
			[]reflect.Value{v(unsortableStruct{2}), v(unsortableStruct{1}), v(unsortableStruct{3})},
			[]reflect.Value{v(unsortableStruct{2}), v(unsortableStruct{1}), v(unsortableStruct{3})},
		},
	}
	cs := spew.ConfigState{DisableMethods: false, SpewKeys: false}
	helpTestSortValues(tests, &cs, t)
}

// TestSortValuesWithSpew ensures the sort functionality for relect.Value
// based sorting works as intended when using spew to stringify keys.
func TestSortValuesWithSpew(t *testing.T) {
	v := reflect.ValueOf

	a := v("a")
	b := v("b")
	c := v("c")
	tests := []sortTestCase{
		// Ints.
		{
			[]reflect.Value{v(2), v(1), v(3)},
			[]reflect.Value{v(1), v(2), v(3)},
		},
		// Strings.
		{
			[]reflect.Value{b, a, c},
			[]reflect.Value{a, b, c},
		},
		// SortableStructs.
		{
			[]reflect.Value{v(sortableStruct{2}), v(sortableStruct{1}), v(sortableStruct{3})},
			[]reflect.Value{v(sortableStruct{1}), v(sortableStruct{2}), v(sortableStruct{3})},
		},
		// UnsortableStructs.
		{
			[]reflect.Value{v(unsortableStruct{2}), v(unsortableStruct{1}), v(unsortableStruct{3})},
			[]reflect.Value{v(unsortableStruct{1}), v(unsortableStruct{2}), v(unsortableStruct{3})},
		},
	}
	cs := spew.ConfigState{DisableMethods: true, SpewKeys: true}
	helpTestSortValues(tests, &cs, t)
}
