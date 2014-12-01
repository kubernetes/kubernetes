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

package spew_test

import (
	"fmt"
	"github.com/davecgh/go-spew/spew"
)

type Flag int

const (
	flagOne Flag = iota
	flagTwo
)

var flagStrings = map[Flag]string{
	flagOne: "flagOne",
	flagTwo: "flagTwo",
}

func (f Flag) String() string {
	if s, ok := flagStrings[f]; ok {
		return s
	}
	return fmt.Sprintf("Unknown flag (%d)", int(f))
}

type Bar struct {
	flag Flag
	data uintptr
}

type Foo struct {
	unexportedField Bar
	ExportedField   map[interface{}]interface{}
}

// This example demonstrates how to use Dump to dump variables to stdout.
func ExampleDump() {
	// The following package level declarations are assumed for this example:
	/*
		type Flag int

		const (
			flagOne Flag = iota
			flagTwo
		)

		var flagStrings = map[Flag]string{
			flagOne: "flagOne",
			flagTwo: "flagTwo",
		}

		func (f Flag) String() string {
			if s, ok := flagStrings[f]; ok {
				return s
			}
			return fmt.Sprintf("Unknown flag (%d)", int(f))
		}

		type Bar struct {
			flag Flag
			data uintptr
		}

		type Foo struct {
			unexportedField Bar
			ExportedField   map[interface{}]interface{}
		}
	*/

	// Setup some sample data structures for the example.
	bar := Bar{Flag(flagTwo), uintptr(0)}
	s1 := Foo{bar, map[interface{}]interface{}{"one": true}}
	f := Flag(5)
	b := []byte{
		0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
		0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
		0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
		0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30,
		0x31, 0x32,
	}

	// Dump!
	spew.Dump(s1, f, b)

	// Output:
	// (spew_test.Foo) {
	//  unexportedField: (spew_test.Bar) {
	//   flag: (spew_test.Flag) flagTwo,
	//   data: (uintptr) <nil>
	//  },
	//  ExportedField: (map[interface {}]interface {}) (len=1) {
	//   (string) (len=3) "one": (bool) true
	//  }
	// }
	// (spew_test.Flag) Unknown flag (5)
	// ([]uint8) (len=34 cap=34) {
	//  00000000  11 12 13 14 15 16 17 18  19 1a 1b 1c 1d 1e 1f 20  |............... |
	//  00000010  21 22 23 24 25 26 27 28  29 2a 2b 2c 2d 2e 2f 30  |!"#$%&'()*+,-./0|
	//  00000020  31 32                                             |12|
	// }
	//
}

// This example demonstrates how to use Printf to display a variable with a
// format string and inline formatting.
func ExamplePrintf() {
	// Create a double pointer to a uint 8.
	ui8 := uint8(5)
	pui8 := &ui8
	ppui8 := &pui8

	// Create a circular data type.
	type circular struct {
		ui8 uint8
		c   *circular
	}
	c := circular{ui8: 1}
	c.c = &c

	// Print!
	spew.Printf("ppui8: %v\n", ppui8)
	spew.Printf("circular: %v\n", c)

	// Output:
	// ppui8: <**>5
	// circular: {1 <*>{1 <*><shown>}}
}

// This example demonstrates how to use a ConfigState.
func ExampleConfigState() {
	// Modify the indent level of the ConfigState only.  The global
	// configuration is not modified.
	scs := spew.ConfigState{Indent: "\t"}

	// Output using the ConfigState instance.
	v := map[string]int{"one": 1}
	scs.Printf("v: %v\n", v)
	scs.Dump(v)

	// Output:
	// v: map[one:1]
	// (map[string]int) (len=1) {
	// 	(string) (len=3) "one": (int) 1
	// }
}

// This example demonstrates how to use ConfigState.Dump to dump variables to
// stdout
func ExampleConfigState_Dump() {
	// See the top-level Dump example for details on the types used in this
	// example.

	// Create two ConfigState instances with different indentation.
	scs := spew.ConfigState{Indent: "\t"}
	scs2 := spew.ConfigState{Indent: " "}

	// Setup some sample data structures for the example.
	bar := Bar{Flag(flagTwo), uintptr(0)}
	s1 := Foo{bar, map[interface{}]interface{}{"one": true}}

	// Dump using the ConfigState instances.
	scs.Dump(s1)
	scs2.Dump(s1)

	// Output:
	// (spew_test.Foo) {
	// 	unexportedField: (spew_test.Bar) {
	// 		flag: (spew_test.Flag) flagTwo,
	// 		data: (uintptr) <nil>
	// 	},
	// 	ExportedField: (map[interface {}]interface {}) (len=1) {
	//		(string) (len=3) "one": (bool) true
	// 	}
	// }
	// (spew_test.Foo) {
	//  unexportedField: (spew_test.Bar) {
	//   flag: (spew_test.Flag) flagTwo,
	//   data: (uintptr) <nil>
	//  },
	//  ExportedField: (map[interface {}]interface {}) (len=1) {
	//   (string) (len=3) "one": (bool) true
	//  }
	// }
	//
}

// This example demonstrates how to use ConfigState.Printf to display a variable
// with a format string and inline formatting.
func ExampleConfigState_Printf() {
	// See the top-level Dump example for details on the types used in this
	// example.

	// Create two ConfigState instances and modify the method handling of the
	// first ConfigState only.
	scs := spew.NewDefaultConfig()
	scs2 := spew.NewDefaultConfig()
	scs.DisableMethods = true

	// Alternatively
	// scs := spew.ConfigState{Indent: " ", DisableMethods: true}
	// scs2 := spew.ConfigState{Indent: " "}

	// This is of type Flag which implements a Stringer and has raw value 1.
	f := flagTwo

	// Dump using the ConfigState instances.
	scs.Printf("f: %v\n", f)
	scs2.Printf("f: %v\n", f)

	// Output:
	// f: 1
	// f: flagTwo
}
