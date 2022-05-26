// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

// This file defines the BuilderMode type and its command-line flag.

import (
	"bytes"
	"fmt"
)

// BuilderMode is a bitmask of options for diagnostics and checking.
//
// *BuilderMode satisfies the flag.Value interface.  Example:
//
// 	var mode = ir.BuilderMode(0)
// 	func init() { flag.Var(&mode, "build", ir.BuilderModeDoc) }
//
type BuilderMode uint

const (
	PrintPackages        BuilderMode = 1 << iota // Print package inventory to stdout
	PrintFunctions                               // Print function IR code to stdout
	PrintSource                                  // Print source code when printing function IR
	LogSource                                    // Log source locations as IR builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	NaiveForm                                    // Build naïve IR form: don't replace local loads/stores with registers
	GlobalDebug                                  // Enable debug info for all packages
)

const BuilderModeDoc = `Options controlling the IR builder.
The value is a sequence of zero or more of these letters:
C	perform sanity [C]hecking of the IR form.
D	include [D]ebug info for every function.
P	print [P]ackage inventory.
F	print [F]unction IR code.
A	print [A]ST nodes responsible for IR instructions
S	log [S]ource locations as IR builder progresses.
N	build [N]aive IR form: don't replace local loads/stores with registers.
`

func (m BuilderMode) String() string {
	var buf bytes.Buffer
	if m&GlobalDebug != 0 {
		buf.WriteByte('D')
	}
	if m&PrintPackages != 0 {
		buf.WriteByte('P')
	}
	if m&PrintFunctions != 0 {
		buf.WriteByte('F')
	}
	if m&PrintSource != 0 {
		buf.WriteByte('A')
	}
	if m&LogSource != 0 {
		buf.WriteByte('S')
	}
	if m&SanityCheckFunctions != 0 {
		buf.WriteByte('C')
	}
	if m&NaiveForm != 0 {
		buf.WriteByte('N')
	}
	return buf.String()
}

// Set parses the flag characters in s and updates *m.
func (m *BuilderMode) Set(s string) error {
	var mode BuilderMode
	for _, c := range s {
		switch c {
		case 'D':
			mode |= GlobalDebug
		case 'P':
			mode |= PrintPackages
		case 'F':
			mode |= PrintFunctions
		case 'A':
			mode |= PrintSource
		case 'S':
			mode |= LogSource
		case 'C':
			mode |= SanityCheckFunctions
		case 'N':
			mode |= NaiveForm
		default:
			return fmt.Errorf("unknown BuilderMode option: %q", c)
		}
	}
	*m = mode
	return nil
}

// Get returns m.
func (m BuilderMode) Get() interface{} { return m }
