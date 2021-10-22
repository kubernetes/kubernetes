// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package debug provides the portable interface to a program being debugged.
package debug

import (
	"fmt"
	"io"
	"strings"
)

// Program is the interface to a (possibly remote) program being debugged.
// The process (if any) and text file associated with it may change during
// the session, but many resources are associated with the Program rather
// than process or text file so they persist across debuggging runs.
type Program interface {
	// Open opens a virtual file associated with the process.
	// Names are things like "text", "mem", "fd/2".
	// Mode is one of "r", "w", "rw".
	// Return values are open File and error.
	// When the target binary is re-run, open files are
	// automatically updated to refer to the corresponding
	// file in the new process.
	Open(name string, mode string) (File, error)

	// Run abandons the current running process, if any,
	// and execs a new instance of the target binary file
	// (which may have changed underfoot).
	// Breakpoints and open files are re-established.
	// The call hangs until the program stops executing,
	// at which point it returns the program status.
	// args contains the command-line arguments for the process.
	Run(args ...string) (Status, error)

	// Stop stops execution of the current process but
	// does not kill it.
	Stop() (Status, error)

	// Resume resumes execution of a stopped process.
	// The call hangs until the program stops executing,
	// at which point it returns the program status.
	Resume() (Status, error)

	// TODO: Step(). Where does the granularity happen,
	// on the proxy end or the debugging control end?

	// Kill kills the current process.
	Kill() (Status, error)

	// Breakpoint sets a breakpoint at the specified address.
	Breakpoint(address uint64) (PCs []uint64, err error)

	// BreakpointAtFunction sets a breakpoint at the start of the specified function.
	BreakpointAtFunction(name string) (PCs []uint64, err error)

	// BreakpointAtLine sets a breakpoint at the specified source line.
	BreakpointAtLine(file string, line uint64) (PCs []uint64, err error)

	// DeleteBreakpoints removes the breakpoints at the specified addresses.
	// Addresses where no breakpoint is set are ignored.
	DeleteBreakpoints(pcs []uint64) error

	// Eval evaluates the expression (typically an address) and returns
	// its string representation(s). Multivalued expressions such as
	// matches for regular expressions return multiple values.
	// TODO: change this to multiple functions with more specific names.
	// Syntax:
	//	re:regexp
	//		Returns a list of symbol names that match the expression
	//	addr:symbol
	//		Returns a one-element list holding the hexadecimal
	//		("0x1234") value of the address of the symbol
	//	val:symbol
	//		Returns a one-element list holding the formatted
	//		value of the symbol
	//	0x1234, 01234, 467
	//		Returns a one-element list holding the name of the
	//		symbol ("main.foo") at that address (hex, octal, decimal).
	Eval(expr string) ([]string, error)

	// Evaluate evaluates an expression.  Accepts a subset of Go expression syntax:
	// basic literals, identifiers, parenthesized expressions, and most operators.
	// Only the len function call is available.
	//
	// The expression can refer to local variables and function parameters of the
	// function where the program is stopped.
	//
	// On success, the type of the value returned will be one of:
	// int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64,
	// complex64, complex128, bool, Pointer, Array, Slice, String, Map, Struct,
	// Channel, Func, or Interface.
	Evaluate(e string) (Value, error)

	// Frames returns up to count stack frames from where the program
	// is currently stopped.
	Frames(count int) ([]Frame, error)

	// VarByName returns a Var referring to a global variable with the given name.
	// TODO: local variables
	VarByName(name string) (Var, error)

	// Value gets the value of a variable by reading the program's memory.
	Value(v Var) (Value, error)

	// MapElement returns Vars for the key and value of a map element specified by
	// a 0-based index.
	MapElement(m Map, index uint64) (Var, Var, error)

	// Goroutines gets the current goroutines.
	Goroutines() ([]*Goroutine, error)
}

type Goroutine struct {
	ID           int64
	Status       GoroutineStatus
	StatusString string // A human-readable string explaining the status in more detail.
	Function     string // Name of the goroutine function.
	Caller       string // Name of the function that created this goroutine.
	StackFrames  []Frame
}

type GoroutineStatus byte

const (
	Running GoroutineStatus = iota
	Queued
	Blocked
)

func (g GoroutineStatus) String() string {
	switch g {
	case Running:
		return "running"
	case Queued:
		return "queued"
	case Blocked:
		return "blocked"
	}
	return "invalid status"
}

func (g *Goroutine) String() string {
	return fmt.Sprintf("goroutine %d [%s] %s -> %s", g.ID, g.StatusString, g.Caller, g.Function)
}

// A reference to a variable in a program.
// TODO: handle variables stored in registers
type Var struct {
	TypeID  uint64 // A type identifier, opaque to the user.
	Address uint64 // The address of the variable.
}

// A value read from a remote program.
type Value interface{}

// Pointer is a Value representing a pointer.
// Note that the TypeID field will be the type of the variable being pointed to,
// not the type of this pointer.
type Pointer struct {
	TypeID  uint64 // A type identifier, opaque to the user.
	Address uint64 // The address of the variable.
}

// Array is a Value representing an array.
type Array struct {
	ElementTypeID uint64
	Address       uint64
	Length        uint64 // Number of elements in the array
	StrideBits    uint64 // Number of bits between array entries
}

// Len returns the number of elements in the array.
func (a Array) Len() uint64 {
	return a.Length
}

// Element returns a Var referring to the given element of the array.
func (a Array) Element(index uint64) Var {
	return Var{
		TypeID:  a.ElementTypeID,
		Address: a.Address + index*(a.StrideBits/8),
	}
}

// Slice is a Value representing a slice.
type Slice struct {
	Array
	Capacity uint64
}

// String is a Value representing a string.
// TODO: a method to access more of a truncated string.
type String struct {
	// Length contains the length of the remote string, in bytes.
	Length uint64
	// String contains the string itself; it may be truncated to fewer bytes than the value of the Length field.
	String string
}

// Map is a Value representing a map.
type Map struct {
	TypeID  uint64
	Address uint64
	Length  uint64 // Number of elements in the map.
}

// Struct is a Value representing a struct.
type Struct struct {
	Fields []StructField
}

// StructField represents a field in a struct object.
type StructField struct {
	Name string
	Var  Var
}

// Channel is a Value representing a channel.
type Channel struct {
	ElementTypeID uint64
	Address       uint64 // Location of the channel struct in memory.
	Buffer        uint64 // Location of the buffer; zero for nil channels.
	Length        uint64 // Number of elements stored in the channel buffer.
	Capacity      uint64 // Capacity of the buffer; zero for unbuffered channels.
	Stride        uint64 // Number of bytes between buffer entries.
	BufferStart   uint64 // Index in the buffer of the element at the head of the queue.
}

// Element returns a Var referring to the given element of the channel's queue.
// If the channel is unbuffered, nil, or if the index is too large, returns a Var with Address == 0.
func (m Channel) Element(index uint64) Var {
	if index >= m.Length {
		return Var{
			TypeID:  m.ElementTypeID,
			Address: 0,
		}
	}
	if index < m.Capacity-m.BufferStart {
		// The element is in the part of the queue that occurs later in the buffer
		// than the head of the queue.
		return Var{
			TypeID:  m.ElementTypeID,
			Address: m.Buffer + (m.BufferStart+index)*m.Stride,
		}
	}
	// The element is in the part of the queue that has wrapped around to the
	// start of the buffer.
	return Var{
		TypeID:  m.ElementTypeID,
		Address: m.Buffer + (m.BufferStart+index-m.Capacity)*m.Stride,
	}
}

// Func is a Value representing a func.
type Func struct {
	Address uint64
}

// Interface is a Value representing an interface.
type Interface struct{}

// The File interface provides access to file-like resources in the program.
// It implements only ReaderAt and WriterAt, not Reader and Writer, because
// random access is a far more common pattern for things like symbol tables,
// and because enormous address space of virtual memory makes routines
// like io.Copy dangerous.
type File interface {
	io.ReaderAt
	io.WriterAt
	io.Closer
}

type Status struct {
	PC, SP uint64
}

type Frame struct {
	// PC is the hardware program counter.
	PC uint64
	// SP is the hardware stack pointer.
	SP uint64
	// File and Line are the source code location of the PC.
	File string
	Line uint64
	// Function is the name of this frame's function.
	Function string
	// FunctionStart is the starting PC of the function.
	FunctionStart uint64
	// Params contains the function's parameters.
	Params []Param
	// Vars contains the function's local variables.
	Vars []LocalVar
}

func (f Frame) String() string {
	params := make([]string, len(f.Params))
	for i, p := range f.Params {
		params[i] = p.Name // TODO: more information
	}
	p := strings.Join(params, ", ")
	off := f.PC - f.FunctionStart
	return fmt.Sprintf("%s(%s)\n\t%s:%d +0x%x", f.Function, p, f.File, f.Line, off)
}

// Param is a parameter of a function.
type Param struct {
	Name string
	Var  Var
}

// LocalVar is a local variable of a function.
type LocalVar struct {
	Name string
	Var  Var
}
