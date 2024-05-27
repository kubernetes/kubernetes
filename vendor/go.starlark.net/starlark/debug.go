package starlark

import (
	"go.starlark.net/syntax"
)

// This file defines an experimental API for the debugging tools.
// Some of these declarations expose details of internal packages.
// (The debugger makes liberal use of exported fields of unexported types.)
// Breaking changes may occur without notice.

// A Binding is the name and position of a binding identifier.
type Binding struct {
	Name string
	Pos  syntax.Position
}

// NumLocals returns the number of local variables of this frame.
// It is zero unless fr.Callable() is a *Function.
func (fr *frame) NumLocals() int { return len(fr.locals) }

// Local returns the binding (name and binding position) and value of
// the i'th local variable of the frame's function.
// Beware: the value may be nil if it has not yet been assigned!
//
// The index i must be less than [NumLocals].
// Local may be called only while the frame is active.
//
// This function is provided only for debugging tools.
func (fr *frame) Local(i int) (Binding, Value) {
	return Binding(fr.callable.(*Function).funcode.Locals[i]), fr.locals[i]
}

// DebugFrame is the debugger API for a frame of the interpreter's call stack.
//
// Most applications have no need for this API; use CallFrame instead.
//
// It may be tempting to use this interface when implementing built-in
// functions. Beware that reflection over the call stack is easily
// abused, leading to built-in functions whose behavior is mysterious
// and unpredictable.
//
// Clients must not retain a DebugFrame nor call any of its methods once
// the current built-in call has returned or execution has resumed
// after a breakpoint as this may have unpredictable effects, including
// but not limited to retention of object that would otherwise be garbage.
type DebugFrame interface {
	Callable() Callable           // returns the frame's function
	NumLocals() int               // returns the number of local variables in this frame
	Local(i int) (Binding, Value) // returns the binding and value of the (Starlark) frame's ith local variable
	Position() syntax.Position    // returns the current position of execution in this frame
}

// DebugFrame returns the debugger interface for
// the specified frame of the interpreter's call stack.
// Frame numbering is as for Thread.CallFrame: 0 <= depth < thread.CallStackDepth().
//
// This function is intended for use in debugging tools.
// Most applications should have no need for it; use CallFrame instead.
func (thread *Thread) DebugFrame(depth int) DebugFrame { return thread.frameAt(depth) }
