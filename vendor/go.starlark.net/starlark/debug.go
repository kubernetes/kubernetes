package starlark

import "go.starlark.net/syntax"

// This file defines an experimental API for the debugging tools.
// Some of these declarations expose details of internal packages.
// (The debugger makes liberal use of exported fields of unexported types.)
// Breaking changes may occur without notice.

// Local returns the value of the i'th local variable.
// It may be nil if not yet assigned.
//
// Local may be called only for frames whose Callable is a *Function (a
// function defined by Starlark source code), and only while the frame
// is active; it will panic otherwise.
//
// This function is provided only for debugging tools.
//
// THIS API IS EXPERIMENTAL AND MAY CHANGE WITHOUT NOTICE.
func (fr *frame) Local(i int) Value { return fr.locals[i] }

// DebugFrame is the debugger API for a frame of the interpreter's call stack.
//
// Most applications have no need for this API; use CallFrame instead.
//
// Clients must not retain a DebugFrame nor call any of its methods once
// the current built-in call has returned or execution has resumed
// after a breakpoint as this may have unpredictable effects, including
// but not limited to retention of object that would otherwise be garbage.
type DebugFrame interface {
	Callable() Callable        // returns the frame's function
	Local(i int) Value         // returns the value of the (Starlark) frame's ith local variable
	Position() syntax.Position // returns the current position of execution in this frame
}

// DebugFrame returns the debugger interface for
// the specified frame of the interpreter's call stack.
// Frame numbering is as for Thread.CallFrame.
//
// This function is intended for use in debugging tools.
// Most applications should have no need for it; use CallFrame instead.
func (thread *Thread) DebugFrame(depth int) DebugFrame { return thread.frameAt(depth) }
