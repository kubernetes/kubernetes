// Copyright 2023 Google LLC
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

// Package functions defines the standard builtin functions supported by the interpreter
package functions

import (
	"context"

	"cel.dev/cel-go/common/types/ref"
)

// Overload defines a named overload of a function, indicating an operand trait
// which must be present on the first argument to the overload as well as one
// of either a unary, binary, or function implementation.
//
// The majority of  operators within the expression language are unary or binary
// and the specializations simplify the call contract for implementers of
// types with operator overloads. Any added complexity is assumed to be handled
// by the generic FunctionOp.
type Overload struct {
	// Operator name as written in an expression or defined within
	// operators.go.
	Operator string

	// Operand trait used to dispatch the call. The zero-value indicates a
	// global function overload or that one of the Unary / Binary / Function
	// definitions should be used to execute the call.
	OperandTrait int

	// Unary defines the overload with a UnaryOp implementation. May be nil.
	Unary UnaryOp

	// Binary defines the overload with a BinaryOp implementation. May be nil.
	Binary BinaryOp

	// Function defines the overload with a FunctionOp implementation. May be nil.
	Function FunctionOp

	// Async defines the overload with an AsyncOp implementation. May be nil.
	Async AsyncOp

	// LateBound specifies whether the Overload has a late-binding that will be resolved
	// at runtime from the activation.
	LateBound bool

	// LateBoundDispatch validates the runtime arguments of a late-bound call against the
	// declared overload signatures and reports the id of the matching overload. May be nil,
	// in which case the arguments are not validated against the declaration.
	LateBoundDispatch LateBoundDispatcher

	// NonStrict specifies whether the Overload will tolerate arguments that
	// are types.Err or types.Unknown.
	NonStrict bool
}

// LateBoundDispatcher reports the id of the declared overload whose signature matches the
// receiver style and runtime arguments of a call, or false if no declared overload matches.
//
// Late-bound implementations are supplied by the activation rather than the declaration, so this
// is the only opportunity to enforce agreement between a call and the signature it was checked
// against.
type LateBoundDispatcher func(memberStyle bool, args ...ref.Val) (string, bool)

// UnaryOp is a function that takes a single value and produces an output.
type UnaryOp func(ref.Val) ref.Val

// BinaryOp is a function that takes two values and produces an output.
type BinaryOp func(ref.Val, ref.Val) ref.Val

// FunctionOp is a function with accepts zero or more arguments and produces
// a value or error as a result.
type FunctionOp func(...ref.Val) ref.Val

// LateBoundOp is the implementation of a function whose binding is supplied at evaluation time
// rather than at declaration time.
//
// Late-bound implementations are resolved from the activation by function name, in a namespace
// which is separate from variables, so a function with multiple overloads is served by a single
// implementation. The id of the overload which matched the call is provided so that such an
// implementation may distinguish between them.
//
// The activation is deliberately not an input to the call. Bindings are supplied for a single
// evaluation alongside the variables of that evaluation, so an implementation which depends on
// evaluation context should capture it in a closure when the activation is constructed.
type LateBoundOp func(overloadID string, args ...ref.Val) ref.Val

// AsyncOp is a function that accepts zero or more arguments and produces
// a value or error asynchronously via a channel.
//
// AsyncOp is an internal interface intended for use by CEL to manage goroutines and
// channels associated with async calls. For public API usage, use BlockingAsyncOp.
// Implementers should listen for context cancellation on the provided context for
// resource cleanup.
type AsyncOp func(context.Context, ...ref.Val) <-chan ref.Val

// BlockingAsyncOp is a function that accepts zero or more arguments and blocks until
// the result is available. When used with AsyncBinding, the framework runs the function
// in its own goroutine and manages channel lifecycle internally.
type BlockingAsyncOp func(context.Context, ...ref.Val) ref.Val
