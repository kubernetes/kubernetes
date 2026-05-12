// Copyright 2018 Google LLC
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

// Package functions defines the standard builtin functions supported by the
// interpreter and as declared within the checker#StandardDeclarations.
package functions

import fn "github.com/google/cel-go/common/functions"

// Overload defines a named overload of a function, indicating an operand trait
// which must be present on the first argument to the overload as well as one
// of either a unary, binary, or function implementation.
//
// The majority of  operators within the expression language are unary or binary
// and the specializations simplify the call contract for implementers of
// types with operator overloads. Any added complexity is assumed to be handled
// by the generic FunctionOp.
type Overload = fn.Overload

// UnaryOp is a function that takes a single value and produces an output.
type UnaryOp = fn.UnaryOp

// BinaryOp is a function that takes two values and produces an output.
type BinaryOp = fn.BinaryOp

// FunctionOp is a function with accepts zero or more arguments and produces
// a value or error as a result.
type FunctionOp = fn.FunctionOp
