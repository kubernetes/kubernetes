// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package summary provides function summaries for a range of standard
// library functions that could be involved in a taint propagation.
// Function summaries describe the taint-propagation behavior of a given
// function, e.g. "if these arguments are tainted, then the following
// arguments/return values should also be tainted".
package summary

import (
	"go/types"
	"strings"

	"github.com/google/go-flow-levee/internal/pkg/utils"
	"golang.org/x/tools/go/ssa"
)

// For returns the summary for a given call if it exists,
// or nil if no summary matches the called function.
func For(call ssa.CallInstruction) *Summary {
	if summ, ok := FuncSummaries[staticFuncName(call)]; ok {
		return &summ
	}
	if summ, ok := InterfaceFuncSummaries[funcKey{methodNameWithoutReceiver(call), sigTypeString(call.Common().Signature())}]; ok {
		return &summ
	}
	return nil
}

// A Summary captures the behavior of a function with respect to taint
// propagation. Specifically: given that at least one of the necessary
// arguments is tainted, which arguments/return values become tainted?
// Note that when it's present, the receiver counts as an argument.
//
// As an example, consider fmt.Fprintf:
//   func Fprintf(w io.Writer, format string, a ...interface{}) (n int, err error) {
// Its Summary is:
//   "fmt.Fprintf": {
//   	ifTainted:   0b110,
//   	taintedArgs: []int{0},
//   },
// In English, this says that if the format string or the varargs slice are
// tainted, then the Writer is tainted.
// (In an actual summary, 0b110 should be written as second | third for readability.)
type Summary struct {
	// IfTainted is a bitset which contains positions for parameters
	// such that if one of these parameters is tainted, taint should
	// be propagated to the arguments and return values.
	// There is a 1-to-1 mapping between the bits and the function's
	// parameters, with the least significant bit corresponding to the
	// first (0th) argument.
	IfTainted int64
	// the positions of the arguments that taint propagates to if one of the
	// positions in ifTainted is tainted
	TaintedArgs []int
	// the positions of the return values that taint propagates to if one of the
	// positions in ifTainted is tainted
	TaintedRets []int
}

func staticFuncName(call ssa.CallInstruction) string {
	if sc := call.Common().StaticCallee(); sc != nil {
		return sc.RelString(call.Parent().Pkg.Pkg)
	}
	return ""
}

func methodNameWithoutReceiver(call ssa.CallInstruction) string {
	cc := call.Common()
	if cc.IsInvoke() {
		return cc.Method.Name()
	}
	if sc := cc.StaticCallee(); sc != nil {
		if sc.Signature.Recv() == nil {
			return ""
		}
		return sc.Name()
	}
	return ""
}

// sigTypeString produces a stripped version of a function's signature, containing
// just the types of the arguments and return values.
// The receiver's type is not included.
// For a function such as:
//   WriteTo(w Writer) (n int64, err error)
// The result is:
//   (Writer)(int64,error)
func sigTypeString(sig *types.Signature) string {
	var b strings.Builder

	b.WriteByte('(')
	paramsPtr := sig.Params()
	if paramsPtr != nil {
		params := *paramsPtr
		for i := 0; i < params.Len(); i++ {
			p := params.At(i)
			b.WriteString(utils.UnqualifiedName(p))
			if i+1 != params.Len() {
				b.WriteByte(',')
			}
		}
	}
	b.WriteByte(')')

	b.WriteByte('(')
	resultsPtr := sig.Results()
	if resultsPtr != nil {
		results := *resultsPtr
		for i := 0; i < results.Len(); i++ {
			p := results.At(i)
			b.WriteString(utils.UnqualifiedName(p))
			if i+1 != results.Len() {
				b.WriteByte(',')
			}
		}
	}
	b.WriteByte(')')

	return b.String()
}
