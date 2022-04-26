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

// Package sanitizer contains the logic responsible for determining whether sources are sanitized
// before they are being sent to sinks.
package sanitizer

import (
	"math"

	"golang.org/x/tools/go/ssa"
)

// Sanitizer removes the taint.
type Sanitizer struct {
	// Call is the underlying call that performs sanitization
	Call *ssa.Call
}

// Dominates returns true if the Sanitizer dominates the supplied instruction.
// In the context of SSA, domination implies that
// if instructions X executes and X dominates Y, then Y is guaranteed to execute and to be
// executed after X.
func (s Sanitizer) Dominates(target ssa.Instruction) bool {
	if s.Call.Parent() != target.Parent() {
		// Instructions are in different functions.
		return false
	}

	if s.Call.Block() == target.Block() {
		sanitizationIdx := math.MaxInt64
		targetIdx := 0
		for i, instr := range s.Call.Block().Instrs {
			if instr == s.Call {
				sanitizationIdx = i
			}
			if instr == target {
				targetIdx = i
				break
			}
		}
		return sanitizationIdx < targetIdx
	}

	return s.Call.Block().Dominates(target.Block())
}
