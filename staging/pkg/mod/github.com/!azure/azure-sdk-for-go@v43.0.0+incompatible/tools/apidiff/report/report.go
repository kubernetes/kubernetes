// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package report

import (
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/delta"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
)

// Package represents a per-package report that contains additive and breaking changes.
type Package struct {
	AdditiveChanges *delta.Content   `json:"additiveChanges,omitempty"`
	BreakingChanges *BreakingChanges `json:"breakingChanges,omitempty"`
}

// HasBreakingChanges returns true if the package report contains breaking changes.
func (r Package) HasBreakingChanges() bool {
	return r.BreakingChanges != nil && !r.BreakingChanges.IsEmpty()
}

// HasAdditiveChanges returns true if the package report contains additive changes.
func (r Package) HasAdditiveChanges() bool {
	return r.AdditiveChanges != nil && !r.AdditiveChanges.IsEmpty()
}

// IsEmpty returns true if the report contains no data (e.g. no changes in exported types).
func (r Package) IsEmpty() bool {
	return (r.AdditiveChanges == nil || r.AdditiveChanges.IsEmpty()) &&
		(r.BreakingChanges == nil || r.BreakingChanges.IsEmpty())
}

// BreakingChanges represents a set of breaking changes.
type BreakingChanges struct {
	Consts     map[string]delta.Signature    `json:"consts,omitempty"`
	Funcs      map[string]delta.FuncSig      `json:"funcs,omitempty"`
	Interfaces map[string]delta.InterfaceDef `json:"interfaces,omitempty"`
	Structs    map[string]delta.StructDef    `json:"structs,omitempty"`
	Removed    *delta.Content                `json:"removed,omitempty"`
}

// IsEmpty returns true if there are no breaking changes.
func (bc BreakingChanges) IsEmpty() bool {
	return len(bc.Consts) == 0 && len(bc.Funcs) == 0 && len(bc.Interfaces) == 0 && len(bc.Structs) == 0 &&
		(bc.Removed == nil || bc.Removed.IsEmpty())
}

// Generate generates a package report based on the delta between lhs and rhs.
// onlyBreakingChanges - pass true to include only breaking changes in the report.
// onlyAdditions - pass true to include only addition changes in the report.
func Generate(lhs, rhs exports.Content, onlyBreakingChanges, onlyAdditions bool) Package {
	r := Package{}
	if !onlyBreakingChanges {
		if adds := delta.GetExports(lhs, rhs); !adds.IsEmpty() {
			r.AdditiveChanges = &adds
		}
	}

	if !onlyAdditions {
		breaks := BreakingChanges{}
		breaks.Consts = delta.GetConstTypeChanges(lhs, rhs)
		breaks.Funcs = delta.GetFuncSigChanges(lhs, rhs)
		breaks.Interfaces = delta.GetInterfaceMethodSigChanges(lhs, rhs)
		breaks.Structs = delta.GetStructFieldChanges(lhs, rhs)
		if removed := delta.GetExports(rhs, lhs); !removed.IsEmpty() {
			breaks.Removed = &removed
		}
		if !breaks.IsEmpty() {
			r.BreakingChanges = &breaks
		}
	}
	return r
}

// ToMarkdown creates a report of the package changes in markdown format.
func (r Package) ToMarkdown() string {
	if r.IsEmpty() {
		return ""
	}
	return formatAsMarkdown(r)
}
