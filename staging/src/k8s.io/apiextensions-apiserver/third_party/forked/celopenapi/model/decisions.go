// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// NewDecision returns an empty Decision instance.
func NewDecision() *Decision {
	return &Decision{}
}

// Decision contains a decision name, or reference to a decision name, and an output expression.
type Decision struct {
	Name      string
	Reference *cel.Ast
	Output    *cel.Ast
}

// DecisionValue represents a named decision and value.
type DecisionValue interface {
	fmt.Stringer

	// Name returns the decision name.
	Name() string

	// IsFinal returns whether the decision value will change with additional rule evaluations.
	//
	// When a decision is final, additional productions and rules which may also trigger the same
	// decision may be skipped.
	IsFinal() bool
}

// SingleDecisionValue extends the DecisionValue which contains a single decision value as well
// as some metadata about the evaluation details and the rule that spawned the value.
type SingleDecisionValue interface {
	DecisionValue

	// Value returns the single value for the decision.
	Value() ref.Val

	// Details returns the evaluation details, if present, that produced the value.
	Details() *cel.EvalDetails

	// RuleID indicate which policy rule id within an instance that produced the decision.
	RuleID() int64
}

// MultiDecisionValue extends the DecisionValue which contains a set of decision values as well as
// the corresponding metadata about how each value was produced.
type MultiDecisionValue interface {
	DecisionValue

	// Values returns the collection of values produced for the decision.
	Values() []ref.Val

	// Details returns the evaluation details for each value in the decision.
	// The value index correponds to the details index. The details may be nil.
	Details() []*cel.EvalDetails

	// RulesIDs returns the rule id within an instance which produce the decision values.
	// The value index corresponds to the rule id index.
	RuleIDs() []int64
}

// DecisionSelector determines whether the given decision is the decision set requested by the
// caller.
type DecisionSelector func(decision string) bool

// NewBoolDecisionValue returns a boolean decision with an initial value.
func NewBoolDecisionValue(name string, value types.Bool) *BoolDecisionValue {
	return &BoolDecisionValue{
		name:  name,
		value: value,
	}
}

// BoolDecisionValue represents the decision value type associated with a decision.
type BoolDecisionValue struct {
	name    string
	value   ref.Val
	isFinal bool
	details *cel.EvalDetails
	ruleID  int64
}

// And logically ANDs the current decision value with the incoming CEL value.
//
// And follows CEL semantics with respect to errors and unknown values where errors may be
// absorbed or short-circuited away by subsequent 'false' values. When unkonwns are encountered
// the unknown values combine and aggregate within the decision. Unknowns may also be absorbed
// per CEL semantics.
func (dv *BoolDecisionValue) And(other ref.Val) *BoolDecisionValue {
	v, vBool := dv.value.(types.Bool)
	if vBool && v == types.False {
		return dv
	}
	o, oBool := other.(types.Bool)
	if oBool && o == types.False {
		dv.value = types.False
		return dv
	}
	if vBool && oBool {
		return dv
	}
	dv.value = logicallyMergeUnkErr(dv.value, other)
	return dv
}

// Details implements the SingleDecisionValue interface method.
func (dv *BoolDecisionValue) Details() *cel.EvalDetails {
	return dv.details
}

// Finalize marks the decision as immutable with additional input and indicates the rule and
// evaluation details which triggered the finalization.
func (dv *BoolDecisionValue) Finalize(details *cel.EvalDetails, rule Rule) DecisionValue {
	dv.details = details
	if rule != nil {
		dv.ruleID = rule.GetID()
	}
	dv.isFinal = true
	return dv
}

// IsFinal returns whether the decision is final.
func (dv *BoolDecisionValue) IsFinal() bool {
	return dv.isFinal
}

// Or logically ORs the decision value with the incoming CEL value.
//
// The ORing logic follows CEL semantics with respect to errors and unknown values.
// Errors may be absorbed or short-circuited away by subsequent 'true' values. When unkonwns are
// encountered the unknown values combine and aggregate within the decision. Unknowns may also be
// absorbed per CEL semantics.
func (dv *BoolDecisionValue) Or(other ref.Val) *BoolDecisionValue {
	v, vBool := dv.value.(types.Bool)
	if vBool && v == types.True {
		return dv
	}
	o, oBool := other.(types.Bool)
	if oBool && o == types.True {
		dv.value = types.True
		return dv
	}
	if vBool && oBool {
		return dv
	}
	dv.value = logicallyMergeUnkErr(dv.value, other)
	return dv
}

// Name implements the DecisionValue interface method.
func (dv *BoolDecisionValue) Name() string {
	return dv.name
}

// RuleID implements the SingleDecisionValue interface method.
func (dv *BoolDecisionValue) RuleID() int64 {
	return dv.ruleID
}

// String renders the decision value to a string for debug purposes.
func (dv *BoolDecisionValue) String() string {
	var buf strings.Builder
	buf.WriteString(dv.name)
	buf.WriteString(": ")
	buf.WriteString(fmt.Sprintf("rule[%d] -> ", dv.ruleID))
	buf.WriteString(fmt.Sprintf("%v", dv.value))
	return buf.String()
}

// Value implements the SingleDecisionValue interface method.
func (dv *BoolDecisionValue) Value() ref.Val {
	return dv.value
}

// NewListDecisionValue returns a named decision value which contains a list of CEL values produced
// by one or more policy instances and / or production rules.
func NewListDecisionValue(name string) *ListDecisionValue {
	return &ListDecisionValue{
		name:    name,
		values:  []ref.Val{},
		details: []*cel.EvalDetails{},
		ruleIDs: []int64{},
	}
}

// ListDecisionValue represents a named decision which collects into a list of values.
type ListDecisionValue struct {
	name    string
	values  []ref.Val
	isFinal bool
	details []*cel.EvalDetails
	ruleIDs []int64
}

// Append accumulates the incoming CEL value into the decision's value list.
func (dv *ListDecisionValue) Append(val ref.Val, det *cel.EvalDetails, rule Rule) {
	dv.values = append(dv.values, val)
	dv.details = append(dv.details, det)
	// Rule ids may be null if the policy is a singleton.
	ruleID := int64(0)
	if rule != nil {
		ruleID = rule.GetID()
	}
	dv.ruleIDs = append(dv.ruleIDs, ruleID)
}

// Details returns the list of evaluation details observed in computing the values in the decision.
// The details indices correlate 1:1 with the value indices.
func (dv *ListDecisionValue) Details() []*cel.EvalDetails {
	return dv.details
}

// Finalize marks the list decision complete.
func (dv *ListDecisionValue) Finalize() DecisionValue {
	dv.isFinal = true
	return dv
}

// IsFinal implements the DecisionValue interface method.
func (dv *ListDecisionValue) IsFinal() bool {
	return dv.isFinal
}

// Name implements the DecisionValue interface method.
func (dv *ListDecisionValue) Name() string {
	return dv.name
}

// RuleIDs returns the list of rule ids which produced the evaluation results.
// The indices of the ruleIDs correlate 1:1 with the value indices.
func (dv *ListDecisionValue) RuleIDs() []int64 {
	return dv.ruleIDs
}

func (dv *ListDecisionValue) String() string {
	var buf strings.Builder
	buf.WriteString(dv.name)
	buf.WriteString(": ")
	for i, v := range dv.values {
		if len(dv.ruleIDs) == len(dv.values) {
			buf.WriteString(fmt.Sprintf("rule[%d] -> ", dv.ruleIDs[i]))
		}
		buf.WriteString(fmt.Sprintf("%v", v))
		buf.WriteString("\n")
		if i < len(dv.values)-1 {
			buf.WriteString("\t")
		}
	}
	return buf.String()
}

// Values implements the MultiDecisionValue interface method.
func (dv *ListDecisionValue) Values() []ref.Val {
	return dv.values
}

func logicallyMergeUnkErr(value, other ref.Val) ref.Val {
	vUnk := types.IsUnknown(value)
	oUnk := types.IsUnknown(other)
	if vUnk && oUnk {
		merged := types.Unknown{}
		merged = append(merged, value.(types.Unknown)...)
		merged = append(merged, other.(types.Unknown)...)
		return merged
	}
	if vUnk {
		return value
	}
	if oUnk {
		return other
	}
	if types.IsError(value) {
		return value
	}
	if types.IsError(other) {
		return other
	}
	return types.NewErr(
		"got values (%v, %v), wanted boolean values",
		value, other)
}
