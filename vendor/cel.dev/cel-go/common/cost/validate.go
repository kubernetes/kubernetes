// Copyright 2026 Google LLC
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

package cost

import (
	"errors"
	"fmt"
	"math"
)

// Validate reports whether the model is well-formed.
//
// A model is checked when it is registered, rather than when it is evaluated, because neither of
// the two evaluation paths can report a problem: an estimator has nowhere to return an error, and
// a tracker which cannot answer an expression would have to choose between charging nothing and
// charging the maximum, one of which lets an expression through and the other of which fails every
// evaluation that sets a cost limit.
//
// The checks apply at every model version, because every model they reject is mispriced at every
// version. They are:
//   - the overload has an ID, and an evaluation cost expression;
//   - every Arg and IntArg index is non-negative;
//   - every element or key projection is applied to an expression which denotes a value;
//   - every ElemTotal, KeyTotal and Length is applied to Arg, Target or Result;
//   - a global overload does not reference the receiver;
//   - every Scale factor is finite and positive;
//   - no Ranged has constant bounds in the wrong order;
//   - the result size expression does not reference Result().
//
// Projections read the value they project from, so ElemOf, KeyOf and the ArgElem, ArgKey,
// TargetElem and TargetKey shorthands are defined over Arg, Target, Result, and the ElemOf and
// KeyOf projections of those. A projection of a computed quantity, such as
// ElemOf(Sum(Arg(0), Arg(1))), has an estimate but no runtime counterpart, and is reported here.
// ElemTotal, KeyTotal and Length estimate a quantity from the size of a single value, so they are
// defined over Arg, Target and Result only.
func (m OverloadModel) Validate() error {
	return m.validate(false)
}

func (m OverloadModel) validate(inPrior bool) error {
	if m.ID == "" {
		return errors.New("overload model requires an ID")
	}
	if m.manual && m.manualEstimator {
		return fmt.Errorf("overload %q cannot combine ManualCost with ManualEstimator", m.ID)
	}
	if m.manual && (m.Cost != nil || m.hasSize) {
		return fmt.Errorf("overload %q cannot combine ManualCost with EvalCost or ResultSize", m.ID)
	}
	if m.manual && (m.estimator == nil || m.tracker == nil) {
		return fmt.Errorf("overload %q ManualCost requires both a FunctionEstimator and a FunctionTracker", m.ID)
	}
	if m.manualEstimator && m.hasSize {
		return fmt.Errorf("overload %q cannot combine ManualEstimator with ResultSize", m.ID)
	}
	if m.manualEstimator && m.estimator == nil {
		return fmt.Errorf("overload %q ManualEstimator requires a FunctionEstimator", m.ID)
	}
	if !m.manual && m.Cost == nil && !(inPrior && (m.manualEstimator || m.hasSize)) {
		return fmt.Errorf("overload %q requires an EvalCost expression or ManualCost", m.ID)
	}
	if m.Cost != nil {
		if err := validateQuantityExpr(m.Cost); err != nil {
			return fmt.Errorf("overload %q eval cost: %w", m.ID, err)
		}
	}
	if m.Size != nil || (m.hasSize && !inPrior) {
		if err := validateQuantityExpr(m.Size); err != nil {
			return fmt.Errorf("overload %q result size: %w", m.ID, err)
		}
	}
	if !m.IsMember && (hasTarget(m.Cost) || hasTarget(m.Size)) {
		return fmt.Errorf("overload %q is global but its model references the receiver; use MemberOverload", m.ID)
	}
	if m.Size != nil && references(m.Size, func(e QuantityExpr) bool { _, ok := e.(resultExpr); return ok }) {
		return fmt.Errorf("overload %q result size references Result()", m.ID)
	}
	return validatePriors(m)
}

func validatePriors(m OverloadModel) error {
	seen := make(map[uint32]struct{}, len(m.priors))
	for _, p := range m.priors {
		if p.before == 0 {
			return fmt.Errorf("overload %q: BeforeVersion requires a version >= 1, got 0", m.ID)
		}
		if _, dup := seen[p.before]; dup {
			return fmt.Errorf("overload %q: duplicate BeforeVersion(%d)", m.ID, p.before)
		}
		seen[p.before] = struct{}{}
		if err := p.model.validate(true); err != nil {
			return fmt.Errorf("overload %q before version %d: %w", m.ID, p.before, err)
		}
		resolved := m.resolvePrior(p.model)
		if !resolved.manual && resolved.Cost == nil {
			return fmt.Errorf("overload %q before version %d: overload %q requires an EvalCost expression or ManualCost", m.ID, p.before, m.ID)
		}
	}
	return nil
}

// references reports whether pred holds for the expression or any of its operands.
func references(expr QuantityExpr, pred func(QuantityExpr) bool) bool {
	if pred(expr) {
		return true
	}
	_, operands := describeQuantityExpr(expr)
	for _, operand := range operands {
		if references(operand, pred) {
			return true
		}
	}
	return false
}

// ValidateOverloadModels reports the first model which is not well-formed.
func ValidateOverloadModels(models ...OverloadModel) error {
	for _, m := range models {
		if err := m.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// validateQuantityExpr reports the first operand of an expression tree which cannot be evaluated.
//
// A nil operand is reported here so that it fails registration rather than panicking when an
// estimate or a charge is first computed.
func validateQuantityExpr(expr QuantityExpr) error {
	if expr == nil {
		return errors.New("nil quantity expression")
	}
	label, operands := describeQuantityExpr(expr)
	switch e := expr.(type) {
	case argExpr:
		if e.index < 0 {
			return fmt.Errorf("Arg index must be non-negative, got %d", e.index)
		}
	case intArgExpr:
		if e.index < 0 {
			return fmt.Errorf("IntArg index must be non-negative, got %d", e.index)
		}
	case scaleExpr:
		if math.IsNaN(e.factor) || math.IsInf(e.factor, 0) || e.factor <= 0 {
			return fmt.Errorf("Scale factor must be finite and positive, got %v", e.factor)
		}
	case rangedExpr:
		lo, loConst := e.minExpr.(constExpr)
		hi, hiConst := e.maxExpr.(constExpr)
		if loConst && hiConst && lo.val > hi.val {
			return fmt.Errorf("Ranged lower bound %d exceeds upper bound %d", lo.val, hi.val)
		}
	}
	// A nil operand is left to the recursion below to report.
	if _, isProjection := expr.(projection); isProjection && operands[0] != nil {
		if _, ok := operands[0].(valueSource); !ok {
			operandLabel, _ := describeQuantityExpr(operands[0])
			return fmt.Errorf(
				"%s is not defined over %s: element and key projections apply to Arg, Target, Result, and ElemOf or KeyOf of those",
				label, operandLabel)
		}
	}
	if _, single := expr.(singleValueProjection); single {
		switch operands[0].(type) {
		case argExpr, targetExpr, resultExpr:
		default:
			operandLabel, _ := describeQuantityExpr(operands[0])
			return fmt.Errorf(
				"%s is not defined over %s: it applies to Arg, Target and Result, whose size bounds it",
				label, operandLabel)
		}
	}
	for _, operand := range operands {
		if err := validateQuantityExpr(operand); err != nil {
			return fmt.Errorf("%s: %w", label, err)
		}
	}
	return nil
}

// projection marks the expressions which read the elements or keys of the value they are applied
// to, and so constrain what that operand may be.
type projection interface {
	QuantityExpr
	projects()
}

func (elemExpr) projects()      {}
func (keyExpr) projects()       {}
func (elemTotalExpr) projects() {}
func (keyTotalExpr) projects()  {}
func (lengthExpr) projects()    {}

// singleValueProjection marks the projections whose estimate is derived from the size of a single
// value, so the operand must denote exactly one value: a total or length of ElemOf(X) would range
// over every element of X, which no single size estimate bounds.
type singleValueProjection interface {
	projection
	singleValue()
}

func (elemTotalExpr) singleValue() {}
func (keyTotalExpr) singleValue()  {}
func (lengthExpr) singleValue()    {}

// describeQuantityExpr reports the constructor name of a quantity expression and its operands.
//
// Every expression type must be listed: an omission makes an expression's operands invisible to
// validation, and reports an unhelpful name in the error message. TestDescribeQuantityExpr covers
// each constructor.
func describeQuantityExpr(expr QuantityExpr) (string, []QuantityExpr) {
	switch e := expr.(type) {
	case constExpr:
		return "Const", nil
	case argExpr:
		return "Arg", nil
	case intArgExpr:
		return "IntArg", nil
	case targetExpr:
		return "Target", nil
	case intTargetExpr:
		return "IntTarget", nil
	case resultExpr:
		return "Result", nil
	case elemExpr:
		return "ElemOf", []QuantityExpr{e.expr}
	case keyExpr:
		return "KeyOf", []QuantityExpr{e.expr}
	case elemTotalExpr:
		return "ElemTotal", []QuantityExpr{e.expr}
	case keyTotalExpr:
		return "KeyTotal", []QuantityExpr{e.expr}
	case lengthExpr:
		return "Length", []QuantityExpr{e.expr}
	case addExpr:
		return "Sum", e.terms
	case subExpr:
		return "Sub", []QuantityExpr{e.lhs, e.rhs}
	case mulExpr:
		return "Mul", e.terms
	case scaleExpr:
		return "Scale", []QuantityExpr{e.expr}
	case minExpr:
		return "Min", []QuantityExpr{e.lhs, e.rhs}
	case maxExpr:
		return "Max", []QuantityExpr{e.lhs, e.rhs}
	case unionExpr:
		return "Union", e.terms
	case rangedExpr:
		return "Ranged", []QuantityExpr{e.minExpr, e.maxExpr}
	case atLeastOneExpr:
		return "AtLeastOneQuantity", []QuantityExpr{e.expr}
	case listExpr:
		return "List", []QuantityExpr{e.lenExpr, e.elemExpr}
	case mapExpr:
		return "Map", []QuantityExpr{e.sizeExpr, e.keyExpr, e.valExpr}
	case optionalExpr:
		return "Optional", []QuantityExpr{e.valueExpr}
	default:
		return fmt.Sprintf("%T", expr), nil
	}
}
