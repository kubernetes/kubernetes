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
	"math"
	"slices"
	"sync"

	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"
)

// TypeContext provides type information for AST nodes.
type TypeContext interface {
	// TargetType returns the deduced type of the receiver/target object, if present.
	TargetType() (*types.Type, bool)

	// ArgType returns the deduced type of the argument at the given 0-based index.
	ArgType(index int) (*types.Type, bool)
}

// EstimateContext provides size and argument evaluation during cost estimation.
type EstimateContext interface {
	TypeContext

	// Arg returns the size estimate for the argument at the given 0-based index.
	Arg(index int) (SizeEstimate, bool)

	// Target returns the size estimate for the receiver/target object, if present.
	Target() (SizeEstimate, bool)

	// Result returns the size estimate of the evaluated result, if known.
	Result() (SizeEstimate, bool)

	// Estimator returns the user-provided Estimator, if configured.
	Estimator() Estimator

	// Size returns the estimated size based on the SizingStrategy configured on the Estimator.
	Size(node AstNode) SizeEstimate

	// ArgValue returns the literal/constant uint64 value of the argument at index, or defaultVal.
	ArgValue(index int, defaultVal uint64) uint64

	// TargetValue returns the literal/constant uint64 value of the receiver/target, or defaultVal.
	TargetValue(defaultVal uint64) uint64
}

// TrackContext provides value size and argument evaluation during runtime cost tracking.
type TrackContext interface {
	TypeContext

	// Arg returns the size of the argument at the given 0-based index.
	Arg(index int) uint64

	// Target returns the size of the receiver/target object, if present.
	Target() uint64

	// Result returns the size of the evaluated result, if present.
	Result() uint64

	// Estimator returns the runtime ActualCostEstimator, if configured.
	Estimator() ActualCostEstimator

	// Size returns the actual runtime size of a value.
	Size(value ref.Val) uint64

	// ArgValue returns the uint64 value of the argument at index, or defaultVal.
	ArgValue(index int, defaultVal uint64) uint64

	// TargetValue returns the uint64 value of the receiver/target, or defaultVal.
	TargetValue(defaultVal uint64) uint64

	// ArgVal returns the value of the argument at the given 0-based index, if present.
	ArgVal(index int) (ref.Val, bool)

	// TargetVal returns the value of the receiver/target object, if present.
	TargetVal() (ref.Val, bool)

	// ResultVal returns the evaluated result value, if present.
	ResultVal() (ref.Val, bool)
}

// QuantityExpr represents a computable size or cost equation.
type QuantityExpr interface {
	estimate(ctx EstimateContext) SizeEstimate
	track(ctx TrackContext) uint64
}

// ModelVersionOf reports the cost model version the given Estimator is configured with, or
// math.MaxUint32 if the estimator does not carry version metadata.
func ModelVersionOf(estimator Estimator) uint32 {
	if v, ok := estimator.(interface{ modelVersion() uint32 }); ok {
		return v.modelVersion()
	}
	return defaultModelVersion
}

// targetInspector is an internal interface for expressions that inspect receiver/target presence.
type targetInspector interface {
	hasTarget() bool
}

// hasTarget returns true if the expression or any of its subexpressions references the target object.
func hasTarget(expr QuantityExpr) bool {
	if expr == nil {
		return false
	}
	if r, ok := expr.(targetInspector); ok {
		return r.hasTarget()
	}
	return false
}

// valueSource is an internal interface for expressions which denote runtime values rather than
// only a quantity derived from them.
//
// Element and key projections read the value they project from, so they are defined only over the
// expressions implementing this interface: Arg, Target, Result, and the element and key
// projections ElemOf and KeyOf. An expression may denote more than one value, as the elements of
// a list do.
//
// OverloadModel.Validate reports a projection over any other expression, so a model which cannot
// answer a projection at runtime is rejected when it is registered rather than mispriced when it
// is evaluated.
type valueSource interface {
	QuantityExpr

	// eachValue calls fn with each value the expression denotes in the given context. An
	// expression which denotes no value in the context makes no calls.
	eachValue(ctx TrackContext, fn func(ref.Val))
}

// eachValueOf calls fn with each value denoted by expr, and makes no calls if expr does not denote
// a value.
func eachValueOf(expr QuantityExpr, ctx TrackContext, fn func(ref.Val)) {
	if src, ok := expr.(valueSource); ok {
		src.eachValue(ctx, fn)
	}
}

// eachElement calls fn with each element of a list, each value of a map, or the value an optional
// holds. Values which are not containers have no elements.
func eachElement(val ref.Val, fn func(ref.Val)) {
	switch v := val.(type) {
	case *types.Optional:
		if v.HasValue() {
			fn(v.GetValue())
		}
	case traits.Lister:
		for it := v.Iterator(); it.HasNext() == types.True; {
			fn(it.Next())
		}
	case traits.Mapper:
		for it := v.Iterator(); it.HasNext() == types.True; {
			fn(v.Get(it.Next()))
		}
	}
}

// eachKey calls fn with each key of a map. Values which are not maps have no keys.
func eachKey(val ref.Val, fn func(ref.Val)) {
	if m, ok := val.(traits.Mapper); ok {
		for it := m.Iterator(); it.HasNext() == types.True; {
			fn(it.Next())
		}
	}
}

// constExpr represents a constant integer quantity.
type constExpr struct {
	val uint64
}

func (c constExpr) estimate(_ EstimateContext) SizeEstimate {
	return FixedSizeEstimate(c.val)
}

func (c constExpr) track(_ TrackContext) uint64 {
	return c.val
}

func (constExpr) hasTarget() bool { return false }

// Const creates a constant quantity expression.
func Const(val uint64) QuantityExpr {
	return constExpr{val: val}
}

// argExpr represents the size of a function argument.
type argExpr struct {
	index int
}

func (a argExpr) estimate(ctx EstimateContext) SizeEstimate {
	if sz, ok := ctx.Arg(a.index); ok {
		return sz
	}
	return UnknownSizeEstimate()
}

func (a argExpr) track(ctx TrackContext) uint64 {
	return ctx.Arg(a.index)
}

func (a argExpr) eachValue(ctx TrackContext, fn func(ref.Val)) {
	if val, ok := ctx.ArgVal(a.index); ok {
		fn(val)
	}
}

func (argExpr) hasTarget() bool { return false }

// Arg creates an expression referencing the size of the argument at the given index.
func Arg(index int) QuantityExpr {
	return argExpr{index: index}
}

// ArgElem creates an expression referencing the element size of the argument at the given index.
func ArgElem(index int) QuantityExpr {
	return ElemOf(Arg(index))
}

// ArgKey creates an expression referencing the key size of the argument at the given index.
func ArgKey(index int) QuantityExpr {
	return KeyOf(Arg(index))
}

// intArgExpr represents the integer value of an argument.
type intArgExpr struct {
	index      int
	defaultVal uint64
}

func (a intArgExpr) estimate(ctx EstimateContext) SizeEstimate {
	if versionOf(ctx) < 1 {
		return FixedSizeEstimate(ctx.ArgValue(a.index, a.defaultVal))
	}
	// A literal reports the same value for any default; a non-literal reports the default.
	lo, hi := ctx.ArgValue(a.index, 0), ctx.ArgValue(a.index, math.MaxUint64)
	if lo != hi {
		return UnknownSizeEstimate()
	}
	return FixedSizeEstimate(lo)
}

func (a intArgExpr) track(ctx TrackContext) uint64 {
	return ctx.ArgValue(a.index, a.defaultVal)
}

func (intArgExpr) hasTarget() bool { return false }

// IntArg creates an expression referencing the integer value of the argument at the given index.
//
// At runtime, an argument which is not an integer reads as defaultVal, and a negative integer reads
// as 0. When estimating, a literal argument reads as its value, and a non-literal argument is
// unbounded from model version 1 (or FixedSizeEstimate(defaultVal) at version 0).
func IntArg(index int, defaultVal uint64) QuantityExpr {
	return intArgExpr{index: index, defaultVal: defaultVal}
}

// targetExpr represents the size of the receiver/target object.
type targetExpr struct{}

func (targetExpr) estimate(ctx EstimateContext) SizeEstimate {
	if sz, ok := ctx.Target(); ok {
		return sz
	}
	return UnknownSizeEstimate()
}

func (targetExpr) track(ctx TrackContext) uint64 {
	return ctx.Target()
}

func (targetExpr) eachValue(ctx TrackContext, fn func(ref.Val)) {
	if val, ok := ctx.TargetVal(); ok {
		fn(val)
	}
}

func (targetExpr) hasTarget() bool { return true }

// Target creates an expression referencing the size of the receiver / target object.
func Target() QuantityExpr {
	return targetExpr{}
}

// TargetElem creates an expression referencing the element size of the receiver / target object.
func TargetElem() QuantityExpr {
	return ElemOf(Target())
}

// elemExpr represents the element size of another quantity expression.
type elemExpr struct {
	expr QuantityExpr
}

func (e elemExpr) estimate(ctx EstimateContext) SizeEstimate {
	sz := e.expr.estimate(ctx)
	if sz.Elem != nil {
		return *sz.Elem
	}
	return UnknownSizeEstimate()
}

// track reports the size of the largest element, matching the estimate, whose Max bounds every
// element rather than describing an average one.
//
// A value with no elements, including a scalar, contributes nothing. The cost of the scan is
// linear in the number of elements; see the element projection notes in docs/cost-model.md.
func (e elemExpr) track(ctx TrackContext) uint64 {
	largest := uint64(0)
	e.eachValue(ctx, func(elem ref.Val) {
		largest = max(largest, ctx.Size(elem))
	})
	return largest
}

func (e elemExpr) eachValue(ctx TrackContext, fn func(ref.Val)) {
	eachValueOf(e.expr, ctx, func(val ref.Val) {
		eachElement(val, fn)
	})
}

func (e elemExpr) hasTarget() bool {
	return hasTarget(e.expr)
}

// ElemOf creates an expression referencing the element size of another expression.
func ElemOf(expr QuantityExpr) QuantityExpr {
	return elemExpr{expr: expr}
}

// elemTotalExpr represents the summed size of the elements of another quantity expression.
type elemTotalExpr struct {
	expr QuantityExpr
}

func (e elemTotalExpr) estimate(ctx EstimateContext) SizeEstimate {
	sz := e.expr.estimate(ctx)
	return totalEstimate(ctx, e.expr, sz, sz.Elem, isListOperand(ctx, e.expr, sz))
}

// singleValueOf resolves the value an ElemTotal, KeyTotal or Length operand denotes. Validate
// restricts those operands to Arg, Target and Result.
func singleValueOf(expr QuantityExpr, ctx TrackContext) (ref.Val, bool) {
	switch x := expr.(type) {
	case argExpr:
		return ctx.ArgVal(x.index)
	case targetExpr:
		return ctx.TargetVal()
	case resultExpr:
		return ctx.ResultVal()
	}
	return nil, false
}

// sumSizes totals the sizes of the parts which each enumerates in the operand's value.
func sumSizes(ctx TrackContext, operand QuantityExpr, each func(ref.Val, func(ref.Val))) uint64 {
	total := uint64(0)
	if val, ok := singleValueOf(operand, ctx); ok {
		each(val, func(part ref.Val) {
			total = SafeAdd(total, ctx.Size(part))
		})
	}
	return total
}

func (e elemTotalExpr) track(ctx TrackContext) uint64 {
	return sumSizes(ctx, e.expr, eachElement)
}

func (e elemTotalExpr) hasTarget() bool {
	return hasTarget(e.expr)
}

// ElemTotal creates an expression referencing the combined size of every element of a list, or
// every value of a map, under the sizing strategy in effect: the length of a flattened list under
// the default strategy, or the content size of a list under the aggregate strategy. See KeyTotal
// for the keys of a map.
//
// The operand must be Arg, Target or Result; see OverloadModel.Validate. The estimate is derived
// from the operand's size under the sizing strategy; see totalEstimate. At runtime this reports
// the exact total.
func ElemTotal(expr QuantityExpr) QuantityExpr {
	return elemTotalExpr{expr: expr}
}

// keyTotalExpr represents the summed size of the keys of another quantity expression.
type keyTotalExpr struct {
	expr QuantityExpr
}

func (k keyTotalExpr) estimate(ctx EstimateContext) SizeEstimate {
	sz := k.expr.estimate(ctx)
	return totalEstimate(ctx, k.expr, sz, sz.Key, false)
}

func (k keyTotalExpr) track(ctx TrackContext) uint64 {
	return sumSizes(ctx, k.expr, eachKey)
}

func (k keyTotalExpr) hasTarget() bool {
	return hasTarget(k.expr)
}

// KeyTotal creates an expression referencing the combined size of every key of a map. A value
// which is not a map has no keys.
//
// The operand must be Arg, Target or Result; see OverloadModel.Validate. The estimate is derived
// from the operand's size under the sizing strategy; see totalEstimate. At runtime this reports
// the exact total.
func KeyTotal(expr QuantityExpr) QuantityExpr {
	return keyTotalExpr{expr: expr}
}

// lengthExpr represents the shallow top-level length of a single value (its CEL size()): the
// number of code points of a string, the number of bytes of a bytes value, the number of top-level
// elements of a list or entries of a map, and 1 for any other value.
type lengthExpr struct {
	expr QuantityExpr
}

// Length creates an expression referencing the shallow top-level length of a single value (its CEL
// size() / ActualSize), independent of the sizing strategy: the number of code points of a string,
// bytes of a bytes value, top-level elements of a list or entries of a map, and 1 for any other
// value. Unlike Arg(i), Target(), or Result(), which report the depth-based total content size
// under AggregateSizingStrategy, Length always reports only the top-level container or value
// length.
//
// The operand must be Arg, Target or Result; see OverloadModel.Validate. Length is a scalar and
// carries no element or key size.
//
// Sizes follow the sizing strategy, so under the aggregate strategy Target() of a string is its
// UTF-8 byte count and Target() of a list is its depth-based aggregate content size. Length lets a
// model charge by top-level length under every strategy while still describing its result's size
// through List and Map.
func Length(expr QuantityExpr) QuantityExpr {
	return lengthExpr{expr: expr}
}

// estimate reports the exact length of a literal operand, as ActualSize counts it at runtime, or
// bounds the operand's length from its estimated size under the sizing strategy.
//
// Because track returns ActualSize(val)—which is 0 for empty strings, bytes, and containers, and 1
// for fixed-width scalars—the returned range is [1, 1] for fixed-width scalars and [lo, hi] within
// [0, MaxUint64] for variable-length values. For soundness (est.Min <= actual), lo must be 0
// whenever the operand may be empty at runtime:
//
//   - Default strategy: the size is the length ([sz.Min, sz.Max]).
//   - Aggregate strategy: every value has an aggregate size >= 1 even when its length is 0 (a
//     string's size is max(1, UTF-8 byte count) with 1–4 bytes per code point; a bytes value's size
//     is max(1, byte length); a container's size is 1 + entry sizes, see countBounds); any other
//     value has length 1. An aggregate size with Min <= 1 therefore yields lo = 0.
//   - Other strategies: [0, MaxUint64], except for scalars whose length is always [1, 1].
//
// The kind of value comes from the operand's checked type, or, for a result whose type is not
// known, from its metadata: an estimate which states element or key sizes is a container.
func (l lengthExpr) estimate(ctx EstimateContext) SizeEstimate {
	if lit, ok := operandLiteral(ctx, l.expr); ok {
		return FixedSizeEstimate(ActualSize(lit))
	}
	sz := l.expr.estimate(ctx)
	kind, known := operandKind(ctx, l.expr)
	if known && hasUnitLength(kind) {
		return FixedSizeEstimate(1)
	}
	switch strategyOf(ctx).(type) {
	case defaultSizingStrategy:
		return RangedSizeEstimate(sz.Min, sz.Max)
	case *aggregateSizingStrategy:
		switch {
		case known && kind == types.StringKind:
			// Size <= 1 may be "" (length 0); n > 1 UTF-8 bytes hold [ceil(n/4), n] code points.
			lo := uint64(0)
			if sz.Min > 1 {
				lo = sz.Min/4 + min(sz.Min%4, 1)
			}
			return RangedSizeEstimate(lo, sz.Max)
		case known && kind == types.BytesKind:
			// Size <= 1 may be b"" (length 0).
			lo := uint64(0)
			if sz.Min > 1 {
				lo = sz.Min
			}
			return RangedSizeEstimate(lo, sz.Max)
		case (known && (kind == types.ListKind || kind == types.MapKind)) || sz.Elem != nil || sz.Key != nil:
			return countBounds(strategyOf(ctx), sz)
		default:
			// Any value: empty strings/bytes/containers have length 0, and length never exceeds size (>= 1).
			return RangedSizeEstimate(0, max(sz.Max, 1))
		}
	default:
		return UnknownSizeEstimate()
	}
}

func (l lengthExpr) track(ctx TrackContext) uint64 {
	val, ok := singleValueOf(l.expr, ctx)
	// A call which fails produces an error rather than a result, which has no length.
	if !ok || types.IsUnknownOrError(val) {
		return 0
	}
	// An optional has the length of the value it holds, as it is estimated.
	if inner, isOpt := optionalValueOf(val); isOpt {
		if inner == nil {
			return 0
		}
		val = inner
	}
	return ActualSize(val)
}

func (l lengthExpr) hasTarget() bool {
	return hasTarget(l.expr)
}

// TargetKey creates an expression referencing the key size of the receiver / target object.
func TargetKey() QuantityExpr {
	return KeyOf(Target())
}

// intTargetExpr represents the integer value of the receiver/target object.
type intTargetExpr struct {
	defaultVal uint64
}

func (t intTargetExpr) estimate(ctx EstimateContext) SizeEstimate {
	if versionOf(ctx) < 1 {
		return FixedSizeEstimate(ctx.TargetValue(t.defaultVal))
	}
	lo, hi := ctx.TargetValue(0), ctx.TargetValue(math.MaxUint64)
	if lo != hi {
		return UnknownSizeEstimate()
	}
	return FixedSizeEstimate(lo)
}

func (t intTargetExpr) track(ctx TrackContext) uint64 {
	return ctx.TargetValue(t.defaultVal)
}

func (intTargetExpr) hasTarget() bool { return true }

// IntTarget creates an expression referencing the integer value of the receiver/target object,
// with the same runtime and estimation rules as IntArg.
func IntTarget(defaultVal uint64) QuantityExpr {
	return intTargetExpr{defaultVal: defaultVal}
}

// keyExpr represents the key size of another quantity expression.
type keyExpr struct {
	expr QuantityExpr
}

func (k keyExpr) estimate(ctx EstimateContext) SizeEstimate {
	sz := k.expr.estimate(ctx)
	if sz.Key != nil {
		return *sz.Key
	}
	// An absent key size is unknown rather than single-byte: map keys may be strings or bytes of
	// any length. Strategies which know the key type is fixed-width populate Key themselves.
	return UnknownSizeEstimate()
}

// track reports the size of the largest key, matching the estimate, whose Max bounds every key
// rather than describing an average one.
//
// A value with no keys, including a list or a scalar, contributes nothing. The cost of the scan is
// linear in the number of entries; see the element projection notes in docs/cost-model.md.
func (k keyExpr) track(ctx TrackContext) uint64 {
	largest := uint64(0)
	k.eachValue(ctx, func(key ref.Val) {
		largest = max(largest, ctx.Size(key))
	})
	return largest
}

func (k keyExpr) eachValue(ctx TrackContext, fn func(ref.Val)) {
	eachValueOf(k.expr, ctx, func(val ref.Val) {
		eachKey(val, fn)
	})
}

func (k keyExpr) hasTarget() bool {
	return hasTarget(k.expr)
}

// KeyOf creates an expression referencing the key size of another expression.
func KeyOf(expr QuantityExpr) QuantityExpr {
	return keyExpr{expr: expr}
}

// resultExpr represents the size of the evaluated result.
type resultExpr struct{}

func (resultExpr) estimate(ctx EstimateContext) SizeEstimate {
	if sz, ok := ctx.Result(); ok {
		return sz
	}
	return UnknownSizeEstimate()
}

func (resultExpr) track(ctx TrackContext) uint64 {
	return ctx.Result()
}

func (resultExpr) eachValue(ctx TrackContext, fn func(ref.Val)) {
	if val, ok := ctx.ResultVal(); ok {
		fn(val)
	}
}

func (resultExpr) hasTarget() bool { return false }

// Result creates an expression referencing the size of the result.
func Result() QuantityExpr {
	return resultExpr{}
}

// addExpr represents the sum of multiple quantity expressions.
type addExpr struct {
	terms []QuantityExpr
}

func (a addExpr) estimate(ctx EstimateContext) SizeEstimate {
	if len(a.terms) == 0 {
		return FixedSizeEstimate(0)
	}
	return foldEstimates(ctx, a.terms, SizeEstimate.Add)
}

func (a addExpr) track(ctx TrackContext) uint64 {
	sum := uint64(0)
	for _, term := range a.terms {
		sum = SafeAdd(sum, term.track(ctx))
	}
	return sum
}

func (a addExpr) hasTarget() bool {
	return slices.ContainsFunc(a.terms, hasTarget)
}

// Sum creates an expression representing the sum of the given terms.
func Sum(terms ...QuantityExpr) QuantityExpr {
	return addExpr{terms: slices.Clone(terms)}
}

// subExpr represents the subtraction of two quantity expressions.
type subExpr struct {
	lhs, rhs QuantityExpr
}

func (s subExpr) estimate(ctx EstimateContext) SizeEstimate {
	lhsVal := s.lhs.estimate(ctx)
	rhsVal := s.rhs.estimate(ctx)
	diff := lhsVal.Subtract(rhsVal)
	// A difference holds some of the minuend's elements, never the subtrahend's.
	diff.Key, diff.Elem = lhsVal.Key, lhsVal.Elem
	return diff
}

func (s subExpr) track(ctx TrackContext) uint64 {
	return SafeSubtract(s.lhs.track(ctx), s.rhs.track(ctx))
}

func (s subExpr) hasTarget() bool {
	return hasTarget(s.lhs) || hasTarget(s.rhs)
}

// Sub creates an expression representing lhs - rhs, saturated at zero.
func Sub(lhs, rhs QuantityExpr) QuantityExpr {
	return subExpr{lhs: lhs, rhs: rhs}
}

// mulExpr represents the product of multiple quantity expressions.
type mulExpr struct {
	terms []QuantityExpr
}

func (m mulExpr) estimate(ctx EstimateContext) SizeEstimate {
	if len(m.terms) == 0 {
		return FixedSizeEstimate(1)
	}
	return foldEstimates(ctx, m.terms, SizeEstimate.Multiply)
}

func (m mulExpr) track(ctx TrackContext) uint64 {
	if len(m.terms) == 0 {
		return 1
	}
	prod := m.terms[0].track(ctx)
	for _, term := range m.terms[1:] {
		prod = SafeMultiply(prod, term.track(ctx))
	}
	return prod
}

func (m mulExpr) hasTarget() bool {
	return slices.ContainsFunc(m.terms, hasTarget)
}

// Mul creates an expression representing the product of the given terms.
func Mul(terms ...QuantityExpr) QuantityExpr {
	return mulExpr{terms: slices.Clone(terms)}
}

// scaleExpr represents an expression scaled by a constant factor.
type scaleExpr struct {
	expr   QuantityExpr
	factor float64
	// rat is the exact rational form of factor; see rationalOf.
	rat rational
}

func (s scaleExpr) estimate(ctx EstimateContext) SizeEstimate {
	val := s.expr.estimate(ctx)
	res := RangedSizeEstimate(
		scale(val.Min, s.factor, s.rat),
		scale(val.Max, s.factor, s.rat),
	)
	res.Key = val.Key
	res.Elem = val.Elem
	return res
}

func (s scaleExpr) track(ctx TrackContext) uint64 {
	return scale(s.expr.track(ctx), s.factor, s.rat)
}

func (s scaleExpr) hasTarget() bool {
	return hasTarget(s.expr)
}

// Scale creates an expression that scales another expression by a constant factor, rounding up.
//
// Factors are constant so that a model's runtime charge depends only on the sizes reported by the
// sizing strategy. A cost which differs by argument type should be modeled with one OverloadModel
// per overload ID, since CEL already dispatches overloads by type. The factor must be finite and
// positive; Validate rejects any other.
//
// The factor is applied as the simplest rational with the same float64 value, so that 0.1 scales by
// exactly 1/10, and the product is computed exactly; an unbounded quantity stays unbounded; and
// when a model is compiled, factors are moved outward so that each product is rounded once.
func Scale(expr QuantityExpr, factor float64) QuantityExpr {
	return scaleExpr{expr: expr, factor: factor, rat: rationalOf(factor)}
}

// Square creates an expression representing the square of another expression.
func Square(expr QuantityExpr) QuantityExpr {
	return Mul(expr, expr)
}

// minExpr represents the minimum of two quantity expressions.
type minExpr struct {
	lhs, rhs QuantityExpr
}

func (m minExpr) estimate(ctx EstimateContext) SizeEstimate {
	return foldEstimates(ctx, []QuantityExpr{m.lhs, m.rhs}, minBounds)
}

func (m minExpr) track(ctx TrackContext) uint64 {
	return min(m.lhs.track(ctx), m.rhs.track(ctx))
}

func (m minExpr) hasTarget() bool {
	return hasTarget(m.lhs) || hasTarget(m.rhs)
}

// Min creates an expression representing the minimum of two quantities.
func Min(lhs, rhs QuantityExpr) QuantityExpr {
	return minExpr{lhs: lhs, rhs: rhs}
}

// maxExpr represents the maximum of two quantity expressions.
type maxExpr struct {
	lhs, rhs QuantityExpr
}

func (m maxExpr) estimate(ctx EstimateContext) SizeEstimate {
	return foldEstimates(ctx, []QuantityExpr{m.lhs, m.rhs}, maxBounds)
}

func (m maxExpr) track(ctx TrackContext) uint64 {
	return max(m.lhs.track(ctx), m.rhs.track(ctx))
}

func (m maxExpr) hasTarget() bool {
	return hasTarget(m.lhs) || hasTarget(m.rhs)
}

// Max creates an expression representing the maximum of two quantities.
func Max(lhs, rhs QuantityExpr) QuantityExpr {
	return maxExpr{lhs: lhs, rhs: rhs}
}

// unionExpr represents the union interval of multiple quantity expressions.
type unionExpr struct {
	terms []QuantityExpr
}

func (u unionExpr) estimate(ctx EstimateContext) SizeEstimate {
	if len(u.terms) == 0 {
		return FixedSizeEstimate(0)
	}
	return foldEstimates(ctx, u.terms, SizeEstimate.Union)
}

func (u unionExpr) track(ctx TrackContext) uint64 {
	res := uint64(0)
	for _, term := range u.terms {
		res = max(res, term.track(ctx))
	}
	return res
}

func (u unionExpr) hasTarget() bool {
	return slices.ContainsFunc(u.terms, hasTarget)
}

// Union creates an expression representing the union interval of multiple expressions.
func Union(terms ...QuantityExpr) QuantityExpr {
	return unionExpr{terms: slices.Clone(terms)}
}

// rangedExpr represents a quantity bounded by minExpr on the bottom and maxExpr on top.
type rangedExpr struct {
	minExpr QuantityExpr
	maxExpr QuantityExpr
}

func (r rangedExpr) estimate(ctx EstimateContext) SizeEstimate {
	return foldEstimates(ctx, []QuantityExpr{r.minExpr, r.maxExpr}, rangedBounds)
}

// foldEstimates estimates the operands and combines the estimates from left to right with op,
// attaching the operands' metadata as combineMeta describes. The estimates of up to four operands,
// as many as the models in this repository use, are held on the stack.
func foldEstimates(ctx EstimateContext, operands []QuantityExpr, op func(SizeEstimate, SizeEstimate) SizeEstimate) SizeEstimate {
	var buf [4]SizeEstimate
	ests := buf[:0]
	for _, operand := range operands {
		ests = append(ests, operand.estimate(ctx))
	}
	res := ests[0]
	for _, est := range ests[1:] {
		res = op(res, est)
	}
	return combineMeta(ctx, RangedSizeEstimate(res.Min, res.Max), operands, ests)
}

func minBounds(lhs, rhs SizeEstimate) SizeEstimate {
	return RangedSizeEstimate(min(lhs.Min, rhs.Min), min(lhs.Max, rhs.Max))
}

func maxBounds(lhs, rhs SizeEstimate) SizeEstimate {
	return RangedSizeEstimate(max(lhs.Min, rhs.Min), max(lhs.Max, rhs.Max))
}

// rangedBounds takes the upper bound from hi and the lower bound from lo, capped by the upper bound
// since a lower bound above it would describe an empty interval.
func rangedBounds(lo, hi SizeEstimate) SizeEstimate {
	return RangedSizeEstimate(min(lo.Min, hi.Max), hi.Max)
}

func (r rangedExpr) track(ctx TrackContext) uint64 {
	return r.maxExpr.track(ctx)
}

func (r rangedExpr) hasTarget() bool {
	return hasTarget(r.minExpr) || hasTarget(r.maxExpr)
}

// Ranged creates an expression where the lower bound is taken from minExpr and the upper bound from maxExpr.
func Ranged(minExpr, maxExpr QuantityExpr) QuantityExpr {
	return rangedExpr{minExpr: minExpr, maxExpr: maxExpr}
}

// AtMost creates an expression representing a quantity bounded between 0 and the upper bound of maxExpr.
func AtMost(maxExpr QuantityExpr) QuantityExpr {
	return Ranged(Const(0), maxExpr)
}

// atLeastOneExpr represents a quantity guaranteed to be at least 1.
type atLeastOneExpr struct {
	expr QuantityExpr
}

func (a atLeastOneExpr) estimate(ctx EstimateContext) SizeEstimate {
	return AtLeastOneSize(a.expr.estimate(ctx))
}

func (a atLeastOneExpr) track(ctx TrackContext) uint64 {
	val := a.expr.track(ctx)
	if val == 0 {
		return 1
	}
	return val
}

func (a atLeastOneExpr) hasTarget() bool {
	return hasTarget(a.expr)
}

// AtLeastOneQuantity creates an expression ensuring the quantity is at least 1.
func AtLeastOneQuantity(expr QuantityExpr) QuantityExpr {
	return atLeastOneExpr{expr: expr}
}

// StringScan creates an expression representing the cost of scanning a string.
func StringScan(expr QuantityExpr) QuantityExpr {
	return Scale(expr, StringTraversalCostFactor)
}

// ListAlloc creates an expression representing list allocation cost with base cost and scaled element count.
func ListAlloc(elemCount QuantityExpr, costFactor float64) QuantityExpr {
	return Sum(Const(ListCreateBaseCost), Scale(elemCount, costFactor))
}

// AllocList creates an expression representing the cost of a call that allocates a list:
// CallCost + ListCreateBaseCost plus the given work terms.
func AllocList(terms ...QuantityExpr) QuantityExpr {
	all := make([]QuantityExpr, 0, len(terms)+1)
	all = append(all, Const(CallCost+ListCreateBaseCost))
	all = append(all, terms...)
	return Sum(all...)
}

// Traversal creates an expression representing the traversal and optional allocation cost over an expression.
func Traversal(expr QuantityExpr, costFactor float64, allocCost uint64) QuantityExpr {
	if allocCost == 0 {
		return Scale(expr, costFactor)
	}
	return Sum(Const(allocCost), Scale(expr, costFactor))
}

// listExpr represents a list size estimate composed of length and element size expressions.
type listExpr struct {
	lenExpr  QuantityExpr
	elemExpr QuantityExpr
}

func (l listExpr) estimate(ctx EstimateContext) SizeEstimate {
	lenSize := l.lenExpr.estimate(ctx)
	elemSize := l.elemExpr.estimate(ctx)
	size := containerSize(ctx, lenSize, elemSize)
	size.Key = nil
	size.Elem = &elemSize
	return size
}

func (l listExpr) track(ctx TrackContext) uint64 {
	return l.lenExpr.track(ctx)
}

func (l listExpr) hasTarget() bool {
	return hasTarget(l.lenExpr) || hasTarget(l.elemExpr)
}

// List creates an expression representing a list with the given length and element size.
//
// The estimate is the size the sizing strategy assigns to such a list: its length under the default
// strategy, and 1 plus the length times the element size under the aggregate strategy. The track is
// the length.
func List(lenExpr, elemExpr QuantityExpr) QuantityExpr {
	return listExpr{lenExpr: lenExpr, elemExpr: elemExpr}
}

// mapExpr represents a map size estimate composed of size, key size, and value size expressions.
type mapExpr struct {
	sizeExpr QuantityExpr
	keyExpr  QuantityExpr
	valExpr  QuantityExpr
}

func (m mapExpr) estimate(ctx EstimateContext) SizeEstimate {
	mapSize := m.sizeExpr.estimate(ctx)
	keySize := m.keyExpr.estimate(ctx)
	valSize := m.valExpr.estimate(ctx)
	size := containerSize(ctx, mapSize, AtLeastOneSize(keySize).Add(AtLeastOneSize(valSize)))
	size.Key = &keySize
	size.Elem = &valSize
	return size
}

func (m mapExpr) track(ctx TrackContext) uint64 {
	return m.sizeExpr.track(ctx)
}

func (m mapExpr) hasTarget() bool {
	return hasTarget(m.sizeExpr) || hasTarget(m.keyExpr) || hasTarget(m.valExpr)
}

// Map creates an expression representing a map with the given entry count, key size and value size.
//
// The estimate is the size the sizing strategy assigns to such a map: its entry count under the
// default strategy, and 1 plus the entry count times the key and value sizes under the aggregate
// strategy. The track is the entry count.
func Map(sizeExpr, keyExpr, valExpr QuantityExpr) QuantityExpr {
	return mapExpr{sizeExpr: sizeExpr, keyExpr: keyExpr, valExpr: valExpr}
}

// optionalExpr represents the size of an optional which holds a value of the given size.
type optionalExpr struct {
	valueExpr QuantityExpr
}

func (o optionalExpr) estimate(ctx EstimateContext) SizeEstimate {
	if versionOf(ctx) < 1 {
		return FixedSizeEstimate(1)
	}
	return optionalSizeOf(o.valueExpr.estimate(ctx))
}

func (o optionalExpr) track(ctx TrackContext) uint64 {
	if trackVersionOf(ctx) < 1 {
		return 1
	}
	return SafeAdd(1, o.valueExpr.track(ctx))
}

func (o optionalExpr) hasTarget() bool {
	return hasTarget(o.valueExpr)
}

// Optional creates an expression representing an optional which holds a value of the given size.
//
// From cost model version 1 an optional is 1 plus the size of the value it holds, under every
// sizing strategy: 1 plus the value's length under the default strategy, and 1 plus its aggregate
// size under the aggregate strategy. An empty optional is 1, which Optional(Const(0)) describes.
// The value is the optional's element, which ElemOf reads, so that value() and orValue() recover its
// size. Version 0 sizes every optional as 1, as cel-go v0.32.0 did.
func Optional(valueExpr QuantityExpr) QuantityExpr {
	return optionalExpr{valueExpr: valueExpr}
}

// OverloadModel pairs an overload ID with its abstract cost and result size expressions.
//
// An OverloadModel is a value built by Overload or MemberOverload, and is not modified afterwards:
// estimators and trackers are compiled from it on request. Validate reports a model which cannot
// be priced, and cel.CostModel and Model call it when the model is registered.
type OverloadModel struct {
	ID              string
	Cost            QuantityExpr
	Size            QuantityExpr
	hasSize         bool
	IsMember        bool
	manual          bool
	manualEstimator bool
	// argumentsOnly marks a short-circuiting operator, whose cost is that of the arguments it
	// evaluates, and which is therefore exempt from the minimum call cost; see minCallCost.
	argumentsOnly bool
	estimator     FunctionEstimator
	tracker       FunctionTracker
	priors        []priorOverload
}

// priorOverload is the model which priced an overload at revisions earlier than before.
type priorOverload struct {
	before uint32
	model  OverloadModel
}

// priorForVersion selects the prior with the smallest before which still exceeds version, so
// that each prior covers the revisions from the previous prior's before up to its own.
func priorForVersion(priors []priorOverload, version uint32) (OverloadModel, bool) {
	var chosen *priorOverload
	for i, p := range priors {
		if version < p.before && (chosen == nil || p.before < chosen.before) {
			chosen = &priors[i]
		}
	}
	if chosen == nil {
		return OverloadModel{}, false
	}
	return chosen.model, true
}

// OverloadModelOption functional options for configuring an OverloadModel.
type OverloadModelOption func(*OverloadModel)

// EvalCost sets the evaluation cost expression for the overload model.
func EvalCost(cost QuantityExpr) OverloadModelOption {
	return func(m *OverloadModel) {
		m.Cost = cost
	}
}

// ResultSize sets the result size expression for the overload model. Passing nil in a
// BeforeVersion entry clears any result size inherited from the enclosing OverloadModel.
func ResultSize(size QuantityExpr) OverloadModelOption {
	return func(m *OverloadModel) {
		m.Size = size
		m.hasSize = true
	}
}

// ManualCost sets a custom FunctionEstimator and FunctionTracker for an overload whose cost or
// result size cannot be expressed declaratively with EvalCost and ResultSize.
//
// Both functions are required; Validate reports a nil one. ManualCost cannot be combined with
// EvalCost, ResultSize, or ManualEstimator on the same OverloadModel or BeforeVersion entry.
func ManualCost(estimator FunctionEstimator, tracker FunctionTracker) OverloadModelOption {
	return func(m *OverloadModel) {
		m.manual = true
		m.estimator = estimator
		m.tracker = tracker
	}
}

// ManualEstimator sets a custom FunctionEstimator for an overload (or BeforeVersion entry) whose
// static estimate cannot be expressed declaratively with EvalCost and ResultSize, while retaining
// the OverloadModel's declarative EvalCost for runtime tracking.
//
// ManualEstimator requires a non-nil FunctionEstimator and cannot be combined with ManualCost or
// ResultSize on the same OverloadModel or BeforeVersion entry.
func ManualEstimator(estimator FunctionEstimator) OverloadModelOption {
	return func(m *OverloadModel) {
		m.manualEstimator = true
		m.estimator = estimator
	}
}

// BeforeVersion records the options that priced the overload before the given model revision.
//
// At revisions earlier than version, the overload is configured by opts (such as EvalCost,
// ResultSize, ManualCost, or ManualEstimator), inheriting the enclosing OverloadModel's ID,
// receiver kind, and any unset EvalCost or ResultSize.
func BeforeVersion(version uint32, opts ...OverloadModelOption) OverloadModelOption {
	return func(m *OverloadModel) {
		m.priors = append(m.priors, priorOverload{
			before: version,
			model:  newOverloadModel(m.ID, m.IsMember, opts),
		})
	}
}

// resolvePrior returns prior with any unset Cost or Size inherited from m.
func (m OverloadModel) resolvePrior(prior OverloadModel) OverloadModel {
	if prior.manual {
		return prior
	}
	if prior.Cost == nil {
		prior.Cost = m.Cost
	}
	if !prior.manualEstimator && !prior.hasSize {
		prior.Size = m.Size
		prior.hasSize = m.hasSize
	}
	return prior
}

// Overload creates a global function OverloadModel with functional options.
func Overload(id string, opts ...OverloadModelOption) OverloadModel {
	return newOverloadModel(id, false, opts)
}

// MemberOverload creates a member function OverloadModel with functional options.
func MemberOverload(id string, opts ...OverloadModelOption) OverloadModel {
	return newOverloadModel(id, true, opts)
}

func newOverloadModel(id string, isMember bool, opts []OverloadModelOption) OverloadModel {
	m := OverloadModel{ID: id, IsMember: isMember}
	for _, opt := range opts {
		opt(&m)
	}
	return m
}

func (m OverloadModel) hasTarget() bool {
	return m.IsMember || hasTarget(m.Cost) || hasTarget(m.Size)
}

var estimatorCtxPool = sync.Pool{
	New: func() any {
		return &estimatorEvalContext{}
	},
}

var trackerCtxPool = sync.Pool{
	New: func() any {
		return &trackerEvalContext{}
	},
}

// FunctionEstimator returns a FunctionEstimator implementing the cost model.
func (m OverloadModel) FunctionEstimator() FunctionEstimator {
	return m.buildEstimator(defaultSizing, defaultModelVersion)
}

// FunctionEstimatorWithOptions returns a FunctionEstimator implementing the cost model, configured
// by the supplied options. Unset options take their defaults: the default SizingStrategy and the
// latest model revision.
func (m OverloadModel) FunctionEstimatorWithOptions(opts ...ModelOption) FunctionEstimator {
	resolved := newModelOptions(opts...)
	return m.buildEstimator(resolved.strategy, resolved.version)
}

func (m OverloadModel) buildEstimator(strategy SizingStrategy, version uint32) FunctionEstimator {
	if prior, ok := priorForVersion(m.priors, version); ok {
		return m.resolvePrior(prior).buildEstimator(strategy, version)
	}
	if m.manual || m.manualEstimator {
		return m.estimator
	}
	hasTarget := m.hasTarget()
	cost := normalize(m.Cost)
	size := normalize(m.Size)
	floor := m.hasCallFloor(version)
	return func(estimator Estimator, target *AstNode, args []AstNode) *CallEstimate {
		if hasTarget && target == nil {
			return nil
		}
		ctx := estimatorCtxPool.Get().(*estimatorEvalContext)
		if ve, ok := estimator.(*versionedEstimator); ok {
			ctx.coster = ve.coster
		}
		ctx.estimator = estimator
		ctx.strategy = strategy
		ctx.version = version
		ctx.target = target
		ctx.args = args
		ctx.hasTarget = hasTarget
		var resSize *SizeEstimate
		if size != nil {
			sz := aggregateResultSize(strategy, size.estimate(ctx))
			resSize = &sz
			// The result size is available to the cost through Result().
			ctx.result = resSize
		}
		costEst := cost.estimate(ctx)
		*ctx = estimatorEvalContext{}
		estimatorCtxPool.Put(ctx)
		if floor {
			costEst.Min, costEst.Max = max(costEst.Min, minCallCost), max(costEst.Max, minCallCost)
		}
		return NewCallEstimate(CostEstimate{Min: costEst.Min, Max: costEst.Max}, resSize)
	}
}

// FunctionTracker returns a FunctionTracker implementing the cost model.
func (m OverloadModel) FunctionTracker() FunctionTracker {
	return m.buildTracker(defaultSizing, defaultModelVersion)
}

// FunctionTrackerWithOptions returns a FunctionTracker implementing the cost model, configured by
// the supplied options.
//
// ModelVersion applies to tracking as it does to estimation, so that a call's estimate and its
// charge are computed by the same rules.
func (m OverloadModel) FunctionTrackerWithOptions(opts ...ModelOption) FunctionTracker {
	resolved := newModelOptions(opts...)
	return m.buildTracker(resolved.strategy, resolved.version)
}

func (m OverloadModel) buildTracker(strategy SizingStrategy, version uint32) FunctionTracker {
	if prior, ok := priorForVersion(m.priors, version); ok {
		return m.resolvePrior(prior).buildTracker(strategy, version)
	}
	if m.manual {
		return m.tracker
	}
	isMember := m.hasTarget()
	cost := normalize(m.Cost)
	floor := m.hasCallFloor(version)
	return func(args []ref.Val, result ref.Val) *uint64 {
		ctx := trackerCtxPool.Get().(*trackerEvalContext)
		ctx.strategy = strategy
		ctx.version = version
		ctx.args = args
		ctx.result = result
		ctx.isMember = isMember
		costVal := cost.track(ctx)
		*ctx = trackerEvalContext{}
		trackerCtxPool.Put(ctx)
		if floor {
			costVal = max(costVal, minCallCost)
		}
		return &costVal
	}
}

// versionOf reports the cost model version of an EstimateContext.
func versionOf(ctx EstimateContext) uint32 {
	if ctx == nil {
		return defaultModelVersion
	}
	if e, ok := ctx.(*estimatorEvalContext); ok && (e.estimator == nil || e.coster != nil) {
		return e.version
	}
	return ModelVersionOf(ctx.Estimator())
}

// estimatorEvalContext provides evaluation context for cost estimation.
type estimatorEvalContext struct {
	coster    *coster
	estimator Estimator
	strategy  SizingStrategy
	version   uint32
	target    *AstNode
	args      []AstNode
	hasTarget bool
	// result is the estimated result size, available to the cost through Result().
	result *SizeEstimate
}

// sizingStrategy implements sizingStrategyContext.
func (e *estimatorEvalContext) sizingStrategy() SizingStrategy {
	return e.strategy
}

// resultType implements resultTypeContext from the estimator, which knows the call being
// estimated.
func (e *estimatorEvalContext) resultType() (*types.Type, bool) {
	if rc, ok := e.estimator.(resultTypeContext); ok {
		return rc.resultType()
	}
	return nil, false
}

// operandNode implements operandNodeContext for Arg and Target operands.
func (e *estimatorEvalContext) operandNode(q QuantityExpr) (AstNode, bool) {
	switch x := q.(type) {
	case argExpr:
		if x.index >= 0 && x.index < len(e.args) {
			return e.args[x.index], true
		}
	case targetExpr:
		if e.target != nil {
			return *e.target, true
		}
	}
	return nil, false
}

// Arg returns the size estimate for the argument at index.
func (e *estimatorEvalContext) Arg(index int) (SizeEstimate, bool) {
	if index < len(e.args) {
		return e.Size(e.args[index]), true
	}
	return UnknownSizeEstimate(), false
}

// Target returns the size estimate for the receiver/target object.
func (e *estimatorEvalContext) Target() (SizeEstimate, bool) {
	if e.target != nil {
		return e.Size(*e.target), true
	}
	return UnknownSizeEstimate(), false
}

// Result returns the size estimate of the result, if the model's result size has been estimated.
func (e *estimatorEvalContext) Result() (SizeEstimate, bool) {
	if e.result != nil {
		return *e.result, true
	}
	return UnknownSizeEstimate(), false
}

// Estimator returns the underlying Estimator.
func (e *estimatorEvalContext) Estimator() Estimator {
	return e.estimator
}

// Size returns the estimated size of an AST node.
func (e *estimatorEvalContext) Size(node AstNode) SizeEstimate {
	if node == nil {
		return UnknownSizeEstimate()
	}
	if sz := node.ComputedSize(); sz != nil {
		return *sz
	}
	if e.coster != nil && node.Expr() != nil {
		if sz := e.coster.computeSize(node.Expr()); sz != nil {
			return *sz
		}
	}
	if sz, ok := e.strategy.EstimateSize(e, node); ok {
		return sz
	}
	if e.estimator != nil {
		if sz := e.estimator.EstimateSize(node); sz != nil {
			return *sz
		}
	}
	return UnknownSizeEstimate()
}

// TargetType returns the type of the receiver/target object.
func (e *estimatorEvalContext) TargetType() (*types.Type, bool) {
	if e.target != nil && (*e.target) != nil {
		return (*e.target).Type(), true
	}
	return nil, false
}

// ArgType returns the type of the argument at index.
func (e *estimatorEvalContext) ArgType(index int) (*types.Type, bool) {
	if index < len(e.args) && e.args[index] != nil {
		return e.args[index].Type(), true
	}
	return nil, false
}

// ArgValue returns the uint64 value of the argument at index, or defaultVal if not a literal int/uint.
func (e *estimatorEvalContext) ArgValue(index int, defaultVal uint64) uint64 {
	if index < len(e.args) && e.args[index] != nil {
		return NodeAsUintValue(e.args[index], defaultVal)
	}
	return defaultVal
}

// TargetValue returns the uint64 value of the receiver/target, or defaultVal if not a literal int/uint.
func (e *estimatorEvalContext) TargetValue(defaultVal uint64) uint64 {
	if e.target != nil && (*e.target) != nil {
		return NodeAsUintValue(*e.target, defaultVal)
	}
	return defaultVal
}

// trackerEvalContext provides evaluation context for runtime cost tracking.
type trackerEvalContext struct {
	estimator ActualCostEstimator
	strategy  SizingStrategy
	version   uint32
	args      []ref.Val
	result    ref.Val
	isMember  bool
}

// valueAsUint returns the non-negative uint64 value of ref.Val (int or uint), or defaultVal.
func valueAsUint(val ref.Val, defaultVal uint64) uint64 {
	if val == nil {
		return defaultVal
	}
	switch v := val.(type) {
	case types.Int:
		if v < 0 {
			return 0
		}
		return uint64(v)
	case types.Uint:
		return uint64(v)
	default:
		return defaultVal
	}
}

// ArgValue returns the uint64 value of the argument at index, or defaultVal.
func (t *trackerEvalContext) ArgValue(index int, defaultVal uint64) uint64 {
	idx := index
	if t.isMember {
		idx = index + 1
	}
	if idx < len(t.args) {
		return valueAsUint(t.args[idx], defaultVal)
	}
	return defaultVal
}

// TargetValue returns the uint64 value of the receiver/target object, or defaultVal.
func (t *trackerEvalContext) TargetValue(defaultVal uint64) uint64 {
	if t.isMember && len(t.args) > 0 {
		return valueAsUint(t.args[0], defaultVal)
	}
	return defaultVal
}

// TargetType returns the type of the receiver/target object.
func (t *trackerEvalContext) TargetType() (*types.Type, bool) {
	if t.isMember && len(t.args) > 0 && t.args[0] != nil {
		if tp, ok := t.args[0].Type().(*types.Type); ok {
			return tp, true
		}
	}
	return nil, false
}

// ArgType returns the type of the argument at index.
func (t *trackerEvalContext) ArgType(index int) (*types.Type, bool) {
	idx := index
	if t.isMember {
		idx = index + 1
	}
	if idx < len(t.args) && t.args[idx] != nil {
		if tp, ok := t.args[idx].Type().(*types.Type); ok {
			return tp, true
		}
	}
	return nil, false
}

// Estimator returns the runtime ActualCostEstimator.
func (t *trackerEvalContext) Estimator() ActualCostEstimator {
	return t.estimator
}

// Size returns the actual runtime size of a value.
func (t *trackerEvalContext) Size(value ref.Val) uint64 {
	if value == nil {
		return 0
	}
	// Inlines defaultSizingStrategy.TrackSize, avoiding an interface call and boxing the context.
	if t.strategy == defaultSizing {
		if opt, isOpt := value.(*types.Optional); isOpt {
			return optionalLength(t.version, opt)
		}
		return ActualSize(value)
	}
	if sz, ok := t.strategy.TrackSize(t, value); ok {
		return sz
	}
	return ActualSize(value)
}

// ArgVal returns the value of the argument at index.
func (t *trackerEvalContext) ArgVal(index int) (ref.Val, bool) {
	idx := index
	if t.isMember {
		idx = index + 1
	}
	if idx < len(t.args) && t.args[idx] != nil {
		return t.args[idx], true
	}
	return nil, false
}

// TargetVal returns the value of the receiver/target object.
func (t *trackerEvalContext) TargetVal() (ref.Val, bool) {
	if t.isMember && len(t.args) > 0 && t.args[0] != nil {
		return t.args[0], true
	}
	return nil, false
}

// ResultVal returns the evaluated result value.
func (t *trackerEvalContext) ResultVal() (ref.Val, bool) {
	if t.result != nil {
		return t.result, true
	}
	return nil, false
}

// Arg returns the runtime size of the argument at index.
func (t *trackerEvalContext) Arg(index int) uint64 {
	if val, ok := t.ArgVal(index); ok {
		return t.Size(val)
	}
	return 0
}

// Target returns the runtime size of the receiver/target object.
func (t *trackerEvalContext) Target() uint64 {
	if val, ok := t.TargetVal(); ok {
		return t.Size(val)
	}
	return 0
}

// Result returns the runtime size of the result.
func (t *trackerEvalContext) Result() uint64 {
	if val, ok := t.ResultVal(); ok {
		return t.Size(val)
	}
	return 0
}

// modelVersion implements trackVersionContext.
func (t *trackerEvalContext) modelVersion() uint32 {
	return t.version
}

// minCallCost is the least a call priced by an OverloadModel costs from cost model version 1, both
// estimated and tracked: a call does at least a unit of work, even over empty arguments, as
// comparing two empty strings does. The short-circuiting operators, ||, && and ?:, and the optional
// or() and orValue(), are exempt, as their cost is that of the arguments they evaluate.
//
// Version 0 reproduces cel-go v0.32.0, which estimated and charged such calls 0.
const minCallCost = uint64(1)

// hasCallFloor reports whether the model's declarative cost is raised to minCallCost at version.
func (m OverloadModel) hasCallFloor(version uint32) bool {
	return version >= 1 && !m.argumentsOnly
}

// argumentsOnly marks the model of a short-circuiting operator, which is exempt from minCallCost.
func argumentsOnly() OverloadModelOption {
	return func(m *OverloadModel) {
		m.argumentsOnly = true
	}
}
