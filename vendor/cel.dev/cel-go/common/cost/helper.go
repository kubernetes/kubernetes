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
	"math/bits"

	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// ContainerSizing describes what the size of a container measures under a sizing strategy.
type ContainerSizing int

const (
	// SizedByLength means a container's size is its length, as under the default sizing strategy.
	SizedByLength ContainerSizing = iota
	// SizedByContent means a container's size is 1 plus the sizes of its entries, as under the
	// aggregate sizing strategy.
	SizedByContent
	// SizedOpaque means the relationship between a container's size and its contents is not known,
	// as under a custom sizing strategy.
	SizedOpaque
)

// ContainerSizingOf reports what a container's size measures for the given Estimator.
//
// A hand-written FunctionEstimator which builds a result's size from lengths, rather than from
// the sizes of its operands, uses it to report the size in the units of the sizing strategy. For
// an Estimator which carries no strategy, sizes are lengths. Prefer describing an overload by an
// OverloadModel, whose List and Map follow the strategy themselves.
func ContainerSizingOf(estimator Estimator) ContainerSizing {
	s, ok := estimator.(sizingStrategyContext)
	if !ok {
		return SizedByLength
	}
	switch s.sizingStrategy().(type) {
	case defaultSizingStrategy:
		return SizedByLength
	case *aggregateSizingStrategy:
		return SizedByContent
	default:
		return SizedOpaque
	}
}

// sizingStrategyContext is implemented by EstimateContext values which expose their sizing
// strategy. It is unexported because EstimateContext is an interface third parties may implement.
type sizingStrategyContext interface {
	sizingStrategy() SizingStrategy
}

// strategyOf returns the sizing strategy of a context. A context which does not report one, such
// as a third-party EstimateContext, is sized by the default strategy, whose sizes are lengths.
func strategyOf(ctx EstimateContext) SizingStrategy {
	if s, ok := ctx.(sizingStrategyContext); ok {
		return s.sizingStrategy()
	}
	return defaultSizing
}

// resultTypeContext is implemented by EstimateContext values which know the checked type of the
// call's result. It is unexported because EstimateContext is an interface third parties may
// implement.
type resultTypeContext interface {
	resultType() (*types.Type, bool)
}

// operandKind returns the kind of the checked type of an Arg, Target or Result operand.
func operandKind(ctx EstimateContext, e QuantityExpr) (types.Kind, bool) {
	var t *types.Type
	var ok bool
	switch x := e.(type) {
	case argExpr:
		t, ok = ctx.ArgType(x.index)
	case targetExpr:
		t, ok = ctx.TargetType()
	case resultExpr:
		if rc, isRC := ctx.(resultTypeContext); isRC {
			t, ok = rc.resultType()
		}
	}
	if !ok || t == nil {
		return types.UnspecifiedKind, false
	}
	switch t.Kind() {
	case types.DynKind, types.AnyKind, types.TypeParamKind, types.UnspecifiedKind, types.ErrorKind:
		return types.UnspecifiedKind, false
	}
	return t.Kind(), true
}

// operandNodeContext is implemented by EstimateContext values which know the AST nodes of the
// call being estimated. It is unexported because EstimateContext is an interface third parties may
// implement.
type operandNodeContext interface {
	operandNode(e QuantityExpr) (AstNode, bool)
}

// operandLiteral returns the value of an Arg or Target operand which is a literal.
func operandLiteral(ctx EstimateContext, e QuantityExpr) (ref.Val, bool) {
	oc, ok := ctx.(operandNodeContext)
	if !ok {
		return nil, false
	}
	node, ok := oc.operandNode(e)
	if !ok || node == nil || node.Expr() == nil || node.Expr().Kind() != ast.LiteralKind {
		return nil, false
	}
	return node.Expr().AsLiteral(), true
}

// hasUnitLength reports whether every value of a kind has length 1: those which are neither
// strings, bytes nor containers.
func hasUnitLength(k types.Kind) bool {
	switch k {
	case types.BoolKind, types.DoubleKind, types.DurationKind, types.IntKind, types.NullTypeKind,
		types.TimestampKind, types.UintKind:
		return true
	default:
		return false
	}
}

// totalEstimate bounds the summed size of one part of a container, its elements or its keys, from
// the container's size estimate sz and the estimated size of one part. whole reports whether the
// part is the container's entire content, as the elements of a list are, rather than a share of
// it, as the keys or values of a map are.
//
//   - Default strategy: a container's size is its length, so the total is the length multiplied
//     by the part size.
//   - Aggregate strategy: a container's size is 1 plus the sizes of its entries, so every part
//     totals at most the content size, sz-1. The content size is also the lower bound when the
//     part is the whole content; otherwise the lower bound is 0.
//   - Other strategies define no relationship between a container's size and its contents, so the
//     total is unknown.
//
// A value which the checker typed as having no elements, such as a string, and a container which
// may not hold anything, total 0. A container which may hold something, but whose part size is not
// stated, has an unknown total under the default strategy.
func totalEstimate(ctx EstimateContext, operand QuantityExpr, sz SizeEstimate, part *SizeEstimate, whole bool) SizeEstimate {
	if !carriesElements(ctx, operand) || !mayHoldElements(ctx, sz) {
		return FixedSizeEstimate(0)
	}
	switch strategyOf(ctx).(type) {
	case defaultSizingStrategy:
		if part == nil {
			return UnknownSizeEstimate()
		}
		total := RangedSizeEstimate(sz.Min, sz.Max).Multiply(*part)
		return RangedSizeEstimate(total.Min, total.Max)
	case *aggregateSizingStrategy:
		hi := SafeSubtract(sz.Max, 1)
		if sz.Max == math.MaxUint64 {
			hi = math.MaxUint64
		}
		lo := uint64(0)
		if whole {
			lo = SafeSubtract(sz.Min, 1)
		}
		return RangedSizeEstimate(min(lo, hi), hi)
	default:
		return UnknownSizeEstimate()
	}
}

// isListOperand reports whether the operand of a total is a list, whose elements are its entire
// content, rather than a map, whose values are only part of it. The checked type decides when it is
// known; otherwise the metadata does, since a list states an element size but no key size.
func isListOperand(ctx EstimateContext, operand QuantityExpr, sz SizeEstimate) bool {
	if kind, known := operandKind(ctx, operand); known {
		return kind == types.ListKind
	}
	return sz.Elem != nil && sz.Key == nil
}

// countBounds bounds the element count of a container from its size estimate under a strategy.
// It bounds comprehension iteration counts; see costComprehension.
//
// The default strategy sizes a container by its length, so the count is the size itself. The
// aggregate strategy sizes a container as 1 plus the sizes of its entries, each of which is at
// least 1, so the count is at least the content size divided by the largest entry. Its upper bound
// is the size itself rather than the content size divided by the smallest entry, so the bound
// remains sound for a size which a custom estimator reports as a length. Other strategies make no
// guarantee, so the count is unknown.
func countBounds(strategy SizingStrategy, sz SizeEstimate) SizeEstimate {
	switch strategy.(type) {
	case defaultSizingStrategy:
		return RangedSizeEstimate(sz.Min, sz.Max)
	case *aggregateSizingStrategy:
		entry := aggregateEntrySize(sz)
		lo := uint64(0)
		if sz.Min > 1 && entry.Max != math.MaxUint64 {
			content := sz.Min - 1
			lo = content / entry.Max
			if content%entry.Max != 0 {
				lo++
			}
		}
		return RangedSizeEstimate(min(lo, sz.Max), sz.Max)
	default:
		return UnknownSizeEstimate()
	}
}

// aggregateEntrySize bounds the aggregate size of one entry of a container: an element of a list,
// or a key and value of a map. Every value has aggregate size at least 1.
func aggregateEntrySize(sz SizeEstimate) SizeEstimate {
	elem := RangedSizeEstimate(1, math.MaxUint64)
	if sz.Elem != nil {
		elem = AtLeastOneSize(RangedSizeEstimate(sz.Elem.Min, sz.Elem.Max))
	}
	if sz.Key == nil {
		return elem
	}
	key := AtLeastOneSize(RangedSizeEstimate(sz.Key.Min, sz.Key.Max))
	sum := key.Add(elem)
	return RangedSizeEstimate(sum.Min, sum.Max)
}

// rational is a non-negative scale factor p/q. A zero q marks a factor with no exact rational form
// within the supported range, which is applied in floating point.
type rational struct {
	p, q uint64
}

// maxRationalDenominator bounds the denominator of an exact scale factor so that products of
// factors, and of a factor with a quantity, can be computed in 128-bit arithmetic.
const maxRationalDenominator = uint64(1) << 32

// rationalOf returns the simplest rational p/q whose float64 value is f, found from the continued
// fraction expansion of f. A decimal factor such as 0.1 is therefore applied as exactly 1/10, which
// is what the author wrote, rather than as the binary value nearest to it.
//
// Factors which are not finite and positive, or whose simplest form needs a denominator above
// maxRationalDenominator, report q == 0.
func rationalOf(f float64) rational {
	if math.IsNaN(f) || math.IsInf(f, 0) || f <= 0 || f >= math.MaxUint64 {
		return rational{}
	}
	// Convergents h/k of the continued fraction of f.
	hPrev, h := uint64(0), uint64(1)
	kPrev, k := uint64(1), uint64(0)
	x := f
	for i := 0; i < 64; i++ {
		a := math.Floor(x)
		if a >= math.MaxUint64 {
			return rational{}
		}
		ai := uint64(a)
		hiH, loH := bits.Mul64(ai, h)
		hiK, loK := bits.Mul64(ai, k)
		if hiH != 0 || hiK != 0 {
			return rational{}
		}
		hNext, carryH := bits.Add64(loH, hPrev, 0)
		kNext, carryK := bits.Add64(loK, kPrev, 0)
		if carryH != 0 || carryK != 0 || kNext > maxRationalDenominator {
			return rational{}
		}
		hPrev, h = h, hNext
		kPrev, k = k, kNext
		if float64(h)/float64(k) == f {
			return rational{p: h, q: k}
		}
		frac := x - a
		if frac == 0 {
			return rational{}
		}
		x = 1 / frac
	}
	return rational{}
}

// mul returns the product of two rationals in lowest terms, and false if the product does not fit
// within the supported range.
func (r rational) mul(o rational) (rational, bool) {
	if r.q == 0 || o.q == 0 {
		return rational{}, false
	}
	// Cross-reduce before multiplying to keep the terms small.
	g1, g2 := gcd(r.p, o.q), gcd(o.p, r.q)
	p1, q2 := r.p/g1, o.q/g1
	p2, q1 := o.p/g2, r.q/g2
	hiP, p := bits.Mul64(p1, p2)
	hiQ, q := bits.Mul64(q1, q2)
	if hiP != 0 || hiQ != 0 || q > maxRationalDenominator {
		return rational{}, false
	}
	return rational{p: p, q: q}, true
}

func gcd(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	if a == 0 {
		return 1
	}
	return a
}

// scale computes ⌈x · f⌉:
//
//   - the factor is applied as its exact rational form when it has one (see rationalOf), so the
//     result is exact for every input, including those above 2^53;
//   - an unbounded input stays unbounded for any positive factor;
//   - a zero input, or a zero or negative factor, yields zero; a NaN factor yields ⊤, so that an
//     invalid factor cannot make a quantity free.
func scale(x uint64, f float64, r rational) uint64 {
	switch {
	case math.IsNaN(f):
		return math.MaxUint64
	case f <= 0 || x == 0:
		return 0
	case x == math.MaxUint64 || math.IsInf(f, 1):
		return math.MaxUint64
	case r.q == 0:
		return SafeMultiplyByFactor(x, f)
	}
	hi, lo := bits.Mul64(x, r.p)
	if hi >= r.q {
		return math.MaxUint64
	}
	quo, rem := bits.Div64(hi, lo, r.q)
	if rem != 0 {
		if quo == math.MaxUint64 {
			return math.MaxUint64
		}
		quo++
	}
	return quo
}

// normalize moves scale factors outward so that each product and each uniformly scaled sum is
// rounded once:
//
//	Scale(Scale(a, f), g)            -> Scale(a, f·g)
//	Mul(Scale(a, f), Scale(b, g), c) -> Scale(Mul(a, b, c), f·g)
//	Sum(Scale(a, f), Scale(b, f))    -> Scale(Sum(a, b), f)
//
// Factors are combined as exact rationals. A combination which has no exact form within the
// supported range is left in place.
func normalize(e QuantityExpr) QuantityExpr {
	if e == nil {
		return nil
	}
	switch x := e.(type) {
	case scaleExpr:
		inner := normalize(x.expr)
		if s, ok := inner.(scaleExpr); ok {
			if r, ok := x.rat.mul(s.rat); ok {
				return scaleExpr{expr: s.expr, factor: x.factor * s.factor, rat: r}
			}
		}
		return scaleExpr{expr: inner, factor: x.factor, rat: x.rat}
	case mulExpr:
		terms := make([]QuantityExpr, 0, len(x.terms))
		for _, t := range x.terms {
			t = normalize(t)
			if m, ok := t.(mulExpr); ok {
				terms = append(terms, m.terms...)
				continue
			}
			terms = append(terms, t)
		}
		factor, rat, hoisted := 1.0, rational{p: 1, q: 1}, false
		plain := make([]QuantityExpr, 0, len(terms))
		for _, t := range terms {
			if s, ok := t.(scaleExpr); ok {
				if r, ok := rat.mul(s.rat); ok {
					factor, rat, hoisted = factor*s.factor, r, true
					plain = append(plain, s.expr)
					continue
				}
			}
			plain = append(plain, t)
		}
		if hoisted {
			return scaleExpr{expr: mulExpr{terms: plain}, factor: factor, rat: rat}
		}
		return mulExpr{terms: plain}
	case addExpr:
		terms := make([]QuantityExpr, len(x.terms))
		for i, t := range x.terms {
			terms[i] = normalize(t)
		}
		if len(terms) > 1 {
			if first, ok := terms[0].(scaleExpr); ok && first.rat.q != 0 {
				inner := make([]QuantityExpr, len(terms))
				same := true
				for i, t := range terms {
					s, ok := t.(scaleExpr)
					if !ok || s.rat != first.rat {
						same = false
						break
					}
					inner[i] = s.expr
				}
				if same {
					return scaleExpr{expr: addExpr{terms: inner}, factor: first.factor, rat: first.rat}
				}
			}
		}
		return addExpr{terms: terms}
	case subExpr:
		return subExpr{lhs: normalize(x.lhs), rhs: normalize(x.rhs)}
	case minExpr:
		return minExpr{lhs: normalize(x.lhs), rhs: normalize(x.rhs)}
	case maxExpr:
		return maxExpr{lhs: normalize(x.lhs), rhs: normalize(x.rhs)}
	case unionExpr:
		terms := make([]QuantityExpr, len(x.terms))
		for i, t := range x.terms {
			terms[i] = normalize(t)
		}
		return unionExpr{terms: terms}
	case rangedExpr:
		return rangedExpr{minExpr: normalize(x.minExpr), maxExpr: normalize(x.maxExpr)}
	case atLeastOneExpr:
		return atLeastOneExpr{expr: normalize(x.expr)}
	case listExpr:
		return listExpr{lenExpr: normalize(x.lenExpr), elemExpr: normalize(x.elemExpr)}
	case mapExpr:
		return mapExpr{sizeExpr: normalize(x.sizeExpr), keyExpr: normalize(x.keyExpr), valExpr: normalize(x.valExpr)}
	case optionalExpr:
		return optionalExpr{valueExpr: normalize(x.valueExpr)}
	default:
		// Atoms and projections contain no scale factors.
		return e
	}
}

// carriesElements reports whether an expression may denote a container, and so whether an absent
// element or key size on its estimate means "unknown" rather than "none".
//
// Constants, integer values and totals are scalars. Arg, Target and Result may be containers,
// unless the checker typed the argument or receiver as something which has no elements, such as a
// string. The element and key projections may be containers, as may any combinator with such an
// operand.
func carriesElements(ctx EstimateContext, e QuantityExpr) bool {
	switch x := e.(type) {
	case argExpr:
		t, ok := ctx.ArgType(x.index)
		return !ok || mayBeContainerType(t)
	case targetExpr:
		t, ok := ctx.TargetType()
		return !ok || mayBeContainerType(t)
	case resultExpr, elemExpr, keyExpr, listExpr, mapExpr, optionalExpr:
		return true
	case subExpr:
		return carriesElements(ctx, x.lhs)
	case scaleExpr:
		return carriesElements(ctx, x.expr)
	case atLeastOneExpr:
		return carriesElements(ctx, x.expr)
	case minExpr:
		return carriesElements(ctx, x.lhs) || carriesElements(ctx, x.rhs)
	case maxExpr:
		return carriesElements(ctx, x.lhs) || carriesElements(ctx, x.rhs)
	case rangedExpr:
		return carriesElements(ctx, x.minExpr) || carriesElements(ctx, x.maxExpr)
	case addExpr:
		return anyCarriesElements(ctx, x.terms)
	case mulExpr:
		return anyCarriesElements(ctx, x.terms)
	case unionExpr:
		return anyCarriesElements(ctx, x.terms)
	default:
		return false
	}
}

func anyCarriesElements(ctx EstimateContext, terms []QuantityExpr) bool {
	for _, t := range terms {
		if carriesElements(ctx, t) {
			return true
		}
	}
	return false
}

// mayBeContainerType reports whether a value of the checked type may hold elements or keys. Only
// the kinds which cannot, such as strings, bytes and numbers, are excluded; an unknown type may be
// anything.
func mayBeContainerType(t *types.Type) bool {
	if t == nil {
		return true
	}
	k := t.Kind()
	return !hasUnitLength(k) && k != types.StringKind && k != types.BytesKind
}

// mergeMeta returns the union of the operands' key and element sizes.
//
// Metadata is attached only when some operand states it. When one does, an operand which may be a
// non-empty container but states no metadata contributes an unknown size, so the union cannot be
// narrower than what that operand may hold. Scalar operands, and empty containers, contribute
// nothing.
func mergeMeta(ctx EstimateContext, exprs []QuantityExpr, ests []SizeEstimate) (key, elem *SizeEstimate) {
	anyKey, anyElem := false, false
	for _, s := range ests {
		anyKey = anyKey || s.Key != nil
		anyElem = anyElem || s.Elem != nil
	}
	if !anyKey && !anyElem {
		return nil, nil
	}
	for i, s := range ests {
		// An operand which may be a non-empty container but states nothing about its contents may
		// hold anything. One which states an element size but no key size is a list, which has no
		// keys.
		nonEmpty := mayHoldElements(ctx, s)
		unstated := nonEmpty && s.Key == nil && s.Elem == nil && carriesElements(ctx, exprs[i])
		if anyElem {
			e := s.Elem
			if e == nil && (unstated || (s.Key != nil && nonEmpty)) {
				e = &unknownSizeEstimate
			}
			elem = mergeSizeEstimatePtr(elem, e)
		}
		if anyKey {
			k := s.Key
			if k == nil && unstated {
				k = &unknownSizeEstimate
			}
			key = mergeSizeEstimatePtr(key, k)
		}
	}
	return key, elem
}

// combineMeta attaches the operands' metadata, as mergeMeta merges it, to a combined estimate.
func combineMeta(ctx EstimateContext, size SizeEstimate, exprs []QuantityExpr, ests []SizeEstimate) SizeEstimate {
	size.Key, size.Elem = mergeMeta(ctx, exprs, ests)
	return size
}

// mayHoldElements reports whether a container with the given size estimate may be non-empty under
// the context's sizing strategy. An empty container has size 0 under the default strategy and size
// 1 under the aggregate strategy.
func mayHoldElements(ctx EstimateContext, s SizeEstimate) bool {
	if _, aggregate := strategyOf(ctx).(*aggregateSizingStrategy); aggregate {
		return s.Max > 1
	}
	return s.Max > 0
}

// aggregateResultSize raises the upper bound of a result size estimated under the aggregate
// strategy to at least 1. That strategy sizes an empty string, bytes value, list or map as 1, so a
// model whose result may be empty, such as a substring which drops the whole receiver, could
// otherwise report an upper bound of 0. Raising an upper bound is sound for any value.
func aggregateResultSize(strategy SizingStrategy, sz SizeEstimate) SizeEstimate {
	if _, aggregate := strategy.(*aggregateSizingStrategy); aggregate && sz.Max == 0 {
		sz.Max = 1
	}
	return sz
}

// containerSize returns the size the context's sizing strategy assigns to a container of n
// entries, each of the given size; see containerSizeFor.
func containerSize(ctx EstimateContext, n, entry SizeEstimate) SizeEstimate {
	return containerSizeFor(strategyOf(ctx), n, entry)
}

// containerSizeFor returns the size a strategy assigns to a container of n entries, each of the
// given size.
//
// The default strategy sizes a container by its length. The aggregate strategy sizes it as 1 plus
// the sizes of its entries, each of which is at least 1. Other strategies define no relationship,
// so the size is unknown.
func containerSizeFor(strategy SizingStrategy, n, entry SizeEstimate) SizeEstimate {
	switch strategy.(type) {
	case defaultSizingStrategy:
		return RangedSizeEstimate(n.Min, n.Max)
	case *aggregateSizingStrategy:
		e := AtLeastOneSize(entry)
		return RangedSizeEstimate(
			SafeAdd(1, SafeMultiply(n.Min, e.Min)),
			SafeAdd(1, SafeMultiply(n.Max, e.Max)),
		)
	default:
		return UnknownSizeEstimate()
	}
}
