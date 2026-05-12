// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
	"strings"
)

//  atom, set, epsilon, action, predicate, rule transitions.
//
//  <p>This is a one way link.  It emanates from a state (usually via a list of
//  transitions) and has a target state.</p>
//
//  <p>Since we never have to change the ATN transitions once we construct it,
//  the states. We'll use the term Edge for the DFA to distinguish them from
//  ATN transitions.</p>

type Transition interface {
	getTarget() ATNState
	setTarget(ATNState)
	getIsEpsilon() bool
	getLabel() *IntervalSet
	getSerializationType() int
	Matches(int, int, int) bool
}

type BaseTransition struct {
	target            ATNState
	isEpsilon         bool
	label             int
	intervalSet       *IntervalSet
	serializationType int
}

func NewBaseTransition(target ATNState) *BaseTransition {

	if target == nil {
		panic("target cannot be nil.")
	}

	t := new(BaseTransition)

	t.target = target
	// Are we epsilon, action, sempred?
	t.isEpsilon = false
	t.intervalSet = nil

	return t
}

func (t *BaseTransition) getTarget() ATNState {
	return t.target
}

func (t *BaseTransition) setTarget(s ATNState) {
	t.target = s
}

func (t *BaseTransition) getIsEpsilon() bool {
	return t.isEpsilon
}

func (t *BaseTransition) getLabel() *IntervalSet {
	return t.intervalSet
}

func (t *BaseTransition) getSerializationType() int {
	return t.serializationType
}

func (t *BaseTransition) Matches(_, _, _ int) bool {
	panic("Not implemented")
}

const (
	TransitionEPSILON    = 1
	TransitionRANGE      = 2
	TransitionRULE       = 3
	TransitionPREDICATE  = 4 // e.g., {isType(input.LT(1))}?
	TransitionATOM       = 5
	TransitionACTION     = 6
	TransitionSET        = 7 // ~(A|B) or ~atom, wildcard, which convert to next 2
	TransitionNOTSET     = 8
	TransitionWILDCARD   = 9
	TransitionPRECEDENCE = 10
)

//goland:noinspection GoUnusedGlobalVariable
var TransitionserializationNames = []string{
	"INVALID",
	"EPSILON",
	"RANGE",
	"RULE",
	"PREDICATE",
	"ATOM",
	"ACTION",
	"SET",
	"NOT_SET",
	"WILDCARD",
	"PRECEDENCE",
}

//var TransitionserializationTypes struct {
//	EpsilonTransition int
//	RangeTransition int
//	RuleTransition int
//	PredicateTransition int
//	AtomTransition int
//	ActionTransition int
//	SetTransition int
//	NotSetTransition int
//	WildcardTransition int
//	PrecedencePredicateTransition int
//}{
//	TransitionEPSILON,
//	TransitionRANGE,
//	TransitionRULE,
//	TransitionPREDICATE,
//	TransitionATOM,
//	TransitionACTION,
//	TransitionSET,
//	TransitionNOTSET,
//	TransitionWILDCARD,
//	TransitionPRECEDENCE
//}

// AtomTransition
// TODO: make all transitions sets? no, should remove set edges
type AtomTransition struct {
	BaseTransition
}

func NewAtomTransition(target ATNState, intervalSet int) *AtomTransition {
	t := &AtomTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionATOM,
			label:             intervalSet,
			isEpsilon:         false,
		},
	}
	t.intervalSet = t.makeLabel()

	return t
}

func (t *AtomTransition) makeLabel() *IntervalSet {
	s := NewIntervalSet()
	s.addOne(t.label)
	return s
}

func (t *AtomTransition) Matches(symbol, _, _ int) bool {
	return t.label == symbol
}

func (t *AtomTransition) String() string {
	return strconv.Itoa(t.label)
}

type RuleTransition struct {
	BaseTransition
	followState           ATNState
	ruleIndex, precedence int
}

func NewRuleTransition(ruleStart ATNState, ruleIndex, precedence int, followState ATNState) *RuleTransition {
	return &RuleTransition{
		BaseTransition: BaseTransition{
			target:            ruleStart,
			isEpsilon:         true,
			serializationType: TransitionRULE,
		},
		ruleIndex:   ruleIndex,
		precedence:  precedence,
		followState: followState,
	}
}

func (t *RuleTransition) Matches(_, _, _ int) bool {
	return false
}

type EpsilonTransition struct {
	BaseTransition
	outermostPrecedenceReturn int
}

func NewEpsilonTransition(target ATNState, outermostPrecedenceReturn int) *EpsilonTransition {
	return &EpsilonTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionEPSILON,
			isEpsilon:         true,
		},
		outermostPrecedenceReturn: outermostPrecedenceReturn,
	}
}

func (t *EpsilonTransition) Matches(_, _, _ int) bool {
	return false
}

func (t *EpsilonTransition) String() string {
	return "epsilon"
}

type RangeTransition struct {
	BaseTransition
	start, stop int
}

func NewRangeTransition(target ATNState, start, stop int) *RangeTransition {
	t := &RangeTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionRANGE,
			isEpsilon:         false,
		},
		start: start,
		stop:  stop,
	}
	t.intervalSet = t.makeLabel()
	return t
}

func (t *RangeTransition) makeLabel() *IntervalSet {
	s := NewIntervalSet()
	s.addRange(t.start, t.stop)
	return s
}

func (t *RangeTransition) Matches(symbol, _, _ int) bool {
	return symbol >= t.start && symbol <= t.stop
}

func (t *RangeTransition) String() string {
	var sb strings.Builder
	sb.WriteByte('\'')
	sb.WriteRune(rune(t.start))
	sb.WriteString("'..'")
	sb.WriteRune(rune(t.stop))
	sb.WriteByte('\'')
	return sb.String()
}

type AbstractPredicateTransition interface {
	Transition
	IAbstractPredicateTransitionFoo()
}

type BaseAbstractPredicateTransition struct {
	BaseTransition
}

func NewBasePredicateTransition(target ATNState) *BaseAbstractPredicateTransition {
	return &BaseAbstractPredicateTransition{
		BaseTransition: BaseTransition{
			target: target,
		},
	}
}

func (a *BaseAbstractPredicateTransition) IAbstractPredicateTransitionFoo() {}

type PredicateTransition struct {
	BaseAbstractPredicateTransition
	isCtxDependent       bool
	ruleIndex, predIndex int
}

func NewPredicateTransition(target ATNState, ruleIndex, predIndex int, isCtxDependent bool) *PredicateTransition {
	return &PredicateTransition{
		BaseAbstractPredicateTransition: BaseAbstractPredicateTransition{
			BaseTransition: BaseTransition{
				target:            target,
				serializationType: TransitionPREDICATE,
				isEpsilon:         true,
			},
		},
		isCtxDependent: isCtxDependent,
		ruleIndex:      ruleIndex,
		predIndex:      predIndex,
	}
}

func (t *PredicateTransition) Matches(_, _, _ int) bool {
	return false
}

func (t *PredicateTransition) getPredicate() *Predicate {
	return NewPredicate(t.ruleIndex, t.predIndex, t.isCtxDependent)
}

func (t *PredicateTransition) String() string {
	return "pred_" + strconv.Itoa(t.ruleIndex) + ":" + strconv.Itoa(t.predIndex)
}

type ActionTransition struct {
	BaseTransition
	isCtxDependent                    bool
	ruleIndex, actionIndex, predIndex int
}

func NewActionTransition(target ATNState, ruleIndex, actionIndex int, isCtxDependent bool) *ActionTransition {
	return &ActionTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionACTION,
			isEpsilon:         true,
		},
		isCtxDependent: isCtxDependent,
		ruleIndex:      ruleIndex,
		actionIndex:    actionIndex,
	}
}

func (t *ActionTransition) Matches(_, _, _ int) bool {
	return false
}

func (t *ActionTransition) String() string {
	return "action_" + strconv.Itoa(t.ruleIndex) + ":" + strconv.Itoa(t.actionIndex)
}

type SetTransition struct {
	BaseTransition
}

func NewSetTransition(target ATNState, set *IntervalSet) *SetTransition {
	t := &SetTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionSET,
		},
	}

	if set != nil {
		t.intervalSet = set
	} else {
		t.intervalSet = NewIntervalSet()
		t.intervalSet.addOne(TokenInvalidType)
	}
	return t
}

func (t *SetTransition) Matches(symbol, _, _ int) bool {
	return t.intervalSet.contains(symbol)
}

func (t *SetTransition) String() string {
	return t.intervalSet.String()
}

type NotSetTransition struct {
	SetTransition
}

func NewNotSetTransition(target ATNState, set *IntervalSet) *NotSetTransition {
	t := &NotSetTransition{
		SetTransition: SetTransition{
			BaseTransition: BaseTransition{
				target:            target,
				serializationType: TransitionNOTSET,
			},
		},
	}
	if set != nil {
		t.intervalSet = set
	} else {
		t.intervalSet = NewIntervalSet()
		t.intervalSet.addOne(TokenInvalidType)
	}

	return t
}

func (t *NotSetTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return symbol >= minVocabSymbol && symbol <= maxVocabSymbol && !t.intervalSet.contains(symbol)
}

func (t *NotSetTransition) String() string {
	return "~" + t.intervalSet.String()
}

type WildcardTransition struct {
	BaseTransition
}

func NewWildcardTransition(target ATNState) *WildcardTransition {
	return &WildcardTransition{
		BaseTransition: BaseTransition{
			target:            target,
			serializationType: TransitionWILDCARD,
		},
	}
}

func (t *WildcardTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return symbol >= minVocabSymbol && symbol <= maxVocabSymbol
}

func (t *WildcardTransition) String() string {
	return "."
}

type PrecedencePredicateTransition struct {
	BaseAbstractPredicateTransition
	precedence int
}

func NewPrecedencePredicateTransition(target ATNState, precedence int) *PrecedencePredicateTransition {
	return &PrecedencePredicateTransition{
		BaseAbstractPredicateTransition: BaseAbstractPredicateTransition{
			BaseTransition: BaseTransition{
				target:            target,
				serializationType: TransitionPRECEDENCE,
				isEpsilon:         true,
			},
		},
		precedence: precedence,
	}
}

func (t *PrecedencePredicateTransition) Matches(_, _, _ int) bool {
	return false
}

func (t *PrecedencePredicateTransition) getPredicate() *PrecedencePredicate {
	return NewPrecedencePredicate(t.precedence)
}

func (t *PrecedencePredicateTransition) String() string {
	return fmt.Sprint(t.precedence) + " >= _p"
}
