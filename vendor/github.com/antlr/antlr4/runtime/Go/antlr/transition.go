// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
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

func (t *BaseTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
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

// TODO: make all transitions sets? no, should remove set edges
type AtomTransition struct {
	*BaseTransition
}

func NewAtomTransition(target ATNState, intervalSet int) *AtomTransition {

	t := new(AtomTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.label = intervalSet // The token type or character value or, signifies special intervalSet.
	t.intervalSet = t.makeLabel()
	t.serializationType = TransitionATOM

	return t
}

func (t *AtomTransition) makeLabel() *IntervalSet {
	s := NewIntervalSet()
	s.addOne(t.label)
	return s
}

func (t *AtomTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return t.label == symbol
}

func (t *AtomTransition) String() string {
	return strconv.Itoa(t.label)
}

type RuleTransition struct {
	*BaseTransition

	followState           ATNState
	ruleIndex, precedence int
}

func NewRuleTransition(ruleStart ATNState, ruleIndex, precedence int, followState ATNState) *RuleTransition {

	t := new(RuleTransition)
	t.BaseTransition = NewBaseTransition(ruleStart)

	t.ruleIndex = ruleIndex
	t.precedence = precedence
	t.followState = followState
	t.serializationType = TransitionRULE
	t.isEpsilon = true

	return t
}

func (t *RuleTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return false
}

type EpsilonTransition struct {
	*BaseTransition

	outermostPrecedenceReturn int
}

func NewEpsilonTransition(target ATNState, outermostPrecedenceReturn int) *EpsilonTransition {

	t := new(EpsilonTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.serializationType = TransitionEPSILON
	t.isEpsilon = true
	t.outermostPrecedenceReturn = outermostPrecedenceReturn
	return t
}

func (t *EpsilonTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return false
}

func (t *EpsilonTransition) String() string {
	return "epsilon"
}

type RangeTransition struct {
	*BaseTransition

	start, stop int
}

func NewRangeTransition(target ATNState, start, stop int) *RangeTransition {

	t := new(RangeTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.serializationType = TransitionRANGE
	t.start = start
	t.stop = stop
	t.intervalSet = t.makeLabel()
	return t
}

func (t *RangeTransition) makeLabel() *IntervalSet {
	s := NewIntervalSet()
	s.addRange(t.start, t.stop)
	return s
}

func (t *RangeTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return symbol >= t.start && symbol <= t.stop
}

func (t *RangeTransition) String() string {
	return "'" + string(t.start) + "'..'" + string(t.stop) + "'"
}

type AbstractPredicateTransition interface {
	Transition
	IAbstractPredicateTransitionFoo()
}

type BaseAbstractPredicateTransition struct {
	*BaseTransition
}

func NewBasePredicateTransition(target ATNState) *BaseAbstractPredicateTransition {

	t := new(BaseAbstractPredicateTransition)
	t.BaseTransition = NewBaseTransition(target)

	return t
}

func (a *BaseAbstractPredicateTransition) IAbstractPredicateTransitionFoo() {}

type PredicateTransition struct {
	*BaseAbstractPredicateTransition

	isCtxDependent       bool
	ruleIndex, predIndex int
}

func NewPredicateTransition(target ATNState, ruleIndex, predIndex int, isCtxDependent bool) *PredicateTransition {

	t := new(PredicateTransition)
	t.BaseAbstractPredicateTransition = NewBasePredicateTransition(target)

	t.serializationType = TransitionPREDICATE
	t.ruleIndex = ruleIndex
	t.predIndex = predIndex
	t.isCtxDependent = isCtxDependent // e.g., $i ref in pred
	t.isEpsilon = true
	return t
}

func (t *PredicateTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return false
}

func (t *PredicateTransition) getPredicate() *Predicate {
	return NewPredicate(t.ruleIndex, t.predIndex, t.isCtxDependent)
}

func (t *PredicateTransition) String() string {
	return "pred_" + strconv.Itoa(t.ruleIndex) + ":" + strconv.Itoa(t.predIndex)
}

type ActionTransition struct {
	*BaseTransition

	isCtxDependent                    bool
	ruleIndex, actionIndex, predIndex int
}

func NewActionTransition(target ATNState, ruleIndex, actionIndex int, isCtxDependent bool) *ActionTransition {

	t := new(ActionTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.serializationType = TransitionACTION
	t.ruleIndex = ruleIndex
	t.actionIndex = actionIndex
	t.isCtxDependent = isCtxDependent // e.g., $i ref in pred
	t.isEpsilon = true
	return t
}

func (t *ActionTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return false
}

func (t *ActionTransition) String() string {
	return "action_" + strconv.Itoa(t.ruleIndex) + ":" + strconv.Itoa(t.actionIndex)
}

type SetTransition struct {
	*BaseTransition
}

func NewSetTransition(target ATNState, set *IntervalSet) *SetTransition {

	t := new(SetTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.serializationType = TransitionSET
	if set != nil {
		t.intervalSet = set
	} else {
		t.intervalSet = NewIntervalSet()
		t.intervalSet.addOne(TokenInvalidType)
	}

	return t
}

func (t *SetTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return t.intervalSet.contains(symbol)
}

func (t *SetTransition) String() string {
	return t.intervalSet.String()
}

type NotSetTransition struct {
	*SetTransition
}

func NewNotSetTransition(target ATNState, set *IntervalSet) *NotSetTransition {

	t := new(NotSetTransition)

	t.SetTransition = NewSetTransition(target, set)

	t.serializationType = TransitionNOTSET

	return t
}

func (t *NotSetTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return symbol >= minVocabSymbol && symbol <= maxVocabSymbol && !t.intervalSet.contains(symbol)
}

func (t *NotSetTransition) String() string {
	return "~" + t.intervalSet.String()
}

type WildcardTransition struct {
	*BaseTransition
}

func NewWildcardTransition(target ATNState) *WildcardTransition {

	t := new(WildcardTransition)
	t.BaseTransition = NewBaseTransition(target)

	t.serializationType = TransitionWILDCARD
	return t
}

func (t *WildcardTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return symbol >= minVocabSymbol && symbol <= maxVocabSymbol
}

func (t *WildcardTransition) String() string {
	return "."
}

type PrecedencePredicateTransition struct {
	*BaseAbstractPredicateTransition

	precedence int
}

func NewPrecedencePredicateTransition(target ATNState, precedence int) *PrecedencePredicateTransition {

	t := new(PrecedencePredicateTransition)
	t.BaseAbstractPredicateTransition = NewBasePredicateTransition(target)

	t.serializationType = TransitionPRECEDENCE
	t.precedence = precedence
	t.isEpsilon = true

	return t
}

func (t *PrecedencePredicateTransition) Matches(symbol, minVocabSymbol, maxVocabSymbol int) bool {
	return false
}

func (t *PrecedencePredicateTransition) getPredicate() *PrecedencePredicate {
	return NewPrecedencePredicate(t.precedence)
}

func (t *PrecedencePredicateTransition) String() string {
	return fmt.Sprint(t.precedence) + " >= _p"
}
