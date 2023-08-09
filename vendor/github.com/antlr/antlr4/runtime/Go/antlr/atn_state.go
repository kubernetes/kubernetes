// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "strconv"

// Constants for serialization.
const (
	ATNStateInvalidType    = 0
	ATNStateBasic          = 1
	ATNStateRuleStart      = 2
	ATNStateBlockStart     = 3
	ATNStatePlusBlockStart = 4
	ATNStateStarBlockStart = 5
	ATNStateTokenStart     = 6
	ATNStateRuleStop       = 7
	ATNStateBlockEnd       = 8
	ATNStateStarLoopBack   = 9
	ATNStateStarLoopEntry  = 10
	ATNStatePlusLoopBack   = 11
	ATNStateLoopEnd        = 12

	ATNStateInvalidStateNumber = -1
)

var ATNStateInitialNumTransitions = 4

type ATNState interface {
	GetEpsilonOnlyTransitions() bool

	GetRuleIndex() int
	SetRuleIndex(int)

	GetNextTokenWithinRule() *IntervalSet
	SetNextTokenWithinRule(*IntervalSet)

	GetATN() *ATN
	SetATN(*ATN)

	GetStateType() int

	GetStateNumber() int
	SetStateNumber(int)

	GetTransitions() []Transition
	SetTransitions([]Transition)
	AddTransition(Transition, int)

	String() string
	hash() int
}

type BaseATNState struct {
	// NextTokenWithinRule caches lookahead during parsing. Not used during construction.
	NextTokenWithinRule *IntervalSet

	// atn is the current ATN.
	atn *ATN

	epsilonOnlyTransitions bool

	// ruleIndex tracks the Rule index because there are no Rule objects at runtime.
	ruleIndex int

	stateNumber int

	stateType int

	// Track the transitions emanating from this ATN state.
	transitions []Transition
}

func NewBaseATNState() *BaseATNState {
	return &BaseATNState{stateNumber: ATNStateInvalidStateNumber, stateType: ATNStateInvalidType}
}

func (as *BaseATNState) GetRuleIndex() int {
	return as.ruleIndex
}

func (as *BaseATNState) SetRuleIndex(v int) {
	as.ruleIndex = v
}
func (as *BaseATNState) GetEpsilonOnlyTransitions() bool {
	return as.epsilonOnlyTransitions
}

func (as *BaseATNState) GetATN() *ATN {
	return as.atn
}

func (as *BaseATNState) SetATN(atn *ATN) {
	as.atn = atn
}

func (as *BaseATNState) GetTransitions() []Transition {
	return as.transitions
}

func (as *BaseATNState) SetTransitions(t []Transition) {
	as.transitions = t
}

func (as *BaseATNState) GetStateType() int {
	return as.stateType
}

func (as *BaseATNState) GetStateNumber() int {
	return as.stateNumber
}

func (as *BaseATNState) SetStateNumber(stateNumber int) {
	as.stateNumber = stateNumber
}

func (as *BaseATNState) GetNextTokenWithinRule() *IntervalSet {
	return as.NextTokenWithinRule
}

func (as *BaseATNState) SetNextTokenWithinRule(v *IntervalSet) {
	as.NextTokenWithinRule = v
}

func (as *BaseATNState) hash() int {
	return as.stateNumber
}

func (as *BaseATNState) String() string {
	return strconv.Itoa(as.stateNumber)
}

func (as *BaseATNState) equals(other interface{}) bool {
	if ot, ok := other.(ATNState); ok {
		return as.stateNumber == ot.GetStateNumber()
	}

	return false
}

func (as *BaseATNState) isNonGreedyExitState() bool {
	return false
}

func (as *BaseATNState) AddTransition(trans Transition, index int) {
	if len(as.transitions) == 0 {
		as.epsilonOnlyTransitions = trans.getIsEpsilon()
	} else if as.epsilonOnlyTransitions != trans.getIsEpsilon() {
		as.epsilonOnlyTransitions = false
	}

	if index == -1 {
		as.transitions = append(as.transitions, trans)
	} else {
		as.transitions = append(as.transitions[:index], append([]Transition{trans}, as.transitions[index:]...)...)
		// TODO: as.transitions.splice(index, 1, trans)
	}
}

type BasicState struct {
	*BaseATNState
}

func NewBasicState() *BasicState {
	b := NewBaseATNState()

	b.stateType = ATNStateBasic

	return &BasicState{BaseATNState: b}
}

type DecisionState interface {
	ATNState

	getDecision() int
	setDecision(int)

	getNonGreedy() bool
	setNonGreedy(bool)
}

type BaseDecisionState struct {
	*BaseATNState
	decision  int
	nonGreedy bool
}

func NewBaseDecisionState() *BaseDecisionState {
	return &BaseDecisionState{BaseATNState: NewBaseATNState(), decision: -1}
}

func (s *BaseDecisionState) getDecision() int {
	return s.decision
}

func (s *BaseDecisionState) setDecision(b int) {
	s.decision = b
}

func (s *BaseDecisionState) getNonGreedy() bool {
	return s.nonGreedy
}

func (s *BaseDecisionState) setNonGreedy(b bool) {
	s.nonGreedy = b
}

type BlockStartState interface {
	DecisionState

	getEndState() *BlockEndState
	setEndState(*BlockEndState)
}

// BaseBlockStartState is the start of a regular (...) block.
type BaseBlockStartState struct {
	*BaseDecisionState
	endState *BlockEndState
}

func NewBlockStartState() *BaseBlockStartState {
	return &BaseBlockStartState{BaseDecisionState: NewBaseDecisionState()}
}

func (s *BaseBlockStartState) getEndState() *BlockEndState {
	return s.endState
}

func (s *BaseBlockStartState) setEndState(b *BlockEndState) {
	s.endState = b
}

type BasicBlockStartState struct {
	*BaseBlockStartState
}

func NewBasicBlockStartState() *BasicBlockStartState {
	b := NewBlockStartState()

	b.stateType = ATNStateBlockStart

	return &BasicBlockStartState{BaseBlockStartState: b}
}

var _ BlockStartState = &BasicBlockStartState{}

// BlockEndState is a terminal node of a simple (a|b|c) block.
type BlockEndState struct {
	*BaseATNState
	startState ATNState
}

func NewBlockEndState() *BlockEndState {
	b := NewBaseATNState()

	b.stateType = ATNStateBlockEnd

	return &BlockEndState{BaseATNState: b}
}

// RuleStopState is the last node in the ATN for a rule, unless that rule is the
// start symbol. In that case, there is one transition to EOF. Later, we might
// encode references to all calls to this rule to compute FOLLOW sets for error
// handling.
type RuleStopState struct {
	*BaseATNState
}

func NewRuleStopState() *RuleStopState {
	b := NewBaseATNState()

	b.stateType = ATNStateRuleStop

	return &RuleStopState{BaseATNState: b}
}

type RuleStartState struct {
	*BaseATNState
	stopState        ATNState
	isPrecedenceRule bool
}

func NewRuleStartState() *RuleStartState {
	b := NewBaseATNState()

	b.stateType = ATNStateRuleStart

	return &RuleStartState{BaseATNState: b}
}

// PlusLoopbackState is a decision state for A+ and (A|B)+. It has two
// transitions: one to the loop back to start of the block, and one to exit.
type PlusLoopbackState struct {
	*BaseDecisionState
}

func NewPlusLoopbackState() *PlusLoopbackState {
	b := NewBaseDecisionState()

	b.stateType = ATNStatePlusLoopBack

	return &PlusLoopbackState{BaseDecisionState: b}
}

// PlusBlockStartState is the start of a (A|B|...)+ loop. Technically it is a
// decision state; we don't use it for code generation. Somebody might need it,
// it is included for completeness. In reality, PlusLoopbackState is the real
// decision-making node for A+.
type PlusBlockStartState struct {
	*BaseBlockStartState
	loopBackState ATNState
}

func NewPlusBlockStartState() *PlusBlockStartState {
	b := NewBlockStartState()

	b.stateType = ATNStatePlusBlockStart

	return &PlusBlockStartState{BaseBlockStartState: b}
}

var _ BlockStartState = &PlusBlockStartState{}

// StarBlockStartState is the block that begins a closure loop.
type StarBlockStartState struct {
	*BaseBlockStartState
}

func NewStarBlockStartState() *StarBlockStartState {
	b := NewBlockStartState()

	b.stateType = ATNStateStarBlockStart

	return &StarBlockStartState{BaseBlockStartState: b}
}

var _ BlockStartState = &StarBlockStartState{}

type StarLoopbackState struct {
	*BaseATNState
}

func NewStarLoopbackState() *StarLoopbackState {
	b := NewBaseATNState()

	b.stateType = ATNStateStarLoopBack

	return &StarLoopbackState{BaseATNState: b}
}

type StarLoopEntryState struct {
	*BaseDecisionState
	loopBackState          ATNState
	precedenceRuleDecision bool
}

func NewStarLoopEntryState() *StarLoopEntryState {
	b := NewBaseDecisionState()

	b.stateType = ATNStateStarLoopEntry

	// False precedenceRuleDecision indicates whether s state can benefit from a precedence DFA during SLL decision making.
	return &StarLoopEntryState{BaseDecisionState: b}
}

// LoopEndState marks the end of a * or + loop.
type LoopEndState struct {
	*BaseATNState
	loopBackState ATNState
}

func NewLoopEndState() *LoopEndState {
	b := NewBaseATNState()

	b.stateType = ATNStateLoopEnd

	return &LoopEndState{BaseATNState: b}
}

// TokensStartState is the Tokens rule start state linking to each lexer rule start state.
type TokensStartState struct {
	*BaseDecisionState
}

func NewTokensStartState() *TokensStartState {
	b := NewBaseDecisionState()

	b.stateType = ATNStateTokenStart

	return &TokensStartState{BaseDecisionState: b}
}
