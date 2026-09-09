// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"os"
	"strconv"
)

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

//goland:noinspection GoUnusedGlobalVariable
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
	Hash() int
	Equals(Collectable[ATNState]) bool
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

func NewATNState() *BaseATNState {
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

func (as *BaseATNState) Hash() int {
	return as.stateNumber
}

func (as *BaseATNState) String() string {
	return strconv.Itoa(as.stateNumber)
}

func (as *BaseATNState) Equals(other Collectable[ATNState]) bool {
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
		_, _ = fmt.Fprintf(os.Stdin, "ATN state %d has both epsilon and non-epsilon transitions.\n", as.stateNumber)
		as.epsilonOnlyTransitions = false
	}

	// TODO: Check code for already present compared to the Java equivalent
	//alreadyPresent := false
	//for _, t := range as.transitions {
	//	if t.getTarget().GetStateNumber() == trans.getTarget().GetStateNumber() {
	//		if t.getLabel() != nil && trans.getLabel() != nil && trans.getLabel().Equals(t.getLabel()) {
	//			alreadyPresent = true
	//			break
	//		}
	//	} else if t.getIsEpsilon() && trans.getIsEpsilon() {
	//		alreadyPresent = true
	//		break
	//	}
	//}
	//if !alreadyPresent {
	if index == -1 {
		as.transitions = append(as.transitions, trans)
	} else {
		as.transitions = append(as.transitions[:index], append([]Transition{trans}, as.transitions[index:]...)...)
		// TODO: as.transitions.splice(index, 1, trans)
	}
	//} else {
	//	_, _ = fmt.Fprintf(os.Stderr, "Transition already present in state %d\n", as.stateNumber)
	//}
}

type BasicState struct {
	BaseATNState
}

func NewBasicState() *BasicState {
	return &BasicState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateBasic,
		},
	}
}

type DecisionState interface {
	ATNState

	getDecision() int
	setDecision(int)

	getNonGreedy() bool
	setNonGreedy(bool)
}

type BaseDecisionState struct {
	BaseATNState
	decision  int
	nonGreedy bool
}

func NewBaseDecisionState() *BaseDecisionState {
	return &BaseDecisionState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateBasic,
		},
		decision: -1,
	}
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
	BaseDecisionState
	endState *BlockEndState
}

func NewBlockStartState() *BaseBlockStartState {
	return &BaseBlockStartState{
		BaseDecisionState: BaseDecisionState{
			BaseATNState: BaseATNState{
				stateNumber: ATNStateInvalidStateNumber,
				stateType:   ATNStateBasic,
			},
			decision: -1,
		},
	}
}

func (s *BaseBlockStartState) getEndState() *BlockEndState {
	return s.endState
}

func (s *BaseBlockStartState) setEndState(b *BlockEndState) {
	s.endState = b
}

type BasicBlockStartState struct {
	BaseBlockStartState
}

func NewBasicBlockStartState() *BasicBlockStartState {
	return &BasicBlockStartState{
		BaseBlockStartState: BaseBlockStartState{
			BaseDecisionState: BaseDecisionState{
				BaseATNState: BaseATNState{
					stateNumber: ATNStateInvalidStateNumber,
					stateType:   ATNStateBlockStart,
				},
			},
		},
	}
}

var _ BlockStartState = &BasicBlockStartState{}

// BlockEndState is a terminal node of a simple (a|b|c) block.
type BlockEndState struct {
	BaseATNState
	startState ATNState
}

func NewBlockEndState() *BlockEndState {
	return &BlockEndState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateBlockEnd,
		},
		startState: nil,
	}
}

// RuleStopState is the last node in the ATN for a rule, unless that rule is the
// start symbol. In that case, there is one transition to EOF. Later, we might
// encode references to all calls to this rule to compute FOLLOW sets for error
// handling.
type RuleStopState struct {
	BaseATNState
}

func NewRuleStopState() *RuleStopState {
	return &RuleStopState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateRuleStop,
		},
	}
}

type RuleStartState struct {
	BaseATNState
	stopState        ATNState
	isPrecedenceRule bool
}

func NewRuleStartState() *RuleStartState {
	return &RuleStartState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateRuleStart,
		},
	}
}

// PlusLoopbackState is a decision state for A+ and (A|B)+. It has two
// transitions: one to the loop back to start of the block, and one to exit.
type PlusLoopbackState struct {
	BaseDecisionState
}

func NewPlusLoopbackState() *PlusLoopbackState {
	return &PlusLoopbackState{
		BaseDecisionState: BaseDecisionState{
			BaseATNState: BaseATNState{
				stateNumber: ATNStateInvalidStateNumber,
				stateType:   ATNStatePlusLoopBack,
			},
		},
	}
}

// PlusBlockStartState is the start of a (A|B|...)+ loop. Technically it is a
// decision state; we don't use it for code generation. Somebody might need it,
// it is included for completeness. In reality, PlusLoopbackState is the real
// decision-making node for A+.
type PlusBlockStartState struct {
	BaseBlockStartState
	loopBackState ATNState
}

func NewPlusBlockStartState() *PlusBlockStartState {
	return &PlusBlockStartState{
		BaseBlockStartState: BaseBlockStartState{
			BaseDecisionState: BaseDecisionState{
				BaseATNState: BaseATNState{
					stateNumber: ATNStateInvalidStateNumber,
					stateType:   ATNStatePlusBlockStart,
				},
			},
		},
	}
}

var _ BlockStartState = &PlusBlockStartState{}

// StarBlockStartState is the block that begins a closure loop.
type StarBlockStartState struct {
	BaseBlockStartState
}

func NewStarBlockStartState() *StarBlockStartState {
	return &StarBlockStartState{
		BaseBlockStartState: BaseBlockStartState{
			BaseDecisionState: BaseDecisionState{
				BaseATNState: BaseATNState{
					stateNumber: ATNStateInvalidStateNumber,
					stateType:   ATNStateStarBlockStart,
				},
			},
		},
	}
}

var _ BlockStartState = &StarBlockStartState{}

type StarLoopbackState struct {
	BaseATNState
}

func NewStarLoopbackState() *StarLoopbackState {
	return &StarLoopbackState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateStarLoopBack,
		},
	}
}

type StarLoopEntryState struct {
	BaseDecisionState
	loopBackState          ATNState
	precedenceRuleDecision bool
}

func NewStarLoopEntryState() *StarLoopEntryState {
	// False precedenceRuleDecision indicates whether s state can benefit from a precedence DFA during SLL decision making.
	return &StarLoopEntryState{
		BaseDecisionState: BaseDecisionState{
			BaseATNState: BaseATNState{
				stateNumber: ATNStateInvalidStateNumber,
				stateType:   ATNStateStarLoopEntry,
			},
		},
	}
}

// LoopEndState marks the end of a * or + loop.
type LoopEndState struct {
	BaseATNState
	loopBackState ATNState
}

func NewLoopEndState() *LoopEndState {
	return &LoopEndState{
		BaseATNState: BaseATNState{
			stateNumber: ATNStateInvalidStateNumber,
			stateType:   ATNStateLoopEnd,
		},
	}
}

// TokensStartState is the Tokens rule start state linking to each lexer rule start state.
type TokensStartState struct {
	BaseDecisionState
}

func NewTokensStartState() *TokensStartState {
	return &TokensStartState{
		BaseDecisionState: BaseDecisionState{
			BaseATNState: BaseATNState{
				stateNumber: ATNStateInvalidStateNumber,
				stateType:   ATNStateTokenStart,
			},
		},
	}
}
