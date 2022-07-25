// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
)

const serializedVersion = 4

type loopEndStateIntPair struct {
	item0 *LoopEndState
	item1 int
}

type blockStartStateIntPair struct {
	item0 BlockStartState
	item1 int
}

type ATNDeserializer struct {
	options *ATNDeserializationOptions
	data    []int32
	pos     int
}

func NewATNDeserializer(options *ATNDeserializationOptions) *ATNDeserializer {
	if options == nil {
		options = &defaultATNDeserializationOptions
	}

	return &ATNDeserializer{options: options}
}

func stringInSlice(a string, list []string) int {
	for i, b := range list {
		if b == a {
			return i
		}
	}

	return -1
}

func (a *ATNDeserializer) Deserialize(data []int32) *ATN {
	a.data = data
	a.pos = 0
	a.checkVersion()

	atn := a.readATN()

	a.readStates(atn)
	a.readRules(atn)
	a.readModes(atn)

	sets := a.readSets(atn, nil)

	a.readEdges(atn, sets)
	a.readDecisions(atn)
	a.readLexerActions(atn)
	a.markPrecedenceDecisions(atn)
	a.verifyATN(atn)

	if a.options.GenerateRuleBypassTransitions() && atn.grammarType == ATNTypeParser {
		a.generateRuleBypassTransitions(atn)
		// Re-verify after modification
		a.verifyATN(atn)
	}

	return atn

}

func (a *ATNDeserializer) checkVersion() {
	version := a.readInt()

	if version != serializedVersion {
		panic("Could not deserialize ATN with version " + strconv.Itoa(version) + " (expected " + strconv.Itoa(serializedVersion) + ").")
	}
}

func (a *ATNDeserializer) readATN() *ATN {
	grammarType := a.readInt()
	maxTokenType := a.readInt()

	return NewATN(grammarType, maxTokenType)
}

func (a *ATNDeserializer) readStates(atn *ATN) {
	nstates := a.readInt()

	// Allocate worst case size.
	loopBackStateNumbers := make([]loopEndStateIntPair, 0, nstates)
	endStateNumbers := make([]blockStartStateIntPair, 0, nstates)

	// Preallocate states slice.
	atn.states = make([]ATNState, 0, nstates)

	for i := 0; i < nstates; i++ {
		stype := a.readInt()

		// Ignore bad types of states
		if stype == ATNStateInvalidType {
			atn.addState(nil)
			continue
		}

		ruleIndex := a.readInt()

		s := a.stateFactory(stype, ruleIndex)

		if stype == ATNStateLoopEnd {
			loopBackStateNumber := a.readInt()

			loopBackStateNumbers = append(loopBackStateNumbers, loopEndStateIntPair{s.(*LoopEndState), loopBackStateNumber})
		} else if s2, ok := s.(BlockStartState); ok {
			endStateNumber := a.readInt()

			endStateNumbers = append(endStateNumbers, blockStartStateIntPair{s2, endStateNumber})
		}

		atn.addState(s)
	}

	// Delay the assignment of loop back and end states until we know all the state
	// instances have been initialized
	for _, pair := range loopBackStateNumbers {
		pair.item0.loopBackState = atn.states[pair.item1]
	}

	for _, pair := range endStateNumbers {
		pair.item0.setEndState(atn.states[pair.item1].(*BlockEndState))
	}

	numNonGreedyStates := a.readInt()
	for j := 0; j < numNonGreedyStates; j++ {
		stateNumber := a.readInt()

		atn.states[stateNumber].(DecisionState).setNonGreedy(true)
	}

	numPrecedenceStates := a.readInt()
	for j := 0; j < numPrecedenceStates; j++ {
		stateNumber := a.readInt()

		atn.states[stateNumber].(*RuleStartState).isPrecedenceRule = true
	}
}

func (a *ATNDeserializer) readRules(atn *ATN) {
	nrules := a.readInt()

	if atn.grammarType == ATNTypeLexer {
		atn.ruleToTokenType = make([]int, nrules)
	}

	atn.ruleToStartState = make([]*RuleStartState, nrules)

	for i := range atn.ruleToStartState {
		s := a.readInt()
		startState := atn.states[s].(*RuleStartState)

		atn.ruleToStartState[i] = startState

		if atn.grammarType == ATNTypeLexer {
			tokenType := a.readInt()

			atn.ruleToTokenType[i] = tokenType
		}
	}

	atn.ruleToStopState = make([]*RuleStopState, nrules)

	for _, state := range atn.states {
		if s2, ok := state.(*RuleStopState); ok {
			atn.ruleToStopState[s2.ruleIndex] = s2
			atn.ruleToStartState[s2.ruleIndex].stopState = s2
		}
	}
}

func (a *ATNDeserializer) readModes(atn *ATN) {
	nmodes := a.readInt()
	atn.modeToStartState = make([]*TokensStartState, nmodes)

	for i := range atn.modeToStartState {
		s := a.readInt()

		atn.modeToStartState[i] = atn.states[s].(*TokensStartState)
	}
}

func (a *ATNDeserializer) readSets(atn *ATN, sets []*IntervalSet) []*IntervalSet {
	m := a.readInt()

	// Preallocate the needed capacity.
	if cap(sets)-len(sets) < m {
		isets := make([]*IntervalSet, len(sets), len(sets)+m)
		copy(isets, sets)
		sets = isets
	}

	for i := 0; i < m; i++ {
		iset := NewIntervalSet()

		sets = append(sets, iset)

		n := a.readInt()
		containsEOF := a.readInt()

		if containsEOF != 0 {
			iset.addOne(-1)
		}

		for j := 0; j < n; j++ {
			i1 := a.readInt()
			i2 := a.readInt()

			iset.addRange(i1, i2)
		}
	}

	return sets
}

func (a *ATNDeserializer) readEdges(atn *ATN, sets []*IntervalSet) {
	nedges := a.readInt()

	for i := 0; i < nedges; i++ {
		var (
			src      = a.readInt()
			trg      = a.readInt()
			ttype    = a.readInt()
			arg1     = a.readInt()
			arg2     = a.readInt()
			arg3     = a.readInt()
			trans    = a.edgeFactory(atn, ttype, src, trg, arg1, arg2, arg3, sets)
			srcState = atn.states[src]
		)

		srcState.AddTransition(trans, -1)
	}

	// Edges for rule stop states can be derived, so they are not serialized
	for _, state := range atn.states {
		for _, t := range state.GetTransitions() {
			var rt, ok = t.(*RuleTransition)

			if !ok {
				continue
			}

			outermostPrecedenceReturn := -1

			if atn.ruleToStartState[rt.getTarget().GetRuleIndex()].isPrecedenceRule {
				if rt.precedence == 0 {
					outermostPrecedenceReturn = rt.getTarget().GetRuleIndex()
				}
			}

			trans := NewEpsilonTransition(rt.followState, outermostPrecedenceReturn)

			atn.ruleToStopState[rt.getTarget().GetRuleIndex()].AddTransition(trans, -1)
		}
	}

	for _, state := range atn.states {
		if s2, ok := state.(BlockStartState); ok {
			// We need to know the end state to set its start state
			if s2.getEndState() == nil {
				panic("IllegalState")
			}

			// Block end states can only be associated to a single block start state
			if s2.getEndState().startState != nil {
				panic("IllegalState")
			}

			s2.getEndState().startState = state
		}

		if s2, ok := state.(*PlusLoopbackState); ok {
			for _, t := range s2.GetTransitions() {
				if t2, ok := t.getTarget().(*PlusBlockStartState); ok {
					t2.loopBackState = state
				}
			}
		} else if s2, ok := state.(*StarLoopbackState); ok {
			for _, t := range s2.GetTransitions() {
				if t2, ok := t.getTarget().(*StarLoopEntryState); ok {
					t2.loopBackState = state
				}
			}
		}
	}
}

func (a *ATNDeserializer) readDecisions(atn *ATN) {
	ndecisions := a.readInt()

	for i := 0; i < ndecisions; i++ {
		s := a.readInt()
		decState := atn.states[s].(DecisionState)

		atn.DecisionToState = append(atn.DecisionToState, decState)
		decState.setDecision(i)
	}
}

func (a *ATNDeserializer) readLexerActions(atn *ATN) {
	if atn.grammarType == ATNTypeLexer {
		count := a.readInt()

		atn.lexerActions = make([]LexerAction, count)

		for i := range atn.lexerActions {
			actionType := a.readInt()
			data1 := a.readInt()
			data2 := a.readInt()
			atn.lexerActions[i] = a.lexerActionFactory(actionType, data1, data2)
		}
	}
}

func (a *ATNDeserializer) generateRuleBypassTransitions(atn *ATN) {
	count := len(atn.ruleToStartState)

	for i := 0; i < count; i++ {
		atn.ruleToTokenType[i] = atn.maxTokenType + i + 1
	}

	for i := 0; i < count; i++ {
		a.generateRuleBypassTransition(atn, i)
	}
}

func (a *ATNDeserializer) generateRuleBypassTransition(atn *ATN, idx int) {
	bypassStart := NewBasicBlockStartState()

	bypassStart.ruleIndex = idx
	atn.addState(bypassStart)

	bypassStop := NewBlockEndState()

	bypassStop.ruleIndex = idx
	atn.addState(bypassStop)

	bypassStart.endState = bypassStop

	atn.defineDecisionState(bypassStart.BaseDecisionState)

	bypassStop.startState = bypassStart

	var excludeTransition Transition
	var endState ATNState

	if atn.ruleToStartState[idx].isPrecedenceRule {
		// Wrap from the beginning of the rule to the StarLoopEntryState
		endState = nil

		for i := 0; i < len(atn.states); i++ {
			state := atn.states[i]

			if a.stateIsEndStateFor(state, idx) != nil {
				endState = state
				excludeTransition = state.(*StarLoopEntryState).loopBackState.GetTransitions()[0]

				break
			}
		}

		if excludeTransition == nil {
			panic("Couldn't identify final state of the precedence rule prefix section.")
		}
	} else {
		endState = atn.ruleToStopState[idx]
	}

	// All non-excluded transitions that currently target end state need to target
	// blockEnd instead
	for i := 0; i < len(atn.states); i++ {
		state := atn.states[i]

		for j := 0; j < len(state.GetTransitions()); j++ {
			transition := state.GetTransitions()[j]

			if transition == excludeTransition {
				continue
			}

			if transition.getTarget() == endState {
				transition.setTarget(bypassStop)
			}
		}
	}

	// All transitions leaving the rule start state need to leave blockStart instead
	ruleToStartState := atn.ruleToStartState[idx]
	count := len(ruleToStartState.GetTransitions())

	for count > 0 {
		bypassStart.AddTransition(ruleToStartState.GetTransitions()[count-1], -1)
		ruleToStartState.SetTransitions([]Transition{ruleToStartState.GetTransitions()[len(ruleToStartState.GetTransitions())-1]})
	}

	// Link the new states
	atn.ruleToStartState[idx].AddTransition(NewEpsilonTransition(bypassStart, -1), -1)
	bypassStop.AddTransition(NewEpsilonTransition(endState, -1), -1)

	MatchState := NewBasicState()

	atn.addState(MatchState)
	MatchState.AddTransition(NewAtomTransition(bypassStop, atn.ruleToTokenType[idx]), -1)
	bypassStart.AddTransition(NewEpsilonTransition(MatchState, -1), -1)
}

func (a *ATNDeserializer) stateIsEndStateFor(state ATNState, idx int) ATNState {
	if state.GetRuleIndex() != idx {
		return nil
	}

	if _, ok := state.(*StarLoopEntryState); !ok {
		return nil
	}

	maybeLoopEndState := state.GetTransitions()[len(state.GetTransitions())-1].getTarget()

	if _, ok := maybeLoopEndState.(*LoopEndState); !ok {
		return nil
	}

	var _, ok = maybeLoopEndState.GetTransitions()[0].getTarget().(*RuleStopState)

	if maybeLoopEndState.(*LoopEndState).epsilonOnlyTransitions && ok {
		return state
	}

	return nil
}

// markPrecedenceDecisions analyzes the StarLoopEntryState states in the
// specified ATN to set the StarLoopEntryState.precedenceRuleDecision field to
// the correct value.
func (a *ATNDeserializer) markPrecedenceDecisions(atn *ATN) {
	for _, state := range atn.states {
		if _, ok := state.(*StarLoopEntryState); !ok {
			continue
		}

		// We analyze the ATN to determine if a ATN decision state is the
		// decision for the closure block that determines whether a
		// precedence rule should continue or complete.
		if atn.ruleToStartState[state.GetRuleIndex()].isPrecedenceRule {
			maybeLoopEndState := state.GetTransitions()[len(state.GetTransitions())-1].getTarget()

			if s3, ok := maybeLoopEndState.(*LoopEndState); ok {
				var _, ok2 = maybeLoopEndState.GetTransitions()[0].getTarget().(*RuleStopState)

				if s3.epsilonOnlyTransitions && ok2 {
					state.(*StarLoopEntryState).precedenceRuleDecision = true
				}
			}
		}
	}
}

func (a *ATNDeserializer) verifyATN(atn *ATN) {
	if !a.options.VerifyATN() {
		return
	}

	// Verify assumptions
	for _, state := range atn.states {
		if state == nil {
			continue
		}

		a.checkCondition(state.GetEpsilonOnlyTransitions() || len(state.GetTransitions()) <= 1, "")

		switch s2 := state.(type) {
		case *PlusBlockStartState:
			a.checkCondition(s2.loopBackState != nil, "")

		case *StarLoopEntryState:
			a.checkCondition(s2.loopBackState != nil, "")
			a.checkCondition(len(s2.GetTransitions()) == 2, "")

			switch s2.transitions[0].getTarget().(type) {
			case *StarBlockStartState:
				_, ok := s2.transitions[1].getTarget().(*LoopEndState)

				a.checkCondition(ok, "")
				a.checkCondition(!s2.nonGreedy, "")

			case *LoopEndState:
				var _, ok = s2.transitions[1].getTarget().(*StarBlockStartState)

				a.checkCondition(ok, "")
				a.checkCondition(s2.nonGreedy, "")

			default:
				panic("IllegalState")
			}

		case *StarLoopbackState:
			a.checkCondition(len(state.GetTransitions()) == 1, "")

			var _, ok = state.GetTransitions()[0].getTarget().(*StarLoopEntryState)

			a.checkCondition(ok, "")

		case *LoopEndState:
			a.checkCondition(s2.loopBackState != nil, "")

		case *RuleStartState:
			a.checkCondition(s2.stopState != nil, "")

		case BlockStartState:
			a.checkCondition(s2.getEndState() != nil, "")

		case *BlockEndState:
			a.checkCondition(s2.startState != nil, "")

		case DecisionState:
			a.checkCondition(len(s2.GetTransitions()) <= 1 || s2.getDecision() >= 0, "")

		default:
			var _, ok = s2.(*RuleStopState)

			a.checkCondition(len(s2.GetTransitions()) <= 1 || ok, "")
		}
	}
}

func (a *ATNDeserializer) checkCondition(condition bool, message string) {
	if !condition {
		if message == "" {
			message = "IllegalState"
		}

		panic(message)
	}
}

func (a *ATNDeserializer) readInt() int {
	v := a.data[a.pos]

	a.pos++

	return int(v) // data is 32 bits but int is at least that big
}

func (a *ATNDeserializer) edgeFactory(atn *ATN, typeIndex, src, trg, arg1, arg2, arg3 int, sets []*IntervalSet) Transition {
	target := atn.states[trg]

	switch typeIndex {
	case TransitionEPSILON:
		return NewEpsilonTransition(target, -1)

	case TransitionRANGE:
		if arg3 != 0 {
			return NewRangeTransition(target, TokenEOF, arg2)
		}

		return NewRangeTransition(target, arg1, arg2)

	case TransitionRULE:
		return NewRuleTransition(atn.states[arg1], arg2, arg3, target)

	case TransitionPREDICATE:
		return NewPredicateTransition(target, arg1, arg2, arg3 != 0)

	case TransitionPRECEDENCE:
		return NewPrecedencePredicateTransition(target, arg1)

	case TransitionATOM:
		if arg3 != 0 {
			return NewAtomTransition(target, TokenEOF)
		}

		return NewAtomTransition(target, arg1)

	case TransitionACTION:
		return NewActionTransition(target, arg1, arg2, arg3 != 0)

	case TransitionSET:
		return NewSetTransition(target, sets[arg1])

	case TransitionNOTSET:
		return NewNotSetTransition(target, sets[arg1])

	case TransitionWILDCARD:
		return NewWildcardTransition(target)
	}

	panic("The specified transition type is not valid.")
}

func (a *ATNDeserializer) stateFactory(typeIndex, ruleIndex int) ATNState {
	var s ATNState

	switch typeIndex {
	case ATNStateInvalidType:
		return nil

	case ATNStateBasic:
		s = NewBasicState()

	case ATNStateRuleStart:
		s = NewRuleStartState()

	case ATNStateBlockStart:
		s = NewBasicBlockStartState()

	case ATNStatePlusBlockStart:
		s = NewPlusBlockStartState()

	case ATNStateStarBlockStart:
		s = NewStarBlockStartState()

	case ATNStateTokenStart:
		s = NewTokensStartState()

	case ATNStateRuleStop:
		s = NewRuleStopState()

	case ATNStateBlockEnd:
		s = NewBlockEndState()

	case ATNStateStarLoopBack:
		s = NewStarLoopbackState()

	case ATNStateStarLoopEntry:
		s = NewStarLoopEntryState()

	case ATNStatePlusLoopBack:
		s = NewPlusLoopbackState()

	case ATNStateLoopEnd:
		s = NewLoopEndState()

	default:
		panic(fmt.Sprintf("state type %d is invalid", typeIndex))
	}

	s.SetRuleIndex(ruleIndex)

	return s
}

func (a *ATNDeserializer) lexerActionFactory(typeIndex, data1, data2 int) LexerAction {
	switch typeIndex {
	case LexerActionTypeChannel:
		return NewLexerChannelAction(data1)

	case LexerActionTypeCustom:
		return NewLexerCustomAction(data1, data2)

	case LexerActionTypeMode:
		return NewLexerModeAction(data1)

	case LexerActionTypeMore:
		return LexerMoreActionINSTANCE

	case LexerActionTypePopMode:
		return LexerPopModeActionINSTANCE

	case LexerActionTypePushMode:
		return NewLexerPushModeAction(data1)

	case LexerActionTypeSkip:
		return LexerSkipActionINSTANCE

	case LexerActionTypeType:
		return NewLexerTypeAction(data1)

	default:
		panic(fmt.Sprintf("lexer action %d is invalid", typeIndex))
	}
}
