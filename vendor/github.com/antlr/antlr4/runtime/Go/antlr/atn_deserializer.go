// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf16"
)

// This is the earliest supported serialized UUID.
// stick to serialized version for now, we don't need a UUID instance
var BaseSerializedUUID = "AADB8D7E-AEEF-4415-AD2B-8204D6CF042E"
var AddedUnicodeSMP = "59627784-3BE5-417A-B9EB-8131A7286089"

// This list contains all of the currently supported UUIDs, ordered by when
// the feature first appeared in this branch.
var SupportedUUIDs = []string{BaseSerializedUUID, AddedUnicodeSMP}

var SerializedVersion = 3

// This is the current serialized UUID.
var SerializedUUID = AddedUnicodeSMP

type LoopEndStateIntPair struct {
	item0 *LoopEndState
	item1 int
}

type BlockStartStateIntPair struct {
	item0 BlockStartState
	item1 int
}

type ATNDeserializer struct {
	deserializationOptions *ATNDeserializationOptions
	data                   []rune
	pos                    int
	uuid                   string
}

func NewATNDeserializer(options *ATNDeserializationOptions) *ATNDeserializer {
	if options == nil {
		options = ATNDeserializationOptionsdefaultOptions
	}

	return &ATNDeserializer{deserializationOptions: options}
}

func stringInSlice(a string, list []string) int {
	for i, b := range list {
		if b == a {
			return i
		}
	}

	return -1
}

// isFeatureSupported determines if a particular serialized representation of an
// ATN supports a particular feature, identified by the UUID used for
// serializing the ATN at the time the feature was first introduced. Feature is
// the UUID marking the first time the feature was supported in the serialized
// ATN. ActualUuid is the UUID of the actual serialized ATN which is currently
// being deserialized. It returns true if actualUuid represents a serialized ATN
// at or after the feature identified by feature was introduced, and otherwise
// false.
func (a *ATNDeserializer) isFeatureSupported(feature, actualUUID string) bool {
	idx1 := stringInSlice(feature, SupportedUUIDs)

	if idx1 < 0 {
		return false
	}

	idx2 := stringInSlice(actualUUID, SupportedUUIDs)

	return idx2 >= idx1
}

func (a *ATNDeserializer) DeserializeFromUInt16(data []uint16) *ATN {
	a.reset(utf16.Decode(data))
	a.checkVersion()
	a.checkUUID()

	atn := a.readATN()

	a.readStates(atn)
	a.readRules(atn)
	a.readModes(atn)

	sets := make([]*IntervalSet, 0)

	// First, deserialize sets with 16-bit arguments <= U+FFFF.
	sets = a.readSets(atn, sets, a.readInt)
	// Next, if the ATN was serialized with the Unicode SMP feature,
	// deserialize sets with 32-bit arguments <= U+10FFFF.
	if (a.isFeatureSupported(AddedUnicodeSMP, a.uuid)) {
		sets = a.readSets(atn, sets, a.readInt32)
	}

	a.readEdges(atn, sets)
	a.readDecisions(atn)
	a.readLexerActions(atn)
	a.markPrecedenceDecisions(atn)
	a.verifyATN(atn)

	if a.deserializationOptions.generateRuleBypassTransitions && atn.grammarType == ATNTypeParser {
		a.generateRuleBypassTransitions(atn)
		// Re-verify after modification
		a.verifyATN(atn)
	}

	return atn

}

func (a *ATNDeserializer) reset(data []rune) {
	temp := make([]rune, len(data))

	for i, c := range data {
		// Don't adjust the first value since that's the version number
		if i == 0 {
			temp[i] = c
		} else if c > 1 {
			temp[i] = c - 2
		} else {
		    temp[i] = c + 65533
		}
	}

	a.data = temp
	a.pos = 0
}

func (a *ATNDeserializer) checkVersion() {
	version := a.readInt()

	if version != SerializedVersion {
		panic("Could not deserialize ATN with version " + strconv.Itoa(version) + " (expected " + strconv.Itoa(SerializedVersion) + ").")
	}
}

func (a *ATNDeserializer) checkUUID() {
	uuid := a.readUUID()

	if stringInSlice(uuid, SupportedUUIDs) < 0 {
		panic("Could not deserialize ATN with UUID: " + uuid + " (expected " + SerializedUUID + " or a legacy UUID).")
	}

	a.uuid = uuid
}

func (a *ATNDeserializer) readATN() *ATN {
	grammarType := a.readInt()
	maxTokenType := a.readInt()

	return NewATN(grammarType, maxTokenType)
}

func (a *ATNDeserializer) readStates(atn *ATN) {
	loopBackStateNumbers := make([]LoopEndStateIntPair, 0)
	endStateNumbers := make([]BlockStartStateIntPair, 0)

	nstates := a.readInt()

	for i := 0; i < nstates; i++ {
		stype := a.readInt()

		// Ignore bad types of states
		if stype == ATNStateInvalidType {
			atn.addState(nil)

			continue
		}

		ruleIndex := a.readInt()

		if ruleIndex == 0xFFFF {
			ruleIndex = -1
		}

		s := a.stateFactory(stype, ruleIndex)

		if stype == ATNStateLoopEnd {
			loopBackStateNumber := a.readInt()

			loopBackStateNumbers = append(loopBackStateNumbers, LoopEndStateIntPair{s.(*LoopEndState), loopBackStateNumber})
		} else if s2, ok := s.(BlockStartState); ok {
			endStateNumber := a.readInt()

			endStateNumbers = append(endStateNumbers, BlockStartStateIntPair{s2, endStateNumber})
		}

		atn.addState(s)
	}

	// Delay the assignment of loop back and end states until we know all the state
	// instances have been initialized
	for j := 0; j < len(loopBackStateNumbers); j++ {
		pair := loopBackStateNumbers[j]

		pair.item0.loopBackState = atn.states[pair.item1]
	}

	for j := 0; j < len(endStateNumbers); j++ {
		pair := endStateNumbers[j]

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
		atn.ruleToTokenType = make([]int, nrules) // TODO: initIntArray(nrules, 0)
	}

	atn.ruleToStartState = make([]*RuleStartState, nrules) // TODO: initIntArray(nrules, 0)

	for i := 0; i < nrules; i++ {
		s := a.readInt()
		startState := atn.states[s].(*RuleStartState)

		atn.ruleToStartState[i] = startState

		if atn.grammarType == ATNTypeLexer {
			tokenType := a.readInt()

			if tokenType == 0xFFFF {
				tokenType = TokenEOF
			}

			atn.ruleToTokenType[i] = tokenType
		}
	}

	atn.ruleToStopState = make([]*RuleStopState, nrules) //initIntArray(nrules, 0)

	for i := 0; i < len(atn.states); i++ {
		state := atn.states[i]

		if s2, ok := state.(*RuleStopState); ok {
			atn.ruleToStopState[s2.ruleIndex] = s2
			atn.ruleToStartState[s2.ruleIndex].stopState = s2
		}
	}
}

func (a *ATNDeserializer) readModes(atn *ATN) {
	nmodes := a.readInt()

	for i := 0; i < nmodes; i++ {
		s := a.readInt()

		atn.modeToStartState = append(atn.modeToStartState, atn.states[s].(*TokensStartState))
	}
}

func (a *ATNDeserializer) readSets(atn *ATN, sets []*IntervalSet, readUnicode func() int) []*IntervalSet {
	m := a.readInt()

	for i := 0; i < m; i++ {
		iset := NewIntervalSet()

		sets = append(sets, iset)

		n := a.readInt()
		containsEOF := a.readInt()

		if containsEOF != 0 {
			iset.addOne(-1)
		}

		for j := 0; j < n; j++ {
			i1 := readUnicode()
			i2 := readUnicode()

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
	for i := 0; i < len(atn.states); i++ {
		state := atn.states[i]

		for j := 0; j < len(state.GetTransitions()); j++ {
			var t, ok = state.GetTransitions()[j].(*RuleTransition)

			if !ok {
				continue
			}

			outermostPrecedenceReturn := -1

			if atn.ruleToStartState[t.getTarget().GetRuleIndex()].isPrecedenceRule {
				if t.precedence == 0 {
					outermostPrecedenceReturn = t.getTarget().GetRuleIndex()
				}
			}

			trans := NewEpsilonTransition(t.followState, outermostPrecedenceReturn)

			atn.ruleToStopState[t.getTarget().GetRuleIndex()].AddTransition(trans, -1)
		}
	}

	for i := 0; i < len(atn.states); i++ {
		state := atn.states[i]

		if s2, ok := state.(*BaseBlockStartState); ok {
			// We need to know the end state to set its start state
			if s2.endState == nil {
				panic("IllegalState")
			}

			// Block end states can only be associated to a single block start state
			if s2.endState.startState != nil {
				panic("IllegalState")
			}

			s2.endState.startState = state
		}

		if s2, ok := state.(*PlusLoopbackState); ok {
			for j := 0; j < len(s2.GetTransitions()); j++ {
				target := s2.GetTransitions()[j].getTarget()

				if t2, ok := target.(*PlusBlockStartState); ok {
					t2.loopBackState = state
				}
			}
		} else if s2, ok := state.(*StarLoopbackState); ok {
			for j := 0; j < len(s2.GetTransitions()); j++ {
				target := s2.GetTransitions()[j].getTarget()

				if t2, ok := target.(*StarLoopEntryState); ok {
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

		atn.lexerActions = make([]LexerAction, count) // initIntArray(count, nil)

		for i := 0; i < count; i++ {
			actionType := a.readInt()
			data1 := a.readInt()

			if data1 == 0xFFFF {
				data1 = -1
			}

			data2 := a.readInt()

			if data2 == 0xFFFF {
				data2 = -1
			}

			lexerAction := a.lexerActionFactory(actionType, data1, data2)

			atn.lexerActions[i] = lexerAction
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
	if !a.deserializationOptions.verifyATN {
		return
	}

	// Verify assumptions
	for i := 0; i < len(atn.states); i++ {
		state := atn.states[i]

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

			switch s2 := state.(type) {
			case *StarBlockStartState:
				var _, ok2 = s2.GetTransitions()[1].getTarget().(*LoopEndState)

				a.checkCondition(ok2, "")
				a.checkCondition(!s2.nonGreedy, "")

			case *LoopEndState:
				var s3, ok2 = s2.GetTransitions()[1].getTarget().(*StarBlockStartState)

				a.checkCondition(ok2, "")
				a.checkCondition(s3.nonGreedy, "")

			default:
				panic("IllegalState")
			}

		case *StarLoopbackState:
			a.checkCondition(len(state.GetTransitions()) == 1, "")

			var _, ok2 = state.GetTransitions()[0].getTarget().(*StarLoopEntryState)

			a.checkCondition(ok2, "")

		case *LoopEndState:
			a.checkCondition(s2.loopBackState != nil, "")

		case *RuleStartState:
			a.checkCondition(s2.stopState != nil, "")

		case *BaseBlockStartState:
			a.checkCondition(s2.endState != nil, "")

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

	return int(v)
}

func (a *ATNDeserializer) readInt32() int {
	var low = a.readInt()
	var high = a.readInt()
	return low | (high << 16)
}

//TODO
//func (a *ATNDeserializer) readLong() int64 {
//    panic("Not implemented")
//    var low = a.readInt32()
//    var high = a.readInt32()
//    return (low & 0x00000000FFFFFFFF) | (high << int32)
//}

func createByteToHex() []string {
	bth := make([]string, 256)

	for i := 0; i < 256; i++ {
		bth[i] = strings.ToUpper(hex.EncodeToString([]byte{byte(i)}))
	}

	return bth
}

var byteToHex = createByteToHex()

func (a *ATNDeserializer) readUUID() string {
	bb := make([]int, 16)

	for i := 7; i >= 0; i-- {
		integer := a.readInt()

		bb[(2*i)+1] = integer & 0xFF
		bb[2*i] = (integer >> 8) & 0xFF
	}

	return byteToHex[bb[0]] + byteToHex[bb[1]] +
		byteToHex[bb[2]] + byteToHex[bb[3]] + "-" +
		byteToHex[bb[4]] + byteToHex[bb[5]] + "-" +
		byteToHex[bb[6]] + byteToHex[bb[7]] + "-" +
		byteToHex[bb[8]] + byteToHex[bb[9]] + "-" +
		byteToHex[bb[10]] + byteToHex[bb[11]] +
		byteToHex[bb[12]] + byteToHex[bb[13]] +
		byteToHex[bb[14]] + byteToHex[bb[15]]
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
