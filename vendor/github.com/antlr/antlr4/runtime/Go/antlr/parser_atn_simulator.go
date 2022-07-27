// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
	"strings"
)

var (
	ParserATNSimulatorDebug            = false
	ParserATNSimulatorListATNDecisions = false
	ParserATNSimulatorDFADebug         = false
	ParserATNSimulatorRetryDebug       = false
	TurnOffLRLoopEntryBranchOpt        = false
)

type ParserATNSimulator struct {
	*BaseATNSimulator

	parser         Parser
	predictionMode int
	input          TokenStream
	startIndex     int
	dfa            *DFA
	mergeCache     *DoubleDict
	outerContext   ParserRuleContext
}

func NewParserATNSimulator(parser Parser, atn *ATN, decisionToDFA []*DFA, sharedContextCache *PredictionContextCache) *ParserATNSimulator {

	p := new(ParserATNSimulator)

	p.BaseATNSimulator = NewBaseATNSimulator(atn, sharedContextCache)

	p.parser = parser
	p.decisionToDFA = decisionToDFA
	// SLL, LL, or LL + exact ambig detection?//
	p.predictionMode = PredictionModeLL
	// LAME globals to avoid parameters!!!!! I need these down deep in predTransition
	p.input = nil
	p.startIndex = 0
	p.outerContext = nil
	p.dfa = nil
	// Each prediction operation uses a cache for merge of prediction contexts.
	//  Don't keep around as it wastes huge amounts of memory. DoubleKeyMap
	//  isn't Synchronized but we're ok since two threads shouldn't reuse same
	//  parser/atnsim object because it can only handle one input at a time.
	//  This maps graphs a and b to merged result c. (a,b)&rarrc. We can avoid
	//  the merge if we ever see a and b again.  Note that (b,a)&rarrc should
	//  also be examined during cache lookup.
	//
	p.mergeCache = nil

	return p
}

func (p *ParserATNSimulator) GetPredictionMode() int {
	return p.predictionMode
}

func (p *ParserATNSimulator) SetPredictionMode(v int) {
	p.predictionMode = v
}

func (p *ParserATNSimulator) reset() {
}

func (p *ParserATNSimulator) AdaptivePredict(input TokenStream, decision int, outerContext ParserRuleContext) int {
	if ParserATNSimulatorDebug || ParserATNSimulatorListATNDecisions {
		fmt.Println("AdaptivePredict decision " + strconv.Itoa(decision) +
			" exec LA(1)==" + p.getLookaheadName(input) +
			" line " + strconv.Itoa(input.LT(1).GetLine()) + ":" +
			strconv.Itoa(input.LT(1).GetColumn()))
	}

	p.input = input
	p.startIndex = input.Index()
	p.outerContext = outerContext

	dfa := p.decisionToDFA[decision]
	p.dfa = dfa
	m := input.Mark()
	index := input.Index()

	defer func() {
		p.dfa = nil
		p.mergeCache = nil // wack cache after each prediction
		input.Seek(index)
		input.Release(m)
	}()

	// Now we are certain to have a specific decision's DFA
	// But, do we still need an initial state?
	var s0 *DFAState
	if dfa.getPrecedenceDfa() {
		// the start state for a precedence DFA depends on the current
		// parser precedence, and is provided by a DFA method.
		s0 = dfa.getPrecedenceStartState(p.parser.GetPrecedence())
	} else {
		// the start state for a "regular" DFA is just s0
		s0 = dfa.getS0()
	}

	if s0 == nil {
		if outerContext == nil {
			outerContext = RuleContextEmpty
		}
		if ParserATNSimulatorDebug || ParserATNSimulatorListATNDecisions {
			fmt.Println("predictATN decision " + strconv.Itoa(dfa.decision) +
				" exec LA(1)==" + p.getLookaheadName(input) +
				", outerContext=" + outerContext.String(p.parser.GetRuleNames(), nil))
		}
		// If p is not a precedence DFA, we check the ATN start state
		// to determine if p ATN start state is the decision for the
		// closure block that determines whether a precedence rule
		// should continue or complete.

		t2 := dfa.atnStartState
		t, ok := t2.(*StarLoopEntryState)
		if !dfa.getPrecedenceDfa() && ok {
			if t.precedenceRuleDecision {
				dfa.setPrecedenceDfa(true)
			}
		}
		fullCtx := false
		s0Closure := p.computeStartState(dfa.atnStartState, RuleContextEmpty, fullCtx)

		if dfa.getPrecedenceDfa() {
			// If p is a precedence DFA, we use applyPrecedenceFilter
			// to convert the computed start state to a precedence start
			// state. We then use DFA.setPrecedenceStartState to set the
			// appropriate start state for the precedence level rather
			// than simply setting DFA.s0.
			//
			dfa.s0.configs = s0Closure
			s0Closure = p.applyPrecedenceFilter(s0Closure)
			s0 = p.addDFAState(dfa, NewDFAState(-1, s0Closure))
			dfa.setPrecedenceStartState(p.parser.GetPrecedence(), s0)
		} else {
			s0 = p.addDFAState(dfa, NewDFAState(-1, s0Closure))
			dfa.setS0(s0)
		}
	}
	alt := p.execATN(dfa, s0, input, index, outerContext)
	if ParserATNSimulatorDebug {
		fmt.Println("DFA after predictATN: " + dfa.String(p.parser.GetLiteralNames(), nil))
	}
	return alt

}

// Performs ATN simulation to compute a predicted alternative based
//  upon the remaining input, but also updates the DFA cache to avoid
//  having to traverse the ATN again for the same input sequence.

// There are some key conditions we're looking for after computing a new
// set of ATN configs (proposed DFA state):
// if the set is empty, there is no viable alternative for current symbol
// does the state uniquely predict an alternative?
// does the state have a conflict that would prevent us from
//   putting it on the work list?

// We also have some key operations to do:
// add an edge from previous DFA state to potentially NewDFA state, D,
//   upon current symbol but only if adding to work list, which means in all
//   cases except no viable alternative (and possibly non-greedy decisions?)
// collecting predicates and adding semantic context to DFA accept states
// adding rule context to context-sensitive DFA accept states
// consuming an input symbol
// Reporting a conflict
// Reporting an ambiguity
// Reporting a context sensitivity
// Reporting insufficient predicates

// cover these cases:
//    dead end
//    single alt
//    single alt + preds
//    conflict
//    conflict + preds
//
func (p *ParserATNSimulator) execATN(dfa *DFA, s0 *DFAState, input TokenStream, startIndex int, outerContext ParserRuleContext) int {

	if ParserATNSimulatorDebug || ParserATNSimulatorListATNDecisions {
		fmt.Println("execATN decision " + strconv.Itoa(dfa.decision) +
			" exec LA(1)==" + p.getLookaheadName(input) +
			" line " + strconv.Itoa(input.LT(1).GetLine()) + ":" + strconv.Itoa(input.LT(1).GetColumn()))
	}

	previousD := s0

	if ParserATNSimulatorDebug {
		fmt.Println("s0 = " + s0.String())
	}
	t := input.LA(1)
	for { // for more work
		D := p.getExistingTargetState(previousD, t)
		if D == nil {
			D = p.computeTargetState(dfa, previousD, t)
		}
		if D == ATNSimulatorError {
			// if any configs in previous dipped into outer context, that
			// means that input up to t actually finished entry rule
			// at least for SLL decision. Full LL doesn't dip into outer
			// so don't need special case.
			// We will get an error no matter what so delay until after
			// decision better error message. Also, no reachable target
			// ATN states in SLL implies LL will also get nowhere.
			// If conflict in states that dip out, choose min since we
			// will get error no matter what.
			e := p.noViableAlt(input, outerContext, previousD.configs, startIndex)
			input.Seek(startIndex)
			alt := p.getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(previousD.configs, outerContext)
			if alt != ATNInvalidAltNumber {
				return alt
			}

			panic(e)
		}
		if D.requiresFullContext && p.predictionMode != PredictionModeSLL {
			// IF PREDS, MIGHT RESOLVE TO SINGLE ALT => SLL (or syntax error)
			conflictingAlts := D.configs.GetConflictingAlts()
			if D.predicates != nil {
				if ParserATNSimulatorDebug {
					fmt.Println("DFA state has preds in DFA sim LL failover")
				}
				conflictIndex := input.Index()
				if conflictIndex != startIndex {
					input.Seek(startIndex)
				}
				conflictingAlts = p.evalSemanticContext(D.predicates, outerContext, true)
				if conflictingAlts.length() == 1 {
					if ParserATNSimulatorDebug {
						fmt.Println("Full LL avoided")
					}
					return conflictingAlts.minValue()
				}
				if conflictIndex != startIndex {
					// restore the index so Reporting the fallback to full
					// context occurs with the index at the correct spot
					input.Seek(conflictIndex)
				}
			}
			if ParserATNSimulatorDFADebug {
				fmt.Println("ctx sensitive state " + outerContext.String(nil, nil) + " in " + D.String())
			}
			fullCtx := true
			s0Closure := p.computeStartState(dfa.atnStartState, outerContext, fullCtx)
			p.ReportAttemptingFullContext(dfa, conflictingAlts, D.configs, startIndex, input.Index())
			alt := p.execATNWithFullContext(dfa, D, s0Closure, input, startIndex, outerContext)
			return alt
		}
		if D.isAcceptState {
			if D.predicates == nil {
				return D.prediction
			}
			stopIndex := input.Index()
			input.Seek(startIndex)
			alts := p.evalSemanticContext(D.predicates, outerContext, true)

			switch alts.length() {
			case 0:
				panic(p.noViableAlt(input, outerContext, D.configs, startIndex))
			case 1:
				return alts.minValue()
			default:
				// Report ambiguity after predicate evaluation to make sure the correct set of ambig alts is Reported.
				p.ReportAmbiguity(dfa, D, startIndex, stopIndex, false, alts, D.configs)
				return alts.minValue()
			}
		}
		previousD = D

		if t != TokenEOF {
			input.Consume()
			t = input.LA(1)
		}
	}

	panic("Should not have reached p state")
}

// Get an existing target state for an edge in the DFA. If the target state
// for the edge has not yet been computed or is otherwise not available,
// p method returns {@code nil}.
//
// @param previousD The current DFA state
// @param t The next input symbol
// @return The existing target DFA state for the given input symbol
// {@code t}, or {@code nil} if the target state for p edge is not
// already cached

func (p *ParserATNSimulator) getExistingTargetState(previousD *DFAState, t int) *DFAState {
	edges := previousD.getEdges()
	if edges == nil || t+1 < 0 || t+1 >= len(edges) {
		return nil
	}

	return previousD.getIthEdge(t + 1)
}

// Compute a target state for an edge in the DFA, and attempt to add the
// computed state and corresponding edge to the DFA.
//
// @param dfa The DFA
// @param previousD The current DFA state
// @param t The next input symbol
//
// @return The computed target DFA state for the given input symbol
// {@code t}. If {@code t} does not lead to a valid DFA state, p method
// returns {@link //ERROR}.

func (p *ParserATNSimulator) computeTargetState(dfa *DFA, previousD *DFAState, t int) *DFAState {
	reach := p.computeReachSet(previousD.configs, t, false)

	if reach == nil {
		p.addDFAEdge(dfa, previousD, t, ATNSimulatorError)
		return ATNSimulatorError
	}
	// create Newtarget state we'll add to DFA after it's complete
	D := NewDFAState(-1, reach)

	predictedAlt := p.getUniqueAlt(reach)

	if ParserATNSimulatorDebug {
		altSubSets := PredictionModegetConflictingAltSubsets(reach)
		fmt.Println("SLL altSubSets=" + fmt.Sprint(altSubSets) +
			", previous=" + previousD.configs.String() +
			", configs=" + reach.String() +
			", predict=" + strconv.Itoa(predictedAlt) +
			", allSubsetsConflict=" +
			fmt.Sprint(PredictionModeallSubsetsConflict(altSubSets)) +
			", conflictingAlts=" + p.getConflictingAlts(reach).String())
	}
	if predictedAlt != ATNInvalidAltNumber {
		// NO CONFLICT, UNIQUELY PREDICTED ALT
		D.isAcceptState = true
		D.configs.SetUniqueAlt(predictedAlt)
		D.setPrediction(predictedAlt)
	} else if PredictionModehasSLLConflictTerminatingPrediction(p.predictionMode, reach) {
		// MORE THAN ONE VIABLE ALTERNATIVE
		D.configs.SetConflictingAlts(p.getConflictingAlts(reach))
		D.requiresFullContext = true
		// in SLL-only mode, we will stop at p state and return the minimum alt
		D.isAcceptState = true
		D.setPrediction(D.configs.GetConflictingAlts().minValue())
	}
	if D.isAcceptState && D.configs.HasSemanticContext() {
		p.predicateDFAState(D, p.atn.getDecisionState(dfa.decision))
		if D.predicates != nil {
			D.setPrediction(ATNInvalidAltNumber)
		}
	}
	// all adds to dfa are done after we've created full D state
	D = p.addDFAEdge(dfa, previousD, t, D)
	return D
}

func (p *ParserATNSimulator) predicateDFAState(dfaState *DFAState, decisionState DecisionState) {
	// We need to test all predicates, even in DFA states that
	// uniquely predict alternative.
	nalts := len(decisionState.GetTransitions())
	// Update DFA so reach becomes accept state with (predicate,alt)
	// pairs if preds found for conflicting alts
	altsToCollectPredsFrom := p.getConflictingAltsOrUniqueAlt(dfaState.configs)
	altToPred := p.getPredsForAmbigAlts(altsToCollectPredsFrom, dfaState.configs, nalts)
	if altToPred != nil {
		dfaState.predicates = p.getPredicatePredictions(altsToCollectPredsFrom, altToPred)
		dfaState.setPrediction(ATNInvalidAltNumber) // make sure we use preds
	} else {
		// There are preds in configs but they might go away
		// when OR'd together like {p}? || NONE == NONE. If neither
		// alt has preds, resolve to min alt
		dfaState.setPrediction(altsToCollectPredsFrom.minValue())
	}
}

// comes back with reach.uniqueAlt set to a valid alt
func (p *ParserATNSimulator) execATNWithFullContext(dfa *DFA, D *DFAState, s0 ATNConfigSet, input TokenStream, startIndex int, outerContext ParserRuleContext) int {

	if ParserATNSimulatorDebug || ParserATNSimulatorListATNDecisions {
		fmt.Println("execATNWithFullContext " + s0.String())
	}

	fullCtx := true
	foundExactAmbig := false
	var reach ATNConfigSet
	previous := s0
	input.Seek(startIndex)
	t := input.LA(1)
	predictedAlt := -1

	for { // for more work
		reach = p.computeReachSet(previous, t, fullCtx)
		if reach == nil {
			// if any configs in previous dipped into outer context, that
			// means that input up to t actually finished entry rule
			// at least for LL decision. Full LL doesn't dip into outer
			// so don't need special case.
			// We will get an error no matter what so delay until after
			// decision better error message. Also, no reachable target
			// ATN states in SLL implies LL will also get nowhere.
			// If conflict in states that dip out, choose min since we
			// will get error no matter what.
			e := p.noViableAlt(input, outerContext, previous, startIndex)
			input.Seek(startIndex)
			alt := p.getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(previous, outerContext)
			if alt != ATNInvalidAltNumber {
				return alt
			}

			panic(e)
		}
		altSubSets := PredictionModegetConflictingAltSubsets(reach)
		if ParserATNSimulatorDebug {
			fmt.Println("LL altSubSets=" + fmt.Sprint(altSubSets) + ", predict=" +
				strconv.Itoa(PredictionModegetUniqueAlt(altSubSets)) + ", resolvesToJustOneViableAlt=" +
				fmt.Sprint(PredictionModeresolvesToJustOneViableAlt(altSubSets)))
		}
		reach.SetUniqueAlt(p.getUniqueAlt(reach))
		// unique prediction?
		if reach.GetUniqueAlt() != ATNInvalidAltNumber {
			predictedAlt = reach.GetUniqueAlt()
			break
		}
		if p.predictionMode != PredictionModeLLExactAmbigDetection {
			predictedAlt = PredictionModeresolvesToJustOneViableAlt(altSubSets)
			if predictedAlt != ATNInvalidAltNumber {
				break
			}
		} else {
			// In exact ambiguity mode, we never try to terminate early.
			// Just keeps scarfing until we know what the conflict is
			if PredictionModeallSubsetsConflict(altSubSets) && PredictionModeallSubsetsEqual(altSubSets) {
				foundExactAmbig = true
				predictedAlt = PredictionModegetSingleViableAlt(altSubSets)
				break
			}
			// else there are multiple non-conflicting subsets or
			// we're not sure what the ambiguity is yet.
			// So, keep going.
		}
		previous = reach
		if t != TokenEOF {
			input.Consume()
			t = input.LA(1)
		}
	}
	// If the configuration set uniquely predicts an alternative,
	// without conflict, then we know that it's a full LL decision
	// not SLL.
	if reach.GetUniqueAlt() != ATNInvalidAltNumber {
		p.ReportContextSensitivity(dfa, predictedAlt, reach, startIndex, input.Index())
		return predictedAlt
	}
	// We do not check predicates here because we have checked them
	// on-the-fly when doing full context prediction.

	//
	// In non-exact ambiguity detection mode, we might	actually be able to
	// detect an exact ambiguity, but I'm not going to spend the cycles
	// needed to check. We only emit ambiguity warnings in exact ambiguity
	// mode.
	//
	// For example, we might know that we have conflicting configurations.
	// But, that does not mean that there is no way forward without a
	// conflict. It's possible to have nonconflicting alt subsets as in:

	// altSubSets=[{1, 2}, {1, 2}, {1}, {1, 2}]

	// from
	//
	//    [(17,1,[5 $]), (13,1,[5 10 $]), (21,1,[5 10 $]), (11,1,[$]),
	//     (13,2,[5 10 $]), (21,2,[5 10 $]), (11,2,[$])]
	//
	// In p case, (17,1,[5 $]) indicates there is some next sequence that
	// would resolve p without conflict to alternative 1. Any other viable
	// next sequence, however, is associated with a conflict.  We stop
	// looking for input because no amount of further lookahead will alter
	// the fact that we should predict alternative 1.  We just can't say for
	// sure that there is an ambiguity without looking further.

	p.ReportAmbiguity(dfa, D, startIndex, input.Index(), foundExactAmbig, reach.Alts(), reach)

	return predictedAlt
}

func (p *ParserATNSimulator) computeReachSet(closure ATNConfigSet, t int, fullCtx bool) ATNConfigSet {
	if ParserATNSimulatorDebug {
		fmt.Println("in computeReachSet, starting closure: " + closure.String())
	}
	if p.mergeCache == nil {
		p.mergeCache = NewDoubleDict()
	}
	intermediate := NewBaseATNConfigSet(fullCtx)

	// Configurations already in a rule stop state indicate reaching the end
	// of the decision rule (local context) or end of the start rule (full
	// context). Once reached, these configurations are never updated by a
	// closure operation, so they are handled separately for the performance
	// advantage of having a smaller intermediate set when calling closure.
	//
	// For full-context reach operations, separate handling is required to
	// ensure that the alternative Matching the longest overall sequence is
	// chosen when multiple such configurations can Match the input.

	var skippedStopStates []*BaseATNConfig

	// First figure out where we can reach on input t
	for _, c := range closure.GetItems() {
		if ParserATNSimulatorDebug {
			fmt.Println("testing " + p.GetTokenName(t) + " at " + c.String())
		}

		if _, ok := c.GetState().(*RuleStopState); ok {
			if fullCtx || t == TokenEOF {
				skippedStopStates = append(skippedStopStates, c.(*BaseATNConfig))
				if ParserATNSimulatorDebug {
					fmt.Println("added " + c.String() + " to SkippedStopStates")
				}
			}
			continue
		}

		for _, trans := range c.GetState().GetTransitions() {
			target := p.getReachableTarget(trans, t)
			if target != nil {
				cfg := NewBaseATNConfig4(c, target)
				intermediate.Add(cfg, p.mergeCache)
				if ParserATNSimulatorDebug {
					fmt.Println("added " + cfg.String() + " to intermediate")
				}
			}
		}
	}

	// Now figure out where the reach operation can take us...
	var reach ATNConfigSet

	// This block optimizes the reach operation for intermediate sets which
	// trivially indicate a termination state for the overall
	// AdaptivePredict operation.
	//
	// The conditions assume that intermediate
	// contains all configurations relevant to the reach set, but p
	// condition is not true when one or more configurations have been
	// withheld in SkippedStopStates, or when the current symbol is EOF.
	//
	if skippedStopStates == nil && t != TokenEOF {
		if len(intermediate.configs) == 1 {
			// Don't pursue the closure if there is just one state.
			// It can only have one alternative just add to result
			// Also don't pursue the closure if there is unique alternative
			// among the configurations.
			reach = intermediate
		} else if p.getUniqueAlt(intermediate) != ATNInvalidAltNumber {
			// Also don't pursue the closure if there is unique alternative
			// among the configurations.
			reach = intermediate
		}
	}
	// If the reach set could not be trivially determined, perform a closure
	// operation on the intermediate set to compute its initial value.
	//
	if reach == nil {
		reach = NewBaseATNConfigSet(fullCtx)
		closureBusy := NewArray2DHashSet(nil, nil)
		treatEOFAsEpsilon := t == TokenEOF
		amount := len(intermediate.configs)
		for k := 0; k < amount; k++ {
			p.closure(intermediate.configs[k], reach, closureBusy, false, fullCtx, treatEOFAsEpsilon)
		}
	}
	if t == TokenEOF {
		// After consuming EOF no additional input is possible, so we are
		// only interested in configurations which reached the end of the
		// decision rule (local context) or end of the start rule (full
		// context). Update reach to contain only these configurations. This
		// handles both explicit EOF transitions in the grammar and implicit
		// EOF transitions following the end of the decision or start rule.
		//
		// When reach==intermediate, no closure operation was performed. In
		// p case, removeAllConfigsNotInRuleStopState needs to check for
		// reachable rule stop states as well as configurations already in
		// a rule stop state.
		//
		// This is handled before the configurations in SkippedStopStates,
		// because any configurations potentially added from that list are
		// already guaranteed to meet p condition whether or not it's
		// required.
		//
		reach = p.removeAllConfigsNotInRuleStopState(reach, reach == intermediate)
	}
	// If SkippedStopStates!=nil, then it contains at least one
	// configuration. For full-context reach operations, these
	// configurations reached the end of the start rule, in which case we
	// only add them back to reach if no configuration during the current
	// closure operation reached such a state. This ensures AdaptivePredict
	// chooses an alternative Matching the longest overall sequence when
	// multiple alternatives are viable.
	//
	if skippedStopStates != nil && ((!fullCtx) || (!PredictionModehasConfigInRuleStopState(reach))) {
		for l := 0; l < len(skippedStopStates); l++ {
			reach.Add(skippedStopStates[l], p.mergeCache)
		}
	}
	if len(reach.GetItems()) == 0 {
		return nil
	}

	return reach
}

//
// Return a configuration set containing only the configurations from
// {@code configs} which are in a {@link RuleStopState}. If all
// configurations in {@code configs} are already in a rule stop state, p
// method simply returns {@code configs}.
//
// <p>When {@code lookToEndOfRule} is true, p method uses
// {@link ATN//NextTokens} for each configuration in {@code configs} which is
// not already in a rule stop state to see if a rule stop state is reachable
// from the configuration via epsilon-only transitions.</p>
//
// @param configs the configuration set to update
// @param lookToEndOfRule when true, p method checks for rule stop states
// reachable by epsilon-only transitions from each configuration in
// {@code configs}.
//
// @return {@code configs} if all configurations in {@code configs} are in a
// rule stop state, otherwise return a Newconfiguration set containing only
// the configurations from {@code configs} which are in a rule stop state
//
func (p *ParserATNSimulator) removeAllConfigsNotInRuleStopState(configs ATNConfigSet, lookToEndOfRule bool) ATNConfigSet {
	if PredictionModeallConfigsInRuleStopStates(configs) {
		return configs
	}
	result := NewBaseATNConfigSet(configs.FullContext())
	for _, config := range configs.GetItems() {
		if _, ok := config.GetState().(*RuleStopState); ok {
			result.Add(config, p.mergeCache)
			continue
		}
		if lookToEndOfRule && config.GetState().GetEpsilonOnlyTransitions() {
			NextTokens := p.atn.NextTokens(config.GetState(), nil)
			if NextTokens.contains(TokenEpsilon) {
				endOfRuleState := p.atn.ruleToStopState[config.GetState().GetRuleIndex()]
				result.Add(NewBaseATNConfig4(config, endOfRuleState), p.mergeCache)
			}
		}
	}
	return result
}

func (p *ParserATNSimulator) computeStartState(a ATNState, ctx RuleContext, fullCtx bool) ATNConfigSet {
	// always at least the implicit call to start rule
	initialContext := predictionContextFromRuleContext(p.atn, ctx)
	configs := NewBaseATNConfigSet(fullCtx)
	for i := 0; i < len(a.GetTransitions()); i++ {
		target := a.GetTransitions()[i].getTarget()
		c := NewBaseATNConfig6(target, i+1, initialContext)
		closureBusy := NewArray2DHashSet(nil, nil)
		p.closure(c, configs, closureBusy, true, fullCtx, false)
	}
	return configs
}

//
// This method transforms the start state computed by
// {@link //computeStartState} to the special start state used by a
// precedence DFA for a particular precedence value. The transformation
// process applies the following changes to the start state's configuration
// set.
//
// <ol>
// <li>Evaluate the precedence predicates for each configuration using
// {@link SemanticContext//evalPrecedence}.</li>
// <li>Remove all configurations which predict an alternative greater than
// 1, for which another configuration that predicts alternative 1 is in the
// same ATN state with the same prediction context. This transformation is
// valid for the following reasons:
// <ul>
// <li>The closure block cannot contain any epsilon transitions which bypass
// the body of the closure, so all states reachable via alternative 1 are
// part of the precedence alternatives of the transformed left-recursive
// rule.</li>
// <li>The "primary" portion of a left recursive rule cannot contain an
// epsilon transition, so the only way an alternative other than 1 can exist
// in a state that is also reachable via alternative 1 is by nesting calls
// to the left-recursive rule, with the outer calls not being at the
// preferred precedence level.</li>
// </ul>
// </li>
// </ol>
//
// <p>
// The prediction context must be considered by p filter to address
// situations like the following.
// </p>
// <code>
// <pre>
// grammar TA
// prog: statement* EOF
// statement: letterA | statement letterA 'b'
// letterA: 'a'
// </pre>
// </code>
// <p>
// If the above grammar, the ATN state immediately before the token
// reference {@code 'a'} in {@code letterA} is reachable from the left edge
// of both the primary and closure blocks of the left-recursive rule
// {@code statement}. The prediction context associated with each of these
// configurations distinguishes between them, and prevents the alternative
// which stepped out to {@code prog} (and then back in to {@code statement}
// from being eliminated by the filter.
// </p>
//
// @param configs The configuration set computed by
// {@link //computeStartState} as the start state for the DFA.
// @return The transformed configuration set representing the start state
// for a precedence DFA at a particular precedence level (determined by
// calling {@link Parser//getPrecedence}).
//
func (p *ParserATNSimulator) applyPrecedenceFilter(configs ATNConfigSet) ATNConfigSet {

	statesFromAlt1 := make(map[int]PredictionContext)
	configSet := NewBaseATNConfigSet(configs.FullContext())

	for _, config := range configs.GetItems() {
		// handle alt 1 first
		if config.GetAlt() != 1 {
			continue
		}
		updatedContext := config.GetSemanticContext().evalPrecedence(p.parser, p.outerContext)
		if updatedContext == nil {
			// the configuration was eliminated
			continue
		}
		statesFromAlt1[config.GetState().GetStateNumber()] = config.GetContext()
		if updatedContext != config.GetSemanticContext() {
			configSet.Add(NewBaseATNConfig2(config, updatedContext), p.mergeCache)
		} else {
			configSet.Add(config, p.mergeCache)
		}
	}
	for _, config := range configs.GetItems() {

		if config.GetAlt() == 1 {
			// already handled
			continue
		}
		// In the future, p elimination step could be updated to also
		// filter the prediction context for alternatives predicting alt>1
		// (basically a graph subtraction algorithm).
		if !config.getPrecedenceFilterSuppressed() {
			context := statesFromAlt1[config.GetState().GetStateNumber()]
			if context != nil && context.equals(config.GetContext()) {
				// eliminated
				continue
			}
		}
		configSet.Add(config, p.mergeCache)
	}
	return configSet
}

func (p *ParserATNSimulator) getReachableTarget(trans Transition, ttype int) ATNState {
	if trans.Matches(ttype, 0, p.atn.maxTokenType) {
		return trans.getTarget()
	}

	return nil
}

func (p *ParserATNSimulator) getPredsForAmbigAlts(ambigAlts *BitSet, configs ATNConfigSet, nalts int) []SemanticContext {

	altToPred := make([]SemanticContext, nalts+1)
	for _, c := range configs.GetItems() {
		if ambigAlts.contains(c.GetAlt()) {
			altToPred[c.GetAlt()] = SemanticContextorContext(altToPred[c.GetAlt()], c.GetSemanticContext())
		}
	}
	nPredAlts := 0
	for i := 1; i <= nalts; i++ {
		pred := altToPred[i]
		if pred == nil {
			altToPred[i] = SemanticContextNone
		} else if pred != SemanticContextNone {
			nPredAlts++
		}
	}
	// nonambig alts are nil in altToPred
	if nPredAlts == 0 {
		altToPred = nil
	}
	if ParserATNSimulatorDebug {
		fmt.Println("getPredsForAmbigAlts result " + fmt.Sprint(altToPred))
	}
	return altToPred
}

func (p *ParserATNSimulator) getPredicatePredictions(ambigAlts *BitSet, altToPred []SemanticContext) []*PredPrediction {
	pairs := make([]*PredPrediction, 0)
	containsPredicate := false
	for i := 1; i < len(altToPred); i++ {
		pred := altToPred[i]
		// unpredicated is indicated by SemanticContextNONE
		if ambigAlts != nil && ambigAlts.contains(i) {
			pairs = append(pairs, NewPredPrediction(pred, i))
		}
		if pred != SemanticContextNone {
			containsPredicate = true
		}
	}
	if !containsPredicate {
		return nil
	}
	return pairs
}

//
// This method is used to improve the localization of error messages by
// choosing an alternative rather than panicing a
// {@link NoViableAltException} in particular prediction scenarios where the
// {@link //ERROR} state was reached during ATN simulation.
//
// <p>
// The default implementation of p method uses the following
// algorithm to identify an ATN configuration which successfully parsed the
// decision entry rule. Choosing such an alternative ensures that the
// {@link ParserRuleContext} returned by the calling rule will be complete
// and valid, and the syntax error will be Reported later at a more
// localized location.</p>
//
// <ul>
// <li>If a syntactically valid path or paths reach the end of the decision rule and
// they are semantically valid if predicated, return the min associated alt.</li>
// <li>Else, if a semantically invalid but syntactically valid path exist
// or paths exist, return the minimum associated alt.
// </li>
// <li>Otherwise, return {@link ATN//INVALID_ALT_NUMBER}.</li>
// </ul>
//
// <p>
// In some scenarios, the algorithm described above could predict an
// alternative which will result in a {@link FailedPredicateException} in
// the parser. Specifically, p could occur if the <em>only</em> configuration
// capable of successfully parsing to the end of the decision rule is
// blocked by a semantic predicate. By choosing p alternative within
// {@link //AdaptivePredict} instead of panicing a
// {@link NoViableAltException}, the resulting
// {@link FailedPredicateException} in the parser will identify the specific
// predicate which is preventing the parser from successfully parsing the
// decision rule, which helps developers identify and correct logic errors
// in semantic predicates.
// </p>
//
// @param configs The ATN configurations which were valid immediately before
// the {@link //ERROR} state was reached
// @param outerContext The is the \gamma_0 initial parser context from the paper
// or the parser stack at the instant before prediction commences.
//
// @return The value to return from {@link //AdaptivePredict}, or
// {@link ATN//INVALID_ALT_NUMBER} if a suitable alternative was not
// identified and {@link //AdaptivePredict} should Report an error instead.
//
func (p *ParserATNSimulator) getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(configs ATNConfigSet, outerContext ParserRuleContext) int {
	cfgs := p.splitAccordingToSemanticValidity(configs, outerContext)
	semValidConfigs := cfgs[0]
	semInvalidConfigs := cfgs[1]
	alt := p.GetAltThatFinishedDecisionEntryRule(semValidConfigs)
	if alt != ATNInvalidAltNumber { // semantically/syntactically viable path exists
		return alt
	}
	// Is there a syntactically valid path with a failed pred?
	if len(semInvalidConfigs.GetItems()) > 0 {
		alt = p.GetAltThatFinishedDecisionEntryRule(semInvalidConfigs)
		if alt != ATNInvalidAltNumber { // syntactically viable path exists
			return alt
		}
	}
	return ATNInvalidAltNumber
}

func (p *ParserATNSimulator) GetAltThatFinishedDecisionEntryRule(configs ATNConfigSet) int {
	alts := NewIntervalSet()

	for _, c := range configs.GetItems() {
		_, ok := c.GetState().(*RuleStopState)

		if c.GetReachesIntoOuterContext() > 0 || (ok && c.GetContext().hasEmptyPath()) {
			alts.addOne(c.GetAlt())
		}
	}
	if alts.length() == 0 {
		return ATNInvalidAltNumber
	}

	return alts.first()
}

// Walk the list of configurations and split them according to
//  those that have preds evaluating to true/false.  If no pred, assume
//  true pred and include in succeeded set.  Returns Pair of sets.
//
//  Create a NewSet so as not to alter the incoming parameter.
//
//  Assumption: the input stream has been restored to the starting point
//  prediction, which is where predicates need to evaluate.

type ATNConfigSetPair struct {
	item0, item1 ATNConfigSet
}

func (p *ParserATNSimulator) splitAccordingToSemanticValidity(configs ATNConfigSet, outerContext ParserRuleContext) []ATNConfigSet {
	succeeded := NewBaseATNConfigSet(configs.FullContext())
	failed := NewBaseATNConfigSet(configs.FullContext())

	for _, c := range configs.GetItems() {
		if c.GetSemanticContext() != SemanticContextNone {
			predicateEvaluationResult := c.GetSemanticContext().evaluate(p.parser, outerContext)
			if predicateEvaluationResult {
				succeeded.Add(c, nil)
			} else {
				failed.Add(c, nil)
			}
		} else {
			succeeded.Add(c, nil)
		}
	}
	return []ATNConfigSet{succeeded, failed}
}

// Look through a list of predicate/alt pairs, returning alts for the
//  pairs that win. A {@code NONE} predicate indicates an alt containing an
//  unpredicated config which behaves as "always true." If !complete
//  then we stop at the first predicate that evaluates to true. This
//  includes pairs with nil predicates.
//
func (p *ParserATNSimulator) evalSemanticContext(predPredictions []*PredPrediction, outerContext ParserRuleContext, complete bool) *BitSet {
	predictions := NewBitSet()
	for i := 0; i < len(predPredictions); i++ {
		pair := predPredictions[i]
		if pair.pred == SemanticContextNone {
			predictions.add(pair.alt)
			if !complete {
				break
			}
			continue
		}

		predicateEvaluationResult := pair.pred.evaluate(p.parser, outerContext)
		if ParserATNSimulatorDebug || ParserATNSimulatorDFADebug {
			fmt.Println("eval pred " + pair.String() + "=" + fmt.Sprint(predicateEvaluationResult))
		}
		if predicateEvaluationResult {
			if ParserATNSimulatorDebug || ParserATNSimulatorDFADebug {
				fmt.Println("PREDICT " + fmt.Sprint(pair.alt))
			}
			predictions.add(pair.alt)
			if !complete {
				break
			}
		}
	}
	return predictions
}

func (p *ParserATNSimulator) closure(config ATNConfig, configs ATNConfigSet, closureBusy Set, collectPredicates, fullCtx, treatEOFAsEpsilon bool) {
	initialDepth := 0
	p.closureCheckingStopState(config, configs, closureBusy, collectPredicates,
		fullCtx, initialDepth, treatEOFAsEpsilon)
}

func (p *ParserATNSimulator) closureCheckingStopState(config ATNConfig, configs ATNConfigSet, closureBusy Set, collectPredicates, fullCtx bool, depth int, treatEOFAsEpsilon bool) {
	if ParserATNSimulatorDebug {
		fmt.Println("closure(" + config.String() + ")")
		fmt.Println("configs(" + configs.String() + ")")
		if config.GetReachesIntoOuterContext() > 50 {
			panic("problem")
		}
	}

	if _, ok := config.GetState().(*RuleStopState); ok {
		// We hit rule end. If we have context info, use it
		// run thru all possible stack tops in ctx
		if !config.GetContext().isEmpty() {
			for i := 0; i < config.GetContext().length(); i++ {
				if config.GetContext().getReturnState(i) == BasePredictionContextEmptyReturnState {
					if fullCtx {
						configs.Add(NewBaseATNConfig1(config, config.GetState(), BasePredictionContextEMPTY), p.mergeCache)
						continue
					} else {
						// we have no context info, just chase follow links (if greedy)
						if ParserATNSimulatorDebug {
							fmt.Println("FALLING off rule " + p.getRuleName(config.GetState().GetRuleIndex()))
						}
						p.closureWork(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEOFAsEpsilon)
					}
					continue
				}
				returnState := p.atn.states[config.GetContext().getReturnState(i)]
				newContext := config.GetContext().GetParent(i) // "pop" return state

				c := NewBaseATNConfig5(returnState, config.GetAlt(), newContext, config.GetSemanticContext())
				// While we have context to pop back from, we may have
				// gotten that context AFTER having falling off a rule.
				// Make sure we track that we are now out of context.
				c.SetReachesIntoOuterContext(config.GetReachesIntoOuterContext())
				p.closureCheckingStopState(c, configs, closureBusy, collectPredicates, fullCtx, depth-1, treatEOFAsEpsilon)
			}
			return
		} else if fullCtx {
			// reached end of start rule
			configs.Add(config, p.mergeCache)
			return
		} else {
			// else if we have no context info, just chase follow links (if greedy)
			if ParserATNSimulatorDebug {
				fmt.Println("FALLING off rule " + p.getRuleName(config.GetState().GetRuleIndex()))
			}
		}
	}
	p.closureWork(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEOFAsEpsilon)
}

// Do the actual work of walking epsilon edges//
func (p *ParserATNSimulator) closureWork(config ATNConfig, configs ATNConfigSet, closureBusy Set, collectPredicates, fullCtx bool, depth int, treatEOFAsEpsilon bool) {
	state := config.GetState()
	// optimization
	if !state.GetEpsilonOnlyTransitions() {
		configs.Add(config, p.mergeCache)
		// make sure to not return here, because EOF transitions can act as
		// both epsilon transitions and non-epsilon transitions.
	}
	for i := 0; i < len(state.GetTransitions()); i++ {
		if i == 0 && p.canDropLoopEntryEdgeInLeftRecursiveRule(config) {
			continue
		}

		t := state.GetTransitions()[i]
		_, ok := t.(*ActionTransition)
		continueCollecting := collectPredicates && !ok
		c := p.getEpsilonTarget(config, t, continueCollecting, depth == 0, fullCtx, treatEOFAsEpsilon)
		if ci, ok := c.(*BaseATNConfig); ok && ci != nil {
			newDepth := depth

			if _, ok := config.GetState().(*RuleStopState); ok {
				// target fell off end of rule mark resulting c as having dipped into outer context
				// We can't get here if incoming config was rule stop and we had context
				// track how far we dip into outer context.  Might
				// come in handy and we avoid evaluating context dependent
				// preds if p is > 0.

				if p.dfa != nil && p.dfa.getPrecedenceDfa() {
					if t.(*EpsilonTransition).outermostPrecedenceReturn == p.dfa.atnStartState.GetRuleIndex() {
						c.setPrecedenceFilterSuppressed(true)
					}
				}

				c.SetReachesIntoOuterContext(c.GetReachesIntoOuterContext() + 1)

				if closureBusy.Add(c) != c {
					// avoid infinite recursion for right-recursive rules
					continue
				}

				configs.SetDipsIntoOuterContext(true) // TODO: can remove? only care when we add to set per middle of p method
				newDepth--
				if ParserATNSimulatorDebug {
					fmt.Println("dips into outer ctx: " + c.String())
				}
			} else {
				if !t.getIsEpsilon() && closureBusy.Add(c) != c {
					// avoid infinite recursion for EOF* and EOF+
					continue
				}
				if _, ok := t.(*RuleTransition); ok {
					// latch when newDepth goes negative - once we step out of the entry context we can't return
					if newDepth >= 0 {
						newDepth++
					}
				}
			}
			p.closureCheckingStopState(c, configs, closureBusy, continueCollecting, fullCtx, newDepth, treatEOFAsEpsilon)
		}
	}
}

func (p *ParserATNSimulator) canDropLoopEntryEdgeInLeftRecursiveRule(config ATNConfig) bool {
	if TurnOffLRLoopEntryBranchOpt {
		return false
	}

	_p := config.GetState()

	// First check to see if we are in StarLoopEntryState generated during
	// left-recursion elimination. For efficiency, also check if
	// the context has an empty stack case. If so, it would mean
	// global FOLLOW so we can't perform optimization
	if startLoop, ok := _p.(StarLoopEntryState); !ok || !startLoop.precedenceRuleDecision || config.GetContext().isEmpty() || config.GetContext().hasEmptyPath() {
		return false
	}

	// Require all return states to return back to the same rule
	// that p is in.
	numCtxs := config.GetContext().length()
	for i := 0; i < numCtxs; i++ {
		returnState := p.atn.states[config.GetContext().getReturnState(i)]
		if returnState.GetRuleIndex() != _p.GetRuleIndex() {
			return false
		}
	}

	decisionStartState := _p.(BlockStartState).GetTransitions()[0].getTarget().(BlockStartState)
	blockEndStateNum := decisionStartState.getEndState().stateNumber
	blockEndState := p.atn.states[blockEndStateNum].(*BlockEndState)

	// Verify that the top of each stack context leads to loop entry/exit
	// state through epsilon edges and w/o leaving rule.

	for i := 0; i < numCtxs; i++ { // for each stack context
		returnStateNumber := config.GetContext().getReturnState(i)
		returnState := p.atn.states[returnStateNumber]

		// all states must have single outgoing epsilon edge
		if len(returnState.GetTransitions()) != 1 || !returnState.GetTransitions()[0].getIsEpsilon() {
			return false
		}

		// Look for prefix op case like 'not expr', (' type ')' expr
		returnStateTarget := returnState.GetTransitions()[0].getTarget()
		if returnState.GetStateType() == ATNStateBlockEnd && returnStateTarget == _p {
			continue
		}

		// Look for 'expr op expr' or case where expr's return state is block end
		// of (...)* internal block; the block end points to loop back
		// which points to p but we don't need to check that
		if returnState == blockEndState {
			continue
		}

		// Look for ternary expr ? expr : expr. The return state points at block end,
		// which points at loop entry state
		if returnStateTarget == blockEndState {
			continue
		}

		// Look for complex prefix 'between expr and expr' case where 2nd expr's
		// return state points at block end state of (...)* internal block
		if returnStateTarget.GetStateType() == ATNStateBlockEnd &&
			len(returnStateTarget.GetTransitions()) == 1 &&
			returnStateTarget.GetTransitions()[0].getIsEpsilon() &&
			returnStateTarget.GetTransitions()[0].getTarget() == _p {
			continue
		}

		// anything else ain't conforming
		return false
	}

	return true
}

func (p *ParserATNSimulator) getRuleName(index int) string {
	if p.parser != nil && index >= 0 {
		return p.parser.GetRuleNames()[index]
	}
	var sb strings.Builder
	sb.Grow(32)

	sb.WriteString("<rule ")
	sb.WriteString(strconv.FormatInt(int64(index), 10))
	sb.WriteByte('>')
	return sb.String()
}

func (p *ParserATNSimulator) getEpsilonTarget(config ATNConfig, t Transition, collectPredicates, inContext, fullCtx, treatEOFAsEpsilon bool) ATNConfig {

	switch t.getSerializationType() {
	case TransitionRULE:
		return p.ruleTransition(config, t.(*RuleTransition))
	case TransitionPRECEDENCE:
		return p.precedenceTransition(config, t.(*PrecedencePredicateTransition), collectPredicates, inContext, fullCtx)
	case TransitionPREDICATE:
		return p.predTransition(config, t.(*PredicateTransition), collectPredicates, inContext, fullCtx)
	case TransitionACTION:
		return p.actionTransition(config, t.(*ActionTransition))
	case TransitionEPSILON:
		return NewBaseATNConfig4(config, t.getTarget())
	case TransitionATOM, TransitionRANGE, TransitionSET:
		// EOF transitions act like epsilon transitions after the first EOF
		// transition is traversed
		if treatEOFAsEpsilon {
			if t.Matches(TokenEOF, 0, 1) {
				return NewBaseATNConfig4(config, t.getTarget())
			}
		}
		return nil
	default:
		return nil
	}
}

func (p *ParserATNSimulator) actionTransition(config ATNConfig, t *ActionTransition) *BaseATNConfig {
	if ParserATNSimulatorDebug {
		fmt.Println("ACTION edge " + strconv.Itoa(t.ruleIndex) + ":" + strconv.Itoa(t.actionIndex))
	}
	return NewBaseATNConfig4(config, t.getTarget())
}

func (p *ParserATNSimulator) precedenceTransition(config ATNConfig,
	pt *PrecedencePredicateTransition, collectPredicates, inContext, fullCtx bool) *BaseATNConfig {

	if ParserATNSimulatorDebug {
		fmt.Println("PRED (collectPredicates=" + fmt.Sprint(collectPredicates) + ") " +
			strconv.Itoa(pt.precedence) + ">=_p, ctx dependent=true")
		if p.parser != nil {
			fmt.Println("context surrounding pred is " + fmt.Sprint(p.parser.GetRuleInvocationStack(nil)))
		}
	}
	var c *BaseATNConfig
	if collectPredicates && inContext {
		if fullCtx {
			// In full context mode, we can evaluate predicates on-the-fly
			// during closure, which dramatically reduces the size of
			// the config sets. It also obviates the need to test predicates
			// later during conflict resolution.
			currentPosition := p.input.Index()
			p.input.Seek(p.startIndex)
			predSucceeds := pt.getPredicate().evaluate(p.parser, p.outerContext)
			p.input.Seek(currentPosition)
			if predSucceeds {
				c = NewBaseATNConfig4(config, pt.getTarget()) // no pred context
			}
		} else {
			newSemCtx := SemanticContextandContext(config.GetSemanticContext(), pt.getPredicate())
			c = NewBaseATNConfig3(config, pt.getTarget(), newSemCtx)
		}
	} else {
		c = NewBaseATNConfig4(config, pt.getTarget())
	}
	if ParserATNSimulatorDebug {
		fmt.Println("config from pred transition=" + c.String())
	}
	return c
}

func (p *ParserATNSimulator) predTransition(config ATNConfig, pt *PredicateTransition, collectPredicates, inContext, fullCtx bool) *BaseATNConfig {

	if ParserATNSimulatorDebug {
		fmt.Println("PRED (collectPredicates=" + fmt.Sprint(collectPredicates) + ") " + strconv.Itoa(pt.ruleIndex) +
			":" + strconv.Itoa(pt.predIndex) + ", ctx dependent=" + fmt.Sprint(pt.isCtxDependent))
		if p.parser != nil {
			fmt.Println("context surrounding pred is " + fmt.Sprint(p.parser.GetRuleInvocationStack(nil)))
		}
	}
	var c *BaseATNConfig
	if collectPredicates && (!pt.isCtxDependent || inContext) {
		if fullCtx {
			// In full context mode, we can evaluate predicates on-the-fly
			// during closure, which dramatically reduces the size of
			// the config sets. It also obviates the need to test predicates
			// later during conflict resolution.
			currentPosition := p.input.Index()
			p.input.Seek(p.startIndex)
			predSucceeds := pt.getPredicate().evaluate(p.parser, p.outerContext)
			p.input.Seek(currentPosition)
			if predSucceeds {
				c = NewBaseATNConfig4(config, pt.getTarget()) // no pred context
			}
		} else {
			newSemCtx := SemanticContextandContext(config.GetSemanticContext(), pt.getPredicate())
			c = NewBaseATNConfig3(config, pt.getTarget(), newSemCtx)
		}
	} else {
		c = NewBaseATNConfig4(config, pt.getTarget())
	}
	if ParserATNSimulatorDebug {
		fmt.Println("config from pred transition=" + c.String())
	}
	return c
}

func (p *ParserATNSimulator) ruleTransition(config ATNConfig, t *RuleTransition) *BaseATNConfig {
	if ParserATNSimulatorDebug {
		fmt.Println("CALL rule " + p.getRuleName(t.getTarget().GetRuleIndex()) + ", ctx=" + config.GetContext().String())
	}
	returnState := t.followState
	newContext := SingletonBasePredictionContextCreate(config.GetContext(), returnState.GetStateNumber())
	return NewBaseATNConfig1(config, t.getTarget(), newContext)
}

func (p *ParserATNSimulator) getConflictingAlts(configs ATNConfigSet) *BitSet {
	altsets := PredictionModegetConflictingAltSubsets(configs)
	return PredictionModeGetAlts(altsets)
}

// Sam pointed out a problem with the previous definition, v3, of
// ambiguous states. If we have another state associated with conflicting
// alternatives, we should keep going. For example, the following grammar
//
// s : (ID | ID ID?) ''
//
// When the ATN simulation reaches the state before '', it has a DFA
// state that looks like: [12|1|[], 6|2|[], 12|2|[]]. Naturally
// 12|1|[] and 12|2|[] conflict, but we cannot stop processing p node
// because alternative to has another way to continue, via [6|2|[]].
// The key is that we have a single state that has config's only associated
// with a single alternative, 2, and crucially the state transitions
// among the configurations are all non-epsilon transitions. That means
// we don't consider any conflicts that include alternative 2. So, we
// ignore the conflict between alts 1 and 2. We ignore a set of
// conflicting alts when there is an intersection with an alternative
// associated with a single alt state in the state&rarrconfig-list map.
//
// It's also the case that we might have two conflicting configurations but
// also a 3rd nonconflicting configuration for a different alternative:
// [1|1|[], 1|2|[], 8|3|[]]. This can come about from grammar:
//
// a : A | A | A B
//
// After Matching input A, we reach the stop state for rule A, state 1.
// State 8 is the state right before B. Clearly alternatives 1 and 2
// conflict and no amount of further lookahead will separate the two.
// However, alternative 3 will be able to continue and so we do not
// stop working on p state. In the previous example, we're concerned
// with states associated with the conflicting alternatives. Here alt
// 3 is not associated with the conflicting configs, but since we can continue
// looking for input reasonably, I don't declare the state done. We
// ignore a set of conflicting alts when we have an alternative
// that we still need to pursue.
//

func (p *ParserATNSimulator) getConflictingAltsOrUniqueAlt(configs ATNConfigSet) *BitSet {
	var conflictingAlts *BitSet
	if configs.GetUniqueAlt() != ATNInvalidAltNumber {
		conflictingAlts = NewBitSet()
		conflictingAlts.add(configs.GetUniqueAlt())
	} else {
		conflictingAlts = configs.GetConflictingAlts()
	}
	return conflictingAlts
}

func (p *ParserATNSimulator) GetTokenName(t int) string {
	if t == TokenEOF {
		return "EOF"
	}

	if p.parser != nil && p.parser.GetLiteralNames() != nil {
		if t >= len(p.parser.GetLiteralNames()) {
			fmt.Println(strconv.Itoa(t) + " ttype out of range: " + strings.Join(p.parser.GetLiteralNames(), ","))
			//			fmt.Println(p.parser.GetInputStream().(TokenStream).GetAllText()) // p seems incorrect
		} else {
			return p.parser.GetLiteralNames()[t] + "<" + strconv.Itoa(t) + ">"
		}
	}

	return strconv.Itoa(t)
}

func (p *ParserATNSimulator) getLookaheadName(input TokenStream) string {
	return p.GetTokenName(input.LA(1))
}

// Used for debugging in AdaptivePredict around execATN but I cut
//  it out for clarity now that alg. works well. We can leave p
//  "dead" code for a bit.
//
func (p *ParserATNSimulator) dumpDeadEndConfigs(nvae *NoViableAltException) {

	panic("Not implemented")

	//    fmt.Println("dead end configs: ")
	//    var decs = nvae.deadEndConfigs
	//
	//    for i:=0; i<len(decs); i++ {
	//
	//    	c := decs[i]
	//        var trans = "no edges"
	//        if (len(c.state.GetTransitions())>0) {
	//            var t = c.state.GetTransitions()[0]
	//            if t2, ok := t.(*AtomTransition); ok {
	//                trans = "Atom "+ p.GetTokenName(t2.label)
	//            } else if t3, ok := t.(SetTransition); ok {
	//                _, ok := t.(*NotSetTransition)
	//
	//                var s string
	//                if (ok){
	//                    s = "~"
	//                }
	//
	//                trans = s + "Set " + t3.set
	//            }
	//        }
	//        fmt.Errorf(c.String(p.parser, true) + ":" + trans)
	//    }
}

func (p *ParserATNSimulator) noViableAlt(input TokenStream, outerContext ParserRuleContext, configs ATNConfigSet, startIndex int) *NoViableAltException {
	return NewNoViableAltException(p.parser, input, input.Get(startIndex), input.LT(1), configs, outerContext)
}

func (p *ParserATNSimulator) getUniqueAlt(configs ATNConfigSet) int {
	alt := ATNInvalidAltNumber
	for _, c := range configs.GetItems() {
		if alt == ATNInvalidAltNumber {
			alt = c.GetAlt() // found first alt
		} else if c.GetAlt() != alt {
			return ATNInvalidAltNumber
		}
	}
	return alt
}

//
// Add an edge to the DFA, if possible. This method calls
// {@link //addDFAState} to ensure the {@code to} state is present in the
// DFA. If {@code from} is {@code nil}, or if {@code t} is outside the
// range of edges that can be represented in the DFA tables, p method
// returns without adding the edge to the DFA.
//
// <p>If {@code to} is {@code nil}, p method returns {@code nil}.
// Otherwise, p method returns the {@link DFAState} returned by calling
// {@link //addDFAState} for the {@code to} state.</p>
//
// @param dfa The DFA
// @param from The source state for the edge
// @param t The input symbol
// @param to The target state for the edge
//
// @return If {@code to} is {@code nil}, p method returns {@code nil}
// otherwise p method returns the result of calling {@link //addDFAState}
// on {@code to}
//
func (p *ParserATNSimulator) addDFAEdge(dfa *DFA, from *DFAState, t int, to *DFAState) *DFAState {
	if ParserATNSimulatorDebug {
		fmt.Println("EDGE " + from.String() + " -> " + to.String() + " upon " + p.GetTokenName(t))
	}
	if to == nil {
		return nil
	}
	to = p.addDFAState(dfa, to) // used existing if possible not incoming
	if from == nil || t < -1 || t > p.atn.maxTokenType {
		return to
	}
	if from.getEdges() == nil {
		from.setEdges(make([]*DFAState, p.atn.maxTokenType+1+1))
	}
	from.setIthEdge(t+1, to) // connect

	if ParserATNSimulatorDebug {
		var names []string
		if p.parser != nil {
			names = p.parser.GetLiteralNames()
		}

		fmt.Println("DFA=\n" + dfa.String(names, nil))
	}
	return to
}

//
// Add state {@code D} to the DFA if it is not already present, and return
// the actual instance stored in the DFA. If a state equivalent to {@code D}
// is already in the DFA, the existing state is returned. Otherwise p
// method returns {@code D} after adding it to the DFA.
//
// <p>If {@code D} is {@link //ERROR}, p method returns {@link //ERROR} and
// does not change the DFA.</p>
//
// @param dfa The dfa
// @param D The DFA state to add
// @return The state stored in the DFA. This will be either the existing
// state if {@code D} is already in the DFA, or {@code D} itself if the
// state was not already present.
//
func (p *ParserATNSimulator) addDFAState(dfa *DFA, d *DFAState) *DFAState {
	if d == ATNSimulatorError {
		return d
	}
	hash := d.hash()
	existing, ok := dfa.getState(hash)
	if ok {
		return existing
	}
	d.stateNumber = dfa.numStates()
	if !d.configs.ReadOnly() {
		d.configs.OptimizeConfigs(p.BaseATNSimulator)
		d.configs.SetReadOnly(true)
	}
	dfa.setState(hash, d)
	if ParserATNSimulatorDebug {
		fmt.Println("adding NewDFA state: " + d.String())
	}
	return d
}

func (p *ParserATNSimulator) ReportAttemptingFullContext(dfa *DFA, conflictingAlts *BitSet, configs ATNConfigSet, startIndex, stopIndex int) {
	if ParserATNSimulatorDebug || ParserATNSimulatorRetryDebug {
		interval := NewInterval(startIndex, stopIndex+1)
		fmt.Println("ReportAttemptingFullContext decision=" + strconv.Itoa(dfa.decision) + ":" + configs.String() +
			", input=" + p.parser.GetTokenStream().GetTextFromInterval(interval))
	}
	if p.parser != nil {
		p.parser.GetErrorListenerDispatch().ReportAttemptingFullContext(p.parser, dfa, startIndex, stopIndex, conflictingAlts, configs)
	}
}

func (p *ParserATNSimulator) ReportContextSensitivity(dfa *DFA, prediction int, configs ATNConfigSet, startIndex, stopIndex int) {
	if ParserATNSimulatorDebug || ParserATNSimulatorRetryDebug {
		interval := NewInterval(startIndex, stopIndex+1)
		fmt.Println("ReportContextSensitivity decision=" + strconv.Itoa(dfa.decision) + ":" + configs.String() +
			", input=" + p.parser.GetTokenStream().GetTextFromInterval(interval))
	}
	if p.parser != nil {
		p.parser.GetErrorListenerDispatch().ReportContextSensitivity(p.parser, dfa, startIndex, stopIndex, prediction, configs)
	}
}

// If context sensitive parsing, we know it's ambiguity not conflict//
func (p *ParserATNSimulator) ReportAmbiguity(dfa *DFA, D *DFAState, startIndex, stopIndex int,
	exact bool, ambigAlts *BitSet, configs ATNConfigSet) {
	if ParserATNSimulatorDebug || ParserATNSimulatorRetryDebug {
		interval := NewInterval(startIndex, stopIndex+1)
		fmt.Println("ReportAmbiguity " + ambigAlts.String() + ":" + configs.String() +
			", input=" + p.parser.GetTokenStream().GetTextFromInterval(interval))
	}
	if p.parser != nil {
		p.parser.GetErrorListenerDispatch().ReportAmbiguity(p.parser, dfa, startIndex, stopIndex, exact, ambigAlts, configs)
	}
}
