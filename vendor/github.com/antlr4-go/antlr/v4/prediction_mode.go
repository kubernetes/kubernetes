// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// This enumeration defines the prediction modes available in ANTLR 4 along with
// utility methods for analyzing configuration sets for conflicts and/or
// ambiguities.

const (
	// PredictionModeSLL represents the SLL(*) prediction mode.
	// This prediction mode ignores the current
	// parser context when making predictions. This is the fastest prediction
	// mode, and provides correct results for many grammars. This prediction
	// mode is more powerful than the prediction mode provided by ANTLR 3, but
	// may result in syntax errors for grammar and input combinations which are
	// not SLL.
	//
	// When using this prediction mode, the parser will either return a correct
	// parse tree (i.e. the same parse tree that would be returned with the
	// [PredictionModeLL] prediction mode), or it will Report a syntax error. If a
	// syntax error is encountered when using the SLL prediction mode,
	// it may be due to either an actual syntax error in the input or indicate
	// that the particular combination of grammar and input requires the more
	// powerful LL prediction abilities to complete successfully.
	//
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.
	//
	PredictionModeSLL = 0

	// PredictionModeLL represents the LL(*) prediction mode.
	// This prediction mode allows the current parser
	// context to be used for resolving SLL conflicts that occur during
	// prediction. This is the fastest prediction mode that guarantees correct
	// parse results for all combinations of grammars with syntactically correct
	// inputs.
	//
	// When using this prediction mode, the parser will make correct decisions
	// for all syntactically-correct grammar and input combinations. However, in
	// cases where the grammar is truly ambiguous this prediction mode might not
	// report a precise answer for exactly which alternatives are
	// ambiguous.
	//
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.
	//
	PredictionModeLL = 1

	// PredictionModeLLExactAmbigDetection represents the LL(*) prediction mode
	// with exact ambiguity detection.
	//
	// In addition to the correctness guarantees provided by the [PredictionModeLL] prediction mode,
	// this prediction mode instructs the prediction algorithm to determine the
	// complete and exact set of ambiguous alternatives for every ambiguous
	// decision encountered while parsing.
	//
	// This prediction mode may be used for diagnosing ambiguities during
	// grammar development. Due to the performance overhead of calculating sets
	// of ambiguous alternatives, this prediction mode should be avoided when
	// the exact results are not necessary.
	//
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.
	//
	PredictionModeLLExactAmbigDetection = 2
)

// PredictionModehasSLLConflictTerminatingPrediction computes the SLL prediction termination condition.
//
// This method computes the SLL prediction termination condition for both of
// the following cases:
//
//   - The usual SLL+LL fallback upon SLL conflict
//   - Pure SLL without LL fallback
//
// # Combined SLL+LL Parsing
//
// When LL-fallback is enabled upon SLL conflict, correct predictions are
// ensured regardless of how the termination condition is computed by this
// method. Due to the substantially higher cost of LL prediction, the
// prediction should only fall back to LL when the additional lookahead
// cannot lead to a unique SLL prediction.
//
// Assuming combined SLL+LL parsing, an SLL configuration set with only
// conflicting subsets should fall back to full LL, even if the
// configuration sets don't resolve to the same alternative, e.g.
//
//	{1,2} and {3,4}
//
// If there is at least one non-conflicting
// configuration, SLL could continue with the hopes that more lookahead will
// resolve via one of those non-conflicting configurations.
//
// Here's the prediction termination rule them: SLL (for SLL+LL parsing)
// stops when it sees only conflicting configuration subsets. In contrast,
// full LL keeps going when there is uncertainty.
//
// # Heuristic
//
// As a heuristic, we stop prediction when we see any conflicting subset
// unless we see a state that only has one alternative associated with it.
// The single-alt-state thing lets prediction continue upon rules like
// (otherwise, it would admit defeat too soon):
//
//	[12|1|[], 6|2|[], 12|2|[]]. s : (ID | ID ID?) ;
//
// When the [ATN] simulation reaches the state before ';', it has a
// [DFA] state that looks like:
//
//	[12|1|[], 6|2|[], 12|2|[]]
//
// Naturally
//
//	12|1|[] and  12|2|[]
//
// conflict, but we cannot stop processing this node because alternative to has another way to continue,
// via
//
//	[6|2|[]]
//
// It also let's us continue for this rule:
//
//	[1|1|[], 1|2|[], 8|3|[]] a : A | A | A B ;
//
// After Matching input A, we reach the stop state for rule A, state 1.
// State 8 is the state immediately before B. Clearly alternatives 1 and 2
// conflict and no amount of further lookahead will separate the two.
// However, alternative 3 will be able to continue, and so we do not stop
// working on this state. In the previous example, we're concerned with
// states associated with the conflicting alternatives. Here alt 3 is not
// associated with the conflicting configs, but since we can continue
// looking for input reasonably, don't declare the state done.
//
// # Pure SLL Parsing
//
// To handle pure SLL parsing, all we have to do is make sure that we
// combine stack contexts for configurations that differ only by semantic
// predicate. From there, we can do the usual SLL termination heuristic.
//
// # Predicates in SLL+LL Parsing
//
// SLL decisions don't evaluate predicates until after they reach [DFA] stop
// states because they need to create the [DFA] cache that works in all
// semantic situations. In contrast, full LL evaluates predicates collected
// during start state computation, so it can ignore predicates thereafter.
// This means that SLL termination detection can totally ignore semantic
// predicates.
//
// Implementation-wise, [ATNConfigSet] combines stack contexts but not
// semantic predicate contexts, so we might see two configurations like the
// following:
//
//	(s, 1, x, {}), (s, 1, x', {p})
//
// Before testing these configurations against others, we have to merge
// x and x' (without modifying the existing configurations).
// For example, we test (x+x')==x” when looking for conflicts in
// the following configurations:
//
//	(s, 1, x, {}), (s, 1, x', {p}), (s, 2, x”, {})
//
// If the configuration set has predicates (as indicated by
// [ATNConfigSet.hasSemanticContext]), this algorithm makes a copy of
// the configurations to strip out all the predicates so that a standard
// [ATNConfigSet] will merge everything ignoring predicates.
func PredictionModehasSLLConflictTerminatingPrediction(mode int, configs *ATNConfigSet) bool {

	// Configs in rule stop states indicate reaching the end of the decision
	// rule (local context) or end of start rule (full context). If all
	// configs meet this condition, then none of the configurations is able
	// to Match additional input, so we terminate prediction.
	//
	if PredictionModeallConfigsInRuleStopStates(configs) {
		return true
	}

	// pure SLL mode parsing
	if mode == PredictionModeSLL {
		// Don't bother with combining configs from different semantic
		// contexts if we can fail over to full LL costs more time
		// since we'll often fail over anyway.
		if configs.hasSemanticContext {
			// dup configs, tossing out semantic predicates
			dup := NewATNConfigSet(false)
			for _, c := range configs.configs {

				//				NewATNConfig({semanticContext:}, c)
				c = NewATNConfig2(c, SemanticContextNone)
				dup.Add(c, nil)
			}
			configs = dup
		}
		// now we have combined contexts for configs with dissimilar predicates
	}
	// pure SLL or combined SLL+LL mode parsing
	altsets := PredictionModegetConflictingAltSubsets(configs)
	return PredictionModehasConflictingAltSet(altsets) && !PredictionModehasStateAssociatedWithOneAlt(configs)
}

// PredictionModehasConfigInRuleStopState checks if any configuration in the given configs is in a
// [RuleStopState]. Configurations meeting this condition have reached
// the end of the decision rule (local context) or end of start rule (full
// context).
//
// The func returns true if any configuration in the supplied configs is in a [RuleStopState]
func PredictionModehasConfigInRuleStopState(configs *ATNConfigSet) bool {
	for _, c := range configs.configs {
		if _, ok := c.GetState().(*RuleStopState); ok {
			return true
		}
	}
	return false
}

// PredictionModeallConfigsInRuleStopStates checks if all configurations in configs are in a
// [RuleStopState]. Configurations meeting this condition have reached
// the end of the decision rule (local context) or end of start rule (full
// context).
//
// the func returns true if all configurations in configs are in a
// [RuleStopState]
func PredictionModeallConfigsInRuleStopStates(configs *ATNConfigSet) bool {

	for _, c := range configs.configs {
		if _, ok := c.GetState().(*RuleStopState); !ok {
			return false
		}
	}
	return true
}

// PredictionModeresolvesToJustOneViableAlt checks full LL prediction termination.
//
// Can we stop looking ahead during [ATN] simulation or is there some
// uncertainty as to which alternative we will ultimately pick, after
// consuming more input? Even if there are partial conflicts, we might know
// that everything is going to resolve to the same minimum alternative. That
// means we can stop since no more lookahead will change that fact. On the
// other hand, there might be multiple conflicts that resolve to different
// minimums. That means we need more look ahead to decide which of those
// alternatives we should predict.
//
// The basic idea is to split the set of configurations 'C', into
// conflicting subsets (s, _, ctx, _) and singleton subsets with
// non-conflicting configurations. Two configurations conflict if they have
// identical [ATNConfig].state and [ATNConfig].context values
// but a different [ATNConfig].alt value, e.g.
//
//	(s, i, ctx, _)
//
// and
//
//	(s, j, ctx, _) ; for i != j
//
// Reduce these configuration subsets to the set of possible alternatives.
// You can compute the alternative subsets in one pass as follows:
//
//	A_s,ctx = {i | (s, i, ctx, _)}
//
// for each configuration in C holding s and ctx fixed.
//
// Or in pseudo-code:
//
//	 for each configuration c in  C:
//		  map[c] U = c.ATNConfig.alt alt  // map hash/equals uses s and x, not alt and not pred
//
// The values in map are the set of
//
//	A_s,ctx
//
// sets.
//
// If
//
//	|A_s,ctx| = 1
//
// then there is no conflict associated with s and ctx.
//
// Reduce the subsets to singletons by choosing a minimum of each subset. If
// the union of these alternative subsets is a singleton, then no amount of
// further lookahead will help us. We will always pick that alternative. If,
// however, there is more than one alternative, then we are uncertain which
// alternative to predict and must continue looking for resolution. We may
// or may not discover an ambiguity in the future, even if there are no
// conflicting subsets this round.
//
// The biggest sin is to terminate early because it means we've made a
// decision but were uncertain as to the eventual outcome. We haven't used
// enough lookahead. On the other hand, announcing a conflict too late is no
// big deal; you will still have the conflict. It's just inefficient. It
// might even look until the end of file.
//
// No special consideration for semantic predicates is required because
// predicates are evaluated on-the-fly for full LL prediction, ensuring that
// no configuration contains a semantic context during the termination
// check.
//
// # Conflicting Configs
//
// Two configurations:
//
//	(s, i, x) and (s, j, x')
//
// conflict when i != j  but x = x'. Because we merge all
// (s, i, _) configurations together, that means that there are at
// most n configurations associated with state s for
// n possible alternatives in the decision. The merged stacks
// complicate the comparison of configuration contexts x and x'.
//
// Sam checks to see if one is a subset of the other by calling
// merge and checking to see if the merged result is either x or x'.
// If the x associated with lowest alternative i
// is the superset, then i is the only possible prediction since the
// others resolve to min(i) as well. However, if x is
// associated with j > i then at least one stack configuration for
// j is not in conflict with alternative i. The algorithm
// should keep going, looking for more lookahead due to the uncertainty.
//
// For simplicity, I'm doing an equality check between x and
// x', which lets the algorithm continue to consume lookahead longer
// than necessary. The reason I like the equality is of course the
// simplicity but also because that is the test you need to detect the
// alternatives that are actually in conflict.
//
// # Continue/Stop Rule
//
// Continue if the union of resolved alternative sets from non-conflicting and
// conflicting alternative subsets has more than one alternative. We are
// uncertain about which alternative to predict.
//
// The complete set of alternatives,
//
//	[i for (_, i, _)]
//
// tells us which alternatives are still in the running for the amount of input we've
// consumed at this point. The conflicting sets let us to strip away
// configurations that won't lead to more states because we resolve
// conflicts to the configuration with a minimum alternate for the
// conflicting set.
//
// Cases
//
//   - no conflicts and more than 1 alternative in set => continue
//   - (s, 1, x), (s, 2, x), (s, 3, z), (s', 1, y), (s', 2, y) yields non-conflicting set
//     {3} ∪ conflicting sets min({1,2}) ∪ min({1,2}) = {1,3} => continue
//   - (s, 1, x), (s, 2, x), (s', 1, y), (s', 2, y), (s”, 1, z) yields non-conflicting set
//     {1} ∪ conflicting sets min({1,2}) ∪ min({1,2}) = {1} => stop and predict 1
//   - (s, 1, x), (s, 2, x), (s', 1, y), (s', 2, y) yields conflicting, reduced sets
//     {1} ∪ {1} = {1} => stop and predict 1, can announce ambiguity {1,2}
//   - (s, 1, x), (s, 2, x), (s', 2, y), (s', 3, y) yields conflicting, reduced sets
//     {1} ∪ {2} = {1,2} => continue
//   - (s, 1, x), (s, 2, x), (s', 2, y), (s', 3, y) yields conflicting, reduced sets
//     {1} ∪ {2} = {1,2} => continue
//   - (s, 1, x), (s, 2, x), (s', 3, y), (s', 4, y) yields conflicting, reduced sets
//     {1} ∪ {3} = {1,3} => continue
//
// # Exact Ambiguity Detection
//
// If all states report the same conflicting set of alternatives, then we
// know we have the exact ambiguity set:
//
//	|A_i| > 1
//
// and
//
//	A_i = A_j ; for all i, j
//
// In other words, we continue examining lookahead until all A_i
// have more than one alternative and all A_i are the same. If
//
//	A={{1,2}, {1,3}}
//
// then regular LL prediction would terminate because the resolved set is {1}.
// To determine what the real ambiguity is, we have to know whether the ambiguity is between one and
// two or one and three so we keep going. We can only stop prediction when
// we need exact ambiguity detection when the sets look like:
//
//	A={{1,2}}
//
// or
//
//	{{1,2},{1,2}}, etc...
func PredictionModeresolvesToJustOneViableAlt(altsets []*BitSet) int {
	return PredictionModegetSingleViableAlt(altsets)
}

// PredictionModeallSubsetsConflict determines if every alternative subset in altsets contains more
// than one alternative.
//
// The func returns true if every [BitSet] in altsets has
// [BitSet].cardinality cardinality > 1
func PredictionModeallSubsetsConflict(altsets []*BitSet) bool {
	return !PredictionModehasNonConflictingAltSet(altsets)
}

// PredictionModehasNonConflictingAltSet determines if any single alternative subset in altsets contains
// exactly one alternative.
//
// The func returns true if altsets contains at least one [BitSet] with
// [BitSet].cardinality cardinality 1
func PredictionModehasNonConflictingAltSet(altsets []*BitSet) bool {
	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		if alts.length() == 1 {
			return true
		}
	}
	return false
}

// PredictionModehasConflictingAltSet determines if any single alternative subset in altsets contains
// more than one alternative.
//
// The func returns true if altsets contains a [BitSet] with
// [BitSet].cardinality cardinality > 1, otherwise false
func PredictionModehasConflictingAltSet(altsets []*BitSet) bool {
	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		if alts.length() > 1 {
			return true
		}
	}
	return false
}

// PredictionModeallSubsetsEqual determines if every alternative subset in altsets is equivalent.
//
// The func returns true if every member of altsets is equal to the others.
func PredictionModeallSubsetsEqual(altsets []*BitSet) bool {
	var first *BitSet

	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		if first == nil {
			first = alts
		} else if alts != first {
			return false
		}
	}

	return true
}

// PredictionModegetUniqueAlt returns the unique alternative predicted by all alternative subsets in
// altsets. If no such alternative exists, this method returns
// [ATNInvalidAltNumber].
//
// @param altsets a collection of alternative subsets
func PredictionModegetUniqueAlt(altsets []*BitSet) int {
	all := PredictionModeGetAlts(altsets)
	if all.length() == 1 {
		return all.minValue()
	}

	return ATNInvalidAltNumber
}

// PredictionModeGetAlts returns the complete set of represented alternatives for a collection of
// alternative subsets. This method returns the union of each [BitSet]
// in altsets, being the set of represented alternatives in altsets.
func PredictionModeGetAlts(altsets []*BitSet) *BitSet {
	all := NewBitSet()
	for _, alts := range altsets {
		all.or(alts)
	}
	return all
}

// PredictionModegetConflictingAltSubsets gets the conflicting alt subsets from a configuration set.
//
//	for each configuration c in configs:
//	   map[c] U= c.ATNConfig.alt // map hash/equals uses s and x, not alt and not pred
func PredictionModegetConflictingAltSubsets(configs *ATNConfigSet) []*BitSet {
	configToAlts := NewJMap[*ATNConfig, *BitSet, *ATNAltConfigComparator[*ATNConfig]](atnAltCfgEqInst, AltSetCollection, "PredictionModegetConflictingAltSubsets()")

	for _, c := range configs.configs {

		alts, ok := configToAlts.Get(c)
		if !ok {
			alts = NewBitSet()
			configToAlts.Put(c, alts)
		}
		alts.add(c.GetAlt())
	}

	return configToAlts.Values()
}

// PredictionModeGetStateToAltMap gets a map from state to alt subset from a configuration set.
//
//	for each configuration c in configs:
//	   map[c.ATNConfig.state] U= c.ATNConfig.alt}
func PredictionModeGetStateToAltMap(configs *ATNConfigSet) *AltDict {
	m := NewAltDict()

	for _, c := range configs.configs {
		alts := m.Get(c.GetState().String())
		if alts == nil {
			alts = NewBitSet()
			m.put(c.GetState().String(), alts)
		}
		alts.(*BitSet).add(c.GetAlt())
	}
	return m
}

func PredictionModehasStateAssociatedWithOneAlt(configs *ATNConfigSet) bool {
	values := PredictionModeGetStateToAltMap(configs).values()
	for i := 0; i < len(values); i++ {
		if values[i].(*BitSet).length() == 1 {
			return true
		}
	}
	return false
}

// PredictionModegetSingleViableAlt gets the single alternative predicted by all alternative subsets in altsets
// if there is one.
//
// TODO: JI - Review this code - it does not seem to do the same thing as the Java code - maybe because [BitSet] is not like the Java utils BitSet
func PredictionModegetSingleViableAlt(altsets []*BitSet) int {
	result := ATNInvalidAltNumber

	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		minAlt := alts.minValue()
		if result == ATNInvalidAltNumber {
			result = minAlt
		} else if result != minAlt { // more than 1 viable alt
			return ATNInvalidAltNumber
		}
	}
	return result
}
