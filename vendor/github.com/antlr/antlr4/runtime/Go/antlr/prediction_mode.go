// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// This enumeration defines the prediction modes available in ANTLR 4 along with
// utility methods for analyzing configuration sets for conflicts and/or
// ambiguities.

const (
	//
	// The SLL(*) prediction mode. This prediction mode ignores the current
	// parser context when making predictions. This is the fastest prediction
	// mode, and provides correct results for many grammars. This prediction
	// mode is more powerful than the prediction mode provided by ANTLR 3, but
	// may result in syntax errors for grammar and input combinations which are
	// not SLL.
	//
	// <p>
	// When using this prediction mode, the parser will either return a correct
	// parse tree (i.e. the same parse tree that would be returned with the
	// {@link //LL} prediction mode), or it will Report a syntax error. If a
	// syntax error is encountered when using the {@link //SLL} prediction mode,
	// it may be due to either an actual syntax error in the input or indicate
	// that the particular combination of grammar and input requires the more
	// powerful {@link //LL} prediction abilities to complete successfully.</p>
	//
	// <p>
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.</p>
	//
	PredictionModeSLL = 0
	//
	// The LL(*) prediction mode. This prediction mode allows the current parser
	// context to be used for resolving SLL conflicts that occur during
	// prediction. This is the fastest prediction mode that guarantees correct
	// parse results for all combinations of grammars with syntactically correct
	// inputs.
	//
	// <p>
	// When using this prediction mode, the parser will make correct decisions
	// for all syntactically-correct grammar and input combinations. However, in
	// cases where the grammar is truly ambiguous this prediction mode might not
	// Report a precise answer for <em>exactly which</em> alternatives are
	// ambiguous.</p>
	//
	// <p>
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.</p>
	//
	PredictionModeLL = 1
	//
	// The LL(*) prediction mode with exact ambiguity detection. In addition to
	// the correctness guarantees provided by the {@link //LL} prediction mode,
	// this prediction mode instructs the prediction algorithm to determine the
	// complete and exact set of ambiguous alternatives for every ambiguous
	// decision encountered while parsing.
	//
	// <p>
	// This prediction mode may be used for diagnosing ambiguities during
	// grammar development. Due to the performance overhead of calculating sets
	// of ambiguous alternatives, this prediction mode should be avoided when
	// the exact results are not necessary.</p>
	//
	// <p>
	// This prediction mode does not provide any guarantees for prediction
	// behavior for syntactically-incorrect inputs.</p>
	//
	PredictionModeLLExactAmbigDetection = 2
)

//
// Computes the SLL prediction termination condition.
//
// <p>
// This method computes the SLL prediction termination condition for both of
// the following cases.</p>
//
// <ul>
// <li>The usual SLL+LL fallback upon SLL conflict</li>
// <li>Pure SLL without LL fallback</li>
// </ul>
//
// <p><strong>COMBINED SLL+LL PARSING</strong></p>
//
// <p>When LL-fallback is enabled upon SLL conflict, correct predictions are
// ensured regardless of how the termination condition is computed by this
// method. Due to the substantially higher cost of LL prediction, the
// prediction should only fall back to LL when the additional lookahead
// cannot lead to a unique SLL prediction.</p>
//
// <p>Assuming combined SLL+LL parsing, an SLL configuration set with only
// conflicting subsets should fall back to full LL, even if the
// configuration sets don't resolve to the same alternative (e.g.
// {@code {1,2}} and {@code {3,4}}. If there is at least one non-conflicting
// configuration, SLL could continue with the hopes that more lookahead will
// resolve via one of those non-conflicting configurations.</p>
//
// <p>Here's the prediction termination rule them: SLL (for SLL+LL parsing)
// stops when it sees only conflicting configuration subsets. In contrast,
// full LL keeps going when there is uncertainty.</p>
//
// <p><strong>HEURISTIC</strong></p>
//
// <p>As a heuristic, we stop prediction when we see any conflicting subset
// unless we see a state that only has one alternative associated with it.
// The single-alt-state thing lets prediction continue upon rules like
// (otherwise, it would admit defeat too soon):</p>
//
// <p>{@code [12|1|[], 6|2|[], 12|2|[]]. s : (ID | ID ID?) '' }</p>
//
// <p>When the ATN simulation reaches the state before {@code ''}, it has a
// DFA state that looks like: {@code [12|1|[], 6|2|[], 12|2|[]]}. Naturally
// {@code 12|1|[]} and {@code 12|2|[]} conflict, but we cannot stop
// processing this node because alternative to has another way to continue,
// via {@code [6|2|[]]}.</p>
//
// <p>It also let's us continue for this rule:</p>
//
// <p>{@code [1|1|[], 1|2|[], 8|3|[]] a : A | A | A B }</p>
//
// <p>After Matching input A, we reach the stop state for rule A, state 1.
// State 8 is the state right before B. Clearly alternatives 1 and 2
// conflict and no amount of further lookahead will separate the two.
// However, alternative 3 will be able to continue and so we do not stop
// working on this state. In the previous example, we're concerned with
// states associated with the conflicting alternatives. Here alt 3 is not
// associated with the conflicting configs, but since we can continue
// looking for input reasonably, don't declare the state done.</p>
//
// <p><strong>PURE SLL PARSING</strong></p>
//
// <p>To handle pure SLL parsing, all we have to do is make sure that we
// combine stack contexts for configurations that differ only by semantic
// predicate. From there, we can do the usual SLL termination heuristic.</p>
//
// <p><strong>PREDICATES IN SLL+LL PARSING</strong></p>
//
// <p>SLL decisions don't evaluate predicates until after they reach DFA stop
// states because they need to create the DFA cache that works in all
// semantic situations. In contrast, full LL evaluates predicates collected
// during start state computation so it can ignore predicates thereafter.
// This means that SLL termination detection can totally ignore semantic
// predicates.</p>
//
// <p>Implementation-wise, {@link ATNConfigSet} combines stack contexts but not
// semantic predicate contexts so we might see two configurations like the
// following.</p>
//
// <p>{@code (s, 1, x, {}), (s, 1, x', {p})}</p>
//
// <p>Before testing these configurations against others, we have to merge
// {@code x} and {@code x'} (without modifying the existing configurations).
// For example, we test {@code (x+x')==x''} when looking for conflicts in
// the following configurations.</p>
//
// <p>{@code (s, 1, x, {}), (s, 1, x', {p}), (s, 2, x'', {})}</p>
//
// <p>If the configuration set has predicates (as indicated by
// {@link ATNConfigSet//hasSemanticContext}), this algorithm makes a copy of
// the configurations to strip out all of the predicates so that a standard
// {@link ATNConfigSet} will merge everything ignoring predicates.</p>
//
func PredictionModehasSLLConflictTerminatingPrediction(mode int, configs ATNConfigSet) bool {
	// Configs in rule stop states indicate reaching the end of the decision
	// rule (local context) or end of start rule (full context). If all
	// configs meet this condition, then none of the configurations is able
	// to Match additional input so we terminate prediction.
	//
	if PredictionModeallConfigsInRuleStopStates(configs) {
		return true
	}
	// pure SLL mode parsing
	if mode == PredictionModeSLL {
		// Don't bother with combining configs from different semantic
		// contexts if we can fail over to full LL costs more time
		// since we'll often fail over anyway.
		if configs.HasSemanticContext() {
			// dup configs, tossing out semantic predicates
			dup := NewBaseATNConfigSet(false)
			for _, c := range configs.GetItems() {

				//				NewBaseATNConfig({semanticContext:}, c)
				c = NewBaseATNConfig2(c, SemanticContextNone)
				dup.Add(c, nil)
			}
			configs = dup
		}
		// now we have combined contexts for configs with dissimilar preds
	}
	// pure SLL or combined SLL+LL mode parsing
	altsets := PredictionModegetConflictingAltSubsets(configs)
	return PredictionModehasConflictingAltSet(altsets) && !PredictionModehasStateAssociatedWithOneAlt(configs)
}

// Checks if any configuration in {@code configs} is in a
// {@link RuleStopState}. Configurations meeting this condition have reached
// the end of the decision rule (local context) or end of start rule (full
// context).
//
// @param configs the configuration set to test
// @return {@code true} if any configuration in {@code configs} is in a
// {@link RuleStopState}, otherwise {@code false}
func PredictionModehasConfigInRuleStopState(configs ATNConfigSet) bool {
	for _, c := range configs.GetItems() {
		if _, ok := c.GetState().(*RuleStopState); ok {
			return true
		}
	}
	return false
}

// Checks if all configurations in {@code configs} are in a
// {@link RuleStopState}. Configurations meeting this condition have reached
// the end of the decision rule (local context) or end of start rule (full
// context).
//
// @param configs the configuration set to test
// @return {@code true} if all configurations in {@code configs} are in a
// {@link RuleStopState}, otherwise {@code false}
func PredictionModeallConfigsInRuleStopStates(configs ATNConfigSet) bool {

	for _, c := range configs.GetItems() {
		if _, ok := c.GetState().(*RuleStopState); !ok {
			return false
		}
	}
	return true
}

//
// Full LL prediction termination.
//
// <p>Can we stop looking ahead during ATN simulation or is there some
// uncertainty as to which alternative we will ultimately pick, after
// consuming more input? Even if there are partial conflicts, we might know
// that everything is going to resolve to the same minimum alternative. That
// means we can stop since no more lookahead will change that fact. On the
// other hand, there might be multiple conflicts that resolve to different
// minimums. That means we need more look ahead to decide which of those
// alternatives we should predict.</p>
//
// <p>The basic idea is to split the set of configurations {@code C}, into
// conflicting subsets {@code (s, _, ctx, _)} and singleton subsets with
// non-conflicting configurations. Two configurations conflict if they have
// identical {@link ATNConfig//state} and {@link ATNConfig//context} values
// but different {@link ATNConfig//alt} value, e.g. {@code (s, i, ctx, _)}
// and {@code (s, j, ctx, _)} for {@code i!=j}.</p>
//
// <p>Reduce these configuration subsets to the set of possible alternatives.
// You can compute the alternative subsets in one pass as follows:</p>
//
// <p>{@code A_s,ctx = {i | (s, i, ctx, _)}} for each configuration in
// {@code C} holding {@code s} and {@code ctx} fixed.</p>
//
// <p>Or in pseudo-code, for each configuration {@code c} in {@code C}:</p>
//
// <pre>
// map[c] U= c.{@link ATNConfig//alt alt} // map hash/equals uses s and x, not
// alt and not pred
// </pre>
//
// <p>The values in {@code map} are the set of {@code A_s,ctx} sets.</p>
//
// <p>If {@code |A_s,ctx|=1} then there is no conflict associated with
// {@code s} and {@code ctx}.</p>
//
// <p>Reduce the subsets to singletons by choosing a minimum of each subset. If
// the union of these alternative subsets is a singleton, then no amount of
// more lookahead will help us. We will always pick that alternative. If,
// however, there is more than one alternative, then we are uncertain which
// alternative to predict and must continue looking for resolution. We may
// or may not discover an ambiguity in the future, even if there are no
// conflicting subsets this round.</p>
//
// <p>The biggest sin is to terminate early because it means we've made a
// decision but were uncertain as to the eventual outcome. We haven't used
// enough lookahead. On the other hand, announcing a conflict too late is no
// big deal you will still have the conflict. It's just inefficient. It
// might even look until the end of file.</p>
//
// <p>No special consideration for semantic predicates is required because
// predicates are evaluated on-the-fly for full LL prediction, ensuring that
// no configuration contains a semantic context during the termination
// check.</p>
//
// <p><strong>CONFLICTING CONFIGS</strong></p>
//
// <p>Two configurations {@code (s, i, x)} and {@code (s, j, x')}, conflict
// when {@code i!=j} but {@code x=x'}. Because we merge all
// {@code (s, i, _)} configurations together, that means that there are at
// most {@code n} configurations associated with state {@code s} for
// {@code n} possible alternatives in the decision. The merged stacks
// complicate the comparison of configuration contexts {@code x} and
// {@code x'}. Sam checks to see if one is a subset of the other by calling
// merge and checking to see if the merged result is either {@code x} or
// {@code x'}. If the {@code x} associated with lowest alternative {@code i}
// is the superset, then {@code i} is the only possible prediction since the
// others resolve to {@code min(i)} as well. However, if {@code x} is
// associated with {@code j>i} then at least one stack configuration for
// {@code j} is not in conflict with alternative {@code i}. The algorithm
// should keep going, looking for more lookahead due to the uncertainty.</p>
//
// <p>For simplicity, I'm doing a equality check between {@code x} and
// {@code x'} that lets the algorithm continue to consume lookahead longer
// than necessary. The reason I like the equality is of course the
// simplicity but also because that is the test you need to detect the
// alternatives that are actually in conflict.</p>
//
// <p><strong>CONTINUE/STOP RULE</strong></p>
//
// <p>Continue if union of resolved alternative sets from non-conflicting and
// conflicting alternative subsets has more than one alternative. We are
// uncertain about which alternative to predict.</p>
//
// <p>The complete set of alternatives, {@code [i for (_,i,_)]}, tells us which
// alternatives are still in the running for the amount of input we've
// consumed at this point. The conflicting sets let us to strip away
// configurations that won't lead to more states because we resolve
// conflicts to the configuration with a minimum alternate for the
// conflicting set.</p>
//
// <p><strong>CASES</strong></p>
//
// <ul>
//
// <li>no conflicts and more than 1 alternative in set =&gt continue</li>
//
// <li> {@code (s, 1, x)}, {@code (s, 2, x)}, {@code (s, 3, z)},
// {@code (s', 1, y)}, {@code (s', 2, y)} yields non-conflicting set
// {@code {3}} U conflicting sets {@code min({1,2})} U {@code min({1,2})} =
// {@code {1,3}} =&gt continue
// </li>
//
// <li>{@code (s, 1, x)}, {@code (s, 2, x)}, {@code (s', 1, y)},
// {@code (s', 2, y)}, {@code (s'', 1, z)} yields non-conflicting set
// {@code {1}} U conflicting sets {@code min({1,2})} U {@code min({1,2})} =
// {@code {1}} =&gt stop and predict 1</li>
//
// <li>{@code (s, 1, x)}, {@code (s, 2, x)}, {@code (s', 1, y)},
// {@code (s', 2, y)} yields conflicting, reduced sets {@code {1}} U
// {@code {1}} = {@code {1}} =&gt stop and predict 1, can announce
// ambiguity {@code {1,2}}</li>
//
// <li>{@code (s, 1, x)}, {@code (s, 2, x)}, {@code (s', 2, y)},
// {@code (s', 3, y)} yields conflicting, reduced sets {@code {1}} U
// {@code {2}} = {@code {1,2}} =&gt continue</li>
//
// <li>{@code (s, 1, x)}, {@code (s, 2, x)}, {@code (s', 3, y)},
// {@code (s', 4, y)} yields conflicting, reduced sets {@code {1}} U
// {@code {3}} = {@code {1,3}} =&gt continue</li>
//
// </ul>
//
// <p><strong>EXACT AMBIGUITY DETECTION</strong></p>
//
// <p>If all states Report the same conflicting set of alternatives, then we
// know we have the exact ambiguity set.</p>
//
// <p><code>|A_<em>i</em>|&gt1</code> and
// <code>A_<em>i</em> = A_<em>j</em></code> for all <em>i</em>, <em>j</em>.</p>
//
// <p>In other words, we continue examining lookahead until all {@code A_i}
// have more than one alternative and all {@code A_i} are the same. If
// {@code A={{1,2}, {1,3}}}, then regular LL prediction would terminate
// because the resolved set is {@code {1}}. To determine what the real
// ambiguity is, we have to know whether the ambiguity is between one and
// two or one and three so we keep going. We can only stop prediction when
// we need exact ambiguity detection when the sets look like
// {@code A={{1,2}}} or {@code {{1,2},{1,2}}}, etc...</p>
//
func PredictionModeresolvesToJustOneViableAlt(altsets []*BitSet) int {
	return PredictionModegetSingleViableAlt(altsets)
}

//
// Determines if every alternative subset in {@code altsets} contains more
// than one alternative.
//
// @param altsets a collection of alternative subsets
// @return {@code true} if every {@link BitSet} in {@code altsets} has
// {@link BitSet//cardinality cardinality} &gt 1, otherwise {@code false}
//
func PredictionModeallSubsetsConflict(altsets []*BitSet) bool {
	return !PredictionModehasNonConflictingAltSet(altsets)
}

//
// Determines if any single alternative subset in {@code altsets} contains
// exactly one alternative.
//
// @param altsets a collection of alternative subsets
// @return {@code true} if {@code altsets} contains a {@link BitSet} with
// {@link BitSet//cardinality cardinality} 1, otherwise {@code false}
//
func PredictionModehasNonConflictingAltSet(altsets []*BitSet) bool {
	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		if alts.length() == 1 {
			return true
		}
	}
	return false
}

//
// Determines if any single alternative subset in {@code altsets} contains
// more than one alternative.
//
// @param altsets a collection of alternative subsets
// @return {@code true} if {@code altsets} contains a {@link BitSet} with
// {@link BitSet//cardinality cardinality} &gt 1, otherwise {@code false}
//
func PredictionModehasConflictingAltSet(altsets []*BitSet) bool {
	for i := 0; i < len(altsets); i++ {
		alts := altsets[i]
		if alts.length() > 1 {
			return true
		}
	}
	return false
}

//
// Determines if every alternative subset in {@code altsets} is equivalent.
//
// @param altsets a collection of alternative subsets
// @return {@code true} if every member of {@code altsets} is equal to the
// others, otherwise {@code false}
//
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

//
// Returns the unique alternative predicted by all alternative subsets in
// {@code altsets}. If no such alternative exists, this method returns
// {@link ATN//INVALID_ALT_NUMBER}.
//
// @param altsets a collection of alternative subsets
//
func PredictionModegetUniqueAlt(altsets []*BitSet) int {
	all := PredictionModeGetAlts(altsets)
	if all.length() == 1 {
		return all.minValue()
	}

	return ATNInvalidAltNumber
}

// Gets the complete set of represented alternatives for a collection of
// alternative subsets. This method returns the union of each {@link BitSet}
// in {@code altsets}.
//
// @param altsets a collection of alternative subsets
// @return the set of represented alternatives in {@code altsets}
//
func PredictionModeGetAlts(altsets []*BitSet) *BitSet {
	all := NewBitSet()
	for _, alts := range altsets {
		all.or(alts)
	}
	return all
}

//
// This func gets the conflicting alt subsets from a configuration set.
// For each configuration {@code c} in {@code configs}:
//
// <pre>
// map[c] U= c.{@link ATNConfig//alt alt} // map hash/equals uses s and x, not
// alt and not pred
// </pre>
//
func PredictionModegetConflictingAltSubsets(configs ATNConfigSet) []*BitSet {
	configToAlts := make(map[int]*BitSet)

	for _, c := range configs.GetItems() {
		key := 31 * c.GetState().GetStateNumber() + c.GetContext().hash()

		alts, ok := configToAlts[key]
		if !ok {
			alts = NewBitSet()
			configToAlts[key] = alts
		}
		alts.add(c.GetAlt())
	}

	values := make([]*BitSet, 0, 10)
	for _, v := range configToAlts {
		values = append(values, v)
	}
	return values
}

//
// Get a map from state to alt subset from a configuration set. For each
// configuration {@code c} in {@code configs}:
//
// <pre>
// map[c.{@link ATNConfig//state state}] U= c.{@link ATNConfig//alt alt}
// </pre>
//
func PredictionModeGetStateToAltMap(configs ATNConfigSet) *AltDict {
	m := NewAltDict()

	for _, c := range configs.GetItems() {
		alts := m.Get(c.GetState().String())
		if alts == nil {
			alts = NewBitSet()
			m.put(c.GetState().String(), alts)
		}
		alts.(*BitSet).add(c.GetAlt())
	}
	return m
}

func PredictionModehasStateAssociatedWithOneAlt(configs ATNConfigSet) bool {
	values := PredictionModeGetStateToAltMap(configs).values()
	for i := 0; i < len(values); i++ {
		if values[i].(*BitSet).length() == 1 {
			return true
		}
	}
	return false
}

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
