// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"strconv"
)

//
// This implementation of {@link ANTLRErrorListener} can be used to identify
// certain potential correctness and performance problems in grammars. "reports"
// are made by calling {@link Parser//NotifyErrorListeners} with the appropriate
// message.
//
// <ul>
// <li><b>Ambiguities</b>: These are cases where more than one path through the
// grammar can Match the input.</li>
// <li><b>Weak context sensitivity</b>: These are cases where full-context
// prediction resolved an SLL conflict to a unique alternative which equaled the
// minimum alternative of the SLL conflict.</li>
// <li><b>Strong (forced) context sensitivity</b>: These are cases where the
// full-context prediction resolved an SLL conflict to a unique alternative,
// <em>and</em> the minimum alternative of the SLL conflict was found to not be
// a truly viable alternative. Two-stage parsing cannot be used for inputs where
// d situation occurs.</li>
// </ul>

type DiagnosticErrorListener struct {
	*DefaultErrorListener

	exactOnly bool
}

func NewDiagnosticErrorListener(exactOnly bool) *DiagnosticErrorListener {

	n := new(DiagnosticErrorListener)

	// whether all ambiguities or only exact ambiguities are Reported.
	n.exactOnly = exactOnly
	return n
}

func (d *DiagnosticErrorListener) ReportAmbiguity(recognizer Parser, dfa *DFA, startIndex, stopIndex int, exact bool, ambigAlts *BitSet, configs ATNConfigSet) {
	if d.exactOnly && !exact {
		return
	}
	msg := "reportAmbiguity d=" +
		d.getDecisionDescription(recognizer, dfa) +
		": ambigAlts=" +
		d.getConflictingAlts(ambigAlts, configs).String() +
		", input='" +
		recognizer.GetTokenStream().GetTextFromInterval(NewInterval(startIndex, stopIndex)) + "'"
	recognizer.NotifyErrorListeners(msg, nil, nil)
}

func (d *DiagnosticErrorListener) ReportAttemptingFullContext(recognizer Parser, dfa *DFA, startIndex, stopIndex int, conflictingAlts *BitSet, configs ATNConfigSet) {

	msg := "reportAttemptingFullContext d=" +
		d.getDecisionDescription(recognizer, dfa) +
		", input='" +
		recognizer.GetTokenStream().GetTextFromInterval(NewInterval(startIndex, stopIndex)) + "'"
	recognizer.NotifyErrorListeners(msg, nil, nil)
}

func (d *DiagnosticErrorListener) ReportContextSensitivity(recognizer Parser, dfa *DFA, startIndex, stopIndex, prediction int, configs ATNConfigSet) {
	msg := "reportContextSensitivity d=" +
		d.getDecisionDescription(recognizer, dfa) +
		", input='" +
		recognizer.GetTokenStream().GetTextFromInterval(NewInterval(startIndex, stopIndex)) + "'"
	recognizer.NotifyErrorListeners(msg, nil, nil)
}

func (d *DiagnosticErrorListener) getDecisionDescription(recognizer Parser, dfa *DFA) string {
	decision := dfa.decision
	ruleIndex := dfa.atnStartState.GetRuleIndex()

	ruleNames := recognizer.GetRuleNames()
	if ruleIndex < 0 || ruleIndex >= len(ruleNames) {
		return strconv.Itoa(decision)
	}
	ruleName := ruleNames[ruleIndex]
	if ruleName == "" {
		return strconv.Itoa(decision)
	}
	return strconv.Itoa(decision) + " (" + ruleName + ")"
}

// Computes the set of conflicting or ambiguous alternatives from a
// configuration set, if that information was not already provided by the
// parser.
//
// @param ReportedAlts The set of conflicting or ambiguous alternatives, as
// Reported by the parser.
// @param configs The conflicting or ambiguous configuration set.
// @return Returns {@code ReportedAlts} if it is not {@code nil}, otherwise
// returns the set of alternatives represented in {@code configs}.
func (d *DiagnosticErrorListener) getConflictingAlts(ReportedAlts *BitSet, set ATNConfigSet) *BitSet {
	if ReportedAlts != nil {
		return ReportedAlts
	}
	result := NewBitSet()
	for _, c := range set.GetItems() {
		result.add(c.GetAlt())
	}

	return result
}
