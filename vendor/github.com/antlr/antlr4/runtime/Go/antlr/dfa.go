// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"sort"
)

type DFA struct {
	// atnStartState is the ATN state in which this was created
	atnStartState DecisionState

	decision int

	// states is all the DFA states. Use Map to get the old state back; Set can only
	// indicate whether it is there.
	states map[int]*DFAState

	s0 *DFAState

	// precedenceDfa is the backing field for isPrecedenceDfa and setPrecedenceDfa.
	// True if the DFA is for a precedence decision and false otherwise.
	precedenceDfa bool
}

func NewDFA(atnStartState DecisionState, decision int) *DFA {
	dfa := &DFA{
		atnStartState: atnStartState,
		decision:      decision,
		states:        make(map[int]*DFAState),
	}
	if s, ok := atnStartState.(*StarLoopEntryState); ok && s.precedenceRuleDecision {
		dfa.precedenceDfa = true
		dfa.s0 = NewDFAState(-1, NewBaseATNConfigSet(false))
		dfa.s0.isAcceptState = false
		dfa.s0.requiresFullContext = false
	}
	return dfa
}

// getPrecedenceStartState gets the start state for the current precedence and
// returns the start state corresponding to the specified precedence if a start
// state exists for the specified precedence and nil otherwise. d must be a
// precedence DFA. See also isPrecedenceDfa.
func (d *DFA) getPrecedenceStartState(precedence int) *DFAState {
	if !d.getPrecedenceDfa() {
		panic("only precedence DFAs may contain a precedence start state")
	}

	// s0.edges is never nil for a precedence DFA
	if precedence < 0 || precedence >= len(d.getS0().getEdges()) {
		return nil
	}

	return d.getS0().getIthEdge(precedence)
}

// setPrecedenceStartState sets the start state for the current precedence. d
// must be a precedence DFA. See also isPrecedenceDfa.
func (d *DFA) setPrecedenceStartState(precedence int, startState *DFAState) {
	if !d.getPrecedenceDfa() {
		panic("only precedence DFAs may contain a precedence start state")
	}

	if precedence < 0 {
		return
	}

	// Synchronization on s0 here is ok. When the DFA is turned into a
	// precedence DFA, s0 will be initialized once and not updated again. s0.edges
	// is never nil for a precedence DFA.
	s0 := d.getS0()
	if precedence >= s0.numEdges() {
		edges := append(s0.getEdges(), make([]*DFAState, precedence+1-s0.numEdges())...)
		s0.setEdges(edges)
		d.setS0(s0)
	}

	s0.setIthEdge(precedence, startState)
}

func (d *DFA) getPrecedenceDfa() bool {
	return d.precedenceDfa
}

// setPrecedenceDfa sets whether d is a precedence DFA. If precedenceDfa differs
// from the current DFA configuration, then d.states is cleared, the initial
// state s0 is set to a new DFAState with an empty outgoing DFAState.edges to
// store the start states for individual precedence values if precedenceDfa is
// true or nil otherwise, and d.precedenceDfa is updated.
func (d *DFA) setPrecedenceDfa(precedenceDfa bool) {
	if d.getPrecedenceDfa() != precedenceDfa {
		d.setStates(make(map[int]*DFAState))

		if precedenceDfa {
			precedenceState := NewDFAState(-1, NewBaseATNConfigSet(false))

			precedenceState.setEdges(make([]*DFAState, 0))
			precedenceState.isAcceptState = false
			precedenceState.requiresFullContext = false
			d.setS0(precedenceState)
		} else {
			d.setS0(nil)
		}

		d.precedenceDfa = precedenceDfa
	}
}

func (d *DFA) getS0() *DFAState {
	return d.s0
}

func (d *DFA) setS0(s *DFAState) {
	d.s0 = s
}

func (d *DFA) getState(hash int) (*DFAState, bool) {
	s, ok := d.states[hash]
	return s, ok
}

func (d *DFA) setStates(states map[int]*DFAState) {
	d.states = states
}

func (d *DFA) setState(hash int, state *DFAState) {
	d.states[hash] = state
}

func (d *DFA) numStates() int {
	return len(d.states)
}

type dfaStateList []*DFAState

func (d dfaStateList) Len() int           { return len(d) }
func (d dfaStateList) Less(i, j int) bool { return d[i].stateNumber < d[j].stateNumber }
func (d dfaStateList) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

// sortedStates returns the states in d sorted by their state number.
func (d *DFA) sortedStates() []*DFAState {
	vs := make([]*DFAState, 0, len(d.states))

	for _, v := range d.states {
		vs = append(vs, v)
	}

	sort.Sort(dfaStateList(vs))

	return vs
}

func (d *DFA) String(literalNames []string, symbolicNames []string) string {
	if d.getS0() == nil {
		return ""
	}

	return NewDFASerializer(d, literalNames, symbolicNames).String()
}

func (d *DFA) ToLexerString() string {
	if d.getS0() == nil {
		return ""
	}

	return NewLexerDFASerializer(d).String()
}
