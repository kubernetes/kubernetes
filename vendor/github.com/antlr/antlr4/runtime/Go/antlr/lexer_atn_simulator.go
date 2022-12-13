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
	LexerATNSimulatorDebug    = false
	LexerATNSimulatorDFADebug = false

	LexerATNSimulatorMinDFAEdge = 0
	LexerATNSimulatorMaxDFAEdge = 127 // forces unicode to stay in ATN

	LexerATNSimulatorMatchCalls = 0
)

type ILexerATNSimulator interface {
	IATNSimulator

	reset()
	Match(input CharStream, mode int) int
	GetCharPositionInLine() int
	GetLine() int
	GetText(input CharStream) string
	Consume(input CharStream)
}

type LexerATNSimulator struct {
	*BaseATNSimulator

	recog              Lexer
	predictionMode     int
	mergeCache         DoubleDict
	startIndex         int
	Line               int
	CharPositionInLine int
	mode               int
	prevAccept         *SimState
	MatchCalls         int
}

func NewLexerATNSimulator(recog Lexer, atn *ATN, decisionToDFA []*DFA, sharedContextCache *PredictionContextCache) *LexerATNSimulator {
	l := new(LexerATNSimulator)

	l.BaseATNSimulator = NewBaseATNSimulator(atn, sharedContextCache)

	l.decisionToDFA = decisionToDFA
	l.recog = recog
	// The current token's starting index into the character stream.
	// Shared across DFA to ATN simulation in case the ATN fails and the
	// DFA did not have a previous accept state. In l case, we use the
	// ATN-generated exception object.
	l.startIndex = -1
	// line number 1..n within the input///
	l.Line = 1
	// The index of the character relative to the beginning of the line
	// 0..n-1///
	l.CharPositionInLine = 0
	l.mode = LexerDefaultMode
	// Used during DFA/ATN exec to record the most recent accept configuration
	// info
	l.prevAccept = NewSimState()
	// done
	return l
}

func (l *LexerATNSimulator) copyState(simulator *LexerATNSimulator) {
	l.CharPositionInLine = simulator.CharPositionInLine
	l.Line = simulator.Line
	l.mode = simulator.mode
	l.startIndex = simulator.startIndex
}

func (l *LexerATNSimulator) Match(input CharStream, mode int) int {
	l.MatchCalls++
	l.mode = mode
	mark := input.Mark()

	defer func() {
		input.Release(mark)
	}()

	l.startIndex = input.Index()
	l.prevAccept.reset()

	dfa := l.decisionToDFA[mode]

	var s0 *DFAState
	l.atn.stateMu.RLock()
	s0 = dfa.getS0()
	l.atn.stateMu.RUnlock()

	if s0 == nil {
		return l.MatchATN(input)
	}

	return l.execATN(input, s0)
}

func (l *LexerATNSimulator) reset() {
	l.prevAccept.reset()
	l.startIndex = -1
	l.Line = 1
	l.CharPositionInLine = 0
	l.mode = LexerDefaultMode
}

func (l *LexerATNSimulator) MatchATN(input CharStream) int {
	startState := l.atn.modeToStartState[l.mode]

	if LexerATNSimulatorDebug {
		fmt.Println("MatchATN mode " + strconv.Itoa(l.mode) + " start: " + startState.String())
	}
	oldMode := l.mode
	s0Closure := l.computeStartState(input, startState)
	suppressEdge := s0Closure.hasSemanticContext
	s0Closure.hasSemanticContext = false

	next := l.addDFAState(s0Closure, suppressEdge)

	predict := l.execATN(input, next)

	if LexerATNSimulatorDebug {
		fmt.Println("DFA after MatchATN: " + l.decisionToDFA[oldMode].ToLexerString())
	}
	return predict
}

func (l *LexerATNSimulator) execATN(input CharStream, ds0 *DFAState) int {

	if LexerATNSimulatorDebug {
		fmt.Println("start state closure=" + ds0.configs.String())
	}
	if ds0.isAcceptState {
		// allow zero-length tokens
		l.captureSimState(l.prevAccept, input, ds0)
	}
	t := input.LA(1)
	s := ds0 // s is current/from DFA state

	for { // while more work
		if LexerATNSimulatorDebug {
			fmt.Println("execATN loop starting closure: " + s.configs.String())
		}

		// As we move src->trg, src->trg, we keep track of the previous trg to
		// avoid looking up the DFA state again, which is expensive.
		// If the previous target was already part of the DFA, we might
		// be able to avoid doing a reach operation upon t. If s!=nil,
		// it means that semantic predicates didn't prevent us from
		// creating a DFA state. Once we know s!=nil, we check to see if
		// the DFA state has an edge already for t. If so, we can just reuse
		// it's configuration set there's no point in re-computing it.
		// This is kind of like doing DFA simulation within the ATN
		// simulation because DFA simulation is really just a way to avoid
		// computing reach/closure sets. Technically, once we know that
		// we have a previously added DFA state, we could jump over to
		// the DFA simulator. But, that would mean popping back and forth
		// a lot and making things more complicated algorithmically.
		// This optimization makes a lot of sense for loops within DFA.
		// A character will take us back to an existing DFA state
		// that already has lots of edges out of it. e.g., .* in comments.
		target := l.getExistingTargetState(s, t)
		if target == nil {
			target = l.computeTargetState(input, s, t)
			// print("Computed:" + str(target))
		}
		if target == ATNSimulatorError {
			break
		}
		// If l is a consumable input element, make sure to consume before
		// capturing the accept state so the input index, line, and char
		// position accurately reflect the state of the interpreter at the
		// end of the token.
		if t != TokenEOF {
			l.Consume(input)
		}
		if target.isAcceptState {
			l.captureSimState(l.prevAccept, input, target)
			if t == TokenEOF {
				break
			}
		}
		t = input.LA(1)
		s = target // flip current DFA target becomes Newsrc/from state
	}

	return l.failOrAccept(l.prevAccept, input, s.configs, t)
}

// Get an existing target state for an edge in the DFA. If the target state
// for the edge has not yet been computed or is otherwise not available,
// l method returns {@code nil}.
//
// @param s The current DFA state
// @param t The next input symbol
// @return The existing target DFA state for the given input symbol
// {@code t}, or {@code nil} if the target state for l edge is not
// already cached
func (l *LexerATNSimulator) getExistingTargetState(s *DFAState, t int) *DFAState {
	if t < LexerATNSimulatorMinDFAEdge || t > LexerATNSimulatorMaxDFAEdge {
		return nil
	}

	l.atn.edgeMu.RLock()
	defer l.atn.edgeMu.RUnlock()
	if s.getEdges() == nil {
		return nil
	}
	target := s.getIthEdge(t - LexerATNSimulatorMinDFAEdge)
	if LexerATNSimulatorDebug && target != nil {
		fmt.Println("reuse state " + strconv.Itoa(s.stateNumber) + " edge to " + strconv.Itoa(target.stateNumber))
	}
	return target
}

// Compute a target state for an edge in the DFA, and attempt to add the
// computed state and corresponding edge to the DFA.
//
// @param input The input stream
// @param s The current DFA state
// @param t The next input symbol
//
// @return The computed target DFA state for the given input symbol
// {@code t}. If {@code t} does not lead to a valid DFA state, l method
// returns {@link //ERROR}.
func (l *LexerATNSimulator) computeTargetState(input CharStream, s *DFAState, t int) *DFAState {
	reach := NewOrderedATNConfigSet()

	// if we don't find an existing DFA state
	// Fill reach starting from closure, following t transitions
	l.getReachableConfigSet(input, s.configs, reach.BaseATNConfigSet, t)

	if len(reach.configs) == 0 { // we got nowhere on t from s
		if !reach.hasSemanticContext {
			// we got nowhere on t, don't panic out l knowledge it'd
			// cause a failover from DFA later.
			l.addDFAEdge(s, t, ATNSimulatorError, nil)
		}
		// stop when we can't Match any more char
		return ATNSimulatorError
	}
	// Add an edge from s to target DFA found/created for reach
	return l.addDFAEdge(s, t, nil, reach.BaseATNConfigSet)
}

func (l *LexerATNSimulator) failOrAccept(prevAccept *SimState, input CharStream, reach ATNConfigSet, t int) int {
	if l.prevAccept.dfaState != nil {
		lexerActionExecutor := prevAccept.dfaState.lexerActionExecutor
		l.accept(input, lexerActionExecutor, l.startIndex, prevAccept.index, prevAccept.line, prevAccept.column)
		return prevAccept.dfaState.prediction
	}

	// if no accept and EOF is first char, return EOF
	if t == TokenEOF && input.Index() == l.startIndex {
		return TokenEOF
	}

	panic(NewLexerNoViableAltException(l.recog, input, l.startIndex, reach))
}

// Given a starting configuration set, figure out all ATN configurations
// we can reach upon input {@code t}. Parameter {@code reach} is a return
// parameter.
func (l *LexerATNSimulator) getReachableConfigSet(input CharStream, closure ATNConfigSet, reach ATNConfigSet, t int) {
	// l is used to Skip processing for configs which have a lower priority
	// than a config that already reached an accept state for the same rule
	SkipAlt := ATNInvalidAltNumber

	for _, cfg := range closure.GetItems() {
		currentAltReachedAcceptState := (cfg.GetAlt() == SkipAlt)
		if currentAltReachedAcceptState && cfg.(*LexerATNConfig).passedThroughNonGreedyDecision {
			continue
		}

		if LexerATNSimulatorDebug {

			fmt.Printf("testing %s at %s\n", l.GetTokenName(t), cfg.String()) // l.recog, true))
		}

		for _, trans := range cfg.GetState().GetTransitions() {
			target := l.getReachableTarget(trans, t)
			if target != nil {
				lexerActionExecutor := cfg.(*LexerATNConfig).lexerActionExecutor
				if lexerActionExecutor != nil {
					lexerActionExecutor = lexerActionExecutor.fixOffsetBeforeMatch(input.Index() - l.startIndex)
				}
				treatEOFAsEpsilon := (t == TokenEOF)
				config := NewLexerATNConfig3(cfg.(*LexerATNConfig), target, lexerActionExecutor)
				if l.closure(input, config, reach,
					currentAltReachedAcceptState, true, treatEOFAsEpsilon) {
					// any remaining configs for l alt have a lower priority
					// than the one that just reached an accept state.
					SkipAlt = cfg.GetAlt()
				}
			}
		}
	}
}

func (l *LexerATNSimulator) accept(input CharStream, lexerActionExecutor *LexerActionExecutor, startIndex, index, line, charPos int) {
	if LexerATNSimulatorDebug {
		fmt.Printf("ACTION %v\n", lexerActionExecutor)
	}
	// seek to after last char in token
	input.Seek(index)
	l.Line = line
	l.CharPositionInLine = charPos
	if lexerActionExecutor != nil && l.recog != nil {
		lexerActionExecutor.execute(l.recog, input, startIndex)
	}
}

func (l *LexerATNSimulator) getReachableTarget(trans Transition, t int) ATNState {
	if trans.Matches(t, 0, LexerMaxCharValue) {
		return trans.getTarget()
	}

	return nil
}

func (l *LexerATNSimulator) computeStartState(input CharStream, p ATNState) *OrderedATNConfigSet {
	configs := NewOrderedATNConfigSet()
	for i := 0; i < len(p.GetTransitions()); i++ {
		target := p.GetTransitions()[i].getTarget()
		cfg := NewLexerATNConfig6(target, i+1, BasePredictionContextEMPTY)
		l.closure(input, cfg, configs, false, false, false)
	}

	return configs
}

// Since the alternatives within any lexer decision are ordered by
// preference, l method stops pursuing the closure as soon as an accept
// state is reached. After the first accept state is reached by depth-first
// search from {@code config}, all other (potentially reachable) states for
// l rule would have a lower priority.
//
// @return {@code true} if an accept state is reached, otherwise
// {@code false}.
func (l *LexerATNSimulator) closure(input CharStream, config *LexerATNConfig, configs ATNConfigSet,
	currentAltReachedAcceptState, speculative, treatEOFAsEpsilon bool) bool {

	if LexerATNSimulatorDebug {
		fmt.Println("closure(" + config.String() + ")") // config.String(l.recog, true) + ")")
	}

	_, ok := config.state.(*RuleStopState)
	if ok {

		if LexerATNSimulatorDebug {
			if l.recog != nil {
				fmt.Printf("closure at %s rule stop %s\n", l.recog.GetRuleNames()[config.state.GetRuleIndex()], config)
			} else {
				fmt.Printf("closure at rule stop %s\n", config)
			}
		}

		if config.context == nil || config.context.hasEmptyPath() {
			if config.context == nil || config.context.isEmpty() {
				configs.Add(config, nil)
				return true
			}

			configs.Add(NewLexerATNConfig2(config, config.state, BasePredictionContextEMPTY), nil)
			currentAltReachedAcceptState = true
		}
		if config.context != nil && !config.context.isEmpty() {
			for i := 0; i < config.context.length(); i++ {
				if config.context.getReturnState(i) != BasePredictionContextEmptyReturnState {
					newContext := config.context.GetParent(i) // "pop" return state
					returnState := l.atn.states[config.context.getReturnState(i)]
					cfg := NewLexerATNConfig2(config, returnState, newContext)
					currentAltReachedAcceptState = l.closure(input, cfg, configs, currentAltReachedAcceptState, speculative, treatEOFAsEpsilon)
				}
			}
		}
		return currentAltReachedAcceptState
	}
	// optimization
	if !config.state.GetEpsilonOnlyTransitions() {
		if !currentAltReachedAcceptState || !config.passedThroughNonGreedyDecision {
			configs.Add(config, nil)
		}
	}
	for j := 0; j < len(config.state.GetTransitions()); j++ {
		trans := config.state.GetTransitions()[j]
		cfg := l.getEpsilonTarget(input, config, trans, configs, speculative, treatEOFAsEpsilon)
		if cfg != nil {
			currentAltReachedAcceptState = l.closure(input, cfg, configs,
				currentAltReachedAcceptState, speculative, treatEOFAsEpsilon)
		}
	}
	return currentAltReachedAcceptState
}

// side-effect: can alter configs.hasSemanticContext
func (l *LexerATNSimulator) getEpsilonTarget(input CharStream, config *LexerATNConfig, trans Transition,
	configs ATNConfigSet, speculative, treatEOFAsEpsilon bool) *LexerATNConfig {

	var cfg *LexerATNConfig

	if trans.getSerializationType() == TransitionRULE {

		rt := trans.(*RuleTransition)
		newContext := SingletonBasePredictionContextCreate(config.context, rt.followState.GetStateNumber())
		cfg = NewLexerATNConfig2(config, trans.getTarget(), newContext)

	} else if trans.getSerializationType() == TransitionPRECEDENCE {
		panic("Precedence predicates are not supported in lexers.")
	} else if trans.getSerializationType() == TransitionPREDICATE {
		// Track traversing semantic predicates. If we traverse,
		// we cannot add a DFA state for l "reach" computation
		// because the DFA would not test the predicate again in the
		// future. Rather than creating collections of semantic predicates
		// like v3 and testing them on prediction, v4 will test them on the
		// fly all the time using the ATN not the DFA. This is slower but
		// semantically it's not used that often. One of the key elements to
		// l predicate mechanism is not adding DFA states that see
		// predicates immediately afterwards in the ATN. For example,

		// a : ID {p1}? | ID {p2}?

		// should create the start state for rule 'a' (to save start state
		// competition), but should not create target of ID state. The
		// collection of ATN states the following ID references includes
		// states reached by traversing predicates. Since l is when we
		// test them, we cannot cash the DFA state target of ID.

		pt := trans.(*PredicateTransition)

		if LexerATNSimulatorDebug {
			fmt.Println("EVAL rule " + strconv.Itoa(trans.(*PredicateTransition).ruleIndex) + ":" + strconv.Itoa(pt.predIndex))
		}
		configs.SetHasSemanticContext(true)
		if l.evaluatePredicate(input, pt.ruleIndex, pt.predIndex, speculative) {
			cfg = NewLexerATNConfig4(config, trans.getTarget())
		}
	} else if trans.getSerializationType() == TransitionACTION {
		if config.context == nil || config.context.hasEmptyPath() {
			// execute actions anywhere in the start rule for a token.
			//
			// TODO: if the entry rule is invoked recursively, some
			// actions may be executed during the recursive call. The
			// problem can appear when hasEmptyPath() is true but
			// isEmpty() is false. In l case, the config needs to be
			// split into two contexts - one with just the empty path
			// and another with everything but the empty path.
			// Unfortunately, the current algorithm does not allow
			// getEpsilonTarget to return two configurations, so
			// additional modifications are needed before we can support
			// the split operation.
			lexerActionExecutor := LexerActionExecutorappend(config.lexerActionExecutor, l.atn.lexerActions[trans.(*ActionTransition).actionIndex])
			cfg = NewLexerATNConfig3(config, trans.getTarget(), lexerActionExecutor)
		} else {
			// ignore actions in referenced rules
			cfg = NewLexerATNConfig4(config, trans.getTarget())
		}
	} else if trans.getSerializationType() == TransitionEPSILON {
		cfg = NewLexerATNConfig4(config, trans.getTarget())
	} else if trans.getSerializationType() == TransitionATOM ||
		trans.getSerializationType() == TransitionRANGE ||
		trans.getSerializationType() == TransitionSET {
		if treatEOFAsEpsilon {
			if trans.Matches(TokenEOF, 0, LexerMaxCharValue) {
				cfg = NewLexerATNConfig4(config, trans.getTarget())
			}
		}
	}
	return cfg
}

// Evaluate a predicate specified in the lexer.
//
// <p>If {@code speculative} is {@code true}, l method was called before
// {@link //consume} for the Matched character. This method should call
// {@link //consume} before evaluating the predicate to ensure position
// sensitive values, including {@link Lexer//GetText}, {@link Lexer//GetLine},
// and {@link Lexer//getcolumn}, properly reflect the current
// lexer state. This method should restore {@code input} and the simulator
// to the original state before returning (i.e. undo the actions made by the
// call to {@link //consume}.</p>
//
// @param input The input stream.
// @param ruleIndex The rule containing the predicate.
// @param predIndex The index of the predicate within the rule.
// @param speculative {@code true} if the current index in {@code input} is
// one character before the predicate's location.
//
// @return {@code true} if the specified predicate evaluates to
// {@code true}.
// /
func (l *LexerATNSimulator) evaluatePredicate(input CharStream, ruleIndex, predIndex int, speculative bool) bool {
	// assume true if no recognizer was provided
	if l.recog == nil {
		return true
	}
	if !speculative {
		return l.recog.Sempred(nil, ruleIndex, predIndex)
	}
	savedcolumn := l.CharPositionInLine
	savedLine := l.Line
	index := input.Index()
	marker := input.Mark()

	defer func() {
		l.CharPositionInLine = savedcolumn
		l.Line = savedLine
		input.Seek(index)
		input.Release(marker)
	}()

	l.Consume(input)
	return l.recog.Sempred(nil, ruleIndex, predIndex)
}

func (l *LexerATNSimulator) captureSimState(settings *SimState, input CharStream, dfaState *DFAState) {
	settings.index = input.Index()
	settings.line = l.Line
	settings.column = l.CharPositionInLine
	settings.dfaState = dfaState
}

func (l *LexerATNSimulator) addDFAEdge(from *DFAState, tk int, to *DFAState, cfgs ATNConfigSet) *DFAState {
	if to == nil && cfgs != nil {
		// leading to l call, ATNConfigSet.hasSemanticContext is used as a
		// marker indicating dynamic predicate evaluation makes l edge
		// dependent on the specific input sequence, so the static edge in the
		// DFA should be omitted. The target DFAState is still created since
		// execATN has the ability to reSynchronize with the DFA state cache
		// following the predicate evaluation step.
		//
		// TJP notes: next time through the DFA, we see a pred again and eval.
		// If that gets us to a previously created (but dangling) DFA
		// state, we can continue in pure DFA mode from there.
		// /
		suppressEdge := cfgs.HasSemanticContext()
		cfgs.SetHasSemanticContext(false)

		to = l.addDFAState(cfgs, true)

		if suppressEdge {
			return to
		}
	}
	// add the edge
	if tk < LexerATNSimulatorMinDFAEdge || tk > LexerATNSimulatorMaxDFAEdge {
		// Only track edges within the DFA bounds
		return to
	}
	if LexerATNSimulatorDebug {
		fmt.Println("EDGE " + from.String() + " -> " + to.String() + " upon " + strconv.Itoa(tk))
	}
	l.atn.edgeMu.Lock()
	defer l.atn.edgeMu.Unlock()
	if from.getEdges() == nil {
		// make room for tokens 1..n and -1 masquerading as index 0
		from.setEdges(make([]*DFAState, LexerATNSimulatorMaxDFAEdge-LexerATNSimulatorMinDFAEdge+1))
	}
	from.setIthEdge(tk-LexerATNSimulatorMinDFAEdge, to) // connect

	return to
}

// Add a NewDFA state if there isn't one with l set of
// configurations already. This method also detects the first
// configuration containing an ATN rule stop state. Later, when
// traversing the DFA, we will know which rule to accept.
func (l *LexerATNSimulator) addDFAState(configs ATNConfigSet, suppressEdge bool) *DFAState {

	proposed := NewDFAState(-1, configs)
	var firstConfigWithRuleStopState ATNConfig

	for _, cfg := range configs.GetItems() {

		_, ok := cfg.GetState().(*RuleStopState)

		if ok {
			firstConfigWithRuleStopState = cfg
			break
		}
	}
	if firstConfigWithRuleStopState != nil {
		proposed.isAcceptState = true
		proposed.lexerActionExecutor = firstConfigWithRuleStopState.(*LexerATNConfig).lexerActionExecutor
		proposed.setPrediction(l.atn.ruleToTokenType[firstConfigWithRuleStopState.GetState().GetRuleIndex()])
	}
	hash := proposed.hash()
	dfa := l.decisionToDFA[l.mode]

	l.atn.stateMu.Lock()
	defer l.atn.stateMu.Unlock()
	existing, ok := dfa.getState(hash)
	if ok {
		proposed = existing
	} else {
		proposed.stateNumber = dfa.numStates()
		configs.SetReadOnly(true)
		proposed.configs = configs
		dfa.setState(hash, proposed)
	}
	if !suppressEdge {
		dfa.setS0(proposed)
	}
	return proposed
}

func (l *LexerATNSimulator) getDFA(mode int) *DFA {
	return l.decisionToDFA[mode]
}

// Get the text Matched so far for the current token.
func (l *LexerATNSimulator) GetText(input CharStream) string {
	// index is first lookahead char, don't include.
	return input.GetTextFromInterval(NewInterval(l.startIndex, input.Index()-1))
}

func (l *LexerATNSimulator) Consume(input CharStream) {
	curChar := input.LA(1)
	if curChar == int('\n') {
		l.Line++
		l.CharPositionInLine = 0
	} else {
		l.CharPositionInLine++
	}
	input.Consume()
}

func (l *LexerATNSimulator) GetCharPositionInLine() int {
	return l.CharPositionInLine
}

func (l *LexerATNSimulator) GetLine() int {
	return l.Line
}

func (l *LexerATNSimulator) GetTokenName(tt int) string {
	if tt == -1 {
		return "EOF"
	}

	var sb strings.Builder
	sb.Grow(6)
	sb.WriteByte('\'')
	sb.WriteRune(rune(tt))
	sb.WriteByte('\'')

	return sb.String()
}

func resetSimState(sim *SimState) {
	sim.index = -1
	sim.line = 0
	sim.column = -1
	sim.dfaState = nil
}

type SimState struct {
	index    int
	line     int
	column   int
	dfaState *DFAState
}

func NewSimState() *SimState {
	s := new(SimState)
	resetSimState(s)
	return s
}

func (s *SimState) reset() {
	resetSimState(s)
}
