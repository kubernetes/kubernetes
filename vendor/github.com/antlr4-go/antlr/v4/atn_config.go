// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
)

const (
	lexerConfig  = iota // Indicates that this ATNConfig is for a lexer
	parserConfig        // Indicates that this ATNConfig is for a parser
)

// ATNConfig is a tuple: (ATN state, predicted alt, syntactic, semantic
// context). The syntactic context is a graph-structured stack node whose
// path(s) to the root is the rule invocation(s) chain used to arrive in the
// state. The semantic context is the tree of semantic predicates encountered
// before reaching an ATN state.
type ATNConfig struct {
	precedenceFilterSuppressed     bool
	state                          ATNState
	alt                            int
	context                        *PredictionContext
	semanticContext                SemanticContext
	reachesIntoOuterContext        int
	cType                          int // lexerConfig or parserConfig
	lexerActionExecutor            *LexerActionExecutor
	passedThroughNonGreedyDecision bool
}

// NewATNConfig6 creates a new ATNConfig instance given a state, alt and context only
func NewATNConfig6(state ATNState, alt int, context *PredictionContext) *ATNConfig {
	return NewATNConfig5(state, alt, context, SemanticContextNone)
}

// NewATNConfig5 creates a new ATNConfig instance given a state, alt, context and semantic context
func NewATNConfig5(state ATNState, alt int, context *PredictionContext, semanticContext SemanticContext) *ATNConfig {
	if semanticContext == nil {
		panic("semanticContext cannot be nil") // TODO: Necessary?
	}

	pac := &ATNConfig{}
	pac.state = state
	pac.alt = alt
	pac.context = context
	pac.semanticContext = semanticContext
	pac.cType = parserConfig
	return pac
}

// NewATNConfig4 creates a new ATNConfig instance given an existing config, and a state only
func NewATNConfig4(c *ATNConfig, state ATNState) *ATNConfig {
	return NewATNConfig(c, state, c.GetContext(), c.GetSemanticContext())
}

// NewATNConfig3 creates a new ATNConfig instance given an existing config, a state and a semantic context
func NewATNConfig3(c *ATNConfig, state ATNState, semanticContext SemanticContext) *ATNConfig {
	return NewATNConfig(c, state, c.GetContext(), semanticContext)
}

// NewATNConfig2 creates a new ATNConfig instance given an existing config, and a context only
func NewATNConfig2(c *ATNConfig, semanticContext SemanticContext) *ATNConfig {
	return NewATNConfig(c, c.GetState(), c.GetContext(), semanticContext)
}

// NewATNConfig1 creates a new ATNConfig instance given an existing config, a state, and a context only
func NewATNConfig1(c *ATNConfig, state ATNState, context *PredictionContext) *ATNConfig {
	return NewATNConfig(c, state, context, c.GetSemanticContext())
}

// NewATNConfig creates a new ATNConfig instance given an existing config, a state, a context and a semantic context, other 'constructors'
// are just wrappers around this one.
func NewATNConfig(c *ATNConfig, state ATNState, context *PredictionContext, semanticContext SemanticContext) *ATNConfig {
	b := &ATNConfig{}
	b.InitATNConfig(c, state, c.GetAlt(), context, semanticContext)
	b.cType = parserConfig
	return b
}

func (a *ATNConfig) InitATNConfig(c *ATNConfig, state ATNState, alt int, context *PredictionContext, semanticContext SemanticContext) {

	a.state = state
	a.alt = alt
	a.context = context
	a.semanticContext = semanticContext
	a.reachesIntoOuterContext = c.GetReachesIntoOuterContext()
	a.precedenceFilterSuppressed = c.getPrecedenceFilterSuppressed()
}

func (a *ATNConfig) getPrecedenceFilterSuppressed() bool {
	return a.precedenceFilterSuppressed
}

func (a *ATNConfig) setPrecedenceFilterSuppressed(v bool) {
	a.precedenceFilterSuppressed = v
}

// GetState returns the ATN state associated with this configuration
func (a *ATNConfig) GetState() ATNState {
	return a.state
}

// GetAlt returns the alternative associated with this configuration
func (a *ATNConfig) GetAlt() int {
	return a.alt
}

// SetContext sets the rule invocation stack associated with this configuration
func (a *ATNConfig) SetContext(v *PredictionContext) {
	a.context = v
}

// GetContext returns the rule invocation stack associated with this configuration
func (a *ATNConfig) GetContext() *PredictionContext {
	return a.context
}

// GetSemanticContext returns the semantic context associated with this configuration
func (a *ATNConfig) GetSemanticContext() SemanticContext {
	return a.semanticContext
}

// GetReachesIntoOuterContext returns the count of references to an outer context from this configuration
func (a *ATNConfig) GetReachesIntoOuterContext() int {
	return a.reachesIntoOuterContext
}

// SetReachesIntoOuterContext sets the count of references to an outer context from this configuration
func (a *ATNConfig) SetReachesIntoOuterContext(v int) {
	a.reachesIntoOuterContext = v
}

// Equals is the default comparison function for an ATNConfig when no specialist implementation is required
// for a collection.
//
// An ATN configuration is equal to another if both have the same state, they
// predict the same alternative, and syntactic/semantic contexts are the same.
func (a *ATNConfig) Equals(o Collectable[*ATNConfig]) bool {
	switch a.cType {
	case lexerConfig:
		return a.LEquals(o)
	case parserConfig:
		return a.PEquals(o)
	default:
		panic("Invalid ATNConfig type")
	}
}

// PEquals is the default comparison function for a Parser ATNConfig when no specialist implementation is required
// for a collection.
//
// An ATN configuration is equal to another if both have the same state, they
// predict the same alternative, and syntactic/semantic contexts are the same.
func (a *ATNConfig) PEquals(o Collectable[*ATNConfig]) bool {
	var other, ok = o.(*ATNConfig)

	if !ok {
		return false
	}
	if a == other {
		return true
	} else if other == nil {
		return false
	}

	var equal bool

	if a.context == nil {
		equal = other.context == nil
	} else {
		equal = a.context.Equals(other.context)
	}

	var (
		nums = a.state.GetStateNumber() == other.state.GetStateNumber()
		alts = a.alt == other.alt
		cons = a.semanticContext.Equals(other.semanticContext)
		sups = a.precedenceFilterSuppressed == other.precedenceFilterSuppressed
	)

	return nums && alts && cons && sups && equal
}

// Hash is the default hash function for a parser ATNConfig, when no specialist hash function
// is required for a collection
func (a *ATNConfig) Hash() int {
	switch a.cType {
	case lexerConfig:
		return a.LHash()
	case parserConfig:
		return a.PHash()
	default:
		panic("Invalid ATNConfig type")
	}
}

// PHash is the default hash function for a parser ATNConfig, when no specialist hash function
// is required for a collection
func (a *ATNConfig) PHash() int {
	var c int
	if a.context != nil {
		c = a.context.Hash()
	}

	h := murmurInit(7)
	h = murmurUpdate(h, a.state.GetStateNumber())
	h = murmurUpdate(h, a.alt)
	h = murmurUpdate(h, c)
	h = murmurUpdate(h, a.semanticContext.Hash())
	return murmurFinish(h, 4)
}

// String returns a string representation of the ATNConfig, usually used for debugging purposes
func (a *ATNConfig) String() string {
	var s1, s2, s3 string

	if a.context != nil {
		s1 = ",[" + fmt.Sprint(a.context) + "]"
	}

	if a.semanticContext != SemanticContextNone {
		s2 = "," + fmt.Sprint(a.semanticContext)
	}

	if a.reachesIntoOuterContext > 0 {
		s3 = ",up=" + fmt.Sprint(a.reachesIntoOuterContext)
	}

	return fmt.Sprintf("(%v,%v%v%v%v)", a.state, a.alt, s1, s2, s3)
}

func NewLexerATNConfig6(state ATNState, alt int, context *PredictionContext) *ATNConfig {
	lac := &ATNConfig{}
	lac.state = state
	lac.alt = alt
	lac.context = context
	lac.semanticContext = SemanticContextNone
	lac.cType = lexerConfig
	return lac
}

func NewLexerATNConfig4(c *ATNConfig, state ATNState) *ATNConfig {
	lac := &ATNConfig{}
	lac.lexerActionExecutor = c.lexerActionExecutor
	lac.passedThroughNonGreedyDecision = checkNonGreedyDecision(c, state)
	lac.InitATNConfig(c, state, c.GetAlt(), c.GetContext(), c.GetSemanticContext())
	lac.cType = lexerConfig
	return lac
}

func NewLexerATNConfig3(c *ATNConfig, state ATNState, lexerActionExecutor *LexerActionExecutor) *ATNConfig {
	lac := &ATNConfig{}
	lac.lexerActionExecutor = lexerActionExecutor
	lac.passedThroughNonGreedyDecision = checkNonGreedyDecision(c, state)
	lac.InitATNConfig(c, state, c.GetAlt(), c.GetContext(), c.GetSemanticContext())
	lac.cType = lexerConfig
	return lac
}

func NewLexerATNConfig2(c *ATNConfig, state ATNState, context *PredictionContext) *ATNConfig {
	lac := &ATNConfig{}
	lac.lexerActionExecutor = c.lexerActionExecutor
	lac.passedThroughNonGreedyDecision = checkNonGreedyDecision(c, state)
	lac.InitATNConfig(c, state, c.GetAlt(), context, c.GetSemanticContext())
	lac.cType = lexerConfig
	return lac
}

//goland:noinspection GoUnusedExportedFunction
func NewLexerATNConfig1(state ATNState, alt int, context *PredictionContext) *ATNConfig {
	lac := &ATNConfig{}
	lac.state = state
	lac.alt = alt
	lac.context = context
	lac.semanticContext = SemanticContextNone
	lac.cType = lexerConfig
	return lac
}

// LHash is the default hash function for Lexer ATNConfig objects, it can be used directly or via
// the default comparator [ObjEqComparator].
func (a *ATNConfig) LHash() int {
	var f int
	if a.passedThroughNonGreedyDecision {
		f = 1
	} else {
		f = 0
	}
	h := murmurInit(7)
	h = murmurUpdate(h, a.state.GetStateNumber())
	h = murmurUpdate(h, a.alt)
	h = murmurUpdate(h, a.context.Hash())
	h = murmurUpdate(h, a.semanticContext.Hash())
	h = murmurUpdate(h, f)
	h = murmurUpdate(h, a.lexerActionExecutor.Hash())
	h = murmurFinish(h, 6)
	return h
}

// LEquals is the default comparison function for Lexer ATNConfig objects, it can be used directly or via
// the default comparator [ObjEqComparator].
func (a *ATNConfig) LEquals(other Collectable[*ATNConfig]) bool {
	var otherT, ok = other.(*ATNConfig)
	if !ok {
		return false
	} else if a == otherT {
		return true
	} else if a.passedThroughNonGreedyDecision != otherT.passedThroughNonGreedyDecision {
		return false
	}

	switch {
	case a.lexerActionExecutor == nil && otherT.lexerActionExecutor == nil:
		return true
	case a.lexerActionExecutor != nil && otherT.lexerActionExecutor != nil:
		if !a.lexerActionExecutor.Equals(otherT.lexerActionExecutor) {
			return false
		}
	default:
		return false // One but not both, are nil
	}

	return a.PEquals(otherT)
}

func checkNonGreedyDecision(source *ATNConfig, target ATNState) bool {
	var ds, ok = target.(DecisionState)

	return source.passedThroughNonGreedyDecision || (ok && ds.getNonGreedy())
}
