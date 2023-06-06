// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
)

// ATNConfig is a tuple: (ATN state, predicted alt, syntactic, semantic
// context). The syntactic context is a graph-structured stack node whose
// path(s) to the root is the rule invocation(s) chain used to arrive at the
// state. The semantic context is the tree of semantic predicates encountered
// before reaching an ATN state.
type ATNConfig interface {
	Equals(o Collectable[ATNConfig]) bool
	Hash() int

	GetState() ATNState
	GetAlt() int
	GetSemanticContext() SemanticContext

	GetContext() PredictionContext
	SetContext(PredictionContext)

	GetReachesIntoOuterContext() int
	SetReachesIntoOuterContext(int)

	String() string

	getPrecedenceFilterSuppressed() bool
	setPrecedenceFilterSuppressed(bool)
}

type BaseATNConfig struct {
	precedenceFilterSuppressed bool
	state                      ATNState
	alt                        int
	context                    PredictionContext
	semanticContext            SemanticContext
	reachesIntoOuterContext    int
}

func NewBaseATNConfig7(old *BaseATNConfig) ATNConfig { // TODO: Dup
	return &BaseATNConfig{
		state:                   old.state,
		alt:                     old.alt,
		context:                 old.context,
		semanticContext:         old.semanticContext,
		reachesIntoOuterContext: old.reachesIntoOuterContext,
	}
}

func NewBaseATNConfig6(state ATNState, alt int, context PredictionContext) *BaseATNConfig {
	return NewBaseATNConfig5(state, alt, context, SemanticContextNone)
}

func NewBaseATNConfig5(state ATNState, alt int, context PredictionContext, semanticContext SemanticContext) *BaseATNConfig {
	if semanticContext == nil {
		panic("semanticContext cannot be nil") // TODO: Necessary?
	}

	return &BaseATNConfig{state: state, alt: alt, context: context, semanticContext: semanticContext}
}

func NewBaseATNConfig4(c ATNConfig, state ATNState) *BaseATNConfig {
	return NewBaseATNConfig(c, state, c.GetContext(), c.GetSemanticContext())
}

func NewBaseATNConfig3(c ATNConfig, state ATNState, semanticContext SemanticContext) *BaseATNConfig {
	return NewBaseATNConfig(c, state, c.GetContext(), semanticContext)
}

func NewBaseATNConfig2(c ATNConfig, semanticContext SemanticContext) *BaseATNConfig {
	return NewBaseATNConfig(c, c.GetState(), c.GetContext(), semanticContext)
}

func NewBaseATNConfig1(c ATNConfig, state ATNState, context PredictionContext) *BaseATNConfig {
	return NewBaseATNConfig(c, state, context, c.GetSemanticContext())
}

func NewBaseATNConfig(c ATNConfig, state ATNState, context PredictionContext, semanticContext SemanticContext) *BaseATNConfig {
	if semanticContext == nil {
		panic("semanticContext cannot be nil")
	}

	return &BaseATNConfig{
		state:                      state,
		alt:                        c.GetAlt(),
		context:                    context,
		semanticContext:            semanticContext,
		reachesIntoOuterContext:    c.GetReachesIntoOuterContext(),
		precedenceFilterSuppressed: c.getPrecedenceFilterSuppressed(),
	}
}

func (b *BaseATNConfig) getPrecedenceFilterSuppressed() bool {
	return b.precedenceFilterSuppressed
}

func (b *BaseATNConfig) setPrecedenceFilterSuppressed(v bool) {
	b.precedenceFilterSuppressed = v
}

func (b *BaseATNConfig) GetState() ATNState {
	return b.state
}

func (b *BaseATNConfig) GetAlt() int {
	return b.alt
}

func (b *BaseATNConfig) SetContext(v PredictionContext) {
	b.context = v
}
func (b *BaseATNConfig) GetContext() PredictionContext {
	return b.context
}

func (b *BaseATNConfig) GetSemanticContext() SemanticContext {
	return b.semanticContext
}

func (b *BaseATNConfig) GetReachesIntoOuterContext() int {
	return b.reachesIntoOuterContext
}

func (b *BaseATNConfig) SetReachesIntoOuterContext(v int) {
	b.reachesIntoOuterContext = v
}

// Equals is the default comparison function for an ATNConfig when no specialist implementation is required
// for a collection.
//
// An ATN configuration is equal to another if both have the same state, they
// predict the same alternative, and syntactic/semantic contexts are the same.
func (b *BaseATNConfig) Equals(o Collectable[ATNConfig]) bool {
	if b == o {
		return true
	} else if o == nil {
		return false
	}

	var other, ok = o.(*BaseATNConfig)

	if !ok {
		return false
	}

	var equal bool

	if b.context == nil {
		equal = other.context == nil
	} else {
		equal = b.context.Equals(other.context)
	}

	var (
		nums = b.state.GetStateNumber() == other.state.GetStateNumber()
		alts = b.alt == other.alt
		cons = b.semanticContext.Equals(other.semanticContext)
		sups = b.precedenceFilterSuppressed == other.precedenceFilterSuppressed
	)

	return nums && alts && cons && sups && equal
}

// Hash is the default hash function for BaseATNConfig, when no specialist hash function
// is required for a collection
func (b *BaseATNConfig) Hash() int {
	var c int
	if b.context != nil {
		c = b.context.Hash()
	}

	h := murmurInit(7)
	h = murmurUpdate(h, b.state.GetStateNumber())
	h = murmurUpdate(h, b.alt)
	h = murmurUpdate(h, c)
	h = murmurUpdate(h, b.semanticContext.Hash())
	return murmurFinish(h, 4)
}

func (b *BaseATNConfig) String() string {
	var s1, s2, s3 string

	if b.context != nil {
		s1 = ",[" + fmt.Sprint(b.context) + "]"
	}

	if b.semanticContext != SemanticContextNone {
		s2 = "," + fmt.Sprint(b.semanticContext)
	}

	if b.reachesIntoOuterContext > 0 {
		s3 = ",up=" + fmt.Sprint(b.reachesIntoOuterContext)
	}

	return fmt.Sprintf("(%v,%v%v%v%v)", b.state, b.alt, s1, s2, s3)
}

type LexerATNConfig struct {
	*BaseATNConfig
	lexerActionExecutor            *LexerActionExecutor
	passedThroughNonGreedyDecision bool
}

func NewLexerATNConfig6(state ATNState, alt int, context PredictionContext) *LexerATNConfig {
	return &LexerATNConfig{BaseATNConfig: NewBaseATNConfig5(state, alt, context, SemanticContextNone)}
}

func NewLexerATNConfig5(state ATNState, alt int, context PredictionContext, lexerActionExecutor *LexerActionExecutor) *LexerATNConfig {
	return &LexerATNConfig{
		BaseATNConfig:       NewBaseATNConfig5(state, alt, context, SemanticContextNone),
		lexerActionExecutor: lexerActionExecutor,
	}
}

func NewLexerATNConfig4(c *LexerATNConfig, state ATNState) *LexerATNConfig {
	return &LexerATNConfig{
		BaseATNConfig:                  NewBaseATNConfig(c, state, c.GetContext(), c.GetSemanticContext()),
		lexerActionExecutor:            c.lexerActionExecutor,
		passedThroughNonGreedyDecision: checkNonGreedyDecision(c, state),
	}
}

func NewLexerATNConfig3(c *LexerATNConfig, state ATNState, lexerActionExecutor *LexerActionExecutor) *LexerATNConfig {
	return &LexerATNConfig{
		BaseATNConfig:                  NewBaseATNConfig(c, state, c.GetContext(), c.GetSemanticContext()),
		lexerActionExecutor:            lexerActionExecutor,
		passedThroughNonGreedyDecision: checkNonGreedyDecision(c, state),
	}
}

func NewLexerATNConfig2(c *LexerATNConfig, state ATNState, context PredictionContext) *LexerATNConfig {
	return &LexerATNConfig{
		BaseATNConfig:                  NewBaseATNConfig(c, state, context, c.GetSemanticContext()),
		lexerActionExecutor:            c.lexerActionExecutor,
		passedThroughNonGreedyDecision: checkNonGreedyDecision(c, state),
	}
}

func NewLexerATNConfig1(state ATNState, alt int, context PredictionContext) *LexerATNConfig {
	return &LexerATNConfig{BaseATNConfig: NewBaseATNConfig5(state, alt, context, SemanticContextNone)}
}

// Hash is the default hash function for LexerATNConfig objects, it can be used directly or via
// the default comparator [ObjEqComparator].
func (l *LexerATNConfig) Hash() int {
	var f int
	if l.passedThroughNonGreedyDecision {
		f = 1
	} else {
		f = 0
	}
	h := murmurInit(7)
	h = murmurUpdate(h, l.state.GetStateNumber())
	h = murmurUpdate(h, l.alt)
	h = murmurUpdate(h, l.context.Hash())
	h = murmurUpdate(h, l.semanticContext.Hash())
	h = murmurUpdate(h, f)
	h = murmurUpdate(h, l.lexerActionExecutor.Hash())
	h = murmurFinish(h, 6)
	return h
}

// Equals is the default comparison function for LexerATNConfig objects, it can be used directly or via
// the default comparator [ObjEqComparator].
func (l *LexerATNConfig) Equals(other Collectable[ATNConfig]) bool {
	if l == other {
		return true
	}
	var othert, ok = other.(*LexerATNConfig)

	if l == other {
		return true
	} else if !ok {
		return false
	} else if l.passedThroughNonGreedyDecision != othert.passedThroughNonGreedyDecision {
		return false
	}

	var b bool

	if l.lexerActionExecutor != nil {
		b = !l.lexerActionExecutor.Equals(othert.lexerActionExecutor)
	} else {
		b = othert.lexerActionExecutor != nil
	}

	if b {
		return false
	}

	return l.BaseATNConfig.Equals(othert.BaseATNConfig)
}

func checkNonGreedyDecision(source *LexerATNConfig, target ATNState) bool {
	var ds, ok = target.(DecisionState)

	return source.passedThroughNonGreedyDecision || (ok && ds.getNonGreedy())
}
