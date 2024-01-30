// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
)

type ATNConfigSet interface {
	Hash() int
	Equals(o Collectable[ATNConfig]) bool
	Add(ATNConfig, *DoubleDict) bool
	AddAll([]ATNConfig) bool

	GetStates() *JStore[ATNState, Comparator[ATNState]]
	GetPredicates() []SemanticContext
	GetItems() []ATNConfig

	OptimizeConfigs(interpreter *BaseATNSimulator)

	Length() int
	IsEmpty() bool
	Contains(ATNConfig) bool
	ContainsFast(ATNConfig) bool
	Clear()
	String() string

	HasSemanticContext() bool
	SetHasSemanticContext(v bool)

	ReadOnly() bool
	SetReadOnly(bool)

	GetConflictingAlts() *BitSet
	SetConflictingAlts(*BitSet)

	Alts() *BitSet

	FullContext() bool

	GetUniqueAlt() int
	SetUniqueAlt(int)

	GetDipsIntoOuterContext() bool
	SetDipsIntoOuterContext(bool)
}

// BaseATNConfigSet is a specialized set of ATNConfig that tracks information
// about its elements and can combine similar configurations using a
// graph-structured stack.
type BaseATNConfigSet struct {
	cachedHash int

	// configLookup is used to determine whether two BaseATNConfigSets are equal. We
	// need all configurations with the same (s, i, _, semctx) to be equal. A key
	// effectively doubles the number of objects associated with ATNConfigs. All
	// keys are hashed by (s, i, _, pi), not including the context. Wiped out when
	// read-only because a set becomes a DFA state.
	configLookup *JStore[ATNConfig, Comparator[ATNConfig]]

	// configs is the added elements.
	configs []ATNConfig

	// TODO: These fields make me pretty uncomfortable, but it is nice to pack up
	// info together because it saves recomputation. Can we track conflicts as they
	// are added to save scanning configs later?
	conflictingAlts *BitSet

	// dipsIntoOuterContext is used by parsers and lexers. In a lexer, it indicates
	// we hit a pred while computing a closure operation. Do not make a DFA state
	// from the BaseATNConfigSet in this case. TODO: How is this used by parsers?
	dipsIntoOuterContext bool

	// fullCtx is whether it is part of a full context LL prediction. Used to
	// determine how to merge $. It is a wildcard with SLL, but not for an LL
	// context merge.
	fullCtx bool

	// Used in parser and lexer. In lexer, it indicates we hit a pred
	// while computing a closure operation. Don't make a DFA state from a.
	hasSemanticContext bool

	// readOnly is whether it is read-only. Do not
	// allow any code to manipulate the set if true because DFA states will point at
	// sets and those must not change. It not, protect other fields; conflictingAlts
	// in particular, which is assigned after readOnly.
	readOnly bool

	// TODO: These fields make me pretty uncomfortable, but it is nice to pack up
	// info together because it saves recomputation. Can we track conflicts as they
	// are added to save scanning configs later?
	uniqueAlt int
}

func (b *BaseATNConfigSet) Alts() *BitSet {
	alts := NewBitSet()
	for _, it := range b.configs {
		alts.add(it.GetAlt())
	}
	return alts
}

func NewBaseATNConfigSet(fullCtx bool) *BaseATNConfigSet {
	return &BaseATNConfigSet{
		cachedHash:   -1,
		configLookup: NewJStore[ATNConfig, Comparator[ATNConfig]](aConfCompInst),
		fullCtx:      fullCtx,
	}
}

// Add merges contexts with existing configs for (s, i, pi, _), where s is the
// ATNConfig.state, i is the ATNConfig.alt, and pi is the
// ATNConfig.semanticContext. We use (s,i,pi) as the key. Updates
// dipsIntoOuterContext and hasSemanticContext when necessary.
func (b *BaseATNConfigSet) Add(config ATNConfig, mergeCache *DoubleDict) bool {
	if b.readOnly {
		panic("set is read-only")
	}

	if config.GetSemanticContext() != SemanticContextNone {
		b.hasSemanticContext = true
	}

	if config.GetReachesIntoOuterContext() > 0 {
		b.dipsIntoOuterContext = true
	}

	existing, present := b.configLookup.Put(config)

	// The config was not already in the set
	//
	if !present {
		b.cachedHash = -1
		b.configs = append(b.configs, config) // Track order here
		return true
	}

	// Merge a previous (s, i, pi, _) with it and save the result
	rootIsWildcard := !b.fullCtx
	merged := merge(existing.GetContext(), config.GetContext(), rootIsWildcard, mergeCache)

	// No need to check for existing.context because config.context is in the cache,
	// since the only way to create new graphs is the "call rule" and here. We cache
	// at both places.
	existing.SetReachesIntoOuterContext(intMax(existing.GetReachesIntoOuterContext(), config.GetReachesIntoOuterContext()))

	// Preserve the precedence filter suppression during the merge
	if config.getPrecedenceFilterSuppressed() {
		existing.setPrecedenceFilterSuppressed(true)
	}

	// Replace the context because there is no need to do alt mapping
	existing.SetContext(merged)

	return true
}

func (b *BaseATNConfigSet) GetStates() *JStore[ATNState, Comparator[ATNState]] {

	// states uses the standard comparator provided by the ATNState instance
	//
	states := NewJStore[ATNState, Comparator[ATNState]](aStateEqInst)

	for i := 0; i < len(b.configs); i++ {
		states.Put(b.configs[i].GetState())
	}

	return states
}

func (b *BaseATNConfigSet) HasSemanticContext() bool {
	return b.hasSemanticContext
}

func (b *BaseATNConfigSet) SetHasSemanticContext(v bool) {
	b.hasSemanticContext = v
}

func (b *BaseATNConfigSet) GetPredicates() []SemanticContext {
	preds := make([]SemanticContext, 0)

	for i := 0; i < len(b.configs); i++ {
		c := b.configs[i].GetSemanticContext()

		if c != SemanticContextNone {
			preds = append(preds, c)
		}
	}

	return preds
}

func (b *BaseATNConfigSet) GetItems() []ATNConfig {
	return b.configs
}

func (b *BaseATNConfigSet) OptimizeConfigs(interpreter *BaseATNSimulator) {
	if b.readOnly {
		panic("set is read-only")
	}

	if b.configLookup.Len() == 0 {
		return
	}

	for i := 0; i < len(b.configs); i++ {
		config := b.configs[i]

		config.SetContext(interpreter.getCachedContext(config.GetContext()))
	}
}

func (b *BaseATNConfigSet) AddAll(coll []ATNConfig) bool {
	for i := 0; i < len(coll); i++ {
		b.Add(coll[i], nil)
	}

	return false
}

// Compare is a hack function just to verify that adding DFAstares to the known
// set works, so long as comparison of ATNConfigSet s works. For that to work, we
// need to make sure that the set of ATNConfigs in two sets are equivalent. We can't
// know the order, so we do this inefficient hack. If this proves the point, then
// we can change the config set to a better structure.
func (b *BaseATNConfigSet) Compare(bs *BaseATNConfigSet) bool {
	if len(b.configs) != len(bs.configs) {
		return false
	}

	for _, c := range b.configs {
		found := false
		for _, c2 := range bs.configs {
			if c.Equals(c2) {
				found = true
				break
			}
		}

		if !found {
			return false
		}

	}
	return true
}

func (b *BaseATNConfigSet) Equals(other Collectable[ATNConfig]) bool {
	if b == other {
		return true
	} else if _, ok := other.(*BaseATNConfigSet); !ok {
		return false
	}

	other2 := other.(*BaseATNConfigSet)

	return b.configs != nil &&
		b.fullCtx == other2.fullCtx &&
		b.uniqueAlt == other2.uniqueAlt &&
		b.conflictingAlts == other2.conflictingAlts &&
		b.hasSemanticContext == other2.hasSemanticContext &&
		b.dipsIntoOuterContext == other2.dipsIntoOuterContext &&
		b.Compare(other2)
}

func (b *BaseATNConfigSet) Hash() int {
	if b.readOnly {
		if b.cachedHash == -1 {
			b.cachedHash = b.hashCodeConfigs()
		}

		return b.cachedHash
	}

	return b.hashCodeConfigs()
}

func (b *BaseATNConfigSet) hashCodeConfigs() int {
	h := 1
	for _, config := range b.configs {
		h = 31*h + config.Hash()
	}
	return h
}

func (b *BaseATNConfigSet) Length() int {
	return len(b.configs)
}

func (b *BaseATNConfigSet) IsEmpty() bool {
	return len(b.configs) == 0
}

func (b *BaseATNConfigSet) Contains(item ATNConfig) bool {
	if b.configLookup == nil {
		panic("not implemented for read-only sets")
	}

	return b.configLookup.Contains(item)
}

func (b *BaseATNConfigSet) ContainsFast(item ATNConfig) bool {
	if b.configLookup == nil {
		panic("not implemented for read-only sets")
	}

	return b.configLookup.Contains(item) // TODO: containsFast is not implemented for Set
}

func (b *BaseATNConfigSet) Clear() {
	if b.readOnly {
		panic("set is read-only")
	}

	b.configs = make([]ATNConfig, 0)
	b.cachedHash = -1
	b.configLookup = NewJStore[ATNConfig, Comparator[ATNConfig]](atnConfCompInst)
}

func (b *BaseATNConfigSet) FullContext() bool {
	return b.fullCtx
}

func (b *BaseATNConfigSet) GetDipsIntoOuterContext() bool {
	return b.dipsIntoOuterContext
}

func (b *BaseATNConfigSet) SetDipsIntoOuterContext(v bool) {
	b.dipsIntoOuterContext = v
}

func (b *BaseATNConfigSet) GetUniqueAlt() int {
	return b.uniqueAlt
}

func (b *BaseATNConfigSet) SetUniqueAlt(v int) {
	b.uniqueAlt = v
}

func (b *BaseATNConfigSet) GetConflictingAlts() *BitSet {
	return b.conflictingAlts
}

func (b *BaseATNConfigSet) SetConflictingAlts(v *BitSet) {
	b.conflictingAlts = v
}

func (b *BaseATNConfigSet) ReadOnly() bool {
	return b.readOnly
}

func (b *BaseATNConfigSet) SetReadOnly(readOnly bool) {
	b.readOnly = readOnly

	if readOnly {
		b.configLookup = nil // Read only, so no need for the lookup cache
	}
}

func (b *BaseATNConfigSet) String() string {
	s := "["

	for i, c := range b.configs {
		s += c.String()

		if i != len(b.configs)-1 {
			s += ", "
		}
	}

	s += "]"

	if b.hasSemanticContext {
		s += ",hasSemanticContext=" + fmt.Sprint(b.hasSemanticContext)
	}

	if b.uniqueAlt != ATNInvalidAltNumber {
		s += ",uniqueAlt=" + fmt.Sprint(b.uniqueAlt)
	}

	if b.conflictingAlts != nil {
		s += ",conflictingAlts=" + b.conflictingAlts.String()
	}

	if b.dipsIntoOuterContext {
		s += ",dipsIntoOuterContext"
	}

	return s
}

type OrderedATNConfigSet struct {
	*BaseATNConfigSet
}

func NewOrderedATNConfigSet() *OrderedATNConfigSet {
	b := NewBaseATNConfigSet(false)

	// This set uses the standard Hash() and Equals() from ATNConfig
	b.configLookup = NewJStore[ATNConfig, Comparator[ATNConfig]](aConfEqInst)

	return &OrderedATNConfigSet{BaseATNConfigSet: b}
}

func hashATNConfig(i interface{}) int {
	o := i.(ATNConfig)
	hash := 7
	hash = 31*hash + o.GetState().GetStateNumber()
	hash = 31*hash + o.GetAlt()
	hash = 31*hash + o.GetSemanticContext().Hash()
	return hash
}

func equalATNConfigs(a, b interface{}) bool {
	if a == nil || b == nil {
		return false
	}

	if a == b {
		return true
	}

	var ai, ok = a.(ATNConfig)
	var bi, ok1 = b.(ATNConfig)

	if !ok || !ok1 {
		return false
	}

	if ai.GetState().GetStateNumber() != bi.GetState().GetStateNumber() {
		return false
	}

	if ai.GetAlt() != bi.GetAlt() {
		return false
	}

	return ai.GetSemanticContext().Equals(bi.GetSemanticContext())
}
