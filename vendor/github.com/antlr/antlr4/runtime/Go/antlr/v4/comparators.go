package antlr

// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

// This file contains all the implementations of custom comparators used for generic collections when the
// Hash() and Equals() funcs supplied by the struct objects themselves need to be overridden. Normally, we would
// put the comparators in the source file for the struct themselves, but given the organization of this code is
// sorta kinda based upon the Java code, I found it confusing trying to find out which comparator was where and used by
// which instantiation of a collection. For instance, an Array2DHashSet in the Java source, when used with ATNConfig
// collections requires three different comparators depending on what the collection is being used for. Collecting - pun intended -
// all the comparators here, makes it much easier to see which implementation of hash and equals is used by which collection.
// It also makes it easy to verify that the Hash() and Equals() functions marry up with the Java implementations.

// ObjEqComparator is the equivalent of the Java ObjectEqualityComparator, which is the default instance of
// Equality comparator. We do not have inheritance in Go, only interfaces, so we use generics to enforce some
// type safety and avoid having to implement this for every type that we want to perform comparison on.
//
// This comparator works by using the standard Hash() and Equals() methods of the type T that is being compared. Which
// allows us to use it in any collection instance that does nto require a special hash or equals implementation.
type ObjEqComparator[T Collectable[T]] struct{}

var (
	aStateEqInst    = &ObjEqComparator[ATNState]{}
	aConfEqInst     = &ObjEqComparator[ATNConfig]{}
	aConfCompInst   = &ATNConfigComparator[ATNConfig]{}
	atnConfCompInst = &BaseATNConfigComparator[ATNConfig]{}
	dfaStateEqInst  = &ObjEqComparator[*DFAState]{}
	semctxEqInst    = &ObjEqComparator[SemanticContext]{}
	atnAltCfgEqInst = &ATNAltConfigComparator[ATNConfig]{}
)

// Equals2 delegates to the Equals() method of type T
func (c *ObjEqComparator[T]) Equals2(o1, o2 T) bool {
	return o1.Equals(o2)
}

// Hash1 delegates to the Hash() method of type T
func (c *ObjEqComparator[T]) Hash1(o T) int {

	return o.Hash()
}

type SemCComparator[T Collectable[T]] struct{}

// ATNConfigComparator is used as the compartor for the configLookup field of an ATNConfigSet
// and has a custom Equals() and Hash() implementation, because equality is not based on the
// standard Hash() and Equals() methods of the ATNConfig type.
type ATNConfigComparator[T Collectable[T]] struct {
}

// Equals2 is a custom comparator for ATNConfigs specifically for configLookup
func (c *ATNConfigComparator[T]) Equals2(o1, o2 ATNConfig) bool {

	// Same pointer, must be equal, even if both nil
	//
	if o1 == o2 {
		return true

	}

	// If either are nil, but not both, then the result is false
	//
	if o1 == nil || o2 == nil {
		return false
	}

	return o1.GetState().GetStateNumber() == o2.GetState().GetStateNumber() &&
		o1.GetAlt() == o2.GetAlt() &&
		o1.GetSemanticContext().Equals(o2.GetSemanticContext())
}

// Hash1 is custom hash implementation for ATNConfigs specifically for configLookup
func (c *ATNConfigComparator[T]) Hash1(o ATNConfig) int {
	hash := 7
	hash = 31*hash + o.GetState().GetStateNumber()
	hash = 31*hash + o.GetAlt()
	hash = 31*hash + o.GetSemanticContext().Hash()
	return hash
}

// ATNAltConfigComparator is used as the comparator for mapping configs to Alt Bitsets
type ATNAltConfigComparator[T Collectable[T]] struct {
}

// Equals2 is a custom comparator for ATNConfigs specifically for configLookup
func (c *ATNAltConfigComparator[T]) Equals2(o1, o2 ATNConfig) bool {

	// Same pointer, must be equal, even if both nil
	//
	if o1 == o2 {
		return true

	}

	// If either are nil, but not both, then the result is false
	//
	if o1 == nil || o2 == nil {
		return false
	}

	return o1.GetState().GetStateNumber() == o2.GetState().GetStateNumber() &&
		o1.GetContext().Equals(o2.GetContext())
}

// Hash1 is custom hash implementation for ATNConfigs specifically for configLookup
func (c *ATNAltConfigComparator[T]) Hash1(o ATNConfig) int {
	h := murmurInit(7)
	h = murmurUpdate(h, o.GetState().GetStateNumber())
	h = murmurUpdate(h, o.GetContext().Hash())
	return murmurFinish(h, 2)
}

// BaseATNConfigComparator is used as the comparator for the configLookup field of a BaseATNConfigSet
// and has a custom Equals() and Hash() implementation, because equality is not based on the
// standard Hash() and Equals() methods of the ATNConfig type.
type BaseATNConfigComparator[T Collectable[T]] struct {
}

// Equals2 is a custom comparator for ATNConfigs specifically for baseATNConfigSet
func (c *BaseATNConfigComparator[T]) Equals2(o1, o2 ATNConfig) bool {

	// Same pointer, must be equal, even if both nil
	//
	if o1 == o2 {
		return true

	}

	// If either are nil, but not both, then the result is false
	//
	if o1 == nil || o2 == nil {
		return false
	}

	return o1.GetState().GetStateNumber() == o2.GetState().GetStateNumber() &&
		o1.GetAlt() == o2.GetAlt() &&
		o1.GetSemanticContext().Equals(o2.GetSemanticContext())
}

// Hash1 is custom hash implementation for ATNConfigs specifically for configLookup, but in fact just
// delegates to the standard Hash() method of the ATNConfig type.
func (c *BaseATNConfigComparator[T]) Hash1(o ATNConfig) int {

	return o.Hash()
}
