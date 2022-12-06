// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"strconv"
)

// Represents {@code $} in local context prediction, which means wildcard.
// {@code//+x =//}.
// /
const (
	BasePredictionContextEmptyReturnState = 0x7FFFFFFF
)

// Represents {@code $} in an array in full context mode, when {@code $}
// doesn't mean wildcard: {@code $ + x = [$,x]}. Here,
// {@code $} = {@link //EmptyReturnState}.
// /

var (
	BasePredictionContextglobalNodeCount = 1
	BasePredictionContextid              = BasePredictionContextglobalNodeCount
)

type PredictionContext interface {
	hash() int
	GetParent(int) PredictionContext
	getReturnState(int) int
	equals(PredictionContext) bool
	length() int
	isEmpty() bool
	hasEmptyPath() bool
	String() string
}

type BasePredictionContext struct {
	cachedHash int
}

func NewBasePredictionContext(cachedHash int) *BasePredictionContext {
	pc := new(BasePredictionContext)
	pc.cachedHash = cachedHash

	return pc
}

func (b *BasePredictionContext) isEmpty() bool {
	return false
}

func calculateHash(parent PredictionContext, returnState int) int {
	h := murmurInit(1)
	h = murmurUpdate(h, parent.hash())
	h = murmurUpdate(h, returnState)
	return murmurFinish(h, 2)
}

var _emptyPredictionContextHash int

func init() {
	_emptyPredictionContextHash = murmurInit(1)
	_emptyPredictionContextHash = murmurFinish(_emptyPredictionContextHash, 0)
}

func calculateEmptyHash() int {
	return _emptyPredictionContextHash
}

// Used to cache {@link BasePredictionContext} objects. Its used for the shared
// context cash associated with contexts in DFA states. This cache
// can be used for both lexers and parsers.

type PredictionContextCache struct {
	cache map[PredictionContext]PredictionContext
}

func NewPredictionContextCache() *PredictionContextCache {
	t := new(PredictionContextCache)
	t.cache = make(map[PredictionContext]PredictionContext)
	return t
}

// Add a context to the cache and return it. If the context already exists,
// return that one instead and do not add a Newcontext to the cache.
// Protect shared cache from unsafe thread access.
//
func (p *PredictionContextCache) add(ctx PredictionContext) PredictionContext {
	if ctx == BasePredictionContextEMPTY {
		return BasePredictionContextEMPTY
	}
	existing := p.cache[ctx]
	if existing != nil {
		return existing
	}
	p.cache[ctx] = ctx
	return ctx
}

func (p *PredictionContextCache) Get(ctx PredictionContext) PredictionContext {
	return p.cache[ctx]
}

func (p *PredictionContextCache) length() int {
	return len(p.cache)
}

type SingletonPredictionContext interface {
	PredictionContext
}

type BaseSingletonPredictionContext struct {
	*BasePredictionContext

	parentCtx   PredictionContext
	returnState int
}

func NewBaseSingletonPredictionContext(parent PredictionContext, returnState int) *BaseSingletonPredictionContext {
	var cachedHash int
	if parent != nil {
		cachedHash = calculateHash(parent, returnState)
	} else {
		cachedHash = calculateEmptyHash()
	}

	s := new(BaseSingletonPredictionContext)
	s.BasePredictionContext = NewBasePredictionContext(cachedHash)

	s.parentCtx = parent
	s.returnState = returnState

	return s
}

func SingletonBasePredictionContextCreate(parent PredictionContext, returnState int) PredictionContext {
	if returnState == BasePredictionContextEmptyReturnState && parent == nil {
		// someone can pass in the bits of an array ctx that mean $
		return BasePredictionContextEMPTY
	}

	return NewBaseSingletonPredictionContext(parent, returnState)
}

func (b *BaseSingletonPredictionContext) length() int {
	return 1
}

func (b *BaseSingletonPredictionContext) GetParent(index int) PredictionContext {
	return b.parentCtx
}

func (b *BaseSingletonPredictionContext) getReturnState(index int) int {
	return b.returnState
}

func (b *BaseSingletonPredictionContext) hasEmptyPath() bool {
	return b.returnState == BasePredictionContextEmptyReturnState
}

func (b *BaseSingletonPredictionContext) equals(other PredictionContext) bool {
	if b == other {
		return true
	} else if _, ok := other.(*BaseSingletonPredictionContext); !ok {
		return false
	} else if b.hash() != other.hash() {
		return false // can't be same if hash is different
	}

	otherP := other.(*BaseSingletonPredictionContext)

	if b.returnState != other.getReturnState(0) {
		return false
	} else if b.parentCtx == nil {
		return otherP.parentCtx == nil
	}

	return b.parentCtx.equals(otherP.parentCtx)
}

func (b *BaseSingletonPredictionContext) hash() int {
	return b.cachedHash
}

func (b *BaseSingletonPredictionContext) String() string {
	var up string

	if b.parentCtx == nil {
		up = ""
	} else {
		up = b.parentCtx.String()
	}

	if len(up) == 0 {
		if b.returnState == BasePredictionContextEmptyReturnState {
			return "$"
		}

		return strconv.Itoa(b.returnState)
	}

	return strconv.Itoa(b.returnState) + " " + up
}

var BasePredictionContextEMPTY = NewEmptyPredictionContext()

type EmptyPredictionContext struct {
	*BaseSingletonPredictionContext
}

func NewEmptyPredictionContext() *EmptyPredictionContext {

	p := new(EmptyPredictionContext)

	p.BaseSingletonPredictionContext = NewBaseSingletonPredictionContext(nil, BasePredictionContextEmptyReturnState)

	return p
}

func (e *EmptyPredictionContext) isEmpty() bool {
	return true
}

func (e *EmptyPredictionContext) GetParent(index int) PredictionContext {
	return nil
}

func (e *EmptyPredictionContext) getReturnState(index int) int {
	return e.returnState
}

func (e *EmptyPredictionContext) equals(other PredictionContext) bool {
	return e == other
}

func (e *EmptyPredictionContext) String() string {
	return "$"
}

type ArrayPredictionContext struct {
	*BasePredictionContext

	parents      []PredictionContext
	returnStates []int
}

func NewArrayPredictionContext(parents []PredictionContext, returnStates []int) *ArrayPredictionContext {
	// Parent can be nil only if full ctx mode and we make an array
	// from {@link //EMPTY} and non-empty. We merge {@link //EMPTY} by using
	// nil parent and
	// returnState == {@link //EmptyReturnState}.
	hash := murmurInit(1)

	for _, parent := range parents {
		hash = murmurUpdate(hash, parent.hash())
	}

	for _, returnState := range returnStates {
		hash = murmurUpdate(hash, returnState)
	}

	hash = murmurFinish(hash, len(parents)<<1)

	c := new(ArrayPredictionContext)
	c.BasePredictionContext = NewBasePredictionContext(hash)

	c.parents = parents
	c.returnStates = returnStates

	return c
}

func (a *ArrayPredictionContext) GetReturnStates() []int {
	return a.returnStates
}

func (a *ArrayPredictionContext) hasEmptyPath() bool {
	return a.getReturnState(a.length()-1) == BasePredictionContextEmptyReturnState
}

func (a *ArrayPredictionContext) isEmpty() bool {
	// since EmptyReturnState can only appear in the last position, we
	// don't need to verify that size==1
	return a.returnStates[0] == BasePredictionContextEmptyReturnState
}

func (a *ArrayPredictionContext) length() int {
	return len(a.returnStates)
}

func (a *ArrayPredictionContext) GetParent(index int) PredictionContext {
	return a.parents[index]
}

func (a *ArrayPredictionContext) getReturnState(index int) int {
	return a.returnStates[index]
}

func (a *ArrayPredictionContext) equals(other PredictionContext) bool {
	if _, ok := other.(*ArrayPredictionContext); !ok {
		return false
	} else if a.cachedHash != other.hash() {
		return false // can't be same if hash is different
	} else {
		otherP := other.(*ArrayPredictionContext)
		return &a.returnStates == &otherP.returnStates && &a.parents == &otherP.parents
	}
}

func (a *ArrayPredictionContext) hash() int {
	return a.BasePredictionContext.cachedHash
}

func (a *ArrayPredictionContext) String() string {
	if a.isEmpty() {
		return "[]"
	}

	s := "["
	for i := 0; i < len(a.returnStates); i++ {
		if i > 0 {
			s = s + ", "
		}
		if a.returnStates[i] == BasePredictionContextEmptyReturnState {
			s = s + "$"
			continue
		}
		s = s + strconv.Itoa(a.returnStates[i])
		if a.parents[i] != nil {
			s = s + " " + a.parents[i].String()
		} else {
			s = s + "nil"
		}
	}

	return s + "]"
}

// Convert a {@link RuleContext} tree to a {@link BasePredictionContext} graph.
// Return {@link //EMPTY} if {@code outerContext} is empty or nil.
// /
func predictionContextFromRuleContext(a *ATN, outerContext RuleContext) PredictionContext {
	if outerContext == nil {
		outerContext = RuleContextEmpty
	}
	// if we are in RuleContext of start rule, s, then BasePredictionContext
	// is EMPTY. Nobody called us. (if we are empty, return empty)
	if outerContext.GetParent() == nil || outerContext == RuleContextEmpty {
		return BasePredictionContextEMPTY
	}
	// If we have a parent, convert it to a BasePredictionContext graph
	parent := predictionContextFromRuleContext(a, outerContext.GetParent().(RuleContext))
	state := a.states[outerContext.GetInvokingState()]
	transition := state.GetTransitions()[0]

	return SingletonBasePredictionContextCreate(parent, transition.(*RuleTransition).followState.GetStateNumber())
}

func merge(a, b PredictionContext, rootIsWildcard bool, mergeCache *DoubleDict) PredictionContext {
	// share same graph if both same
	if a == b {
		return a
	}

	ac, ok1 := a.(*BaseSingletonPredictionContext)
	bc, ok2 := b.(*BaseSingletonPredictionContext)

	if ok1 && ok2 {
		return mergeSingletons(ac, bc, rootIsWildcard, mergeCache)
	}
	// At least one of a or b is array
	// If one is $ and rootIsWildcard, return $ as// wildcard
	if rootIsWildcard {
		if _, ok := a.(*EmptyPredictionContext); ok {
			return a
		}
		if _, ok := b.(*EmptyPredictionContext); ok {
			return b
		}
	}
	// convert singleton so both are arrays to normalize
	if _, ok := a.(*BaseSingletonPredictionContext); ok {
		a = NewArrayPredictionContext([]PredictionContext{a.GetParent(0)}, []int{a.getReturnState(0)})
	}
	if _, ok := b.(*BaseSingletonPredictionContext); ok {
		b = NewArrayPredictionContext([]PredictionContext{b.GetParent(0)}, []int{b.getReturnState(0)})
	}
	return mergeArrays(a.(*ArrayPredictionContext), b.(*ArrayPredictionContext), rootIsWildcard, mergeCache)
}

//
// Merge two {@link SingletonBasePredictionContext} instances.
//
// <p>Stack tops equal, parents merge is same return left graph.<br>
// <embed src="images/SingletonMerge_SameRootSamePar.svg"
// type="image/svg+xml"/></p>
//
// <p>Same stack top, parents differ merge parents giving array node, then
// remainders of those graphs. A Newroot node is created to point to the
// merged parents.<br>
// <embed src="images/SingletonMerge_SameRootDiffPar.svg"
// type="image/svg+xml"/></p>
//
// <p>Different stack tops pointing to same parent. Make array node for the
// root where both element in the root point to the same (original)
// parent.<br>
// <embed src="images/SingletonMerge_DiffRootSamePar.svg"
// type="image/svg+xml"/></p>
//
// <p>Different stack tops pointing to different parents. Make array node for
// the root where each element points to the corresponding original
// parent.<br>
// <embed src="images/SingletonMerge_DiffRootDiffPar.svg"
// type="image/svg+xml"/></p>
//
// @param a the first {@link SingletonBasePredictionContext}
// @param b the second {@link SingletonBasePredictionContext}
// @param rootIsWildcard {@code true} if this is a local-context merge,
// otherwise false to indicate a full-context merge
// @param mergeCache
// /
func mergeSingletons(a, b *BaseSingletonPredictionContext, rootIsWildcard bool, mergeCache *DoubleDict) PredictionContext {
	if mergeCache != nil {
		previous := mergeCache.Get(a.hash(), b.hash())
		if previous != nil {
			return previous.(PredictionContext)
		}
		previous = mergeCache.Get(b.hash(), a.hash())
		if previous != nil {
			return previous.(PredictionContext)
		}
	}

	rootMerge := mergeRoot(a, b, rootIsWildcard)
	if rootMerge != nil {
		if mergeCache != nil {
			mergeCache.set(a.hash(), b.hash(), rootMerge)
		}
		return rootMerge
	}
	if a.returnState == b.returnState {
		parent := merge(a.parentCtx, b.parentCtx, rootIsWildcard, mergeCache)
		// if parent is same as existing a or b parent or reduced to a parent,
		// return it
		if parent == a.parentCtx {
			return a // ax + bx = ax, if a=b
		}
		if parent == b.parentCtx {
			return b // ax + bx = bx, if a=b
		}
		// else: ax + ay = a'[x,y]
		// merge parents x and y, giving array node with x,y then remainders
		// of those graphs. dup a, a' points at merged array
		// Newjoined parent so create Newsingleton pointing to it, a'
		spc := SingletonBasePredictionContextCreate(parent, a.returnState)
		if mergeCache != nil {
			mergeCache.set(a.hash(), b.hash(), spc)
		}
		return spc
	}
	// a != b payloads differ
	// see if we can collapse parents due to $+x parents if local ctx
	var singleParent PredictionContext
	if a == b || (a.parentCtx != nil && a.parentCtx == b.parentCtx) { // ax +
		// bx =
		// [a,b]x
		singleParent = a.parentCtx
	}
	if singleParent != nil { // parents are same
		// sort payloads and use same parent
		payloads := []int{a.returnState, b.returnState}
		if a.returnState > b.returnState {
			payloads[0] = b.returnState
			payloads[1] = a.returnState
		}
		parents := []PredictionContext{singleParent, singleParent}
		apc := NewArrayPredictionContext(parents, payloads)
		if mergeCache != nil {
			mergeCache.set(a.hash(), b.hash(), apc)
		}
		return apc
	}
	// parents differ and can't merge them. Just pack together
	// into array can't merge.
	// ax + by = [ax,by]
	payloads := []int{a.returnState, b.returnState}
	parents := []PredictionContext{a.parentCtx, b.parentCtx}
	if a.returnState > b.returnState { // sort by payload
		payloads[0] = b.returnState
		payloads[1] = a.returnState
		parents = []PredictionContext{b.parentCtx, a.parentCtx}
	}
	apc := NewArrayPredictionContext(parents, payloads)
	if mergeCache != nil {
		mergeCache.set(a.hash(), b.hash(), apc)
	}
	return apc
}

//
// Handle case where at least one of {@code a} or {@code b} is
// {@link //EMPTY}. In the following diagrams, the symbol {@code $} is used
// to represent {@link //EMPTY}.
//
// <h2>Local-Context Merges</h2>
//
// <p>These local-context merge operations are used when {@code rootIsWildcard}
// is true.</p>
//
// <p>{@link //EMPTY} is superset of any graph return {@link //EMPTY}.<br>
// <embed src="images/LocalMerge_EmptyRoot.svg" type="image/svg+xml"/></p>
//
// <p>{@link //EMPTY} and anything is {@code //EMPTY}, so merged parent is
// {@code //EMPTY} return left graph.<br>
// <embed src="images/LocalMerge_EmptyParent.svg" type="image/svg+xml"/></p>
//
// <p>Special case of last merge if local context.<br>
// <embed src="images/LocalMerge_DiffRoots.svg" type="image/svg+xml"/></p>
//
// <h2>Full-Context Merges</h2>
//
// <p>These full-context merge operations are used when {@code rootIsWildcard}
// is false.</p>
//
// <p><embed src="images/FullMerge_EmptyRoots.svg" type="image/svg+xml"/></p>
//
// <p>Must keep all contexts {@link //EMPTY} in array is a special value (and
// nil parent).<br>
// <embed src="images/FullMerge_EmptyRoot.svg" type="image/svg+xml"/></p>
//
// <p><embed src="images/FullMerge_SameRoot.svg" type="image/svg+xml"/></p>
//
// @param a the first {@link SingletonBasePredictionContext}
// @param b the second {@link SingletonBasePredictionContext}
// @param rootIsWildcard {@code true} if this is a local-context merge,
// otherwise false to indicate a full-context merge
// /
func mergeRoot(a, b SingletonPredictionContext, rootIsWildcard bool) PredictionContext {
	if rootIsWildcard {
		if a == BasePredictionContextEMPTY {
			return BasePredictionContextEMPTY // // + b =//
		}
		if b == BasePredictionContextEMPTY {
			return BasePredictionContextEMPTY // a +// =//
		}
	} else {
		if a == BasePredictionContextEMPTY && b == BasePredictionContextEMPTY {
			return BasePredictionContextEMPTY // $ + $ = $
		} else if a == BasePredictionContextEMPTY { // $ + x = [$,x]
			payloads := []int{b.getReturnState(-1), BasePredictionContextEmptyReturnState}
			parents := []PredictionContext{b.GetParent(-1), nil}
			return NewArrayPredictionContext(parents, payloads)
		} else if b == BasePredictionContextEMPTY { // x + $ = [$,x] ($ is always first if present)
			payloads := []int{a.getReturnState(-1), BasePredictionContextEmptyReturnState}
			parents := []PredictionContext{a.GetParent(-1), nil}
			return NewArrayPredictionContext(parents, payloads)
		}
	}
	return nil
}

//
// Merge two {@link ArrayBasePredictionContext} instances.
//
// <p>Different tops, different parents.<br>
// <embed src="images/ArrayMerge_DiffTopDiffPar.svg" type="image/svg+xml"/></p>
//
// <p>Shared top, same parents.<br>
// <embed src="images/ArrayMerge_ShareTopSamePar.svg" type="image/svg+xml"/></p>
//
// <p>Shared top, different parents.<br>
// <embed src="images/ArrayMerge_ShareTopDiffPar.svg" type="image/svg+xml"/></p>
//
// <p>Shared top, all shared parents.<br>
// <embed src="images/ArrayMerge_ShareTopSharePar.svg"
// type="image/svg+xml"/></p>
//
// <p>Equal tops, merge parents and reduce top to
// {@link SingletonBasePredictionContext}.<br>
// <embed src="images/ArrayMerge_EqualTop.svg" type="image/svg+xml"/></p>
// /
func mergeArrays(a, b *ArrayPredictionContext, rootIsWildcard bool, mergeCache *DoubleDict) PredictionContext {
	if mergeCache != nil {
		previous := mergeCache.Get(a.hash(), b.hash())
		if previous != nil {
			return previous.(PredictionContext)
		}
		previous = mergeCache.Get(b.hash(), a.hash())
		if previous != nil {
			return previous.(PredictionContext)
		}
	}
	// merge sorted payloads a + b => M
	i := 0 // walks a
	j := 0 // walks b
	k := 0 // walks target M array

	mergedReturnStates := make([]int, len(a.returnStates)+len(b.returnStates))
	mergedParents := make([]PredictionContext, len(a.returnStates)+len(b.returnStates))
	// walk and merge to yield mergedParents, mergedReturnStates
	for i < len(a.returnStates) && j < len(b.returnStates) {
		aParent := a.parents[i]
		bParent := b.parents[j]
		if a.returnStates[i] == b.returnStates[j] {
			// same payload (stack tops are equal), must yield merged singleton
			payload := a.returnStates[i]
			// $+$ = $
			bothDollars := payload == BasePredictionContextEmptyReturnState && aParent == nil && bParent == nil
			axAX := (aParent != nil && bParent != nil && aParent == bParent) // ax+ax
			// ->
			// ax
			if bothDollars || axAX {
				mergedParents[k] = aParent // choose left
				mergedReturnStates[k] = payload
			} else { // ax+ay -> a'[x,y]
				mergedParent := merge(aParent, bParent, rootIsWildcard, mergeCache)
				mergedParents[k] = mergedParent
				mergedReturnStates[k] = payload
			}
			i++ // hop over left one as usual
			j++ // but also Skip one in right side since we merge
		} else if a.returnStates[i] < b.returnStates[j] { // copy a[i] to M
			mergedParents[k] = aParent
			mergedReturnStates[k] = a.returnStates[i]
			i++
		} else { // b > a, copy b[j] to M
			mergedParents[k] = bParent
			mergedReturnStates[k] = b.returnStates[j]
			j++
		}
		k++
	}
	// copy over any payloads remaining in either array
	if i < len(a.returnStates) {
		for p := i; p < len(a.returnStates); p++ {
			mergedParents[k] = a.parents[p]
			mergedReturnStates[k] = a.returnStates[p]
			k++
		}
	} else {
		for p := j; p < len(b.returnStates); p++ {
			mergedParents[k] = b.parents[p]
			mergedReturnStates[k] = b.returnStates[p]
			k++
		}
	}
	// trim merged if we combined a few that had same stack tops
	if k < len(mergedParents) { // write index < last position trim
		if k == 1 { // for just one merged element, return singleton top
			pc := SingletonBasePredictionContextCreate(mergedParents[0], mergedReturnStates[0])
			if mergeCache != nil {
				mergeCache.set(a.hash(), b.hash(), pc)
			}
			return pc
		}
		mergedParents = mergedParents[0:k]
		mergedReturnStates = mergedReturnStates[0:k]
	}

	M := NewArrayPredictionContext(mergedParents, mergedReturnStates)

	// if we created same array as a or b, return that instead
	// TODO: track whether this is possible above during merge sort for speed
	if M == a {
		if mergeCache != nil {
			mergeCache.set(a.hash(), b.hash(), a)
		}
		return a
	}
	if M == b {
		if mergeCache != nil {
			mergeCache.set(a.hash(), b.hash(), b)
		}
		return b
	}
	combineCommonParents(mergedParents)

	if mergeCache != nil {
		mergeCache.set(a.hash(), b.hash(), M)
	}
	return M
}

//
// Make pass over all <em>M</em> {@code parents} merge any {@code equals()}
// ones.
// /
func combineCommonParents(parents []PredictionContext) {
	uniqueParents := make(map[PredictionContext]PredictionContext)

	for p := 0; p < len(parents); p++ {
		parent := parents[p]
		if uniqueParents[parent] == nil {
			uniqueParents[parent] = parent
		}
	}
	for q := 0; q < len(parents); q++ {
		parents[q] = uniqueParents[parents[q]]
	}
}

func getCachedBasePredictionContext(context PredictionContext, contextCache *PredictionContextCache, visited map[PredictionContext]PredictionContext) PredictionContext {

	if context.isEmpty() {
		return context
	}
	existing := visited[context]
	if existing != nil {
		return existing
	}
	existing = contextCache.Get(context)
	if existing != nil {
		visited[context] = existing
		return existing
	}
	changed := false
	parents := make([]PredictionContext, context.length())
	for i := 0; i < len(parents); i++ {
		parent := getCachedBasePredictionContext(context.GetParent(i), contextCache, visited)
		if changed || parent != context.GetParent(i) {
			if !changed {
				parents = make([]PredictionContext, context.length())
				for j := 0; j < context.length(); j++ {
					parents[j] = context.GetParent(j)
				}
				changed = true
			}
			parents[i] = parent
		}
	}
	if !changed {
		contextCache.add(context)
		visited[context] = context
		return context
	}
	var updated PredictionContext
	if len(parents) == 0 {
		updated = BasePredictionContextEMPTY
	} else if len(parents) == 1 {
		updated = SingletonBasePredictionContextCreate(parents[0], context.getReturnState(0))
	} else {
		updated = NewArrayPredictionContext(parents, context.(*ArrayPredictionContext).GetReturnStates())
	}
	contextCache.add(updated)
	visited[updated] = updated
	visited[context] = updated

	return updated
}
