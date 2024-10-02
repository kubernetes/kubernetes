package antlr

// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

import (
	"container/list"
	"runtime/debug"
	"sort"
	"sync"
)

// Collectable is an interface that a struct should implement if it is to be
// usable as a key in these collections.
type Collectable[T any] interface {
	Hash() int
	Equals(other Collectable[T]) bool
}

type Comparator[T any] interface {
	Hash1(o T) int
	Equals2(T, T) bool
}

type CollectionSource int
type CollectionDescriptor struct {
	SybolicName string
	Description string
}

const (
	UnknownCollection CollectionSource = iota
	ATNConfigLookupCollection
	ATNStateCollection
	DFAStateCollection
	ATNConfigCollection
	PredictionContextCollection
	SemanticContextCollection
	ClosureBusyCollection
	PredictionVisitedCollection
	MergeCacheCollection
	PredictionContextCacheCollection
	AltSetCollection
	ReachSetCollection
)

var CollectionDescriptors = map[CollectionSource]CollectionDescriptor{
	UnknownCollection: {
		SybolicName: "UnknownCollection",
		Description: "Unknown collection type. Only used if the target author thought it was an unimportant collection.",
	},
	ATNConfigCollection: {
		SybolicName: "ATNConfigCollection",
		Description: "ATNConfig collection. Used to store the ATNConfigs for a particular state in the ATN." +
			"For instance, it is used to store the results of the closure() operation in the ATN.",
	},
	ATNConfigLookupCollection: {
		SybolicName: "ATNConfigLookupCollection",
		Description: "ATNConfigLookup collection. Used to store the ATNConfigs for a particular state in the ATN." +
			"This is used to prevent duplicating equivalent states in an ATNConfigurationSet.",
	},
	ATNStateCollection: {
		SybolicName: "ATNStateCollection",
		Description: "ATNState collection. This is used to store the states of the ATN.",
	},
	DFAStateCollection: {
		SybolicName: "DFAStateCollection",
		Description: "DFAState collection. This is used to store the states of the DFA.",
	},
	PredictionContextCollection: {
		SybolicName: "PredictionContextCollection",
		Description: "PredictionContext collection. This is used to store the prediction contexts of the ATN and cache computes.",
	},
	SemanticContextCollection: {
		SybolicName: "SemanticContextCollection",
		Description: "SemanticContext collection. This is used to store the semantic contexts of the ATN.",
	},
	ClosureBusyCollection: {
		SybolicName: "ClosureBusyCollection",
		Description: "ClosureBusy collection. This is used to check and prevent infinite recursion right recursive rules." +
			"It stores ATNConfigs that are currently being processed in the closure() operation.",
	},
	PredictionVisitedCollection: {
		SybolicName: "PredictionVisitedCollection",
		Description: "A map that records whether we have visited a particular context when searching through cached entries.",
	},
	MergeCacheCollection: {
		SybolicName: "MergeCacheCollection",
		Description: "A map that records whether we have already merged two particular contexts and can save effort by not repeating it.",
	},
	PredictionContextCacheCollection: {
		SybolicName: "PredictionContextCacheCollection",
		Description: "A map that records whether we have already created a particular context and can save effort by not computing it again.",
	},
	AltSetCollection: {
		SybolicName: "AltSetCollection",
		Description: "Used to eliminate duplicate alternatives in an ATN config set.",
	},
	ReachSetCollection: {
		SybolicName: "ReachSetCollection",
		Description: "Used as merge cache to prevent us needing to compute the merge of two states if we have already done it.",
	},
}

// JStore implements a container that allows the use of a struct to calculate the key
// for a collection of values akin to map. This is not meant to be a full-blown HashMap but just
// serve the needs of the ANTLR Go runtime.
//
// For ease of porting the logic of the runtime from the master target (Java), this collection
// operates in a similar way to Java, in that it can use any struct that supplies a Hash() and Equals()
// function as the key. The values are stored in a standard go map which internally is a form of hashmap
// itself, the key for the go map is the hash supplied by the key object. The collection is able to deal with
// hash conflicts by using a simple slice of values associated with the hash code indexed bucket. That isn't
// particularly efficient, but it is simple, and it works. As this is specifically for the ANTLR runtime, and
// we understand the requirements, then this is fine - this is not a general purpose collection.
type JStore[T any, C Comparator[T]] struct {
	store      map[int][]T
	len        int
	comparator Comparator[T]
	stats      *JStatRec
}

func NewJStore[T any, C Comparator[T]](comparator Comparator[T], cType CollectionSource, desc string) *JStore[T, C] {

	if comparator == nil {
		panic("comparator cannot be nil")
	}

	s := &JStore[T, C]{
		store:      make(map[int][]T, 1),
		comparator: comparator,
	}
	if collectStats {
		s.stats = &JStatRec{
			Source:      cType,
			Description: desc,
		}

		// Track where we created it from  if we are being asked to do so
		if runtimeConfig.statsTraceStacks {
			s.stats.CreateStack = debug.Stack()
		}
		Statistics.AddJStatRec(s.stats)
	}
	return s
}

// Put will store given value in the collection. Note that the key for storage is generated from
// the value itself - this is specifically because that is what ANTLR needs - this would not be useful
// as any kind of general collection.
//
// If the key has a hash conflict, then the value will be added to the slice of values associated with the
// hash, unless the value is already in the slice, in which case the existing value is returned. Value equivalence is
// tested by calling the equals() method on the key.
//
// # If the given value is already present in the store, then the existing value is returned as v and exists is set to true
//
// If the given value is not present in the store, then the value is added to the store and returned as v and exists is set to false.
func (s *JStore[T, C]) Put(value T) (v T, exists bool) {

	if collectStats {
		s.stats.Puts++
	}
	kh := s.comparator.Hash1(value)

	var hClash bool
	for _, v1 := range s.store[kh] {
		hClash = true
		if s.comparator.Equals2(value, v1) {
			if collectStats {
				s.stats.PutHits++
				s.stats.PutHashConflicts++
			}
			return v1, true
		}
		if collectStats {
			s.stats.PutMisses++
		}
	}
	if collectStats && hClash {
		s.stats.PutHashConflicts++
	}
	s.store[kh] = append(s.store[kh], value)

	if collectStats {
		if len(s.store[kh]) > s.stats.MaxSlotSize {
			s.stats.MaxSlotSize = len(s.store[kh])
		}
	}
	s.len++
	if collectStats {
		s.stats.CurSize = s.len
		if s.len > s.stats.MaxSize {
			s.stats.MaxSize = s.len
		}
	}
	return value, false
}

// Get will return the value associated with the key - the type of the key is the same type as the value
// which would not generally be useful, but this is a specific thing for ANTLR where the key is
// generated using the object we are going to store.
func (s *JStore[T, C]) Get(key T) (T, bool) {
	if collectStats {
		s.stats.Gets++
	}
	kh := s.comparator.Hash1(key)
	var hClash bool
	for _, v := range s.store[kh] {
		hClash = true
		if s.comparator.Equals2(key, v) {
			if collectStats {
				s.stats.GetHits++
				s.stats.GetHashConflicts++
			}
			return v, true
		}
		if collectStats {
			s.stats.GetMisses++
		}
	}
	if collectStats {
		if hClash {
			s.stats.GetHashConflicts++
		}
		s.stats.GetNoEnt++
	}
	return key, false
}

// Contains returns true if the given key is present in the store
func (s *JStore[T, C]) Contains(key T) bool {
	_, present := s.Get(key)
	return present
}

func (s *JStore[T, C]) SortedSlice(less func(i, j T) bool) []T {
	vs := make([]T, 0, len(s.store))
	for _, v := range s.store {
		vs = append(vs, v...)
	}
	sort.Slice(vs, func(i, j int) bool {
		return less(vs[i], vs[j])
	})

	return vs
}

func (s *JStore[T, C]) Each(f func(T) bool) {
	for _, e := range s.store {
		for _, v := range e {
			f(v)
		}
	}
}

func (s *JStore[T, C]) Len() int {
	return s.len
}

func (s *JStore[T, C]) Values() []T {
	vs := make([]T, 0, len(s.store))
	for _, e := range s.store {
		vs = append(vs, e...)
	}
	return vs
}

type entry[K, V any] struct {
	key K
	val V
}

type JMap[K, V any, C Comparator[K]] struct {
	store      map[int][]*entry[K, V]
	len        int
	comparator Comparator[K]
	stats      *JStatRec
}

func NewJMap[K, V any, C Comparator[K]](comparator Comparator[K], cType CollectionSource, desc string) *JMap[K, V, C] {
	m := &JMap[K, V, C]{
		store:      make(map[int][]*entry[K, V], 1),
		comparator: comparator,
	}
	if collectStats {
		m.stats = &JStatRec{
			Source:      cType,
			Description: desc,
		}
		// Track where we created it from  if we are being asked to do so
		if runtimeConfig.statsTraceStacks {
			m.stats.CreateStack = debug.Stack()
		}
		Statistics.AddJStatRec(m.stats)
	}
	return m
}

func (m *JMap[K, V, C]) Put(key K, val V) (V, bool) {
	if collectStats {
		m.stats.Puts++
	}
	kh := m.comparator.Hash1(key)

	var hClash bool
	for _, e := range m.store[kh] {
		hClash = true
		if m.comparator.Equals2(e.key, key) {
			if collectStats {
				m.stats.PutHits++
				m.stats.PutHashConflicts++
			}
			return e.val, true
		}
		if collectStats {
			m.stats.PutMisses++
		}
	}
	if collectStats {
		if hClash {
			m.stats.PutHashConflicts++
		}
	}
	m.store[kh] = append(m.store[kh], &entry[K, V]{key, val})
	if collectStats {
		if len(m.store[kh]) > m.stats.MaxSlotSize {
			m.stats.MaxSlotSize = len(m.store[kh])
		}
	}
	m.len++
	if collectStats {
		m.stats.CurSize = m.len
		if m.len > m.stats.MaxSize {
			m.stats.MaxSize = m.len
		}
	}
	return val, false
}

func (m *JMap[K, V, C]) Values() []V {
	vs := make([]V, 0, len(m.store))
	for _, e := range m.store {
		for _, v := range e {
			vs = append(vs, v.val)
		}
	}
	return vs
}

func (m *JMap[K, V, C]) Get(key K) (V, bool) {
	if collectStats {
		m.stats.Gets++
	}
	var none V
	kh := m.comparator.Hash1(key)
	var hClash bool
	for _, e := range m.store[kh] {
		hClash = true
		if m.comparator.Equals2(e.key, key) {
			if collectStats {
				m.stats.GetHits++
				m.stats.GetHashConflicts++
			}
			return e.val, true
		}
		if collectStats {
			m.stats.GetMisses++
		}
	}
	if collectStats {
		if hClash {
			m.stats.GetHashConflicts++
		}
		m.stats.GetNoEnt++
	}
	return none, false
}

func (m *JMap[K, V, C]) Len() int {
	return m.len
}

func (m *JMap[K, V, C]) Delete(key K) {
	kh := m.comparator.Hash1(key)
	for i, e := range m.store[kh] {
		if m.comparator.Equals2(e.key, key) {
			m.store[kh] = append(m.store[kh][:i], m.store[kh][i+1:]...)
			m.len--
			return
		}
	}
}

func (m *JMap[K, V, C]) Clear() {
	m.store = make(map[int][]*entry[K, V])
}

type JPCMap struct {
	store *JMap[*PredictionContext, *JMap[*PredictionContext, *PredictionContext, *ObjEqComparator[*PredictionContext]], *ObjEqComparator[*PredictionContext]]
	size  int
	stats *JStatRec
}

func NewJPCMap(cType CollectionSource, desc string) *JPCMap {
	m := &JPCMap{
		store: NewJMap[*PredictionContext, *JMap[*PredictionContext, *PredictionContext, *ObjEqComparator[*PredictionContext]], *ObjEqComparator[*PredictionContext]](pContextEqInst, cType, desc),
	}
	if collectStats {
		m.stats = &JStatRec{
			Source:      cType,
			Description: desc,
		}
		// Track where we created it from  if we are being asked to do so
		if runtimeConfig.statsTraceStacks {
			m.stats.CreateStack = debug.Stack()
		}
		Statistics.AddJStatRec(m.stats)
	}
	return m
}

func (pcm *JPCMap) Get(k1, k2 *PredictionContext) (*PredictionContext, bool) {
	if collectStats {
		pcm.stats.Gets++
	}
	// Do we have a map stored by k1?
	//
	m2, present := pcm.store.Get(k1)
	if present {
		if collectStats {
			pcm.stats.GetHits++
		}
		// We found a map of values corresponding to k1, so now we need to look up k2 in that map
		//
		return m2.Get(k2)
	}
	if collectStats {
		pcm.stats.GetMisses++
	}
	return nil, false
}

func (pcm *JPCMap) Put(k1, k2, v *PredictionContext) {

	if collectStats {
		pcm.stats.Puts++
	}
	// First does a map already exist for k1?
	//
	if m2, present := pcm.store.Get(k1); present {
		if collectStats {
			pcm.stats.PutHits++
		}
		_, present = m2.Put(k2, v)
		if !present {
			pcm.size++
			if collectStats {
				pcm.stats.CurSize = pcm.size
				if pcm.size > pcm.stats.MaxSize {
					pcm.stats.MaxSize = pcm.size
				}
			}
		}
	} else {
		// No map found for k1, so we create it, add in our value, then store is
		//
		if collectStats {
			pcm.stats.PutMisses++
			m2 = NewJMap[*PredictionContext, *PredictionContext, *ObjEqComparator[*PredictionContext]](pContextEqInst, pcm.stats.Source, pcm.stats.Description+" map entry")
		} else {
			m2 = NewJMap[*PredictionContext, *PredictionContext, *ObjEqComparator[*PredictionContext]](pContextEqInst, PredictionContextCacheCollection, "map entry")
		}

		m2.Put(k2, v)
		pcm.store.Put(k1, m2)
		pcm.size++
	}
}

type JPCMap2 struct {
	store map[int][]JPCEntry
	size  int
	stats *JStatRec
}

type JPCEntry struct {
	k1, k2, v *PredictionContext
}

func NewJPCMap2(cType CollectionSource, desc string) *JPCMap2 {
	m := &JPCMap2{
		store: make(map[int][]JPCEntry, 1000),
	}
	if collectStats {
		m.stats = &JStatRec{
			Source:      cType,
			Description: desc,
		}
		// Track where we created it from  if we are being asked to do so
		if runtimeConfig.statsTraceStacks {
			m.stats.CreateStack = debug.Stack()
		}
		Statistics.AddJStatRec(m.stats)
	}
	return m
}

func dHash(k1, k2 *PredictionContext) int {
	return k1.cachedHash*31 + k2.cachedHash
}

func (pcm *JPCMap2) Get(k1, k2 *PredictionContext) (*PredictionContext, bool) {
	if collectStats {
		pcm.stats.Gets++
	}

	h := dHash(k1, k2)
	var hClash bool
	for _, e := range pcm.store[h] {
		hClash = true
		if e.k1.Equals(k1) && e.k2.Equals(k2) {
			if collectStats {
				pcm.stats.GetHits++
				pcm.stats.GetHashConflicts++
			}
			return e.v, true
		}
		if collectStats {
			pcm.stats.GetMisses++
		}
	}
	if collectStats {
		if hClash {
			pcm.stats.GetHashConflicts++
		}
		pcm.stats.GetNoEnt++
	}
	return nil, false
}

func (pcm *JPCMap2) Put(k1, k2, v *PredictionContext) (*PredictionContext, bool) {
	if collectStats {
		pcm.stats.Puts++
	}
	h := dHash(k1, k2)
	var hClash bool
	for _, e := range pcm.store[h] {
		hClash = true
		if e.k1.Equals(k1) && e.k2.Equals(k2) {
			if collectStats {
				pcm.stats.PutHits++
				pcm.stats.PutHashConflicts++
			}
			return e.v, true
		}
		if collectStats {
			pcm.stats.PutMisses++
		}
	}
	if collectStats {
		if hClash {
			pcm.stats.PutHashConflicts++
		}
	}
	pcm.store[h] = append(pcm.store[h], JPCEntry{k1, k2, v})
	pcm.size++
	if collectStats {
		pcm.stats.CurSize = pcm.size
		if pcm.size > pcm.stats.MaxSize {
			pcm.stats.MaxSize = pcm.size
		}
	}
	return nil, false
}

type VisitEntry struct {
	k *PredictionContext
	v *PredictionContext
}
type VisitRecord struct {
	store map[*PredictionContext]*PredictionContext
	len   int
	stats *JStatRec
}

type VisitList struct {
	cache *list.List
	lock  sync.RWMutex
}

var visitListPool = VisitList{
	cache: list.New(),
	lock:  sync.RWMutex{},
}

// NewVisitRecord returns a new VisitRecord instance from the pool if available.
// Note that this "map" uses a pointer as a key because we are emulating the behavior of
// IdentityHashMap in Java, which uses the `==` operator to compare whether the keys are equal,
// which means is the key the same reference to an object rather than is it .equals() to another
// object.
func NewVisitRecord() *VisitRecord {
	visitListPool.lock.Lock()
	el := visitListPool.cache.Front()
	defer visitListPool.lock.Unlock()
	var vr *VisitRecord
	if el == nil {
		vr = &VisitRecord{
			store: make(map[*PredictionContext]*PredictionContext),
		}
		if collectStats {
			vr.stats = &JStatRec{
				Source:      PredictionContextCacheCollection,
				Description: "VisitRecord",
			}
			// Track where we created it from  if we are being asked to do so
			if runtimeConfig.statsTraceStacks {
				vr.stats.CreateStack = debug.Stack()
			}
		}
	} else {
		vr = el.Value.(*VisitRecord)
		visitListPool.cache.Remove(el)
		vr.store = make(map[*PredictionContext]*PredictionContext)
	}
	if collectStats {
		Statistics.AddJStatRec(vr.stats)
	}
	return vr
}

func (vr *VisitRecord) Release() {
	vr.len = 0
	vr.store = nil
	if collectStats {
		vr.stats.MaxSize = 0
		vr.stats.CurSize = 0
		vr.stats.Gets = 0
		vr.stats.GetHits = 0
		vr.stats.GetMisses = 0
		vr.stats.GetHashConflicts = 0
		vr.stats.GetNoEnt = 0
		vr.stats.Puts = 0
		vr.stats.PutHits = 0
		vr.stats.PutMisses = 0
		vr.stats.PutHashConflicts = 0
		vr.stats.MaxSlotSize = 0
	}
	visitListPool.lock.Lock()
	visitListPool.cache.PushBack(vr)
	visitListPool.lock.Unlock()
}

func (vr *VisitRecord) Get(k *PredictionContext) (*PredictionContext, bool) {
	if collectStats {
		vr.stats.Gets++
	}
	v := vr.store[k]
	if v != nil {
		if collectStats {
			vr.stats.GetHits++
		}
		return v, true
	}
	if collectStats {
		vr.stats.GetNoEnt++
	}
	return nil, false
}

func (vr *VisitRecord) Put(k, v *PredictionContext) (*PredictionContext, bool) {
	if collectStats {
		vr.stats.Puts++
	}
	vr.store[k] = v
	vr.len++
	if collectStats {
		vr.stats.CurSize = vr.len
		if vr.len > vr.stats.MaxSize {
			vr.stats.MaxSize = vr.len
		}
	}
	return v, false
}
