package diskv

import (
	"sync"

	"github.com/petar/GoLLRB/llrb"
)

// Index is a generic interface for things that can
// provide an ordered list of keys.
type Index interface {
	Initialize(less LessFunction, keys <-chan string)
	Insert(key string)
	Delete(key string)
	Keys(from string, n int) []string
}

// LessFunction is used to initialize an Index of keys in a specific order.
type LessFunction func(string, string) bool

// llrbString is a custom data type that satisfies the LLRB Less interface,
// making the strings it wraps sortable by the LLRB package.
type llrbString struct {
	s string
	l LessFunction
}

// Less satisfies the llrb.Less interface using the llrbString's LessFunction.
func (s llrbString) Less(i llrb.Item) bool {
	return s.l(s.s, i.(llrbString).s)
}

// LLRBIndex is an implementation of the Index interface
// using Petar Maymounkov's LLRB tree.
type LLRBIndex struct {
	sync.RWMutex
	LessFunction
	*llrb.LLRB
}

// Initialize populates the LLRB tree with data from the keys channel,
// according to the passed less function. It's destructive to the LLRBIndex.
func (i *LLRBIndex) Initialize(less LessFunction, keys <-chan string) {
	i.Lock()
	defer i.Unlock()
	i.LessFunction = less
	i.LLRB = rebuild(less, keys)
}

// Insert inserts the given key (only) into the LLRB tree.
func (i *LLRBIndex) Insert(key string) {
	i.Lock()
	defer i.Unlock()
	if i.LLRB == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}
	i.LLRB.ReplaceOrInsert(llrbString{s: key, l: i.LessFunction})
}

// Delete removes the given key (only) from the LLRB tree.
func (i *LLRBIndex) Delete(key string) {
	i.Lock()
	defer i.Unlock()
	if i.LLRB == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}
	i.LLRB.Delete(llrbString{s: key, l: i.LessFunction})
}

// Keys yields a maximum of n keys in order. If the passed 'from' key is empty,
// Keys will return the first n keys. If the passed 'from' key is non-empty, the
// first key in the returned slice will be the key that immediately follows the
// passed key, in key order.
func (i *LLRBIndex) Keys(from string, n int) []string {
	i.RLock()
	defer i.RUnlock()

	if i.LLRB == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}

	if i.LLRB.Len() <= 0 {
		return []string{}
	}

	llrbFrom := llrbString{s: from, l: i.LessFunction}
	skipFirst := true
	if len(from) <= 0 || !i.LLRB.Has(llrbFrom) {
		// no such key, so start at the top
		llrbFrom = i.LLRB.Min().(llrbString)
		skipFirst = false
	}

	keys := []string{}
	iterator := func(i llrb.Item) bool {
		keys = append(keys, i.(llrbString).s)
		return len(keys) < n
	}
	i.LLRB.AscendGreaterOrEqual(llrbFrom, iterator)

	if skipFirst && len(keys) > 0 {
		keys = keys[1:]
	}

	return keys
}

// rebuildIndex does the work of regenerating the index
// with the given keys.
func rebuild(less LessFunction, keys <-chan string) *llrb.LLRB {
	tree := llrb.New()
	for key := range keys {
		tree.ReplaceOrInsert(llrbString{s: key, l: less})
	}
	return tree
}
