package diskv

import (
	"sync"

	"github.com/google/btree"
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

// btreeString is a custom data type that satisfies the BTree Less interface,
// making the strings it wraps sortable by the BTree package.
type btreeString struct {
	s string
	l LessFunction
}

// Less satisfies the BTree.Less interface using the btreeString's LessFunction.
func (s btreeString) Less(i btree.Item) bool {
	return s.l(s.s, i.(btreeString).s)
}

// BTreeIndex is an implementation of the Index interface using google/btree.
type BTreeIndex struct {
	sync.RWMutex
	LessFunction
	*btree.BTree
}

// Initialize populates the BTree tree with data from the keys channel,
// according to the passed less function. It's destructive to the BTreeIndex.
func (i *BTreeIndex) Initialize(less LessFunction, keys <-chan string) {
	i.Lock()
	defer i.Unlock()
	i.LessFunction = less
	i.BTree = rebuild(less, keys)
}

// Insert inserts the given key (only) into the BTree tree.
func (i *BTreeIndex) Insert(key string) {
	i.Lock()
	defer i.Unlock()
	if i.BTree == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}
	i.BTree.ReplaceOrInsert(btreeString{s: key, l: i.LessFunction})
}

// Delete removes the given key (only) from the BTree tree.
func (i *BTreeIndex) Delete(key string) {
	i.Lock()
	defer i.Unlock()
	if i.BTree == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}
	i.BTree.Delete(btreeString{s: key, l: i.LessFunction})
}

// Keys yields a maximum of n keys in order. If the passed 'from' key is empty,
// Keys will return the first n keys. If the passed 'from' key is non-empty, the
// first key in the returned slice will be the key that immediately follows the
// passed key, in key order.
func (i *BTreeIndex) Keys(from string, n int) []string {
	i.RLock()
	defer i.RUnlock()

	if i.BTree == nil || i.LessFunction == nil {
		panic("uninitialized index")
	}

	if i.BTree.Len() <= 0 {
		return []string{}
	}

	btreeFrom := btreeString{s: from, l: i.LessFunction}
	skipFirst := true
	if len(from) <= 0 || !i.BTree.Has(btreeFrom) {
		// no such key, so fabricate an always-smallest item
		btreeFrom = btreeString{s: "", l: func(string, string) bool { return true }}
		skipFirst = false
	}

	keys := []string{}
	iterator := func(i btree.Item) bool {
		keys = append(keys, i.(btreeString).s)
		return len(keys) < n
	}
	i.BTree.AscendGreaterOrEqual(btreeFrom, iterator)

	if skipFirst && len(keys) > 0 {
		keys = keys[1:]
	}

	return keys
}

// rebuildIndex does the work of regenerating the index
// with the given keys.
func rebuild(less LessFunction, keys <-chan string) *btree.BTree {
	tree := btree.New(2)
	for key := range keys {
		tree.ReplaceOrInsert(btreeString{s: key, l: less})
	}
	return tree
}
