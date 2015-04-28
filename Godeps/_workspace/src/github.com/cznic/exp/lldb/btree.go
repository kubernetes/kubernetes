// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/cznic/bufs"
	"github.com/cznic/fileutil"
	"github.com/cznic/sortutil"
)

const (
	kData             = 256         // [1, 512]
	kIndex            = 256         // [2, 2048]
	kKV               = 19          // Size of the key/value field in btreeDataPage
	kSz               = kKV - 1 - 7 // Content prefix size
	kH                = kKV - 7     // Content field offset for handle
	tagBTreeDataPage  = 1
	tagBTreeIndexPage = 0
)

// BTree is a B+tree[1][2], i.e. a variant which speeds up
// enumeration/iteration of the BTree. According to its origin it can be
// volatile (backed only by memory) or non-volatile (backed by a non-volatile
// Allocator).
//
// The specific implementation of BTrees in this package are B+trees with
// delayed split/concatenation (discussed in e.g. [3]).
//
// Note: No BTree methods returns io.EOF for physical Filer reads/writes.  The
// io.EOF is returned only by bTreeEnumerator methods to indicate "no more K-V
// pair".
//
//  [1]: http://en.wikipedia.org/wiki/B+tree
//  [2]: http://zgking.com:8080/home/donghui/publications/books/dshandbook_BTree.pdf
//  [3]: http://people.cs.aau.dk/~simas/aalg06/UbiquitBtree.pdf
type BTree struct {
	store   btreeStore
	root    btree
	collate func(a, b []byte) int
	serial  uint64
}

// NewBTree returns a new, memory-only BTree.
func NewBTree(collate func(a, b []byte) int) *BTree {
	store := newMemBTreeStore()
	root, err := newBTree(store)
	if err != nil { // should not happen
		panic(err.Error())
	}

	return &BTree{store, root, collate, 0}
}

// IsMem reports if t is a memory only BTree.
func (t *BTree) IsMem() (r bool) {
	_, r = t.store.(*memBTreeStore)
	return
}

// Clear empties the tree.
func (t *BTree) Clear() (err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	return t.root.clear(t.store)
}

// Delete deletes key and its associated value from the tree.
func (t *BTree) Delete(key []byte) (err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	_, err = t.root.extract(t.store, nil, t.collate, key)
	return
}

// DeleteAny deletes one key and its associated value from the tree. If the
// tree is empty on return then empty is true.
func (t *BTree) DeleteAny() (empty bool, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	return t.root.deleteAny(t.store)
}

func elem(v interface{}) string {
	switch x := v.(type) {
	default:
		panic("internal error")
	case nil:
		return "nil"
	case bool:
		if x {
			return "true"
		}

		return "false"
	case int64:
		return fmt.Sprint(x)
	case uint64:
		return fmt.Sprint(x)
	case float64:
		s := fmt.Sprintf("%g", x)
		if !strings.Contains(s, ".") {
			s += "."
		}
		return s
	case complex128:
		s := fmt.Sprint(x)
		return s[1 : len(s)-1]
	case []byte:
		return fmt.Sprintf("[]byte{% 02x}", x)
	case string:
		return fmt.Sprintf("%q", x)
	}
}

// Dump outputs a human readable dump of t to w. It is usable iff t keys and
// values are encoded scalars (see EncodeScalars). Intended use is only for
// examples or debugging. Some type information is lost in the rendering, for
// example a float value '17.' and an integer value '17' may both output as
// '17'.
func (t *BTree) Dump(w io.Writer) (err error) {
	enum, err := t.seekFirst()
	if err != nil {
		return
	}

	for {
		bkey, bval, err := enum.current()
		if err != nil {
			return err
		}

		key, err := DecodeScalars(bkey)
		if err != nil {
			return err
		}

		val, err := DecodeScalars(bval)
		if err != nil {
			return err
		}

		kk := []string{}
		if key == nil {
			kk = []string{"null"}
		}
		for _, v := range key {
			kk = append(kk, elem(v))
		}
		vv := []string{}
		if val == nil {
			vv = []string{"null"}
		}
		for _, v := range val {
			vv = append(vv, elem(v))
		}
		skey := strings.Join(kk, ", ")
		sval := strings.Join(vv, ", ")
		if len(vv) > 1 {
			sval = fmt.Sprintf("[]interface{%s}", sval)
		}
		if _, err = fmt.Fprintf(w, "%s → %s\n", skey, sval); err != nil {
			return err
		}

		err = enum.next()
		if err != nil {
			if fileutil.IsEOF(err) {
				err = nil
				break
			}

			return err
		}
	}
	return
}

// Extract is a combination of Get and Delete. If the key exists in the tree,
// it is returned (like Get) and also deleted from a tree in a more efficient
// way which doesn't walk it twice.  The returned slice may be a sub-slice of
// buf if buf was large enough to hold the entire content.  Otherwise, a newly
// allocated slice will be returned.  It is valid to pass a nil buf.
func (t *BTree) Extract(buf, key []byte) (value []byte, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	return t.root.extract(t.store, buf, t.collate, key)
}

// First returns the first KV pair of the tree, if it exists. Otherwise key == nil
// and value == nil.
func (t *BTree) First() (key, value []byte, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	var p btreeDataPage
	if _, p, err = t.root.first(t.store); err != nil || p == nil {
		return
	}

	if key, err = p.key(t.store, 0); err != nil {
		return
	}

	value, err = p.value(t.store, 0)
	return
}

// Get returns the value associated with key, or nil if no such value exists.
// The returned slice may be a sub-slice of buf if buf was large enough to hold
// the entire content.  Otherwise, a newly allocated slice will be returned.
// It is valid to pass a nil buf.
func (t *BTree) Get(buf, key []byte) (value []byte, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	buffer := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(buffer)
	if buffer, err = t.root.get(t.store, buffer, t.collate, key); buffer == nil || err != nil {
		return
	}

	value = need(len(buffer), buf)
	copy(value, buffer)
	return
}

// Handle reports t's handle.
func (t *BTree) Handle() int64 {
	return int64(t.root)
}

// Last returns the last KV pair of the tree, if it exists. Otherwise key == nil
// and value == nil.
func (t *BTree) Last() (key, value []byte, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	var p btreeDataPage
	if _, p, err = t.root.last(t.store); err != nil || p == nil {
		return
	}

	index := p.len() - 1
	if key, err = p.key(t.store, index); err != nil {
		return
	}

	value, err = p.value(t.store, index)
	return
}

// Put combines Get and Set in a more efficient way where the tree is walked
// only once.  The upd(ater) receives the current (key, old-value), if that
// exists or (key, nil) otherwise.  It can then return a (new-value, true, nil)
// to create or overwrite the existing value in the KV pair, or (whatever,
// false, nil) if it decides not to create or not to update the value of the KV
// pair.
//
// 	tree.Set(k, v)
//
// conceptually equals
//
// 	tree.Put(k, func(k, v []byte){ return v, true }([]byte, bool))
//
// modulo the differing return values.
//
// The returned slice may be a sub-slice of buf if buf was large enough to hold
// the entire content.  Otherwise, a newly allocated slice will be returned.
// It is valid to pass a nil buf.
func (t *BTree) Put(buf, key []byte, upd func(key, old []byte) (new []byte, write bool, err error)) (old []byte, written bool, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	return t.root.put2(buf, t.store, t.collate, key, upd)
}

// Seek returns an Enumerator with "position" or an error of any. Normally the
// position is on a KV pair such that key >= KV.key. Then hit is key == KV.key.
// The position is possibly "after" the last KV pair, but that is not an error.
func (t *BTree) Seek(key []byte) (enum *BTreeEnumerator, hit bool, err error) {
	enum0, hit, err := t.seek(key)
	if err != nil {
		return
	}

	enum = &BTreeEnumerator{
		enum:     enum0,
		firstHit: hit,
		key:      append([]byte(nil), key...),
	}
	return
}

func (t *BTree) seek(key []byte) (enum *bTreeEnumerator, hit bool, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	r := &bTreeEnumerator{t: t, collate: t.collate, serial: t.serial}
	if r.p, r.index, hit, err = t.root.seek(t.store, r.collate, key); err != nil {
		return
	}

	enum = r
	return
}

// IndexSeek returns an Enumerator with "position" or an error of any. Normally
// the position is on a KV pair such that key >= KV.key. Then hit is key ==
// KV.key.  The position is possibly "after" the last KV pair, but that is not
// an error.  The collate function originally passed to CreateBTree is used for
// enumerating the tree but a custom collate function c is used for IndexSeek.
func (t *BTree) IndexSeek(key []byte, c func(a, b []byte) int) (enum *BTreeEnumerator, hit bool, err error) { //TODO +test
	enum0, hit, err := t.indexSeek(key, c)
	if err != nil {
		return
	}

	enum = &BTreeEnumerator{
		enum:     enum0,
		firstHit: hit,
		key:      append([]byte(nil), key...),
	}
	return
}

func (t *BTree) indexSeek(key []byte, c func(a, b []byte) int) (enum *bTreeEnumerator, hit bool, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	r := &bTreeEnumerator{t: t, collate: t.collate, serial: t.serial}
	if r.p, r.index, hit, err = t.root.seek(t.store, c, key); err != nil {
		return
	}

	enum = r
	return
}

// seekFirst returns an enumerator positioned on the first KV pair in the tree,
// if any.  For an empty tree, err == io.EOF is returend.
func (t *BTree) SeekFirst() (enum *BTreeEnumerator, err error) {
	enum0, err := t.seekFirst()
	if err != nil {
		return
	}

	var key []byte
	if key, _, err = enum0.current(); err != nil {
		return
	}

	enum = &BTreeEnumerator{
		enum:     enum0,
		firstHit: true,
		key:      append([]byte(nil), key...),
	}
	return
}

func (t *BTree) seekFirst() (enum *bTreeEnumerator, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	var p btreeDataPage
	if _, p, err = t.root.first(t.store); err == nil && p == nil {
		err = io.EOF
	}
	if err != nil {
		return
	}

	return &bTreeEnumerator{t: t, collate: t.collate, p: p, index: 0, serial: t.serial}, nil
}

// seekLast returns an enumerator positioned on the last KV pair in the tree,
// if any.  For an empty tree, err == io.EOF is returend.
func (t *BTree) SeekLast() (enum *BTreeEnumerator, err error) {
	enum0, err := t.seekLast()
	if err != nil {
		return
	}

	var key []byte
	if key, _, err = enum0.current(); err != nil {
		return
	}

	enum = &BTreeEnumerator{
		enum:     enum0,
		firstHit: true,
		key:      append([]byte(nil), key...),
	}
	return
}

func (t *BTree) seekLast() (enum *bTreeEnumerator, err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	var p btreeDataPage
	if _, p, err = t.root.last(t.store); err == nil && p == nil {
		err = io.EOF
	}
	if err != nil {
		return
	}

	return &bTreeEnumerator{t: t, collate: t.collate, p: p, index: p.len() - 1, serial: t.serial}, nil
}

// Set sets the value associated with key. Any previous value, if existed, is
// overwritten by the new one.
func (t *BTree) Set(key, value []byte) (err error) {
	if t == nil {
		err = errors.New("BTree method invoked on nil receiver")
		return
	}

	t.serial++
	dst := bufs.GCache.Get(maxBuf)
	_, err = t.root.put(dst, t.store, t.collate, key, value, true)
	bufs.GCache.Put(dst)
	return
}

// bTreeEnumerator is a closure of a BTree and a position. It is returned from
// BTree.seek.
//
// NOTE: bTreeEnumerator cannot be used after its BTree was mutated after the
// bTreeEnumerator was acquired from any of the seek, seekFirst, seekLast
// methods.
type bTreeEnumerator struct {
	t       *BTree
	collate func(a, b []byte) int
	p       btreeDataPage
	index   int
	serial  uint64
}

// Current returns the KV pair the enumerator is currently positioned on. If
// the position is before the first KV pair in the tree or after the last KV
// pair in the tree then err == io.EOF is returned.
//
// If the enumerator has been invalidated by updating the tree, ErrINVAL is
// returned.
func (e *bTreeEnumerator) current() (key, value []byte, err error) {
	if e == nil {
		err = errors.New("bTreeEnumerator method invoked on nil receiver")
		return
	}

	if e.serial != e.t.serial {
		err = &ErrINVAL{Src: "bTreeEnumerator invalidated by updating the tree"}
		return
	}

	if e.p == nil || e.index == e.p.len() {
		return nil, nil, io.EOF
	}

	if key, err = e.p.key(e.t.store, e.index); err != nil {
		return
	}

	value, err = e.p.value(e.t.store, e.index)
	return
}

// Next attempts to position the enumerator onto the next KV pair wrt the
// current position. If there is no "next" KV pair, io.EOF is returned.
//
// If the enumerator has been invalidated by updating the tree, ErrINVAL is
// returned.
func (e *bTreeEnumerator) next() (err error) {
	if e == nil {
		err = errors.New("bTreeEnumerator method invoked on nil receiver")
		return
	}

	if e.serial != e.t.serial {
		err = &ErrINVAL{Src: "bTreeEnumerator invalidated by updating the tree"}
		return
	}

	if e.p == nil {
		return io.EOF
	}

	switch {
	case e.index < e.p.len()-1:
		e.index++
	default:
		ph := e.p.next()
		if ph == 0 {
			err = io.EOF
			break
		}

		if e.p, err = e.t.store.Get(e.p, ph); err != nil {
			e.p = nil
			return
		}
		e.index = 0
	}
	return
}

// Prev attempts to position the enumerator onto the previous KV pair wrt the
// current position. If there is no "previous" KV pair, io.EOF is returned.
//
// If the enumerator has been invalidated by updating the tree, ErrINVAL is
// returned.
func (e *bTreeEnumerator) prev() (err error) {
	if e == nil {
		err = errors.New("bTreeEnumerator method invoked on nil receiver")
		return
	}

	if e.serial != e.t.serial {
		err = &ErrINVAL{Src: "bTreeEnumerator invalidated by updating the tree"}
		return
	}

	if e.p == nil {
		return io.EOF
	}

	switch {
	case e.index > 0:
		e.index--
	default:
		ph := e.p.prev()
		if ph == 0 {
			err = io.EOF
			break
		}

		if e.p, err = e.t.store.Get(e.p, ph); err != nil {
			e.p = nil
			return
		}
		e.index = e.p.len() - 1
	}
	return
}

// BTreeEnumerator captures the state of enumerating a tree. It is returned
// from the Seek* methods.  The enumerator is aware of any mutations made to
// the tree in the process of enumerating it and automatically resumes the
// enumeration.
type BTreeEnumerator struct {
	enum     *bTreeEnumerator
	err      error
	key      []byte
	firstHit bool
}

// Next returns the currently enumerated KV pair, if it exists and moves to the
// next KV in the key collation order. If there is no KV pair to return, err ==
// io.EOF is returned.
func (e *BTreeEnumerator) Next() (key, value []byte, err error) {
	if err = e.err; err != nil {
		return
	}

	canRetry := true
retry:
	if key, value, err = e.enum.current(); err != nil {
		if _, ok := err.(*ErrINVAL); !ok || !canRetry {
			e.err = err
			return
		}

		canRetry = false
		var hit bool
		if e.enum, hit, err = e.enum.t.seek(e.key); err != nil {
			e.err = err
			return
		}

		if !e.firstHit && hit {
			err = e.enum.next()
			if err != nil {
				e.err = err
				return
			}
		}

		goto retry
	}

	e.firstHit = false
	e.key = append([]byte(nil), key...)
	e.err = e.enum.next()
	return
}

// Prev returns the currently enumerated KV pair, if it exists and moves to the
// previous KV in the key collation order. If there is no KV pair to return,
// err == io.EOF is returned.
func (e *BTreeEnumerator) Prev() (key, value []byte, err error) {
	if err = e.err; err != nil {
		return
	}

	canRetry := true
retry:
	if key, value, err = e.enum.current(); err != nil {
		if _, ok := err.(*ErrINVAL); !ok || !canRetry {
			e.err = err
			return
		}

		canRetry = false
		var hit bool
		if e.enum, hit, err = e.enum.t.seek(e.key); err != nil {
			e.err = err
			return
		}

		if !e.firstHit && hit {
			err = e.enum.prev()
			if err != nil {
				e.err = err
				return
			}
		}

		goto retry
	}

	e.firstHit = false
	e.key = append([]byte(nil), key...)
	e.err = e.enum.prev()
	return
}

// CreateBTree creates a new BTree in store. It returns the tree, its (freshly
// assigned) handle (for OpenBTree or RemoveBTree) or an error, if any.
func CreateBTree(store *Allocator, collate func(a, b []byte) int) (bt *BTree, handle int64, err error) {
	r := &BTree{store: store, collate: collate}
	if r.root, err = newBTree(store); err != nil {
		return
	}

	return r, int64(r.root), nil
}

// OpenBTree opens a store's BTree using handle. It returns the tree or an
// error, if any. The same tree may be opened more than once, but operations on
// the separate instances should not ever overlap or void the other instances.
// However, the intended API usage is to open the same tree handle only once
// (handled by some upper layer "dispatcher").
func OpenBTree(store *Allocator, collate func(a, b []byte) int, handle int64) (bt *BTree, err error) {
	r := &BTree{store: store, root: btree(handle), collate: collate}
	b := bufs.GCache.Get(7)
	defer bufs.GCache.Put(b)
	if b, err = store.Get(b, handle); err != nil {
		return
	}

	if len(b) != 7 {
		return nil, &ErrILSEQ{Off: h2off(handle), More: "btree.go:671"}
	}

	return r, nil
}

// RemoveBTree removes tree, represented by handle from store. Empty trees are
// cheap, each uses only few bytes of the store. If there's a chance that a
// tree will eventually get reused (non empty again), it's recommended to
// not/never remove it.  One advantage of such approach is a stable handle of
// such tree.
func RemoveBTree(store *Allocator, handle int64) (err error) {
	tree, err := OpenBTree(store, nil, handle)
	if err != nil {
		return
	}

	if err = tree.Clear(); err != nil {
		return
	}

	return store.Free(handle)
}

type btreeStore interface {
	Alloc(b []byte) (handle int64, err error)
	Free(handle int64) (err error)
	Get(dst []byte, handle int64) (b []byte, err error)
	Realloc(handle int64, b []byte) (err error)
}

// Read only zero bytes
var zeros [2 * kKV]byte

func init() {
	if kData < 1 || kData > 512 {
		panic(fmt.Errorf("kData %d: out of limits", kData))
	}

	if kIndex < 2 || kIndex > 2048 {
		panic(fmt.Errorf("kIndex %d: out of limits", kIndex))
	}

	if kKV < 8 || kKV > 23 {
		panic(fmt.Errorf("kKV %d: out of limits", kKV))
	}

	if n := len(zeros); n < 15 {
		panic(fmt.Errorf("not enough zeros: %d", n))
	}
}

type memBTreeStore struct {
	h int64
	m map[int64][]byte
}

func newMemBTreeStore() *memBTreeStore {
	return &memBTreeStore{h: 0, m: map[int64][]byte{}}
}

func (s *memBTreeStore) String() string {
	var a sortutil.Int64Slice
	for k := range s.m {
		a = append(a, k)
	}
	sort.Sort(a)
	var sa []string
	for _, k := range a {
		sa = append(sa, fmt.Sprintf("%#x:|% x|", k, s.m[k]))
	}
	return strings.Join(sa, "\n")
}

func (s *memBTreeStore) Alloc(b []byte) (handle int64, err error) {
	s.h++
	handle = s.h
	s.m[handle] = bpack(b)
	return
}

func (s *memBTreeStore) Free(handle int64) (err error) {
	if _, ok := s.m[handle]; !ok {
		return &ErrILSEQ{Type: ErrOther, Off: h2off(handle), More: "btree.go:754"}
	}

	delete(s.m, handle)
	return
}

func (s *memBTreeStore) Get(dst []byte, handle int64) (b []byte, err error) {
	r, ok := s.m[handle]
	if !ok {
		return nil, &ErrILSEQ{Type: ErrOther, Off: h2off(handle), More: "btree.go:764"}
	}

	b = need(len(r), dst)
	copy(b, r)
	return
}

func (s *memBTreeStore) Realloc(handle int64, b []byte) (err error) {
	if _, ok := s.m[handle]; !ok {
		return &ErrILSEQ{Type: ErrOther, Off: h2off(handle), More: "btree.go:774"}
	}

	s.m[handle] = bpack(b)
	return
}

/*

0...0 (1 bytes):
Flag

	  0
	+---+
	| 0 |
	+---+

0 indicates an index page

1...count*14-1
"array" of items, 14 bytes each. Count of items in kIndex-1..2*kIndex+2

	Count = (len(raw) - 8) / 14

	  0..6     7..13
	+-------+----------+
	| Child | DataPage |
	+-------+----------+

	Child    == handle of a child index page
	DataPage == handle of a data page

Offsets into the raw []byte:
Child[X]    == 1+14*X
DataPage[X] == 8+14*X

*/
type btreeIndexPage []byte

func newBTreeIndexPage(leftmostChild int64) (p btreeIndexPage) {
	p = bufs.GCache.Get(1 + (kIndex+1)*2*7)[:8]
	p[0] = tagBTreeIndexPage
	h2b(p[1:], leftmostChild)
	return
}

func (p btreeIndexPage) len() int {
	return (len(p) - 8) / 14
}

func (p btreeIndexPage) child(index int) int64 {
	return b2h(p[1+14*index:])
}

func (p btreeIndexPage) setChild(index int, dp int64) {
	h2b(p[1+14*index:], dp)
}

func (p btreeIndexPage) dataPage(index int) int64 {
	return b2h(p[8+14*index:])
}

func (p btreeIndexPage) setDataPage(index int, dp int64) {
	h2b(p[8+14*index:], dp)
}

func (q btreeIndexPage) insert(index int) btreeIndexPage {
	switch len0 := q.len(); {
	case index < len0:
		has := len(q)
		need := has + 14
		switch {
		case cap(q) >= need:
			q = q[:need]
		default:
			q = append(q, zeros[:14]...)
		}
		copy(q[8+14*(index+1):8+14*(index+1)+2*(len0-index)*7], q[8+14*index:])
	case index == len0:
		has := len(q)
		need := has + 14
		switch {
		case cap(q) >= need:
			q = q[:need]
		default:
			q = append(q, zeros[:14]...)
		}
	}
	return q
}

func (p btreeIndexPage) insert3(index int, dataPage, child int64) btreeIndexPage {
	p = p.insert(index)
	p.setDataPage(index, dataPage)
	p.setChild(index+1, child)
	return p
}

func (p btreeIndexPage) cmp(a btreeStore, c func(a, b []byte) int, keyA []byte, keyBIndex int) (int, error) {
	b := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(b)
	dp, err := a.Get(b, p.dataPage(keyBIndex))
	if err != nil {
		return 0, err
	}

	return btreeDataPage(dp).cmp(a, c, keyA, 0)
}

func (q btreeIndexPage) setLen(n int) btreeIndexPage {
	q = q[:cap(q)]
	need := 8 + 14*n
	if need < len(q) {
		return q[:need]
	}
	return append(q, make([]byte, need-len(q))...)
}

func (p btreeIndexPage) split(a btreeStore, root btree, ph *int64, parent int64, parentIndex int, index *int) (btreeIndexPage, error) {
	right := newBTreeIndexPage(0)
	canRecycle := true
	defer func() {
		if canRecycle {
			bufs.GCache.Put(right)
		}
	}()
	right = right.setLen(kIndex)
	copy(right[1:1+(2*kIndex+1)*7], p[1+14*(kIndex+1):])
	p = p.setLen(kIndex)
	if err := a.Realloc(*ph, p); err != nil {
		return nil, err
	}

	rh, err := a.Alloc(right)
	if err != nil {
		return nil, err
	}

	if parentIndex >= 0 {
		var pp btreeIndexPage = bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(pp)
		if pp, err = a.Get(pp, parent); err != nil {
			return nil, err
		}
		pp = pp.insert3(parentIndex, p.dataPage(kIndex), rh)
		if err = a.Realloc(parent, pp); err != nil {
			return nil, err
		}

	} else {
		nr := newBTreeIndexPage(*ph)
		defer bufs.GCache.Put(nr)
		nr = nr.insert3(0, p.dataPage(kIndex), rh)
		nrh, err := a.Alloc(nr)
		if err != nil {
			return nil, err
		}

		if err = a.Realloc(int64(root), h2b(make([]byte, 7), nrh)); err != nil {
			return nil, err
		}
	}
	if *index > kIndex {
		p = right
		canRecycle = false
		*ph = rh
		*index -= kIndex + 1
	}
	return p, nil
}

// p is dirty on return
func (p btreeIndexPage) extract(index int) btreeIndexPage {
	n := p.len() - 1
	if index < n {
		sz := (n-index)*14 + 7
		copy(p[1+14*index:1+14*index+sz], p[1+14*(index+1):])
	}
	return p.setLen(n)
}

// must persist all changes made
func (p btreeIndexPage) underflow(a btreeStore, root, iroot, parent int64, ph *int64, parentIndex int, index *int) (btreeIndexPage, error) {
	lh, rh, err := checkSiblings(a, parent, parentIndex)
	if err != nil {
		return nil, err
	}

	var left btreeIndexPage = bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(left)

	if lh != 0 {
		if left, err = a.Get(left, lh); err != nil {
			return nil, err
		}

		if lc := btreeIndexPage(left).len(); lc > kIndex {
			var pp = bufs.GCache.Get(maxBuf)
			defer bufs.GCache.Put(pp)
			if pp, err = a.Get(pp, parent); err != nil {
				return nil, err
			}

			pc := p.len()
			p = p.setLen(pc + 1)
			di, si, sz := 1+1*14, 1+0*14, (2*pc+1)*7
			copy(p[di:di+sz], p[si:])
			p.setChild(0, btreeIndexPage(left).child(lc))
			p.setDataPage(0, btreeIndexPage(pp).dataPage(parentIndex-1))
			*index++
			btreeIndexPage(pp).setDataPage(parentIndex-1, btreeIndexPage(left).dataPage(lc-1))
			left = left.setLen(lc - 1)
			if err = a.Realloc(parent, pp); err != nil {
				return nil, err
			}

			if err = a.Realloc(*ph, p); err != nil {
				return nil, err
			}

			return p, a.Realloc(lh, left)
		}
	}

	if rh != 0 {
		right := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(right)
		if right, err = a.Get(right, rh); err != nil {
			return nil, err
		}

		if rc := btreeIndexPage(right).len(); rc > kIndex {
			pp := bufs.GCache.Get(maxBuf)
			defer bufs.GCache.Put(pp)
			if pp, err = a.Get(pp, parent); err != nil {
				return nil, err
			}

			pc := p.len()
			p = p.setLen(pc + 1)
			p.setDataPage(pc, btreeIndexPage(pp).dataPage(parentIndex))
			pc++
			p.setChild(pc, btreeIndexPage(right).child(0))
			btreeIndexPage(pp).setDataPage(parentIndex, btreeIndexPage(right).dataPage(0))
			di, si, sz := 1+0*14, 1+1*14, (2*rc+1)*7
			copy(right[di:di+sz], right[si:])
			right = btreeIndexPage(right).setLen(rc - 1)
			if err = a.Realloc(parent, pp); err != nil {
				return nil, err
			}

			if err = a.Realloc(*ph, p); err != nil {
				return nil, err
			}

			return p, a.Realloc(rh, right)
		}
	}

	if lh != 0 {
		*index += left.len() + 1
		if left, err = left.concat(a, root, iroot, parent, lh, *ph, parentIndex-1); err != nil {
			return p, err
		}

		p, *ph = left, lh
		return p, nil
	}

	return p.concat(a, root, iroot, parent, *ph, rh, parentIndex)
}

// must persist all changes made
func (p btreeIndexPage) concat(a btreeStore, root, iroot, parent, ph, rh int64, parentIndex int) (btreeIndexPage, error) {
	pp := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(pp)
	pp, err := a.Get(pp, parent)
	if err != nil {
		return nil, err
	}

	right := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(right)
	if right, err = a.Get(right, rh); err != nil {
		return nil, err
	}

	pc := p.len()
	rc := btreeIndexPage(right).len()
	p = p.setLen(pc + rc + 1)
	p.setDataPage(pc, btreeIndexPage(pp).dataPage(parentIndex))
	di, si, sz := 1+14*(pc+1), 1+0*14, (2*rc+1)*7
	copy(p[di:di+sz], right[si:])
	if err := a.Realloc(ph, p); err != nil {
		return nil, err
	}

	if err := a.Free(rh); err != nil {
		return nil, err
	}

	if pc := btreeIndexPage(pp).len(); pc > 1 {
		if parentIndex < pc-1 {
			di, si, sz := 8+parentIndex*14, 8+(parentIndex+1)*14, 2*(pc-1-parentIndex)*7
			copy(pp[di:si+sz], pp[si:])
		}
		pp = btreeIndexPage(pp).setLen(pc - 1)
		return p, a.Realloc(parent, pp)
	}

	if err := a.Free(iroot); err != nil {
		return nil, err
	}

	b7 := bufs.GCache.Get(7)
	defer bufs.GCache.Put(b7)
	return p, a.Realloc(root, h2b(b7[:7], ph))
}

/*

0...0 (1 bytes):
Flag

	  0
	+---+
	| 1 |
	+---+

1 indicates a data page

1...14 (14 bytes)

	  1..7  8..14
	+------+------+
	| Prev | Next |
	+------+------+

	Prev, Next == Handles of the data pages doubly linked list

	Count = (len(raw) - 15) / (2*kKV)

15...count*2*kKV-1
"array" of items, 2*kKV bytes each. Count of items in kData-1..2*kData

Item
	  0..kKV-1   kKV..2*kKV-1
	+----------+--------------+
	|   Key    |    Value     |
	+----------+--------------+

Key/Value encoding

Length 0...kKV-1

	  0    1...N    N+1...kKV-1
	+---+---------+-------------+
	| N |  Data   |  Padding    |
	+---+---------+-------------+

	N       == content length
	Data    == Key or Value content
	Padding == MUST be zero bytes

Length >= kKV

	   0     1...kkV-8   kKV-7...kkV-1
	+------+-----------+--------------+
	| 0xFF |   Data    |      H       |
	+------+-----------+--------------+

	Data == Key or Value content, first kKV-7 bytes
	H    == Handle to THE REST of the content, w/o the first bytes in Data.

Offsets into the raw []byte:
Key[X]   == 15+2*kKV*X
Value[X] == 15+kKV+2*kKV*X
*/
type btreeDataPage []byte

func newBTreeDataPage() (p btreeDataPage) {
	p = bufs.GCache.Cget(1 + 2*7 + (kData+1)*2*kKV)[:1+2*7]
	p[0] = tagBTreeDataPage
	return
}

func newBTreeDataPageAlloc(a btreeStore) (p btreeDataPage, h int64, err error) {
	p = newBTreeDataPage()
	h, err = a.Alloc(p)
	return
}

func (p btreeDataPage) len() int {
	return (len(p) - 15) / (2 * kKV)
}

func (q btreeDataPage) setLen(n int) btreeDataPage {
	q = q[:cap(q)]
	need := 15 + 2*kKV*n
	if need < len(q) {
		return q[:need]
	}
	return append(q, make([]byte, need-len(q))...)
}

func (p btreeDataPage) prev() int64 {
	return b2h(p[1:])
}

func (p btreeDataPage) next() int64 {
	return b2h(p[8:])
}

func (p btreeDataPage) setPrev(h int64) {
	h2b(p[1:], h)
}

func (p btreeDataPage) setNext(h int64) {
	h2b(p[8:], h)
}

func (q btreeDataPage) insert(index int) btreeDataPage {
	switch len0 := q.len(); {
	case index < len0:
		has := len(q)
		need := has + 2*kKV
		switch {
		case cap(q) >= need:
			q = q[:need]
		default:
			q = append(q, zeros[:2*kKV]...)
		}
		q.copy(q, index+1, index, len0-index)
		return q
	case index == len0:
		has := len(q)
		need := has + 2*kKV
		switch {
		case cap(q) >= need:
			return q[:need]
		default:
			return append(q, zeros[:2*kKV]...)
		}
	}
	panic("internal error")
}

func (p btreeDataPage) contentField(off int) (b []byte, h int64) {
	p = p[off:]
	switch n := int(p[0]); {
	case n >= kKV: // content has a handle
		b = append([]byte(nil), p[1:1+kSz]...)
		h = b2h(p[kH:])
	default: // content is embedded
		b, h = append([]byte(nil), p[1:1+n]...), 0
	}
	return
}

func (p btreeDataPage) content(a btreeStore, off int) (b []byte, err error) {
	b, h := p.contentField(off)
	if h == 0 {
		return
	}

	// content has a handle
	b2, err := a.Get(nil, h) //TODO buffers: Later, not a public API
	if err != nil {
		return nil, err
	}

	return append(b, b2...), nil
}

func (p btreeDataPage) setContent(a btreeStore, off int, b []byte) (err error) {
	p = p[off:]
	switch {
	case p[0] >= kKV: // existing content has a handle
		switch n := len(b); {
		case n < kKV:
			p[0] = byte(n)
			if err = a.Free(b2h(p[kH:])); err != nil {
				return
			}
			copy(p[1:], b)
		default:
			// reuse handle
			copy(p[1:1+kSz], b)
			return a.Realloc(b2h(p[kH:]), b[kSz:])
		}
	default: // existing content is embedded
		switch n := len(b); {
		case n < kKV:
			p[0] = byte(n)
			copy(p[1:], b)
		default:
			p[0] = 0xff
			copy(p[1:1+kSz], b)
			h, err := a.Alloc(b[kSz:])
			if err != nil {
				return err
			}

			h2b(p[kH:], h)
		}
	}
	return
}

func (p btreeDataPage) keyField(index int) (b []byte, h int64) {
	return p.contentField(15 + 2*kKV*index)
}

func (p btreeDataPage) key(a btreeStore, index int) (b []byte, err error) {
	return p.content(a, 15+2*kKV*index)
}

func (p btreeDataPage) valueField(index int) (b []byte, h int64) {
	return p.contentField(15 + kKV + 2*kKV*index)
}

func (p btreeDataPage) value(a btreeStore, index int) (b []byte, err error) {
	return p.content(a, 15+kKV+2*kKV*index)
}

func (p btreeDataPage) valueCopy(a btreeStore, index int) (b []byte, err error) {
	if b, err = p.content(a, 15+kKV+2*kKV*index); err != nil {
		return
	}

	return append([]byte(nil), b...), nil
}

func (p btreeDataPage) setKey(a btreeStore, index int, key []byte) (err error) {
	return p.setContent(a, 15+2*kKV*index, key)
}

func (p btreeDataPage) setValue(a btreeStore, index int, value []byte) (err error) {
	return p.setContent(a, 15+kKV+2*kKV*index, value)
}

func (p btreeDataPage) cmp(a btreeStore, c func(a, b []byte) int, keyA []byte, keyBIndex int) (y int, err error) {
	var keyB []byte
	if keyB, err = p.content(a, 15+2*kKV*keyBIndex); err != nil {
		return
	}

	return c(keyA, keyB), nil
}

func (p btreeDataPage) copy(src btreeDataPage, di, si, n int) {
	do, so := 15+2*kKV*di, 15+2*kKV*si
	copy(p[do:do+2*kKV*n], src[so:])
}

// {p,left} dirty on exit
func (p btreeDataPage) moveLeft(left btreeDataPage, n int) (btreeDataPage, btreeDataPage) {
	nl, np := left.len(), p.len()
	left = left.setLen(nl + n)
	left.copy(p, nl, 0, n)
	p.copy(p, 0, n, np-n)
	return p.setLen(np - n), left
}

func (p btreeDataPage) moveRight(right btreeDataPage, n int) (btreeDataPage, btreeDataPage) {
	nr, np := right.len(), p.len()
	right = right.setLen(nr + n)
	right.copy(right, n, 0, nr)
	right.copy(p, 0, np-n, n)
	return p.setLen(np - n), right
}

func (p btreeDataPage) insertItem(a btreeStore, index int, key, value []byte) (btreeDataPage, error) {
	p = p.insert(index)
	di, sz := 15+2*kKV*index, 2*kKV
	copy(p[di:di+sz], zeros[:sz])
	if err := p.setKey(a, index, key); err != nil {
		return nil, err
	}
	return p, p.setValue(a, index, value)
}

func (p btreeDataPage) split(a btreeStore, root, ph, parent int64, parentIndex, index int, key, value []byte) (btreeDataPage, error) {
	right, rh, err := newBTreeDataPageAlloc(a)
	// fails defer bufs.GCache.Put(right)
	if err != nil {
		return nil, err
	}

	if next := p.next(); next != 0 {
		right.setNext(p.next())
		nxh := right.next()
		nx := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(nx)
		if nx, err = a.Get(nx, nxh); err != nil {
			return nil, err
		}

		btreeDataPage(nx).setPrev(rh)
		if err = a.Realloc(nxh, nx); err != nil {
			return nil, err
		}
	}

	p.setNext(rh)
	right.setPrev(ph)
	right = right.setLen(kData)
	right.copy(p, 0, kData, kData)
	p = p.setLen(kData)

	if parentIndex >= 0 {
		var pp btreeIndexPage = bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(pp)
		if pp, err = a.Get(pp, parent); err != nil {
			return nil, err
		}

		pp = pp.insert3(parentIndex, rh, rh)
		if err = a.Realloc(parent, pp); err != nil {
			return nil, err
		}

	} else {
		nr := newBTreeIndexPage(ph)
		defer bufs.GCache.Put(nr)
		nr = nr.insert3(0, rh, rh)
		nrh, err := a.Alloc(nr)
		if err != nil {
			return nil, err
		}

		if err = a.Realloc(root, h2b(make([]byte, 7), nrh)); err != nil {
			return nil, err
		}

	}
	if index > kData {
		if right, err = right.insertItem(a, index-kData, key, value); err != nil {
			return nil, err
		}
	} else {
		if p, err = p.insertItem(a, index, key, value); err != nil {
			return nil, err
		}
	}
	if err = a.Realloc(ph, p); err != nil {
		return nil, err
	}

	return p, a.Realloc(rh, right)
}

func (p btreeDataPage) overflow(a btreeStore, root, ph, parent int64, parentIndex, index int, key, value []byte) (btreeDataPage, error) {
	leftH, rightH, err := checkSiblings(a, parent, parentIndex)
	if err != nil {
		return nil, err
	}

	if leftH != 0 {
		left := btreeDataPage(bufs.GCache.Get(maxBuf))
		defer bufs.GCache.Put(left)
		if left, err = a.Get(left, leftH); err != nil {
			return nil, err
		}

		if left.len() < 2*kData {

			p, left = p.moveLeft(left, 1)
			if err = a.Realloc(leftH, left); err != nil {
				return nil, err
			}

			if p, err = p.insertItem(a, index-1, key, value); err != nil {
				return nil, err
			}

			return p, a.Realloc(ph, p)
		}
	}

	if rightH != 0 {
		right := btreeDataPage(bufs.GCache.Get(maxBuf))
		defer bufs.GCache.Put(right)
		if right, err = a.Get(right, rightH); err != nil {
			return nil, err
		}

		if right.len() < 2*kData {
			if index < 2*kData {
				p, right = p.moveRight(right, 1)
				if err = a.Realloc(rightH, right); err != nil {
					return nil, err
				}

				if p, err = p.insertItem(a, index, key, value); err != nil {
					return nil, err
				}

				return p, a.Realloc(ph, p)
			} else {
				if right, err = right.insertItem(a, 0, key, value); err != nil {
					return nil, err
				}

				return p, a.Realloc(rightH, right)
			}
		}
	}
	return p.split(a, root, ph, parent, parentIndex, index, key, value)
}

func (p btreeDataPage) swap(a btreeStore, di int, value []byte, canOverwrite bool) (oldValue []byte, err error) {
	if oldValue, err = p.value(a, di); err != nil {
		return
	}

	if !canOverwrite {
		return
	}

	oldValue = append([]byte(nil), oldValue...)
	err = p.setValue(a, di, value)
	return
}

type btreePage []byte

func (p btreePage) isIndex() bool {
	return p[0] == tagBTreeIndexPage
}

func (p btreePage) len() int {
	if p.isIndex() {
		return btreeIndexPage(p).len()
	}

	return btreeDataPage(p).len()
}

func (p btreePage) find(a btreeStore, c func(a, b []byte) int, key []byte) (index int, ok bool, err error) {
	l := 0
	h := p.len() - 1
	isIndex := p.isIndex()
	if c == nil {
		c = bytes.Compare
	}
	for l <= h {
		index = (l + h) >> 1
		var cmp int
		if isIndex {
			if cmp, err = btreeIndexPage(p).cmp(a, c, key, index); err != nil {
				return
			}
		} else {
			if cmp, err = btreeDataPage(p).cmp(a, c, key, index); err != nil {
				return
			}
		}
		switch ok = cmp == 0; {
		case cmp > 0:
			l = index + 1
		case ok:
			return
		default:
			h = index - 1
		}
	}
	return l, false, nil
}

// p is dirty after extract!
func (p btreeDataPage) extract(a btreeStore, index int) (btreeDataPage, []byte, error) {
	value, err := p.valueCopy(a, index)
	if err != nil {
		return nil, nil, err
	}

	if _, h := p.keyField(index); h != 0 {
		if err = a.Free(h); err != nil {
			return nil, nil, err
		}
	}

	if _, h := p.valueField(index); h != 0 {
		if err = a.Free(h); err != nil {
			return nil, nil, err
		}
	}

	n := p.len() - 1
	if index < n {
		p.copy(p, index, index+1, n-index)
	}
	return p.setLen(n), value, nil
}

func checkSiblings(a btreeStore, parent int64, parentIndex int) (left, right int64, err error) {
	if parentIndex >= 0 {
		var p btreeIndexPage = bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(p)
		if p, err = a.Get(p, parent); err != nil {
			return
		}

		if parentIndex > 0 {
			left = p.child(parentIndex - 1)
		}
		if parentIndex < p.len() {
			right = p.child(parentIndex + 1)
		}
	}
	return
}

// underflow must persist all changes made.
func (p btreeDataPage) underflow(a btreeStore, root, iroot, parent, ph int64, parentIndex int) (err error) {
	lh, rh, err := checkSiblings(a, parent, parentIndex)
	if err != nil {
		return err
	}

	if lh != 0 {
		left := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(left)
		if left, err = a.Get(left, lh); err != nil {
			return err
		}

		if btreeDataPage(left).len()+p.len() >= 2*kData {
			left, p = btreeDataPage(left).moveRight(p, 1)
			if err = a.Realloc(lh, left); err != nil {
				return err
			}

			return a.Realloc(ph, p)
		}
	}

	if rh != 0 {
		right := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(right)
		if right, err = a.Get(right, rh); err != nil {
			return err
		}

		if p.len()+btreeDataPage(right).len() > 2*kData {
			right, p = btreeDataPage(right).moveLeft(p, 1)
			if err = a.Realloc(rh, right); err != nil {
				return err
			}

			return a.Realloc(ph, p)
		}
	}

	if lh != 0 {
		left := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(left)
		if left, err = a.Get(left, lh); err != nil {
			return err
		}

		if err = a.Realloc(ph, p); err != nil {
			return err
		}

		return btreeDataPage(left).concat(a, root, iroot, parent, lh, ph, parentIndex-1)
	}

	return p.concat(a, root, iroot, parent, ph, rh, parentIndex)
}

// concat must persist all changes made.
func (p btreeDataPage) concat(a btreeStore, root, iroot, parent, ph, rh int64, parentIndex int) (err error) {
	right := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(right)
	if right, err = a.Get(right, rh); err != nil {
		return err
	}

	right, p = btreeDataPage(right).moveLeft(p, btreeDataPage(right).len())
	nxh := btreeDataPage(right).next()
	if nxh != 0 {
		nx := bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(nx)
		if nx, err = a.Get(nx, nxh); err != nil {
			return err
		}

		btreeDataPage(nx).setPrev(ph)
		if err = a.Realloc(nxh, nx); err != nil {
			return err
		}
	}
	p.setNext(nxh)
	if err = a.Free(rh); err != nil {
		return err
	}

	pp := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(pp)
	if pp, err = a.Get(pp, parent); err != nil {
		return err
	}

	if btreeIndexPage(pp).len() > 1 {
		pp = btreeIndexPage(pp).extract(parentIndex)
		btreeIndexPage(pp).setChild(parentIndex, ph)
		if err = a.Realloc(parent, pp); err != nil {
			return err
		}

		return a.Realloc(ph, p)
	}

	if err = a.Free(iroot); err != nil {
		return err
	}

	if err = a.Realloc(ph, p); err != nil {
		return err
	}

	var b7 [7]byte
	return a.Realloc(root, h2b(b7[:], ph))
}

// external "root" is stable and contains the real root.
type btree int64

func newBTree(a btreeStore) (btree, error) {
	r, err := a.Alloc(zeros[:7])
	return btree(r), err
}

func (root btree) String(a btreeStore) string {
	r := bufs.GCache.Get(16)
	defer bufs.GCache.Put(r)
	r, err := a.Get(r, int64(root))
	if err != nil {
		panic(err)
	}

	iroot := b2h(r)
	m := map[int64]bool{int64(root): true}

	s := []string{fmt.Sprintf("tree %#x -> %#x\n====", root, iroot)}
	if iroot == 0 {
		return s[0]
	}

	var f func(int64, string)
	f = func(h int64, ind string) {
		if m[h] {
			return
		}

		m[h] = true
		var b btreePage = bufs.GCache.Get(maxBuf)
		defer bufs.GCache.Put(b)
		var err error
		if b, err = a.Get(b, h); err != nil {
			panic(err)
		}

		s = append(s, fmt.Sprintf("%s@%#x", ind, h))
		switch b.isIndex() {
		case true:
			da := []int64{}
			b := btreeIndexPage(b)
			for i := 0; i < b.len(); i++ {
				c, d := b.child(i), b.dataPage(i)
				s = append(s, fmt.Sprintf("%schild[%d] %#x dataPage[%d] %#x", ind, i, c, i, d))
				da = append(da, c)
				da = append(da, d)
			}
			i := b.len()
			c := b.child(i)
			s = append(s, fmt.Sprintf("%schild[%d] %#x", ind, i, c))
			for _, c := range da {
				f(c, ind+"  ")
			}
			f(c, ind+"  ")
		case false:
			b := btreeDataPage(b)
			s = append(s, fmt.Sprintf("%sprev %#x next %#x", ind, b.prev(), b.next()))
			for i := 0; i < b.len(); i++ {
				k, err := b.key(a, i)
				if err != nil {
					panic(err)
				}

				v, err := b.value(a, i)
				if err != nil {
					panic(err)
				}

				s = append(s, fmt.Sprintf("%sK[%d]|% x| V[%d]|% x|", ind, i, k, i, v))
			}
		}
	}

	f(int64(iroot), "")
	return strings.Join(s, "\n")
}

func (root btree) put(dst []byte, a btreeStore, c func(a, b []byte) int, key, value []byte, canOverwrite bool) (prev []byte, err error) {
	prev, _, err = root.put2(dst, a, c, key, func(key, old []byte) (new []byte, write bool, err error) {
		new, write = value, true
		return
	})
	return
}

func (root btree) put2(dst []byte, a btreeStore, c func(a, b []byte) int, key []byte, upd func(key, old []byte) (new []byte, write bool, err error)) (old []byte, written bool, err error) {
	var r, value []byte
	if r, err = a.Get(dst, int64(root)); err != nil {
		return
	}

	iroot := b2h(r)
	var h int64
	if iroot == 0 {
		p := newBTreeDataPage()
		defer bufs.GCache.Put(p)
		if value, written, err = upd(key, nil); err != nil || !written {
			return
		}

		if p, err = p.insertItem(a, 0, key, value); err != nil {
			return
		}

		h, err = a.Alloc(p)
		if err != nil {
			return nil, true, err
		}

		err = a.Realloc(int64(root), h2b(r, h)[:7])
		return
	}

	parentIndex := -1
	var parent int64
	ph := iroot

	p := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(p)

	for {
		if p, err = a.Get(p[:cap(p)], ph); err != nil {
			return
		}

		var index int
		var ok bool

		if index, ok, err = btreePage(p).find(a, c, key); err != nil {
			return
		}

		switch {
		case ok: // Key found
			if btreePage(p).isIndex() {
				ph = btreeIndexPage(p).dataPage(index)
				if p, err = a.Get(p, ph); err != nil {
					return
				}

				if old, err = btreeDataPage(p).valueCopy(a, 0); err != nil {
					return
				}

				if value, written, err = upd(key, old); err != nil || !written {
					return
				}

				if _, err = btreeDataPage(p).swap(a, 0, value, true); err != nil {
					return
				}

				err = a.Realloc(ph, p)
				return
			}

			if old, err = btreeDataPage(p).valueCopy(a, index); err != nil {
				return
			}

			if value, written, err = upd(key, old); err != nil || !written {
				return
			}

			if _, err = btreeDataPage(p).swap(a, index, value, true); err != nil {
				return
			}

			err = a.Realloc(ph, p)
			return
		case btreePage(p).isIndex():
			if btreePage(p).len() > 2*kIndex {
				if p, err = btreeIndexPage(p).split(a, root, &ph, parent, parentIndex, &index); err != nil {
					return
				}
			}
			parentIndex = index
			parent = ph
			ph = btreeIndexPage(p).child(index)
		default:
			if value, written, err = upd(key, nil); err != nil || !written {
				return
			}

			if btreePage(p).len() < 2*kData { // page is not full
				if p, err = btreeDataPage(p).insertItem(a, index, key, value); err != nil {
					return
				}

				err = a.Realloc(ph, p)
				return
			}

			// page is full
			p, err = btreeDataPage(p).overflow(a, int64(root), ph, parent, parentIndex, index, key, value)
			return
		}
	}
}

//TODO actually use 'dst' to return 'value'
func (root btree) get(a btreeStore, dst []byte, c func(a, b []byte) int, key []byte) (b []byte, err error) {
	var r []byte
	if r, err = a.Get(dst, int64(root)); err != nil {
		return
	}

	iroot := b2h(r)
	if iroot == 0 {
		return
	}

	ph := iroot

	for {
		var p btreePage
		if p, err = a.Get(p, ph); err != nil {
			return
		}

		var index int
		var ok bool
		if index, ok, err = p.find(a, c, key); err != nil {
			return
		}

		switch {
		case ok:
			if p.isIndex() {
				dh := btreeIndexPage(p).dataPage(index)
				dp, err := a.Get(dst, dh)
				if err != nil {
					return nil, err
				}

				return btreeDataPage(dp).value(a, 0)
			}

			return btreeDataPage(p).value(a, index)
		case p.isIndex():
			ph = btreeIndexPage(p).child(index)
		default:
			return
		}
	}
}

//TODO actually use 'dst' to return 'value'
func (root btree) extract(a btreeStore, dst []byte, c func(a, b []byte) int, key []byte) (value []byte, err error) {
	var r []byte
	if r, err = a.Get(dst, int64(root)); err != nil {
		return
	}

	iroot := b2h(r)
	if iroot == 0 {
		return
	}

	ph := iroot
	parentIndex := -1
	var parent int64

	p := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(p)

	for {
		if p, err = a.Get(p[:cap(p)], ph); err != nil {
			return
		}

		var index int
		var ok bool
		if index, ok, err = btreePage(p).find(a, c, key); err != nil {
			return
		}

		if ok {
			if btreePage(p).isIndex() {
				dph := btreeIndexPage(p).dataPage(index)
				dp, err := a.Get(dst, dph)
				if err != nil {
					return nil, err
				}

				if btreeDataPage(dp).len() > kData {
					if dp, value, err = btreeDataPage(dp).extract(a, 0); err != nil {
						return nil, err
					}

					return value, a.Realloc(dph, dp)
				}

				if btreeIndexPage(p).len() < kIndex && ph != iroot {
					var err error
					if p, err = btreeIndexPage(p).underflow(a, int64(root), iroot, parent, &ph, parentIndex, &index); err != nil {
						return nil, err
					}
				}
				parentIndex = index + 1
				parent = ph
				ph = btreeIndexPage(p).child(parentIndex)
				continue
			}

			p, value, err = btreeDataPage(p).extract(a, index)
			if btreePage(p).len() >= kData {
				err = a.Realloc(ph, p)
				return
			}

			if ph != iroot {
				err = btreeDataPage(p).underflow(a, int64(root), iroot, parent, ph, parentIndex)
				return
			}

			if btreePage(p).len() == 0 {
				if err = a.Free(ph); err != nil {
					return
				}

				err = a.Realloc(int64(root), zeros[:7])
				return
			}
			err = a.Realloc(ph, p)
			return
		}

		if !btreePage(p).isIndex() {
			return
		}

		if btreePage(p).len() < kIndex && ph != iroot {
			if p, err = btreeIndexPage(p).underflow(a, int64(root), iroot, parent, &ph, parentIndex, &index); err != nil {
				return nil, err
			}
		}
		parentIndex = index
		parent = ph
		ph = btreeIndexPage(p).child(index)
	}
}

func (root btree) deleteAny(a btreeStore) (bool, error) {
	r := bufs.GCache.Get(7)
	defer bufs.GCache.Put(r)
	var err error
	if r, err = a.Get(r, int64(root)); err != nil {
		return false, err
	}

	iroot := b2h(r)
	if iroot == 0 {
		return true, nil
	}

	ph := iroot
	parentIndex := -1
	var parent int64
	p := bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(p)

	for {
		if p, err = a.Get(p, ph); err != nil {
			return false, err
		}

		index := btreePage(p).len() / 2
		if btreePage(p).isIndex() {
			dph := btreeIndexPage(p).dataPage(index)
			dp := bufs.GCache.Get(maxBuf)
			defer bufs.GCache.Put(dp)
			if dp, err = a.Get(dp, dph); err != nil {
				return false, err
			}

			if btreeDataPage(dp).len() > kData {
				if dp, _, err = btreeDataPage(dp).extract(a, 0); err != nil {
					return false, err
				}

				return false, a.Realloc(dph, dp)
			}

			if btreeIndexPage(p).len() < kIndex && ph != iroot {
				if p, err = btreeIndexPage(p).underflow(a, int64(root), iroot, parent, &ph, parentIndex, &index); err != nil {
					return false, err
				}
			}
			parentIndex = index + 1
			parent = ph
			ph = btreeIndexPage(p).child(parentIndex)
			continue
		}

		p, _, err = btreeDataPage(p).extract(a, index)
		if btreePage(p).len() >= kData {
			err = a.Realloc(ph, p)
			return false, err
		}

		if ph != iroot {
			err = btreeDataPage(p).underflow(a, int64(root), iroot, parent, ph, parentIndex)
			return false, err
		}

		if btreePage(p).len() == 0 {
			if err = a.Free(ph); err != nil {
				return true, err
			}

			return true, a.Realloc(int64(root), zeros[:7])
		}

		return false, a.Realloc(ph, p)
	}
}

func (root btree) first(a btreeStore) (ph int64, p btreeDataPage, err error) {
	r := bufs.GCache.Get(7)
	defer bufs.GCache.Put(r)
	if r, err = a.Get(r, int64(root)); err != nil {
		return
	}

	for ph = b2h(r); ph != 0; ph = btreeIndexPage(p).child(0) {
		if p, err = a.Get(p, ph); err != nil {
			return
		}

		if !btreePage(p).isIndex() {
			break
		}
	}

	return
}

func (root btree) last(a btreeStore) (ph int64, p btreeDataPage, err error) {
	r := bufs.GCache.Get(7)
	defer bufs.GCache.Put(r)
	if r, err = a.Get(r, int64(root)); err != nil {
		return
	}

	for ph = b2h(r); ph != 0; ph = btreeIndexPage(p).child(btreeIndexPage(p).len()) {
		if p, err = a.Get(p, ph); err != nil {
			return
		}

		if !btreePage(p).isIndex() {
			break
		}
	}

	return
}

// key >= p[index].key
func (root btree) seek(a btreeStore, c func(a, b []byte) int, key []byte) (p btreeDataPage, index int, equal bool, err error) {
	r := bufs.GCache.Get(7)
	defer bufs.GCache.Put(r)
	if r, err = a.Get(r, int64(root)); err != nil {
		return
	}

	for ph := b2h(r); ph != 0; ph = btreeIndexPage(p).child(index) {
		if p, err = a.Get(p, ph); err != nil {
			break
		}

		if index, equal, err = btreePage(p).find(a, c, key); err != nil {
			break
		}

		if equal {
			if !btreePage(p).isIndex() {
				break
			}

			p, err = a.Get(p, btreeIndexPage(p).dataPage(index))
			index = 0
			break
		}

		if !btreePage(p).isIndex() {
			break
		}
	}
	return
}

func (root btree) clear(a btreeStore) (err error) {
	r := bufs.GCache.Get(7)
	defer bufs.GCache.Put(r)
	if r, err = a.Get(r, int64(root)); err != nil {
		return
	}

	iroot := b2h(r)
	if iroot == 0 {
		return
	}

	if err = root.clear2(a, iroot); err != nil {
		return
	}

	var b [7]byte
	return a.Realloc(int64(root), b[:])
}

func (root btree) clear2(a btreeStore, ph int64) (err error) {
	var p = bufs.GCache.Get(maxBuf)
	defer bufs.GCache.Put(p)
	if p, err = a.Get(p, ph); err != nil {
		return
	}

	switch btreePage(p).isIndex() {
	case true:
		ip := btreeIndexPage(p)
		for i := 0; i <= ip.len(); i++ {
			root.clear2(a, ip.child(i))

		}
	case false:
		dp := btreeDataPage(p)
		for i := 0; i < dp.len(); i++ {
			if err = dp.setKey(a, i, nil); err != nil {
				return
			}

			if err = dp.setValue(a, i, nil); err != nil {
				return
			}
		}
	}
	return a.Free(ph)
}
