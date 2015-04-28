// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plain memory storage back end.

package ql

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"math/big"
	"time"
)

var (
	_ btreeIndex    = (*memIndex)(nil)
	_ btreeIterator = (*memBTreeIterator)(nil)
	_ indexIterator = (*xenumerator2)(nil)
	_ storage       = (*mem)(nil)
	_ temp          = (*memTemp)(nil)
)

type memIndex struct {
	m      *mem
	t      *xtree
	unique bool
}

func newMemIndex(m *mem, unique bool) *memIndex {
	return &memIndex{t: xtreeNew(), unique: unique, m: m}
}

func (x *memIndex) Clear() error {
	x.m.newUndo(undoClearX, 0, []interface{}{x, x.t})
	x.t = xtreeNew()
	return nil
}

func (x *memIndex) Create(indexedValue interface{}, h int64) error {
	t := x.t
	switch {
	case !x.unique:
		k := indexKey{indexedValue, h}
		x.m.newUndo(undoCreateX, 0, []interface{}{x, k})
		t.Set(k, 0)
	case indexedValue == nil: // unique, NULL
		k := indexKey{nil, h}
		x.m.newUndo(undoCreateX, 0, []interface{}{x, k})
		t.Set(k, 0)
	default: // unique, non NULL
		k := indexKey{indexedValue, 0}
		if _, ok := t.Get(k); ok { //LATER need .Put
			return fmt.Errorf("cannot insert into unique index: duplicate value: %v", indexedValue)
		}

		x.m.newUndo(undoCreateX, 0, []interface{}{x, k})
		t.Set(k, int(h))
	}
	return nil
}

func (x *memIndex) Delete(indexedValue interface{}, h int64) error {
	t := x.t
	var k indexKey
	var v interface{}
	var ok, okv bool
	switch {
	case !x.unique:
		k = indexKey{indexedValue, h}
		v, okv = t.Get(k)
		ok = t.delete(k)
	case indexedValue == nil: // unique, NULL
		k = indexKey{nil, h}
		v, okv = t.Get(k)
		ok = t.delete(k)
	default: // unique, non NULL
		k = indexKey{indexedValue, 0}
		v, okv = t.Get(k)
		ok = t.delete(k)
	}
	if ok {
		if okv {
			x.m.newUndo(undoDeleteX, int64(v.(int)), []interface{}{x, k})
		}
		return nil
	}

	return fmt.Errorf("internal error 047")
}

func (x *memIndex) Drop() error {
	x.m.newUndo(undoDropX, 0, []interface{}{x, *x})
	*x = memIndex{}
	return nil
}

func (x *memIndex) Seek(indexedValue interface{}) (indexIterator, bool, error) {
	it, hit := x.t.Seek(indexKey{indexedValue, 0})
	return &xenumerator2{*it, x.unique}, hit, nil
}

func (x *memIndex) SeekFirst() (iter indexIterator, err error) {
	it, err := x.t.SeekFirst()
	if err != nil {
		return nil, err
	}

	return &xenumerator2{*it, x.unique}, nil
}

func (x *memIndex) SeekLast() (iter indexIterator, err error) {
	it, err := x.t.SeekLast()
	if err != nil {
		return nil, err
	}

	return &xenumerator2{*it, x.unique}, nil
}

type xenumerator2 struct {
	it     xenumerator
	unique bool
}

func (it *xenumerator2) Next() (interface{}, int64, error) {
	k, h, err := it.it.Next()
	if err != nil {
		return nil, -1, err
	}

	switch it.unique {
	case true:
		if k.value == nil {
			return nil, k.h, nil
		}

		return k.value, h, nil
	default:
		return k.value, k.h, nil
	}
}

func (it *xenumerator2) Prev() (interface{}, int64, error) {
	k, h, err := it.it.Prev()
	if err != nil {
		return nil, -1, err
	}

	switch it.unique {
	case true:
		if k.value == nil {
			return nil, k.h, nil
		}

		return k.value, h, nil
	default:
		return k.value, k.h, nil
	}
}

type memBTreeIterator enumerator

func (it *memBTreeIterator) Next() (k, v []interface{}, err error) {
	return (*enumerator)(it).Next()
}

type memTemp struct {
	tree  *tree
	store *mem
}

func (t *memTemp) BeginTransaction() (err error) {
	return nil
}

func (t *memTemp) Get(k []interface{}) (v []interface{}, err error) {
	v, _ = t.tree.Get(k)
	return
}

func (t *memTemp) Create(data ...interface{}) (h int64, err error) {
	s := t.store
	switch n := len(s.recycler); {
	case n != 0:
		h = int64(s.recycler[n-1])
		s.recycler = s.recycler[:n-1]
		s.data[h] = s.clone(data...)
	default:
		h = int64(len(s.data))
		s.data = append(s.data, s.clone(data...))
	}
	return
}

func (t *memTemp) Read(dst []interface{}, h int64, cols ...*col) (data []interface{}, err error) {
	return t.store.Read(dst, h, cols...)
}

func (*memTemp) Drop() (err error) { return }

func (t *memTemp) Set(k, v []interface{}) (err error) {
	t.tree.Set(append([]interface{}(nil), k...), t.store.clone(v...))
	return
}

func (t *memTemp) SeekFirst() (e btreeIterator, err error) {
	en, err := t.tree.SeekFirst()
	if err != nil {
		return
	}

	return (*memBTreeIterator)(en), nil
}

const (
	undoCreateNewHandle = iota
	undoCreateRecycledHandle
	undoUpdate
	undoDelete
	undoClearX  // {0: *memIndex, 1: *xtree}
	undoCreateX // {0: *memIndex, 1: indexKey}
	undoDeleteX // {0: *memIndex, 1: indexKey}
	undoDropX   // {0: *memIndex, 1: memIndex}
)

type undo struct {
	tag  int
	h    int64
	data []interface{}
}

type undos struct {
	list   []undo
	parent *undos
}

type mem struct {
	data     [][]interface{}
	id       int64
	recycler []int
	tnl      int
	rollback *undos
}

func newMemStorage() (s *mem, err error) {
	s = &mem{data: [][]interface{}{nil}}
	if err = s.BeginTransaction(); err != nil {
		return nil, err
	}

	h, err := s.Create()
	if h != 1 {
		log.Panic("internal error 048")
	}

	if err = s.Commit(); err != nil {
		return nil, err
	}

	return
}

func (s *mem) OpenIndex(unique bool, handle int64) (btreeIndex, error) { // Never called on the memory backend.
	panic("internal error 049")
}

func (s *mem) newUndo(tag int, h int64, data []interface{}) {
	s.rollback.list = append(s.rollback.list, undo{tag, h, data})
}

func (s *mem) Acid() bool { return false }

func (s *mem) Close() (err error) {
	*s = mem{}
	return
}

func (s *mem) CreateIndex(unique bool) ( /* handle */ int64, btreeIndex, error) {
	return -1, newMemIndex(s, unique), nil // handle of memIndex should never be used
}

func (s *mem) Name() string { return fmt.Sprintf("/proc/self/mem/%p", s) } // fake, non existing name

// OpenMem returns a new, empty DB backed by the process' memory. The back end
// has no limits on field/record/table/DB size other than memory available to
// the process.
func OpenMem() (db *DB, err error) {
	s, err := newMemStorage()
	if err != nil {
		return
	}

	if db, err = newDB(s); err != nil {
		return nil, err
	}

	db.isMem = true
	return db, nil
}

func (s *mem) Verify() (allocs int64, err error) {
	for _, v := range s.recycler {
		if s.data[v] != nil {
			return 0, fmt.Errorf("corrupted: non nil free handle %d", s.data[v])
		}
	}

	for _, v := range s.data {
		if v != nil {
			allocs++
		}
	}

	if allocs != int64(len(s.data))-1-int64(len(s.recycler)) {
		return 0, fmt.Errorf("corrupted: len(data) %d, len(recycler) %d, allocs %d", len(s.data), len(s.recycler), allocs)
	}

	return
}

func (s *mem) String() string {
	var b bytes.Buffer
	for i, v := range s.data {
		b.WriteString(fmt.Sprintf("s.data[%d] %#v\n", i, v))
	}
	for i, v := range s.recycler {
		b.WriteString(fmt.Sprintf("s.recycler[%d] %v\n", i, v))
	}
	return b.String()
}

func (s *mem) CreateTemp(asc bool) (_ temp, err error) {
	st, err := newMemStorage()
	if err != nil {
		return
	}

	return &memTemp{
		tree:  treeNew(collators[asc]),
		store: st,
	}, nil
}

func (s *mem) ResetID() (err error) {
	s.id = 0
	return
}

func (s *mem) ID() (id int64, err error) {
	s.id++
	return s.id, nil
}

func (s *mem) clone(data ...interface{}) []interface{} {
	r := make([]interface{}, len(data))
	for i, v := range data {
		switch x := v.(type) {
		case nil:
			// nop
		case idealComplex:
			r[i] = complex128(x)
		case idealFloat:
			r[i] = float64(x)
		case idealInt:
			r[i] = int64(x)
		case idealRune:
			r[i] = int32(x)
		case idealUint:
			r[i] = uint64(x)
		case bool:
			r[i] = x
		case complex64:
			r[i] = x
		case complex128:
			r[i] = x
		case float32:
			r[i] = x
		case float64:
			r[i] = x
		case int:
			r[i] = int64(x)
		case int8:
			r[i] = x
		case int16:
			r[i] = x
		case int32:
			r[i] = x
		case int64:
			r[i] = x
		case string:
			r[i] = x
		case uint:
			r[i] = uint64(x)
		case uint8:
			r[i] = x
		case uint16:
			r[i] = x
		case uint32:
			r[i] = x
		case uint64:
			r[i] = x
		case []byte:
			r[i] = append([]byte(nil), x...)
		case *big.Int:
			r[i] = big.NewInt(0).Set(x)
		case *big.Rat:
			r[i] = big.NewRat(1, 2).Set(x)
		case time.Time:
			t := x
			r[i] = t
		case time.Duration:
			r[i] = x
		case map[string]interface{}: // map of ids of a cross join
			r[i] = x
		default:
			log.Panic("internal error 050")
		}
	}
	return r
}

func (s *mem) Create(data ...interface{}) (h int64, err error) {
	switch n := len(s.recycler); {
	case n != 0:
		h = int64(s.recycler[n-1])
		s.recycler = s.recycler[:n-1]
		s.data[h] = s.clone(data...)
		r := s.rollback
		r.list = append(r.list, undo{
			tag: undoCreateRecycledHandle,
			h:   h,
		})
	default:
		h = int64(len(s.data))
		s.data = append(s.data, s.clone(data...))
		r := s.rollback
		r.list = append(r.list, undo{
			tag: undoCreateNewHandle,
			h:   h,
		})
	}
	return
}

func (s *mem) Read(dst []interface{}, h int64, cols ...*col) (data []interface{}, err error) {
	if i := int(h); i != 0 && i < len(s.data) {
		d := s.clone(s.data[h]...)
		if cols == nil {
			return d, nil
		}

		for n, dn := len(cols)+2, len(d); dn < n; dn++ {
			d = append(d, nil)
		}
		return d, nil
	}

	return nil, errNoDataForHandle
}

func (s *mem) UpdateRow(h int64, _ []*col, data ...interface{}) (err error) {
	return s.Update(h, data...)
}

func (s *mem) Update(h int64, data ...interface{}) (err error) {
	r := s.rollback
	r.list = append(r.list, undo{
		tag:  undoUpdate,
		h:    h,
		data: s.data[h],
	})
	s.data[h] = s.clone(data...)
	return
}

func (s *mem) Delete(h int64, _ ...*col) (err error) {
	r := s.rollback
	r.list = append(r.list, undo{
		tag:  undoDelete,
		h:    h,
		data: s.data[h],
	})
	s.recycler = append(s.recycler, int(h))
	s.data[h] = nil
	return
}

func (s *mem) BeginTransaction() (err error) {
	s.rollback = &undos{parent: s.rollback}
	s.tnl++
	return nil
}

func (s *mem) Rollback() (err error) {
	if s.tnl == 0 {
		return errRollbackNotInTransaction
	}

	list := s.rollback.list
	for i := len(list) - 1; i >= 0; i-- {
		undo := list[i]
		switch h, data := int(undo.h), undo.data; undo.tag {
		case undoCreateNewHandle:
			d := s.data
			s.data = d[:len(d)-1]
		case undoCreateRecycledHandle:
			s.data[h] = nil
			r := s.recycler
			s.recycler = append(r, h)
		case undoUpdate:
			s.data[h] = data
		case undoDelete:
			s.data[h] = data
			s.recycler = s.recycler[:len(s.recycler)-1]
		case undoClearX:
			x, t := data[0].(*memIndex), data[1].(*xtree)
			x.t = t
		case undoCreateX:
			x, k := data[0].(*memIndex), data[1].(indexKey)
			x.t.delete(k)
		case undoDeleteX:
			x, k := data[0].(*memIndex), data[1].(indexKey)
			x.t.Set(k, h)
		case undoDropX:
			x, v := data[0].(*memIndex), data[1].(memIndex)
			*x = v
		default:
			log.Panic("internal error 051")
		}
	}

	s.tnl--
	s.rollback = s.rollback.parent
	return nil
}

func (s *mem) Commit() (err error) {
	if s.tnl == 0 {
		return errCommitNotInTransaction
	}

	s.tnl--
	s.rollback = s.rollback.parent
	return nil
}

// Transaction index B+Tree
//LATER make it just a wrapper of the implementation in btree.go.

type (
	xd struct { // data page
		c  int
		xd [2*kd + 1]xde
		n  *xd
		p  *xd
	}

	xde struct { // xd element
		k indexKey
		v int
	}

	// xenumerator captures the state of enumerating a tree. It is returned
	// from the Seek* methods. The enumerator is aware of any mutations
	// made to the tree in the process of enumerating it and automatically
	// resumes the enumeration at the proper key, if possible.
	//
	// However, once an xenumerator returns io.EOF to signal "no more
	// items", it does no more attempt to "resync" on tree mutation(s).  In
	// other words, io.EOF from an Enumaretor is "sticky" (idempotent).
	xenumerator struct {
		err error
		hit bool
		i   int
		k   indexKey
		q   *xd
		t   *xtree
		ver int64
	}

	// xtree is a B+tree.
	xtree struct {
		c     int
		first *xd
		last  *xd
		r     interface{}
		ver   int64
	}

	xxe struct { // xx element
		ch  interface{}
		sep *xd
	}

	xx struct { // index page
		c  int
		xx [2*kx + 2]xxe
	}
)

func (a *indexKey) cmp(b *indexKey) int {
	r := collate1(a.value, b.value)
	if r != 0 {
		return r
	}

	return int(a.h) - int(b.h)
}

var ( // R/O zero values
	zxd  xd
	zxde xde
	zxx  xx
	zxxe xxe
)

func xclr(q interface{}) {
	switch xx := q.(type) {
	case *xx:
		for i := 0; i <= xx.c; i++ { // Ch0 Sep0 ... Chn-1 Sepn-1 Chn
			xclr(xx.xx[i].ch)
		}
		*xx = zxx // GC
	case *xd:
		*xx = zxd // GC
	}
}

// -------------------------------------------------------------------------- xx

func xnewX(ch0 interface{}) *xx {
	r := &xx{}
	r.xx[0].ch = ch0
	return r
}

func (q *xx) extract(i int) {
	q.c--
	if i < q.c {
		copy(q.xx[i:], q.xx[i+1:q.c+1])
		q.xx[q.c].ch = q.xx[q.c+1].ch
		q.xx[q.c].sep = nil // GC
		q.xx[q.c+1] = zxxe  // GC
	}
}

func (q *xx) insert(i int, xd *xd, ch interface{}) *xx {
	c := q.c
	if i < c {
		q.xx[c+1].ch = q.xx[c].ch
		copy(q.xx[i+2:], q.xx[i+1:c])
		q.xx[i+1].sep = q.xx[i].sep
	}
	c++
	q.c = c
	q.xx[i].sep = xd
	q.xx[i+1].ch = ch
	return q
}

func (q *xx) siblings(i int) (l, r *xd) {
	if i >= 0 {
		if i > 0 {
			l = q.xx[i-1].ch.(*xd)
		}
		if i < q.c {
			r = q.xx[i+1].ch.(*xd)
		}
	}
	return
}

// -------------------------------------------------------------------------- xd

func (l *xd) mvL(r *xd, c int) {
	copy(l.xd[l.c:], r.xd[:c])
	copy(r.xd[:], r.xd[c:r.c])
	l.c += c
	r.c -= c
}

func (l *xd) mvR(r *xd, c int) {
	copy(r.xd[c:], r.xd[:r.c])
	copy(r.xd[:c], l.xd[l.c-c:])
	r.c += c
	l.c -= c
}

// ----------------------------------------------------------------------- xtree

// xtreeNew returns a newly created, empty xtree. The compare function is used
// for key collation.
func xtreeNew() *xtree {
	return &xtree{}
}

// Clear removes all K/V pairs from the tree.
func (t *xtree) Clear() {
	if t.r == nil {
		return
	}

	xclr(t.r)
	t.c, t.first, t.last, t.r = 0, nil, nil, nil
	t.ver++
}

func (t *xtree) cat(p *xx, q, r *xd, pi int) {
	t.ver++
	q.mvL(r, r.c)
	if r.n != nil {
		r.n.p = q
	} else {
		t.last = q
	}
	q.n = r.n
	if p.c > 1 {
		p.extract(pi)
		p.xx[pi].ch = q
	} else {
		t.r = q
	}
}

func (t *xtree) catX(p, q, r *xx, pi int) {
	t.ver++
	q.xx[q.c].sep = p.xx[pi].sep
	copy(q.xx[q.c+1:], r.xx[:r.c])
	q.c += r.c + 1
	q.xx[q.c].ch = r.xx[r.c].ch
	if p.c > 1 {
		p.c--
		pc := p.c
		if pi < pc {
			p.xx[pi].sep = p.xx[pi+1].sep
			copy(p.xx[pi+1:], p.xx[pi+2:pc+1])
			p.xx[pc].ch = p.xx[pc+1].ch
			p.xx[pc].sep = nil  // GC
			p.xx[pc+1].ch = nil // GC
		}
		return
	}

	t.r = q
}

//Delete removes the k's KV pair, if it exists, in which case Delete returns
//true.
func (t *xtree) delete(k indexKey) (ok bool) {
	pi := -1
	var p *xx
	q := t.r
	if q == nil {
		return
	}

	for {
		var i int
		i, ok = t.find(q, k)
		if ok {
			switch xx := q.(type) {
			case *xx:
				dp := xx.xx[i].sep
				switch {
				case dp.c > kd:
					t.extract(dp, 0)
				default:
					if xx.c < kx && q != t.r {
						t.underflowX(p, &xx, pi, &i)
					}
					pi = i + 1
					p = xx
					q = xx.xx[pi].ch
					ok = false
					continue
				}
			case *xd:
				t.extract(xx, i)
				if xx.c >= kd {
					return
				}

				if q != t.r {
					t.underflow(p, xx, pi)
				} else if t.c == 0 {
					t.Clear()
				}
			}
			return
		}

		switch xx := q.(type) {
		case *xx:
			if xx.c < kx && q != t.r {
				t.underflowX(p, &xx, pi, &i)
			}
			pi = i
			p = xx
			q = xx.xx[i].ch
		case *xd:
			return
		}
	}
}

func (t *xtree) extract(q *xd, i int) { // (r int64) {
	t.ver++
	//r = q.xd[i].v // prepared for Extract
	q.c--
	if i < q.c {
		copy(q.xd[i:], q.xd[i+1:q.c+1])
	}
	q.xd[q.c] = zxde // GC
	t.c--
	return
}

func (t *xtree) find(q interface{}, k indexKey) (i int, ok bool) {
	var mk indexKey
	l := 0
	switch xx := q.(type) {
	case *xx:
		h := xx.c - 1
		for l <= h {
			m := (l + h) >> 1
			mk = xx.xx[m].sep.xd[0].k
			switch cmp := k.cmp(&mk); {
			case cmp > 0:
				l = m + 1
			case cmp == 0:
				return m, true
			default:
				h = m - 1
			}
		}
	case *xd:
		h := xx.c - 1
		for l <= h {
			m := (l + h) >> 1
			mk = xx.xd[m].k
			switch cmp := k.cmp(&mk); {
			case cmp > 0:
				l = m + 1
			case cmp == 0:
				return m, true
			default:
				h = m - 1
			}
		}
	}
	return l, false
}

// First returns the first item of the tree in the key collating order, or
// (nil, nil) if the tree is empty.
func (t *xtree) First() (k indexKey, v int) {
	if q := t.first; q != nil {
		q := &q.xd[0]
		k, v = q.k, q.v
	}
	return
}

// Get returns the value associated with k and true if it exists. Otherwise Get
// returns (nil, false).
func (t *xtree) Get(k indexKey) (v int, ok bool) {
	q := t.r
	if q == nil {
		return
	}

	for {
		var i int
		if i, ok = t.find(q, k); ok {
			switch xx := q.(type) {
			case *xx:
				return xx.xx[i].sep.xd[0].v, true
			case *xd:
				return xx.xd[i].v, true
			}
		}
		switch xx := q.(type) {
		case *xx:
			q = xx.xx[i].ch
		default:
			return
		}
	}
}

func (t *xtree) insert(q *xd, i int, k indexKey, v int) *xd {
	t.ver++
	c := q.c
	if i < c {
		copy(q.xd[i+1:], q.xd[i:c])
	}
	c++
	q.c = c
	q.xd[i].k, q.xd[i].v = k, v
	t.c++
	return q
}

// Last returns the last item of the tree in the key collating order, or (nil,
// nil) if the tree is empty.
func (t *xtree) Last() (k indexKey, v int) {
	if q := t.last; q != nil {
		q := &q.xd[q.c-1]
		k, v = q.k, q.v
	}
	return
}

// Len returns the number of items in the tree.
func (t *xtree) Len() int {
	return t.c
}

func (t *xtree) overflow(p *xx, q *xd, pi, i int, k indexKey, v int) {
	t.ver++
	l, r := p.siblings(pi)

	if l != nil && l.c < 2*kd {
		l.mvL(q, 1)
		t.insert(q, i-1, k, v)
		return
	}

	if r != nil && r.c < 2*kd {
		if i < 2*kd {
			q.mvR(r, 1)
			t.insert(q, i, k, v)
		} else {
			t.insert(r, 0, k, v)
		}
		return
	}

	t.split(p, q, pi, i, k, v)
}

// Seek returns an xenumerator positioned on a an item such that k >= item's
// key. ok reports if k == item.key The xenumerator's position is possibly
// after the last item in the tree.
func (t *xtree) Seek(k indexKey) (e *xenumerator, ok bool) {
	q := t.r
	if q == nil {
		e = &xenumerator{nil, false, 0, k, nil, t, t.ver}
		return
	}

	for {
		var i int
		if i, ok = t.find(q, k); ok {
			switch xx := q.(type) {
			case *xx:
				e = &xenumerator{nil, ok, 0, k, xx.xx[i].sep, t, t.ver}
				return
			case *xd:
				e = &xenumerator{nil, ok, i, k, xx, t, t.ver}
				return
			}
		}
		switch xx := q.(type) {
		case *xx:
			q = xx.xx[i].ch
		case *xd:
			e = &xenumerator{nil, ok, i, k, xx, t, t.ver}
			return
		}
	}
}

// SeekFirst returns an enumerator positioned on the first KV pair in the tree,
// if any. For an empty tree, err == io.EOF is returned and e will be nil.
func (t *xtree) SeekFirst() (e *xenumerator, err error) {
	q := t.first
	if q == nil {
		return nil, io.EOF
	}

	return &xenumerator{nil, true, 0, q.xd[0].k, q, t, t.ver}, nil
}

// SeekLast returns an enumerator positioned on the last KV pair in the tree,
// if any. For an empty tree, err == io.EOF is returned and e will be nil.
func (t *xtree) SeekLast() (e *xenumerator, err error) {
	q := t.last
	if q == nil {
		return nil, io.EOF
	}

	return &xenumerator{nil, true, q.c - 1, q.xd[q.c-1].k, q, t, t.ver}, nil
}

// Set sets the value associated with k.
func (t *xtree) Set(k indexKey, v int) {
	pi := -1
	var p *xx
	q := t.r
	if q != nil {
		for {
			i, ok := t.find(q, k)
			if ok {
				switch xx := q.(type) {
				case *xx:
					xx.xx[i].sep.xd[0].v = v
				case *xd:
					xx.xd[i].v = v
				}
				return
			}

			switch xx := q.(type) {
			case *xx:
				if xx.c > 2*kx {
					t.splitX(p, &xx, pi, &i)
				}
				pi = i
				p = xx
				q = xx.xx[i].ch
			case *xd:
				switch {
				case xx.c < 2*kd:
					t.insert(xx, i, k, v)
				default:
					t.overflow(p, xx, pi, i, k, v)
				}
				return
			}
		}
	}

	z := t.insert(&xd{}, 0, k, v)
	t.r, t.first, t.last = z, z, z
	return
}

func (t *xtree) split(p *xx, q *xd, pi, i int, k indexKey, v int) {
	t.ver++
	r := &xd{}
	if q.n != nil {
		r.n = q.n
		r.n.p = r
	} else {
		t.last = r
	}
	q.n = r
	r.p = q

	copy(r.xd[:], q.xd[kd:2*kd])
	for i := range q.xd[kd:] {
		q.xd[kd+i] = zxde
	}
	q.c = kd
	r.c = kd
	if pi >= 0 {
		p.insert(pi, r, r)
	} else {
		t.r = xnewX(q).insert(0, r, r)
	}
	if i > kd {
		t.insert(r, i-kd, k, v)
		return
	}

	t.insert(q, i, k, v)
}

func (t *xtree) splitX(p *xx, pp **xx, pi int, i *int) {
	t.ver++
	q := *pp
	r := &xx{}
	copy(r.xx[:], q.xx[kx+1:])
	q.c = kx
	r.c = kx
	if pi >= 0 {
		p.insert(pi, q.xx[kx].sep, r)
	} else {
		t.r = xnewX(q).insert(0, q.xx[kx].sep, r)
	}
	q.xx[kx].sep = nil
	for i := range q.xx[kx+1:] {
		q.xx[kx+i+1] = zxxe
	}
	if *i > kx {
		*pp = r
		*i -= kx + 1
	}
}

func (t *xtree) underflow(p *xx, q *xd, pi int) {
	t.ver++
	l, r := p.siblings(pi)

	if l != nil && l.c+q.c >= 2*kd {
		l.mvR(q, 1)
	} else if r != nil && q.c+r.c >= 2*kd {
		q.mvL(r, 1)
		r.xd[r.c] = zxde // GC
	} else if l != nil {
		t.cat(p, l, q, pi-1)
	} else {
		t.cat(p, q, r, pi)
	}
}

func (t *xtree) underflowX(p *xx, pp **xx, pi int, i *int) {
	t.ver++
	var l, r *xx
	q := *pp

	if pi >= 0 {
		if pi > 0 {
			l = p.xx[pi-1].ch.(*xx)
		}
		if pi < p.c {
			r = p.xx[pi+1].ch.(*xx)
		}
	}

	if l != nil && l.c > kx {
		q.xx[q.c+1].ch = q.xx[q.c].ch
		copy(q.xx[1:], q.xx[:q.c])
		q.xx[0].ch = l.xx[l.c].ch
		q.xx[0].sep = p.xx[pi-1].sep
		q.c++
		*i++
		l.c--
		p.xx[pi-1].sep = l.xx[l.c].sep
		return
	}

	if r != nil && r.c > kx {
		q.xx[q.c].sep = p.xx[pi].sep
		q.c++
		q.xx[q.c].ch = r.xx[0].ch
		p.xx[pi].sep = r.xx[0].sep
		copy(r.xx[:], r.xx[1:r.c])
		r.c--
		rc := r.c
		r.xx[rc].ch = r.xx[rc+1].ch
		r.xx[rc].sep = nil
		r.xx[rc+1].ch = nil
		return
	}

	if l != nil {
		*i += l.c + 1
		t.catX(p, l, q, pi-1)
		*pp = l
		return
	}

	t.catX(p, q, r, pi)
}

// ----------------------------------------------------------------- xenumerator

// Next returns the currently enumerated item, if it exists and moves to the
// next item in the key collation order. If there is no item to return, err ==
// io.EOF is returned.
func (e *xenumerator) Next() (k indexKey, v int64, err error) {
	if err = e.err; err != nil {
		return
	}

	if e.ver != e.t.ver {
		f, hit := e.t.Seek(e.k)
		if !e.hit && hit {
			if err = f.next(); err != nil {
				return
			}
		}

		*e = *f
	}
	if e.q == nil {
		e.err, err = io.EOF, io.EOF
		return
	}

	if e.i >= e.q.c {
		if err = e.next(); err != nil {
			return
		}
	}

	i := e.q.xd[e.i]
	k, v = i.k, int64(i.v)
	e.k, e.hit = k, false
	e.next()
	return
}

func (e *xenumerator) next() error {
	if e.q == nil {
		e.err = io.EOF
		return io.EOF
	}

	switch {
	case e.i < e.q.c-1:
		e.i++
	default:
		if e.q, e.i = e.q.n, 0; e.q == nil {
			e.err = io.EOF
		}
	}
	return e.err
}

// Prev returns the currently enumerated item, if it exists and moves to the
// previous item in the key collation order. If there is no item to return, err
// == io.EOF is returned.
func (e *xenumerator) Prev() (k indexKey, v int64, err error) {
	if err = e.err; err != nil {
		return
	}

	if e.ver != e.t.ver {
		f, hit := e.t.Seek(e.k)
		if !e.hit && hit {
			if err = f.prev(); err != nil {
				return
			}
		}

		*e = *f
	}
	if e.q == nil {
		e.err, err = io.EOF, io.EOF
		return
	}

	if e.i >= e.q.c {
		if err = e.next(); err != nil {
			return
		}
	}

	i := e.q.xd[e.i]
	k, v = i.k, int64(i.v)
	e.k, e.hit = k, false
	e.prev()
	return
}

func (e *xenumerator) prev() error {
	if e.q == nil {
		e.err = io.EOF
		return io.EOF
	}

	switch {
	case e.i > 0:
		e.i--
	default:
		if e.q = e.q.p; e.q == nil {
			e.err = io.EOF
			break
		}

		e.i = e.q.c - 1
	}
	return e.err
}
