// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"io"
)

const (
	kx = 128 //DONE benchmark tune this number if using custom key/value type(s).
	kd = 64  //DONE benchmark tune this number if using custom key/value type(s).
)

type (
	// cmp compares a and b. Return value is:
	//
	//	< 0 if a <  b
	//	  0 if a == b
	//	> 0 if a >  b
	//
	cmp func(a, b []interface{}) int

	d struct { // data page
		c int
		d [2*kd + 1]de
		n *d
		p *d
	}

	de struct { // d element
		k []interface{}
		v []interface{}
	}

	enumerator struct {
		err error
		hit bool
		i   int
		k   []interface{}
		q   *d
		t   *tree
		ver int64
	}

	// tree is a B+tree.
	tree struct {
		c     int
		cmp   cmp
		first *d
		last  *d
		r     interface{}
		ver   int64
	}

	xe struct { // x element
		ch  interface{}
		sep *d
	}

	x struct { // index page
		c int
		x [2*kx + 2]xe
	}
)

var ( // R/O zero values
	zd  d
	zde de
	zx  x
	zxe xe
)

func clr(q interface{}) {
	switch x := q.(type) {
	case *x:
		for i := 0; i <= x.c; i++ { // Ch0 Sep0 ... Chn-1 Sepn-1 Chn
			clr(x.x[i].ch)
		}
		*x = zx // GC
	case *d:
		*x = zd // GC
	}
}

// -------------------------------------------------------------------------- x

func newX(ch0 interface{}) *x {
	r := &x{}
	r.x[0].ch = ch0
	return r
}

func (q *x) extract(i int) {
	q.c--
	if i < q.c {
		copy(q.x[i:], q.x[i+1:q.c+1])
		q.x[q.c].ch = q.x[q.c+1].ch
		q.x[q.c].sep = nil // GC
		q.x[q.c+1] = zxe   // GC
	}
}

func (q *x) insert(i int, d *d, ch interface{}) *x {
	c := q.c
	if i < c {
		q.x[c+1].ch = q.x[c].ch
		copy(q.x[i+2:], q.x[i+1:c])
		q.x[i+1].sep = q.x[i].sep
	}
	c++
	q.c = c
	q.x[i].sep = d
	q.x[i+1].ch = ch
	return q
}

func (q *x) siblings(i int) (l, r *d) {
	if i >= 0 {
		if i > 0 {
			l = q.x[i-1].ch.(*d)
		}
		if i < q.c {
			r = q.x[i+1].ch.(*d)
		}
	}
	return
}

// -------------------------------------------------------------------------- d

func (l *d) mvL(r *d, c int) {
	copy(l.d[l.c:], r.d[:c])
	copy(r.d[:], r.d[c:r.c])
	l.c += c
	r.c -= c
}

func (l *d) mvR(r *d, c int) {
	copy(r.d[c:], r.d[:r.c])
	copy(r.d[:c], l.d[l.c-c:])
	r.c += c
	l.c -= c
}

// ----------------------------------------------------------------------- tree

// treeNew returns a newly created, empty tree. The compare function is used
// for key collation.
func treeNew(cmp cmp) *tree {
	return &tree{cmp: cmp}
}

// Clear removes all K/V pairs from the tree.
func (t *tree) Clear() {
	if t.r == nil {
		return
	}

	clr(t.r)
	t.c, t.first, t.last, t.r = 0, nil, nil, nil
	t.ver++
}

func (t *tree) cat(p *x, q, r *d, pi int) {
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
		p.x[pi].ch = q
	} else {
		t.r = q
	}
}

func (t *tree) catX(p, q, r *x, pi int) {
	t.ver++
	q.x[q.c].sep = p.x[pi].sep
	copy(q.x[q.c+1:], r.x[:r.c])
	q.c += r.c + 1
	q.x[q.c].ch = r.x[r.c].ch
	if p.c > 1 {
		p.c--
		pc := p.c
		if pi < pc {
			p.x[pi].sep = p.x[pi+1].sep
			copy(p.x[pi+1:], p.x[pi+2:pc+1])
			p.x[pc].ch = p.x[pc+1].ch
			p.x[pc].sep = nil  // GC
			p.x[pc+1].ch = nil // GC
		}
		return
	}

	t.r = q
}

//Delete removes the k's KV pair, if it exists, in which case Delete returns
//true.
func (t *tree) Delete(k []interface{}) (ok bool) {
	pi := -1
	var p *x
	q := t.r
	if q == nil {
		return
	}

	for {
		var i int
		i, ok = t.find(q, k)
		if ok {
			switch x := q.(type) {
			case *x:
				dp := x.x[i].sep
				switch {
				case dp.c > kd:
					t.extract(dp, 0)
				default:
					if x.c < kx && q != t.r {
						t.underflowX(p, &x, pi, &i)
					}
					pi = i + 1
					p = x
					q = x.x[pi].ch
					ok = false
					continue
				}
			case *d:
				t.extract(x, i)
				if x.c >= kd {
					return
				}

				if q != t.r {
					t.underflow(p, x, pi)
				} else if t.c == 0 {
					t.Clear()
				}
			}
			return
		}

		switch x := q.(type) {
		case *x:
			if x.c < kx && q != t.r {
				t.underflowX(p, &x, pi, &i)
			}
			pi = i
			p = x
			q = x.x[i].ch
		case *d:
			return
		}
	}
}

func (t *tree) extract(q *d, i int) { // (r []interface{}) {
	t.ver++
	//r = q.d[i].v // prepared for Extract
	q.c--
	if i < q.c {
		copy(q.d[i:], q.d[i+1:q.c+1])
	}
	q.d[q.c] = zde // GC
	t.c--
	return
}

func (t *tree) find(q interface{}, k []interface{}) (i int, ok bool) {
	var mk []interface{}
	l := 0
	switch x := q.(type) {
	case *x:
		h := x.c - 1
		for l <= h {
			m := (l + h) >> 1
			mk = x.x[m].sep.d[0].k
			switch cmp := t.cmp(k, mk); {
			case cmp > 0:
				l = m + 1
			case cmp == 0:
				return m, true
			default:
				h = m - 1
			}
		}
	case *d:
		h := x.c - 1
		for l <= h {
			m := (l + h) >> 1
			mk = x.d[m].k
			switch cmp := t.cmp(k, mk); {
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
func (t *tree) First() (k []interface{}, v []interface{}) {
	if q := t.first; q != nil {
		q := &q.d[0]
		k, v = q.k, q.v
	}
	return
}

// Get returns the value associated with k and true if it exists. Otherwise Get
// returns (nil, false).
func (t *tree) Get(k []interface{}) (v []interface{}, ok bool) {
	q := t.r
	if q == nil {
		return
	}

	for {
		var i int
		if i, ok = t.find(q, k); ok {
			switch x := q.(type) {
			case *x:
				return x.x[i].sep.d[0].v, true
			case *d:
				return x.d[i].v, true
			}
		}
		switch x := q.(type) {
		case *x:
			q = x.x[i].ch
		default:
			return
		}
	}
}

func (t *tree) insert(q *d, i int, k []interface{}, v []interface{}) *d {
	t.ver++
	c := q.c
	if i < c {
		copy(q.d[i+1:], q.d[i:c])
	}
	c++
	q.c = c
	q.d[i].k, q.d[i].v = k, v
	t.c++
	return q
}

// Last returns the last item of the tree in the key collating order, or (nil,
// nil) if the tree is empty.
func (t *tree) Last() (k []interface{}, v []interface{}) {
	if q := t.last; q != nil {
		q := &q.d[q.c-1]
		k, v = q.k, q.v
	}
	return
}

// Len returns the number of items in the tree.
func (t *tree) Len() int {
	return t.c
}

func (t *tree) overflow(p *x, q *d, pi, i int, k []interface{}, v []interface{}) {
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

// Seek returns an enumerator positioned on a an item such that k >= item's
// key. ok reports if k == item.key The enumerator's position is possibly
// after the last item in the tree.
func (t *tree) Seek(k []interface{}) (e *enumerator, ok bool) {
	q := t.r
	if q == nil {
		e = &enumerator{nil, false, 0, k, nil, t, t.ver}
		return
	}

	for {
		var i int
		if i, ok = t.find(q, k); ok {
			switch x := q.(type) {
			case *x:
				e = &enumerator{nil, ok, 0, k, x.x[i].sep, t, t.ver}
				return
			case *d:
				e = &enumerator{nil, ok, i, k, x, t, t.ver}
				return
			}
		}
		switch x := q.(type) {
		case *x:
			q = x.x[i].ch
		case *d:
			e = &enumerator{nil, ok, i, k, x, t, t.ver}
			return
		}
	}
}

// SeekFirst returns an enumerator positioned on the first KV pair in the tree,
// if any. For an empty tree, err == io.EOF is returned and e will be nil.
func (t *tree) SeekFirst() (e *enumerator, err error) {
	q := t.first
	if q == nil {
		return nil, io.EOF
	}

	return &enumerator{nil, true, 0, q.d[0].k, q, t, t.ver}, nil
}

// SeekLast returns an enumerator positioned on the last KV pair in the tree,
// if any. For an empty tree, err == io.EOF is returned and e will be nil.
func (t *tree) SeekLast() (e *enumerator, err error) {
	q := t.last
	if q == nil {
		return nil, io.EOF
	}

	return &enumerator{nil, true, q.c - 1, q.d[q.c-1].k, q, t, t.ver}, nil
}

// Set sets the value associated with k.
func (t *tree) Set(k []interface{}, v []interface{}) {
	pi := -1
	var p *x
	q := t.r
	if q != nil {
		for {
			i, ok := t.find(q, k)
			if ok {
				switch x := q.(type) {
				case *x:
					x.x[i].sep.d[0].v = v
				case *d:
					x.d[i].v = v
				}
				return
			}

			switch x := q.(type) {
			case *x:
				if x.c > 2*kx {
					t.splitX(p, &x, pi, &i)
				}
				pi = i
				p = x
				q = x.x[i].ch
			case *d:
				switch {
				case x.c < 2*kd:
					t.insert(x, i, k, v)
				default:
					t.overflow(p, x, pi, i, k, v)
				}
				return
			}
		}
	}

	z := t.insert(&d{}, 0, k, v)
	t.r, t.first, t.last = z, z, z
	return
}

func (t *tree) split(p *x, q *d, pi, i int, k []interface{}, v []interface{}) {
	t.ver++
	r := &d{}
	if q.n != nil {
		r.n = q.n
		r.n.p = r
	} else {
		t.last = r
	}
	q.n = r
	r.p = q

	copy(r.d[:], q.d[kd:2*kd])
	for i := range q.d[kd:] {
		q.d[kd+i] = zde
	}
	q.c = kd
	r.c = kd
	if pi >= 0 {
		p.insert(pi, r, r)
	} else {
		t.r = newX(q).insert(0, r, r)
	}
	if i > kd {
		t.insert(r, i-kd, k, v)
		return
	}

	t.insert(q, i, k, v)
}

func (t *tree) splitX(p *x, pp **x, pi int, i *int) {
	t.ver++
	q := *pp
	r := &x{}
	copy(r.x[:], q.x[kx+1:])
	q.c = kx
	r.c = kx
	if pi >= 0 {
		p.insert(pi, q.x[kx].sep, r)
	} else {
		t.r = newX(q).insert(0, q.x[kx].sep, r)
	}
	q.x[kx].sep = nil
	for i := range q.x[kx+1:] {
		q.x[kx+i+1] = zxe
	}
	if *i > kx {
		*pp = r
		*i -= kx + 1
	}
}

func (t *tree) underflow(p *x, q *d, pi int) {
	t.ver++
	l, r := p.siblings(pi)

	if l != nil && l.c+q.c >= 2*kd {
		l.mvR(q, 1)
	} else if r != nil && q.c+r.c >= 2*kd {
		q.mvL(r, 1)
		r.d[r.c] = zde // GC
	} else if l != nil {
		t.cat(p, l, q, pi-1)
	} else {
		t.cat(p, q, r, pi)
	}
}

func (t *tree) underflowX(p *x, pp **x, pi int, i *int) {
	t.ver++
	var l, r *x
	q := *pp

	if pi >= 0 {
		if pi > 0 {
			l = p.x[pi-1].ch.(*x)
		}
		if pi < p.c {
			r = p.x[pi+1].ch.(*x)
		}
	}

	if l != nil && l.c > kx {
		q.x[q.c+1].ch = q.x[q.c].ch
		copy(q.x[1:], q.x[:q.c])
		q.x[0].ch = l.x[l.c].ch
		q.x[0].sep = p.x[pi-1].sep
		q.c++
		*i++
		l.c--
		p.x[pi-1].sep = l.x[l.c].sep
		return
	}

	if r != nil && r.c > kx {
		q.x[q.c].sep = p.x[pi].sep
		q.c++
		q.x[q.c].ch = r.x[0].ch
		p.x[pi].sep = r.x[0].sep
		copy(r.x[:], r.x[1:r.c])
		r.c--
		rc := r.c
		r.x[rc].ch = r.x[rc+1].ch
		r.x[rc].sep = nil
		r.x[rc+1].ch = nil
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

// ----------------------------------------------------------------- enumerator

// Next returns the currently enumerated item, if it exists and moves to the
// next item in the key collation order. If there is no item to return, err ==
// io.EOF is returned.
func (e *enumerator) Next() (k []interface{}, v []interface{}, err error) {
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

	i := e.q.d[e.i]
	k, v = i.k, i.v
	e.k, e.hit = k, false
	e.next()
	return
}

func (e *enumerator) next() error {
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
func (e *enumerator) Prev() (k []interface{}, v []interface{}, err error) {
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

	i := e.q.d[e.i]
	k, v = i.k, i.v
	e.k, e.hit = k, false
	e.prev()
	return
}

func (e *enumerator) prev() error {
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
