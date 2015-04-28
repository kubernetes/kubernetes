// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"

	"github.com/cznic/fileutil"
	"github.com/cznic/mathutil"
)

var (
	testFrom = flag.Uint("from", 0, "test I [-from, -N)")
	noGrow   = flag.Bool("noGrow", false, "check only embeded keys/values")
)

func verifyPageLinks(a btreeStore, tree btree, n int) (err error) {
	var p btreeDataPage
	var ph int64
	if ph, p, err = tree.first(a); err != nil {
		return
	}

	if n == 0 {
		if ph != 0 || p != nil {
			return fmt.Errorf("first() should returned nil page")
		}

		ph2, p2, err := tree.last(a)
		if err != nil {
			return err
		}

		if ph2 != 0 || p2 != nil {
			return fmt.Errorf("last() should returned nil page")
		}

	}

	n0 := n
	var prev int64
	var lastKey []byte
	for ph != 0 {
		if p, err = a.Get(nil, ph); err != nil {
			return
		}

		if g, e := p.prev(), prev; g != e {
			return fmt.Errorf("broken L-R DLL chain: p %d p.prev %d, exp %d", ph, g, e)
		}

		for i := 0; i < p.len(); i++ {
			key, err := p.key(a, i)
			if err != nil {
				return err
			}

			if key == nil {
				return fmt.Errorf("nil key")
			}

			if lastKey != nil && !(bytes.Compare(lastKey, key) < 0) {
				return fmt.Errorf("L-R key ordering broken")
			}

			lastKey = key
			n--
		}

		prev, ph = ph, p.next()
	}

	if n != 0 {
		return fmt.Errorf("# of keys off by %d (L-R)", n)
	}

	n = n0
	if ph, p, err = tree.last(a); err != nil {
		return
	}

	lastKey = nil
	var next int64

	for ph != 0 {
		if p, err = a.Get(nil, ph); err != nil {
			return
		}

		if g, e := p.next(), next; g != e {
			return fmt.Errorf("broken R-L DLL chain")
		}

		for i := p.len() - 1; i >= 0; i-- {
			key, err := p.key(a, i)
			if err != nil {
				return err
			}

			if key == nil {
				return fmt.Errorf("nil key")
			}

			if lastKey != nil && !(bytes.Compare(key, lastKey) < 0) {
				return fmt.Errorf("R-L key ordering broken")
			}

			lastKey = key
			n--
		}

		next, ph = ph, p.prev()
	}

	if n != 0 {
		return fmt.Errorf("# of keys off by %d (R-L)", n)
	}

	return
}

func testBTreePut1(t *testing.T, nf func() btreeStore, grow, from, to, xor int64) (tree btree) {
	if *noGrow {
		grow = 0
	}

	a := nf()
	if a == nil {
		t.Fatal(a)
	}

	var err error
	tree, err = newBTree(a)
	if err != nil {
		t.Fatal(err)
	}

	if err := verifyPageLinks(a, tree, 0); err != nil {
		t.Fatal(err)
	}

	// Write and read back
	var k, v, prevValue [7]byte

	n := 0
	for i := from; i < to; i++ {
		h2b(k[:], 0x0100000000+i^xor)
		h2b(v[:], 0x0200000000+i^xor)
		kk := append(make([]byte, grow*i), k[:]...)
		vv := append(make([]byte, grow*i), v[:]...)
		prev, err := tree.put(nil, a, nil, kk, vv, true)
		if err != nil || len(prev) != 0 {
			t.Fatal(i, prev, err)
		}

		var buf []byte
		if buf, err = tree.get(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(buf, vv) {
			t.Fatalf("\nK %sG %sE %s%s", hex.Dump(kk), hex.Dump(buf), hex.Dump(vv), tree.String(a))
		}

		n++
	}

	if err := verifyPageLinks(a, tree, n); err != nil {
		t.Fatalf("%s\n%s", err, tree.String(a))
	}

	// Overwrite, read and extract
	for i := from; i < to; i++ {
		h2b(k[:], 0x0100000000+i^xor)
		h2b(prevValue[:], 0x0200000000+i^xor)
		h2b(v[:], 0x0300000000+i^xor)
		kk := append(make([]byte, grow*i), k[:]...)
		vv := append(make([]byte, grow*i), v[:]...)
		expPrev := append(make([]byte, grow*i), prevValue[:]...)
		gotPrev, err := tree.put(nil, a, nil, kk, vv, true)
		if err != nil {
			t.Fatal(i, err)
		}

		if !bytes.Equal(gotPrev, expPrev) {
			t.Fatalf("\nK %sG %sE %s%s", hex.Dump(kk), hex.Dump(gotPrev), hex.Dump(expPrev), tree.String(a))
		}

		var buf []byte
		if buf, err = tree.get(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(buf, vv) {
			t.Fatalf("\n%s%s%s%s", hex.Dump(kk), hex.Dump(buf), hex.Dump(vv), tree.String(a))
		}

		buf = nil
		if buf, err = tree.extract(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(buf, vv) {
			t.Fatalf("i %d, from [%d, %d)\nK %sG %sE %s%s", i, from, to, hex.Dump(kk), hex.Dump(buf), hex.Dump(vv), tree.String(a))
		}

		buf = nil
		if buf, err = tree.get(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if buf != nil {
			t.Fatalf("\nK %sB %s%s", hex.Dump(kk), hex.Dump(buf), tree.String(a))
		}

		buf = nil
		if buf, err = tree.extract(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if buf != nil {
			t.Fatalf("\n%s\n%s%s", hex.Dump(kk), hex.Dump(buf), tree.String(a))
		}

		n--
		if err := verifyPageLinks(a, tree, n); err != nil {
			t.Fatalf("%s\n%s", err, tree.String(a))
		}
	}

	return
}

var xors = [...]int64{0, 0xffffffff, 0x55555555, 0xaaaaaaaa}

func TestBTreePutGetExtract(t *testing.T) {
	N := int64(3 * kData)
	from := int64(*testFrom)

	for grow := 0; grow < 2; grow++ {
		for _, x := range xors {
			var s *memBTreeStore
			tree := testBTreePut1(t, func() btreeStore { s = newMemBTreeStore(); return s }, int64(grow), from, N, x)
			if err := verifyPageLinks(s, tree, 0); err != nil {
				t.Fatal(err)
			}

			if g, e := len(s.m), 1; g != e {
				t.Fatalf("leak(s) %d %d\n%s", g, e, s)
			}
		}
	}
}

func testBTreePut2(t *testing.T, nf func() btreeStore, grow, n int) (tree btree) {
	if *noGrow {
		grow = 0
	}
	rng, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		t.Fatal(err)
	}

	a := nf()
	if a == nil {
		t.Fatal(a)
	}

	tree, err = newBTree(a)
	if err != nil {
		t.Fatal(err)
	}

	var k, v [7]byte
	for i := 0; i < n; i++ {
		ik, iv := int64(rng.Next()), int64(rng.Next())
		h2b(k[:], ik)
		h2b(v[:], iv)
		kk := append(make([]byte, grow*i), k[:]...)
		vv := append(make([]byte, grow*i), v[:]...)
		prev, err := tree.put(nil, a, nil, kk, vv, true)
		if err != nil || len(prev) != 0 {
			t.Fatal(i, prev, err)
		}

		var buf []byte
		if buf, err = tree.get(a, nil, nil, kk); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(buf, vv) {
			t.Fatalf("\n%s%s%s%s", hex.Dump(kk), hex.Dump(buf), hex.Dump(vv), tree.String(a))
		}
	}

	if err := verifyPageLinks(a, tree, n); err != nil {
		t.Fatalf("%s\n%s\n", err, tree.String(a))
	}

	rng.Seek(0)
	for i := 0; i < n; i++ {
		ik, iv := int64(rng.Next()), int64(rng.Next())
		h2b(k[:], ik)
		h2b(v[:], iv)
		kk := append(make([]byte, grow*i), k[:]...)
		vv := append(make([]byte, grow*i), v[:]...)
		var buf []byte
		buf, err := tree.extract(a, nil, nil, kk)
		if err != nil {
			t.Fatal(i, err)
		}

		if !bytes.Equal(buf, vv) {
			t.Fatalf("\n%s\n%s\n%s\n%s", hex.Dump(kk), hex.Dump(buf), hex.Dump(vv), tree.String(a))
		}

		if err := verifyPageLinks(a, tree, n-i-1); err != nil {
			t.Fatalf("%s\n%s", err, tree.String(a))
		}
	}

	return
}

func TestBTreePutGetExtractRnd(t *testing.T) {
	N := *testN

	for grow := 0; grow < 2; grow++ {
		var s *memBTreeStore
		tree := testBTreePut2(t, func() btreeStore { s = newMemBTreeStore(); return s }, grow, N)
		if err := verifyPageLinks(s, tree, 0); err != nil {
			t.Fatal(err)
		}

		if g, e := len(s.m), 1; g != e {
			t.Fatalf("leak(s) %d %d\n%s", g, e, s)
		}
	}
}

func benchmarkBTreePut(b *testing.B, v []byte) {
	b.StopTimer()
	rng := rand.New(rand.NewSource(42))
	ka := make([][7]byte, b.N)
	for _, v := range ka {
		h2b(v[:], int64(rng.Int63()))
	}
	a := newMemBTreeStore()
	tree, err := newBTree(a)
	if err != nil {
		b.Fatal(err)
	}

	runtime.GC()
	b.StartTimer()
	for _, k := range ka {
		tree.put(nil, a, bytes.Compare, k[:], v, true)
	}
}

func BenchmarkBTreePut1(b *testing.B) {
	v := make([]byte, 1)
	benchmarkBTreePut(b, v)
}

func BenchmarkBTreePut8(b *testing.B) {
	v := make([]byte, 8)
	benchmarkBTreePut(b, v)
}

func BenchmarkBTreePut16(b *testing.B) {
	v := make([]byte, 16)
	benchmarkBTreePut(b, v)
}

func BenchmarkBTreePut32(b *testing.B) {
	v := make([]byte, 32)
	benchmarkBTreePut(b, v)
}

func benchmarkBTreeGet(b *testing.B, v []byte) {
	b.StopTimer()
	rng := rand.New(rand.NewSource(42))
	ka := make([][7]byte, b.N)
	for _, v := range ka {
		h2b(v[:], int64(rng.Int63()))
	}
	a := newMemBTreeStore()
	tree, err := newBTree(a)
	if err != nil {
		b.Fatal(err)
	}

	for _, k := range ka {
		tree.put(nil, a, bytes.Compare, k[:], v, true)
	}
	buf := make([]byte, len(v))
	runtime.GC()
	b.StartTimer()
	for _, k := range ka {
		tree.get(a, buf, bytes.Compare, k[:])
	}
}

func BenchmarkBTreeGet1(b *testing.B) {
	v := make([]byte, 1)
	benchmarkBTreeGet(b, v)
}

func BenchmarkBTreeGet8(b *testing.B) {
	v := make([]byte, 8)
	benchmarkBTreeGet(b, v)
}

func BenchmarkBTreeGet16(b *testing.B) {
	v := make([]byte, 16)
	benchmarkBTreeGet(b, v)
}

func BenchmarkBTreeGet32(b *testing.B) {
	v := make([]byte, 32)
	benchmarkBTreeGet(b, v)
}

func TestbTreeSeek(t *testing.T) {
	N := int64(*testN)

	tree := NewBTree(nil)

	// Fill
	for i := int64(1); i <= N; i++ {
		tree.Set(enc8(10*i), enc8(10*i+1))
	}

	// Check
	a, root := tree.store, tree.root
	for i := int64(1); i <= N; i++ {
		// Exact match
		lowKey := enc8(10*i - 1)
		key := enc8(10 * i)
		highKey := enc8(10*i + 1)
		p, index, eq, err := root.seek(a, nil, key)
		if err != nil {
			t.Fatal(err)
		}

		if !eq {
			t.Fatal(i)
		}

		if btreePage(p).isIndex() {
			t.Fatal(i, "need btreeDataPage")
		}

		dp := btreeDataPage(p)
		n := dp.len()
		if n < 0 || n > 2*kData {
			t.Fatal(i, n)
		}

		if index < 0 || index >= n {
			t.Fatal(index)
		}

		g, err := dp.key(a, index)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(g, key) {
			t.Fatal(i)
		}

		g, err = dp.value(a, index)
		if err != nil {
			t.Fatal(err)
		}

		value := enc8(10*i + 1)
		if !bytes.Equal(g, value) {
			t.Fatal(i)
		}

		// Nonexistent "low" key. Search for 9 should return the key 10.
		p, index, eq, err = root.seek(a, nil, lowKey)
		if err != nil {
			t.Fatal(err)
		}

		if eq {
			t.Fatal(i)
		}

		if btreePage(p).isIndex() {
			t.Fatal(i, "need btreeDataPage")
		}

		dp = btreeDataPage(p)
		n = dp.len()
		if n < 0 || n > 2*kData {
			t.Fatal(i, n)
		}

		if index < 0 || index > n {
			t.Fatal(index, n)
		}

		if index == n {
			ph := dp.next()
			index = 0
			if dp, err = a.Get(p, ph); err != nil {
				t.Fatal(err)
			}
		}

		g, err = dp.key(a, index)
		if err != nil {
			t.Fatal(err)
		}

		expKey := key
		if !bytes.Equal(g, expKey) {
			fmt.Println(root.String(a))
			//t.Fatalf("%d low|% x| g|% x| e|% x|", i, lowKey, g, expKey)
		}

		g, err = dp.value(a, index)
		if err != nil {
			t.Fatal(err)
		}

		value = enc8(10*i + 1)
		if !bytes.Equal(g, value) {
			t.Fatal(i)
		}

		// Nonexistent "high" key. Search for 11 should return the key 20.
		p, index, eq, err = root.seek(a, nil, highKey)
		if err != nil {
			t.Fatal(err)
		}

		if eq {
			t.Fatal(i)
		}

		if btreePage(p).isIndex() {
			t.Fatal(i, "need btreeDataPage")
		}

		dp = btreeDataPage(p)
		n = dp.len()
		if n < 0 || n > 2*kData {
			t.Fatal(i, n)
		}

		if index < 0 || index > n {
			t.Fatal(index, n)
		}

		if index == n {
			ph := dp.next()
			if i == N {
				if ph != 0 {
					t.Fatal(ph)
				}

				continue
			}

			index = 0
			if dp, err = a.Get(p, ph); err != nil {
				t.Fatal(err)
			}
		}

		g, err = dp.key(a, index)
		if err != nil {
			t.Fatal(err)
		}

		expKey = enc8(10 * (i + 1))
		if !bytes.Equal(g, expKey) {
			//fmt.Println(root.String(a))
			t.Fatalf("%d low|% x| g|% x| e|% x|", i, lowKey, g, expKey)
		}

		g, err = dp.value(a, index)
		if err != nil {
			t.Fatal(err)
		}

		value = enc8(10*(i+1) + 1)
		if !bytes.Equal(g, value) {
			t.Fatal(i)
		}

	}
}

func enc8(n int64) []byte {
	var b [8]byte
	h2b(b[:], n)
	return b[:]
}

func dec8(b []byte) (int64, error) {
	if len(b) != 0 {
		return 0, fmt.Errorf("dec8: len != 8 but %d", len(b))
	}

	return b2h(b), nil
}

func TestbTreeNext(t *testing.T) {
	N := int64(*testN)

	tree := NewBTree(nil)
	enum, _, err := tree.seek(enc8(0))
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err = enum.current(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.next(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.prev(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	// Fill
	for i := int64(1); i <= N; i++ {
		tree.Set(enc8(10*i), enc8(10*i+1))
	}

	var eq bool

	enum, eq, err = tree.seek(enc8(0))
	if err != nil {
		t.Fatal(err)
	}

	if eq {
		t.Fatal(eq)
	}

	// index: 0
	if _, _, err = enum.current(); err != nil {
		t.Fatal(err)
	}

	if err = enum.next(); N > 1 && err != nil {
		t.Fatal(err)
	}

	enum, eq, err = tree.seek(enc8(N * 10))
	if err != nil {
		t.Fatal(err)
	}

	if !eq {
		t.Fatal(eq)
	}

	// index: N-1
	if _, _, err = enum.current(); err != nil {
		t.Fatal(err)
	}

	if err = enum.next(); N > 1 && !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	enum, eq, err = tree.seek(enc8(N*10 + 1))
	if err != nil {
		t.Fatal(err)
	}

	if eq {
		t.Fatal(eq)
	}

	// index: N
	if _, _, err = enum.current(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.next(); N > 1 && !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	enum, _, err = tree.seek(enc8(0))
	if err != nil {
		t.Fatal(err)
	}

	for i := int64(1); i <= N; i++ {
		expKey, expValue := enc8(10*i), enc8(10*i+1)
		k, v, err := enum.current()
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(k, expKey) {
			t.Fatal(i)
		}

		if !bytes.Equal(v, expValue) {
			t.Fatal(i)
		}

		switch {
		case i == N:
			if err := enum.next(); !fileutil.IsEOF(err) {
				t.Fatal(err)
			}
		default:
			if err := enum.next(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestbTreePrev(t *testing.T) {
	N := int64(*testN)

	tree := NewBTree(nil)
	enum, _, err := tree.seek(enc8(0))
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err = enum.current(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.next(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.prev(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	// Fill
	for i := int64(1); i <= N; i++ {
		tree.Set(enc8(10*i), enc8(10*i+1))
	}

	var eq bool

	enum, eq, err = tree.seek(enc8(0))
	if err != nil {
		t.Fatal(err)
	}

	if eq {
		t.Fatal(eq)
	}

	// index: 0
	if _, _, err = enum.current(); err != nil {
		t.Fatal(err)
	}

	if err = enum.prev(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	enum, eq, err = tree.seek(enc8(N * 10))
	if err != nil {
		t.Fatal(err)
	}

	if !eq {
		t.Fatal(eq)
	}

	// index: N-1
	if _, _, err = enum.current(); err != nil {
		t.Fatal(err)
	}

	if err = enum.prev(); N > 1 && err != nil {
		t.Fatal(err)
	}

	enum, eq, err = tree.seek(enc8(N*10 + 1))
	if err != nil {
		t.Fatal(err)
	}

	if eq {
		t.Fatal(eq)
	}

	// index: N
	if _, _, err = enum.current(); !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	if err = enum.prev(); err != nil {
		t.Fatal(err)
	}

	enum, _, err = tree.seek(enc8(N * 10))
	if err != nil {
		t.Fatal(err)
	}

	for i := N; i >= 1; i-- {
		expKey, expValue := enc8(10*i), enc8(10*i+1)
		k, v, err := enum.current()
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(k, expKey) {
			t.Fatalf("%d k|% x| expK|% x| %s\n", i, k, expKey, tree.root.String(tree.store))
		}

		if !bytes.Equal(v, expValue) {
			t.Fatal(i)
		}

		switch {
		case i == 1:
			if err := enum.prev(); !fileutil.IsEOF(err) {
				t.Fatal(err)
			}
		default:
			if err := enum.prev(); err != nil {
				t.Fatal(i, err)
			}
		}
	}
}

func TestBTreeClear(t *testing.T) {
	N := int64(*testN)

	var err error
	var p []byte
	for n := int64(0); n <= N; n = n*3/2 + 1 {
		tree := NewBTree(nil)
		for i := int64(0); i < n; i++ {
			k := append(make([]byte, kKV), enc8(10*i+1)...)
			v := append(make([]byte, kKV+1), enc8(10*i+2)...)
			if err = tree.Set(k, v); err != nil {
				t.Fatal(err)
			}
		}

		if err = tree.Clear(); err != nil {
			t.Fatal(err)
		}

		if g, e := len(tree.store.(*memBTreeStore).m), 1; g != e {
			t.Fatalf("%v %v %v\n%s", n, g, e, tree.store.(*memBTreeStore).String())
		}

		if p, err = tree.store.Get(p, 1); err != nil {
			t.Fatal(err)
		}

		if g, e := p, zeros[:7]; len(g) != 0 && !bytes.Equal(g, e) {
			t.Fatalf("|% x| |% x|", g, e)
		}
	}
}

func TestBTreeRemove(t *testing.T) {
	N := int64(*testN)

	for n := int64(0); n <= N; n = n*3/2 + 1 {
		f := NewMemFiler()
		store, err := NewAllocator(f, &Options{})
		if err != nil {
			t.Fatal(err)
		}

		sz0, err := f.Size()
		if err != nil {
			t.Fatal(err)
		}

		tree, handle, err := CreateBTree(store, nil)
		if err != nil {
			t.Fatal(err)
		}

		for i := int64(0); i < n; i++ {
			k := append(make([]byte, kKV), enc8(10*i+1)...)
			v := append(make([]byte, kKV+1), enc8(10*i+2)...)
			if err = tree.Set(k, v); err != nil {
				t.Fatal(err)
			}
		}

		if err = RemoveBTree(store, handle); err != nil {
			t.Fatal(err)
		}

		sz, err := f.Size()
		if err != nil {
			t.Fatal(err)
		}

		if g, e := sz-sz0, int64(0); g != e {
			t.Fatal(g, e)
		}
	}
}

func collate(a, b []byte) (r int) {
	da, err := DecodeScalars(a)
	if err != nil {
		panic(err)
	}

	db, err := DecodeScalars(b)
	if err != nil {
		panic(err)
	}

	r, err = Collate(da, db, nil)
	if err != nil {
		panic(err)
	}

	return
}

func TestBTreeCollatingBug(t *testing.T) {
	tree := NewBTree(collate)

	date, err := EncodeScalars("Date")
	if err != nil {
		t.Fatal(err)
	}

	customer, err := EncodeScalars("Customer")
	if err != nil {
		t.Fatal(err)
	}

	if g, e := collate(customer, date), -1; g != e {
		t.Fatal(g, e)
	}

	if g, e := collate(date, customer), 1; g != e {
		t.Fatal(g, e)
	}

	err = tree.Set(date, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = tree.Set(customer, nil)
	if err != nil {
		t.Fatal(err)
	}

	var b bytes.Buffer
	tree.Dump(&b)
	t.Logf("\n%s", b.String())

	key, _, err := tree.First()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := key, customer; !bytes.Equal(g, e) {
		t.Fatal(g, e)
	}

}

func TestExtract(t *testing.T) { // Test of the exported wrapper only, .extract tested elsewhere
	bt := NewBTree(nil)
	bt.Set([]byte("a"), []byte("b"))
	bt.Set([]byte("c"), []byte("d"))
	bt.Set([]byte("e"), []byte("f"))

	if v, err := bt.Get(nil, []byte("a")); string(v) != "b" || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Get(nil, []byte("c")); string(v) != "d" || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Get(nil, []byte("e")); string(v) != "f" || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Extract(nil, []byte("c")); string(v) != "d" || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Get(nil, []byte("a")); string(v) != "b" || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Get(nil, []byte("c")); v != nil || err != nil {
		t.Fatal(v, err)
	}

	if v, err := bt.Get(nil, []byte("e")); string(v) != "f" || err != nil {
		t.Fatal(v, err)
	}
}

func TestFirst(t *testing.T) {
	bt := NewBTree(nil)

	if k, v, err := bt.First(); k != nil || v != nil || err != nil {
		t.Fatal(k, v, err)
	}

	bt.Set([]byte("a"), []byte("b"))
	bt.Set([]byte("c"), []byte("d"))

	if k, v, err := bt.First(); string(k) != "a" || string(v) != "b" || err != nil {
		t.Fatal(k, v, err)
	}

	if err := bt.Delete([]byte("a")); err != nil {
		t.Fatal(err)
	}

	if k, v, err := bt.First(); string(k) != "c" || string(v) != "d" || err != nil {
		t.Fatal(k, v, err)
	}

	if err := bt.Delete([]byte("c")); err != nil {
		t.Fatal(err)
	}

	if k, v, err := bt.First(); k != nil || v != nil || err != nil {
		t.Fatal(k, v, err)
	}
}

func TestLast(t *testing.T) {
	bt := NewBTree(nil)

	if k, v, err := bt.First(); k != nil || v != nil || err != nil {
		t.Fatal(k, v, err)
	}

	bt.Set([]byte("a"), []byte("b"))
	bt.Set([]byte("c"), []byte("d"))

	if k, v, err := bt.Last(); string(k) != "c" || string(v) != "d" || err != nil {
		t.Fatal(k, v, err)
	}

	if err := bt.Delete([]byte("c")); err != nil {
		t.Fatal(err)
	}

	if k, v, err := bt.First(); string(k) != "a" || string(v) != "b" || err != nil {
		t.Fatal(k, v, err)
	}

	if err := bt.Delete([]byte("a")); err != nil {
		t.Fatal(err)
	}

	if k, v, err := bt.First(); k != nil || v != nil || err != nil {
		t.Fatal(k, v, err)
	}
}

func TestseekFirst(t *testing.T) {
	bt := NewBTree(nil)

	enum, err := bt.seekFirst()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	bt.Set([]byte("c"), []byte("d"))
	enum, err = bt.seekFirst()
	if err != nil {
		t.Fatal(err)
	}

	err = enum.prev()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	err = enum.next()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	k, v, err := enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "c" || string(v) != "d" {
		t.Fatal(k, v)
	}

	bt.Set([]byte("a"), []byte("b"))
	enum, err = bt.seekFirst()
	if err != nil {
		t.Fatal(err)
	}

	err = enum.prev()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	k, v, err = enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "a" || string(v) != "b" {
		t.Fatal(k, v)
	}

	err = enum.next()
	if err != nil {
		t.Fatal(err)
	}

	k, v, err = enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "c" || string(v) != "d" {
		t.Fatal(k, v)
	}
}

func TestseekLast(t *testing.T) {
	bt := NewBTree(nil)

	enum, err := bt.seekFirst()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	bt.Set([]byte("a"), []byte("b"))
	enum, err = bt.seekFirst()
	if err != nil {
		t.Fatal(err)
	}

	err = enum.prev()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	err = enum.next()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	k, v, err := enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "a" || string(v) != "b" {
		t.Fatal(k, v)
	}

	bt.Set([]byte("c"), []byte("d"))
	enum, err = bt.seekLast()
	if err != nil {
		t.Fatal(err)
	}

	err = enum.next()
	if !fileutil.IsEOF(err) {
		t.Fatal(err)
	}

	k, v, err = enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "c" || string(v) != "d" {
		t.Fatal(k, v)
	}

	err = enum.prev()
	if err != nil {
		t.Fatal(err)
	}

	k, v, err = enum.current()
	if err != nil {
		t.Fatal(err)
	}

	if string(k) != "a" || string(v) != "b" {
		t.Fatal(k, v)
	}
}

func TestDeleteAny(t *testing.T) {
	const N = 1e4
	rng := rand.New(rand.NewSource(42))
	ref := map[uint32]bool{}
	tr := NewBTree(nil)
	data := []byte{42}
	var key [4]byte
	for i := 0; i < N; i++ {
		k := uint32(rng.Int())
		binary.LittleEndian.PutUint32(key[:], k)
		if err := tr.Set(key[:], data); err != nil {
			t.Fatal(err)
		}

		ref[k] = true
	}

	for i := len(ref); i != 0; i-- {
		empty, err := tr.DeleteAny()
		if err != nil {
			t.Fatal(err)
		}

		if empty && i != 1 {
			t.Fatal(i)
		}
	}
}

func benchmarkBTreeSetFiler(b *testing.B, f Filer, sz int) {
	if err := f.BeginUpdate(); err != nil {
		b.Error(err)
		return
	}

	a, err := NewAllocator(f, &Options{})
	if err != nil {
		b.Error(err)
		return
	}

	tr, _, err := CreateBTree(a, nil)
	if err != nil {
		f.EndUpdate()
		b.Error(err)
		return
	}

	if err = f.EndUpdate(); err != nil {
		b.Error(err)
		return
	}

	keys := make([][8]byte, b.N)
	for i := range keys {
		binary.BigEndian.PutUint64(keys[i][:], uint64(i))
	}
	v := make([]byte, sz)
	runtime.GC()
	b.ResetTimer()
	for _, k := range keys {
		if err = f.BeginUpdate(); err != nil {
			b.Error(err)
			return
		}

		if err := tr.Set(k[:], v); err != nil {
			f.EndUpdate()
			b.Error(err)
			return
		}

		if err = f.EndUpdate(); err != nil {
			b.Error(err)
			return
		}
	}
}

func benchmarkBTreeSetMemFiler(b *testing.B, sz int) {
	f := NewMemFiler()
	benchmarkBTreeSetFiler(b, f, sz)
}

func BenchmarkBTreeSetMemFiler0(b *testing.B) {
	benchmarkBTreeSetMemFiler(b, 0)
}

func BenchmarkBTreeSetMemFiler1e1(b *testing.B) {
	benchmarkBTreeSetMemFiler(b, 1e1)
}

func BenchmarkBTreeSetMemFiler1e2(b *testing.B) {
	benchmarkBTreeSetMemFiler(b, 1e2)
}

func BenchmarkBTreeSetMemFiler1e3(b *testing.B) {
	benchmarkBTreeSetMemFiler(b, 1e3)
}

func benchmarkBTreeSetSimpleFileFiler(b *testing.B, sz int) {
	dir, testDbName := temp()
	defer os.RemoveAll(dir)

	f, err := os.OpenFile(testDbName, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		b.Fatal(err)
	}

	defer f.Close()

	benchmarkBTreeSetFiler(b, NewSimpleFileFiler(f), sz)
}

func BenchmarkBTreeSetSimpleFileFiler0(b *testing.B) {
	benchmarkBTreeSetSimpleFileFiler(b, 0)
}

func BenchmarkBTreeSetSimpleFileFiler1e1(b *testing.B) {
	benchmarkBTreeSetSimpleFileFiler(b, 1e1)
}

func BenchmarkBTreeSetSimpleFileFiler1e2(b *testing.B) {
	benchmarkBTreeSetSimpleFileFiler(b, 1e2)
}

func BenchmarkBTreeSetSimpleFileFiler1e3(b *testing.B) {
	benchmarkBTreeSetSimpleFileFiler(b, 1e3)
}

func benchmarkBTreeSetRollbackFiler(b *testing.B, sz int) {
	dir, testDbName := temp()
	defer os.RemoveAll(dir)

	f, err := os.OpenFile(testDbName, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		b.Fatal(err)
	}

	defer f.Close()

	g := NewSimpleFileFiler(f)
	var filer *RollbackFiler
	if filer, err = NewRollbackFiler(
		g,
		func(sz int64) error {
			if err = g.Truncate(sz); err != nil {
				return err
			}

			return g.Sync()
		},
		g,
	); err != nil {
		b.Error(err)
		return
	}

	benchmarkBTreeSetFiler(b, filer, sz)
}

func BenchmarkBTreeSetRollbackFiler0(b *testing.B) {
	benchmarkBTreeSetRollbackFiler(b, 0)
}

func BenchmarkBTreeSetRollbackFiler1e1(b *testing.B) {
	benchmarkBTreeSetRollbackFiler(b, 1e1)
}

func BenchmarkBTreeSetRollbackFiler1e2(b *testing.B) {
	benchmarkBTreeSetRollbackFiler(b, 1e2)
}

func BenchmarkBTreeSetRollbackFiler1e3(b *testing.B) {
	benchmarkBTreeSetRollbackFiler(b, 1e3)
}

func benchmarkBTreeSetACIDFiler(b *testing.B, sz int) {
	dir, testDbName := temp()
	defer os.RemoveAll(dir)

	f, err := os.OpenFile(testDbName, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		b.Fatal(err)
	}

	defer f.Close()

	wal, err := os.OpenFile(testDbName+".wal", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		b.Fatal(err)
	}

	defer wal.Close()

	filer, err := NewACIDFiler(NewSimpleFileFiler(f), wal)
	if err != nil {
		b.Error(err)
		return
	}

	benchmarkBTreeSetFiler(b, filer, sz)
}

func BenchmarkBTreeSetACIDFiler0(b *testing.B) {
	benchmarkBTreeSetACIDFiler(b, 0)
}

func BenchmarkBTreeSetACIDFiler1e1(b *testing.B) {
	benchmarkBTreeSetACIDFiler(b, 1e1)
}

func BenchmarkBTreeSetACIDFiler1e2(b *testing.B) {
	benchmarkBTreeSetACIDFiler(b, 1e2)
}

func BenchmarkBTreeSetACIDFiler1e3(b *testing.B) {
	benchmarkBTreeSetACIDFiler(b, 1e3)
}

func testbTreeEnumeratorInvalidating(t *testing.T, mutate func(b *BTree) error) {
	b := NewBTree(nil)
	if err := b.Set([]byte{1}, []byte{2}); err != nil {
		t.Fatal(err)
	}

	if err := b.Set([]byte{3}, []byte{4}); err != nil {
		t.Fatal(err)
	}

	e, err := b.seekFirst()
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err = e.current(); err != nil {
		t.Fatal(err)
	}

	if err = e.next(); err != nil {
		t.Fatal(err)
	}

	if err = e.prev(); err != nil {
		t.Fatal(err)
	}

	if err = mutate(b); err != nil {
		t.Fatal(err)
	}

	if _, _, err = e.current(); err == nil {
		t.Fatal(err)
	}

	if _, ok := err.(*ErrINVAL); !ok {
		t.Fatalf("%T", err)
	}

	err = e.next()
	if err == nil {
		t.Fatal(err)
	}

	if _, ok := err.(*ErrINVAL); !ok {
		t.Fatalf("%T", err)
	}

	err = e.prev()
	if err == nil {
		t.Fatal(err)
	}

	if _, ok := err.(*ErrINVAL); !ok {
		t.Fatalf("%T", err)
	}

}

func TestBTreeEnumeratorInvalidating(t *testing.T) {
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error { return b.Clear() })
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error { return b.Delete([]byte{1}) })
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error { _, err := b.DeleteAny(); return err })
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error { _, err := b.Extract(nil, []byte{1}); return err })
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error {
		_, _, err := b.Put(
			nil,
			[]byte{1},
			func(k, o []byte) ([]byte, bool, error) { return nil, false, nil },
		)
		return err
	})
	testbTreeEnumeratorInvalidating(t, func(b *BTree) error { return b.Set([]byte{4}, []byte{5}) })
}

func n2b(n int) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], uint64(n))
	return b[:]
}

func b2n(b []byte) int {
	if len(b) != 8 {
		return mathutil.MinInt
	}

	return int(binary.BigEndian.Uint64(b))
}

func TestBTreeSeekNext(t *testing.T) {
	// seeking within 3 keys: 10, 20, 30
	table := []struct {
		k    int
		hit  bool
		keys []int
	}{
		{5, false, []int{10, 20, 30}},
		{10, true, []int{10, 20, 30}},
		{15, false, []int{20, 30}},
		{20, true, []int{20, 30}},
		{25, false, []int{30}},
		{30, true, []int{30}},
		{35, false, []int{}},
	}

	for i, test := range table {
		up := test.keys
		db := NewBTree(nil)

		if err := db.Set(n2b(10), n2b(100)); err != nil {
			t.Fatal(i, err)
		}

		if err := db.Set(n2b(20), n2b(200)); err != nil {
			t.Fatal(i, err)
		}

		if err := db.Set(n2b(30), n2b(300)); err != nil {
			t.Fatal(i, err)
		}

		for brokenSerial := 0; brokenSerial < 16; brokenSerial++ {
			en, hit, err := db.Seek(n2b(test.k))
			if err != nil {
				t.Fatal(err)
			}

			if g, e := hit, test.hit; g != e {
				t.Fatal(i, g, e)
			}

			j := 0
			for {
				if brokenSerial&(1<<uint(j)) != 0 {
					if err := db.Set(n2b(20), n2b(200)); err != nil {
						t.Fatal(i, err)
					}
				}

				k, v, err := en.Next()
				if err != nil {
					if !fileutil.IsEOF(err) {
						t.Fatal(i, err)
					}

					break
				}

				if g, e := len(k), 8; g != e {
					t.Fatal(i, g, e)
				}

				if j >= len(up) {
					t.Fatal(i, j, brokenSerial)
				}

				if g, e := b2n(k), up[j]; g != e {
					t.Fatal(i, j, brokenSerial, g, e)
				}

				if g, e := len(v), 8; g != e {
					t.Fatal(i, g, e)
				}

				if g, e := b2n(v), 10*up[j]; g != e {
					t.Fatal(i, g, e)
				}

				j++

			}

			if g, e := j, len(up); g != e {
				t.Fatal(i, j, g, e)
			}
		}

	}
}

func TestBTreeSeekPrev(t *testing.T) {
	// seeking within 3 keys: 10, 20, 30
	table := []struct {
		k    int
		hit  bool
		keys []int
	}{
		{5, false, []int{10}},
		{10, true, []int{10}},
		{15, false, []int{20, 10}},
		{20, true, []int{20, 10}},
		{25, false, []int{30, 20, 10}},
		{30, true, []int{30, 20, 10}},
		{35, false, []int{}},
	}

	for i, test := range table {
		down := test.keys
		db := NewBTree(nil)
		if err := db.Set(n2b(10), n2b(100)); err != nil {
			t.Fatal(i, err)
		}

		if err := db.Set(n2b(20), n2b(200)); err != nil {
			t.Fatal(i, err)
		}

		if err := db.Set(n2b(30), n2b(300)); err != nil {
			t.Fatal(i, err)
		}

		for brokenSerial := 0; brokenSerial < 16; brokenSerial++ {
			en, hit, err := db.Seek(n2b(test.k))
			if err != nil {
				t.Fatal(err)
			}

			if g, e := hit, test.hit; g != e {
				t.Fatal(i, g, e)
			}

			j := 0
			for {
				if brokenSerial&(1<<uint(j)) != 0 {
					if err := db.Set(n2b(20), n2b(200)); err != nil {
						t.Fatal(i, err)
					}
				}

				k, v, err := en.Prev()
				if err != nil {
					if !fileutil.IsEOF(err) {
						t.Fatal(i, err)
					}

					break
				}

				if g, e := len(k), 8; g != e {
					t.Fatal(i, g, e)
				}

				if j >= len(down) {
					t.Fatal(i, j, brokenSerial)
				}

				if g, e := b2n(k), down[j]; g != e {
					t.Fatal(i, j, brokenSerial, g, e)
				}

				if g, e := len(v), 8; g != e {
					t.Fatal(i, g, e)
				}

				if g, e := b2n(v), 10*down[j]; g != e {
					t.Fatal(i, g, e)
				}

				j++

			}

			if g, e := j, len(down); g != e {
				t.Fatal(i, j, g, e)
			}
		}

	}
}

func TestBTreeSeekFirst(t *testing.T) {
	db := NewBTree(nil)
	en, err := db.SeekFirst()
	if err == nil {
		t.Fatal(err)
	}

	if err := db.Set(n2b(100), n2b(1000)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekFirst(); err != nil {
		t.Fatal(err)
	}

	k, v, err := en.Next()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 100; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 1000; g != e {
		t.Fatal(g, e)
	}

	if err := db.Set(n2b(110), n2b(1100)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekFirst(); err != nil {
		t.Fatal(err)
	}

	if k, v, err = en.Next(); err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 100; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 1000; g != e {
		t.Fatal(g, e)
	}

	if err := db.Set(n2b(90), n2b(900)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekFirst(); err != nil {
		t.Fatal(err)
	}

	if k, v, err = en.Next(); err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 90; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 900; g != e {
		t.Fatal(g, e)
	}

}

func TestBTreeSeekLast(t *testing.T) {
	db := NewBTree(nil)
	en, err := db.SeekLast()
	if err == nil {
		t.Fatal(err)
	}

	if err := db.Set(n2b(100), n2b(1000)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekLast(); err != nil {
		t.Fatal(err)
	}

	k, v, err := en.Next()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 100; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 1000; g != e {
		t.Fatal(g, e)
	}

	if err := db.Set(n2b(90), n2b(900)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekLast(); err != nil {
		t.Fatal(err)
	}

	if k, v, err = en.Next(); err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 100; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 1000; g != e {
		t.Fatal(g, e)
	}

	if err := db.Set(n2b(110), n2b(1100)); err != nil {
		t.Fatal(err)
	}

	if en, err = db.SeekLast(); err != nil {
		t.Fatal(err)
	}

	if k, v, err = en.Next(); err != nil {
		t.Fatal(err)
	}

	if g, e := b2n(k), 110; g != e {
		t.Fatal(g, e)
	}

	if g, e := b2n(v), 1100; g != e {
		t.Fatal(g, e)
	}

}

// https://code.google.com/p/camlistore/issues/detail?id=216
func TestBug216(t *testing.T) {
	const S = 2*kKV + 2 // 2*kKV+1 ok
	const N = 300000
	rng, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		t.Fatal(err)
	}
	k := make([]byte, S/2)
	v := make([]byte, S-S/2)
	tr := NewBTree(nil)
	for i := 0; i < N; i++ {
		for i := range k {
			k[i] = byte(rng.Next())
		}
		for i := range v {
			v[i] = byte(rng.Next())
		}

		if err := tr.Set(h2b(k, int64(i)), h2b(v, int64(i))); err != nil {
			t.Fatal(i, err)
		}

		if (i+1)%10000 == 0 {
			//dbg("%v", i+1)
		}
	}
}
