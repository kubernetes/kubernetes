// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"math/rand"
	"testing"

	"github.com/cznic/fileutil"
	"github.com/cznic/mathutil"
)

func (f *bitFiler) dump(w io.Writer) {
	fmt.Fprintf(w, "bitFiler @ %p, size: %d(%#x)\n", f, f.size, f.size)
	for k, v := range f.m {
		fmt.Fprintf(w, "bitPage @ %p: pgI %d(%#x): %#v\n", v, k, k, *v)
	}
}

func filerBytes(f Filer) []byte {
	sz, err := f.Size()
	if err != nil {
		panic(err)
	}

	b := make([]byte, int(sz))
	n, err := f.ReadAt(b, 0)
	if n != len(b) {
		panic(fmt.Errorf("sz %d n %d err %v", sz, n, err))
	}

	return b
}

func cmpFilerBytes(t *testing.T, fg, fe Filer) {
	g, e := filerBytes(fg), filerBytes(fe)
	if !bytes.Equal(g, e) {
		t.Fatalf("Filer content doesn't match: got\n%sexp:\n%s", hex.Dump(g), hex.Dump(e))
	}
}

func TestRollbackFiler0(t *testing.T) {
	var r *RollbackFiler
	f, g := NewMemFiler(), NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		t.Fatal(err)
	}

	if err = r.BeginUpdate(); err != nil {
		t.Fatal(err)
	}

	if err = r.EndUpdate(); err != nil {
		t.Fatal(err)
	}

	cmpFilerBytes(t, f, g)
}

func TestRollbackFiler1(t *testing.T) {
	const (
		N = 1e6
		O = 1234
	)

	var r *RollbackFiler
	f, g := NewMemFiler(), NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		t.Fatal(err)
	}

	if err = r.BeginUpdate(); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	b := make([]byte, N)
	for i := range b {
		b[i] = byte(rng.Int())
	}

	if _, err = g.WriteAt(b, O); err != nil {
		t.Fatal(err)
	}

	if _, err = r.WriteAt(b, O); err != nil {
		t.Fatal(err)
	}

	b = filerBytes(f)
	if n := len(b); n != 0 {
		t.Fatal(n)
	}

	if err = r.EndUpdate(); err != nil {
		t.Fatal(err)
	}

	cmpFilerBytes(t, f, g)
}

func TestRollbackFiler2(t *testing.T) {
	const (
		N = 1e6
		O = 1234
	)

	var r *RollbackFiler
	f, g := NewMemFiler(), NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		t.Fatal(err)
	}

	if err = r.BeginUpdate(); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	b := make([]byte, N)
	for i := range b {
		b[i] = byte(rng.Int())
	}

	if _, err = r.WriteAt(b, O); err != nil {
		t.Fatal(err)
	}

	b = filerBytes(f)
	if n := len(b); n != 0 {
		t.Fatal(n)
	}

	if err = r.Rollback(); err != nil {
		t.Fatal(err)
	}

	cmpFilerBytes(t, f, g)
}

func rndBytes(rng *rand.Rand, n int) []byte {
	r := make([]byte, n)
	for i := range r {
		r[i] = byte(rng.Int())
	}
	return r
}

func TestRollbackFiler3(t *testing.T) {
	var r *RollbackFiler
	f := NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		t.Fatal(err)
	}

	n, err := r.ReadAt([]byte{0}, 0)
	if n != 0 || !fileutil.IsEOF(err) {
		t.Fatal(n, err)
	}

	n, err = r.ReadAt([]byte{0}, 1e6)
	if n != 0 || !fileutil.IsEOF(err) {
		t.Fatal(n, err)
	}

	if err = r.BeginUpdate(); err != nil { // BeginUpdate: 0 -> 1
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))

	buf := rndBytes(rng, 100)
	if n, err := r.WriteAt(buf, 1e6); n != 100 || err != nil {
		t.Fatal(err)
	}

	buf = make([]byte, 100)
	if n, err := r.ReadAt(buf, 1e6-200); n != 100 || err != nil {
		t.Fatal(err)
	}

	for i, v := range buf {
		if v != 0 {
			t.Fatal(i, v)
		}
	}

	if err := r.Truncate(1e5); err != nil {
		t.Fatal(err)
	}

	if err = r.BeginUpdate(); err != nil { // BeginUpdate: 1 -> 2
		t.Fatal(err)
	}

	if n, err := r.ReadAt(buf, 1e6); n != 0 || err == nil {
		t.Fatal(n, err)
	}

	if err := r.Truncate(2e6); err != nil {
		t.Fatal(err)
	}

	if err = r.BeginUpdate(); err != nil { // BeginUpdate: 2 -> 3
		t.Fatal(err)
	}

	if n, err := r.ReadAt(buf, 1e6); n == 0 || err != nil {
		t.Fatal(n, err)
	}

	for i, v := range buf {
		if v != 0 {
			t.Fatal(i, v)
		}
	}
}

func TestRollbackFiler4(t *testing.T) {
	const (
		maxSize    = 1e6
		maxChange  = maxSize/100 + 4
		maxChanges = 10
		maxNest    = 3
	)

	var r *RollbackFiler
	f := NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))

	ref := make([]byte, 2*maxSize)
	for i := range ref {
		ref[i] = byte(rng.Int())
	}

	var finalSize int

	var fn func(int, int, []byte) (int, []byte)
	fn = func(nest, inSize int, in []byte) (outSize int, out []byte) {
		defer func() {
			for i := outSize; i < len(out); i++ {
				out[i] = 0
			}
			finalSize = mathutil.Max(finalSize, outSize)
		}()

		out = make([]byte, len(in), 2*maxSize)
		copy(out, in)
		if err := r.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		for i := 0; i < maxChanges; i++ {
			changeLen := rng.Intn(maxChange) + 4
			changeOff := rng.Intn(maxSize * 3 / 2)
			b := make([]byte, changeLen)
			for i := range b {
				b[i] = byte(rng.Int())
			}
			if n, err := r.WriteAt(b, int64(changeOff)); n != len(b) || err != nil {
				t.Fatal(n, len(b), err)
			}
		}

		if err := r.Rollback(); err != nil {
			t.Fatal(err)
		}

		if err := r.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		for i := 0; i < maxChanges; i++ {
			changeLen := rng.Intn(maxChange) + 4
			changeOff := rng.Intn(maxSize * 3 / 2)
			b := make([]byte, changeLen)
			for i := range b {
				b[i] = byte(rng.Int())
			}
			if n, err := r.WriteAt(b, int64(changeOff)); n != len(b) || err != nil {
				t.Fatal(n, len(b), err)
			}
			copy(out[changeOff:], b)
			copy(ref[changeOff:], b)
		}

		newSize := rng.Intn(maxSize*3/2) + 4
		if nest == maxNest {
			if err := r.EndUpdate(); err != nil {
				t.Fatal(err)
			}

			return newSize, out
		}

		outSize, out = fn(nest+1, newSize, out)
		if err := r.EndUpdate(); err != nil {
			t.Fatal(err)
		}

		return
	}

	sz, result := fn(0, maxSize, ref)
	if g, e := sz, finalSize; g != e {
		t.Fatal(err)
	}

	g, e := result[:sz], ref[:sz]
	if !bytes.Equal(g, e) {
		if len(g) == len(e) {
			x := make([]byte, len(g))
			for i := range x {
				if g[i] != e[i] {
					x[i] = 'X'
				}
			}
			//t.Logf("Data diff\n%s", hex.Dump(x))
		}
		//t.Fatalf("Data don't match: got\n%sexp:\n%s", hex.Dump(g), hex.Dump(e))
		t.Fatalf("Data don't match")
	}
}

func BenchmarkRollbackFiler(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	type t struct {
		off int64
		b   []byte
	}
	a := []t{}
	for rem := b.N; rem > 0; {
		off := rng.Int63()
		n := mathutil.Min(rng.Intn(1e3)+1, rem)
		a = append(a, t{off, rndBytes(rng, n)})
		rem -= n
	}

	var r *RollbackFiler
	f := NewMemFiler()

	checkpoint := func(sz int64) (err error) {
		return f.Truncate(sz)
	}

	r, err := NewRollbackFiler(f, checkpoint, f)
	if err != nil {
		b.Fatal(err)
	}

	if err := r.BeginUpdate(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for _, v := range a {
		if _, err := r.WriteAt(v.b, v.off); err != nil {
			b.Fatal(err)
		}
	}
}
