// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"encoding/hex"
	"io/ioutil"
	"math/rand"
	"os"
	"runtime"
	"testing"

	"github.com/cznic/fileutil"
)

// Bench knobs.
const (
	filerTestChunkSize = 32e3
	filerTotalSize     = 10e6
)

type newFunc func() Filer

type testFileFiler struct {
	Filer
}

func (t *testFileFiler) Close() (err error) {
	n := t.Name()
	err = t.Filer.Close()
	if errDel := os.Remove(n); errDel != nil && err == nil {
		err = errDel
	}
	return
}

var (
	newFileFiler = func() Filer {
		file, err := ioutil.TempFile("", "lldb-test-file")
		if err != nil {
			panic(err)
		}

		return &testFileFiler{NewSimpleFileFiler(file)}
	}

	newOSFileFiler = func() Filer {
		file, err := ioutil.TempFile("", "lldb-test-osfile")
		if err != nil {
			panic(err)
		}

		return &testFileFiler{NewOSFiler(file)}
	}

	newMemFiler = func() Filer {
		return NewMemFiler()
	}

	nwBitFiler = func() Filer {
		f, err := newBitFiler(NewMemFiler())
		if err != nil {
			panic(err)
		}

		return f
	}

	newRollbackFiler = func() Filer {
		f := NewMemFiler()

		var r Filer

		checkpoint := func(sz int64) (err error) {
			return f.Truncate(sz)
		}

		r, err := NewRollbackFiler(f, checkpoint, f)
		if err != nil {
			panic(err)
		}

		return r
	}
)

func TestFilerNesting(t *testing.T) {
	testFilerNesting(t, newFileFiler)
	testFilerNesting(t, newOSFileFiler)
	testFilerNesting(t, newMemFiler)
	testFilerNesting(t, newRollbackFiler)
}

func testFilerNesting(t *testing.T, nf newFunc) {
	// Check {Create, Close} works.
	f := nf()
	t.Log(f.Name())
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	// Check {Create, EndUpdate} doesn't work.
	f = nf()
	t.Log(f.Name())
	if err := f.EndUpdate(); err == nil {
		f.Close()
		t.Fatal("unexpected success")
	}

	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	// Check {Create, BeginUpdate, Close} doesn't work.
	f = nf()
	t.Log(f.Name())
	f.BeginUpdate()

	if err := f.Close(); err == nil {
		t.Fatal("unexpected success")
	}

	// Check {Create, BeginUpdate, EndUpdate, Close} works.
	f = nf()
	t.Log(f.Name())
	f.BeginUpdate()
	if err := f.EndUpdate(); err != nil {
		f.Close()
		t.Fatal(err)
	}

	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestFilerTruncate(t *testing.T) {
	testFilerTruncate(t, newFileFiler)
	testFilerTruncate(t, newOSFileFiler)
	testFilerTruncate(t, newMemFiler)
	testFilerTruncate(t, nwBitFiler)
	testFilerTruncate(t, newRollbackFiler)
}

func testFilerTruncate(t *testing.T, nf newFunc) {
	f := nf()
	t.Log(f.Name())
	defer func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	}()

	if _, ok := f.(*RollbackFiler); ok {
		if err := f.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		defer func() {
			if err := f.EndUpdate(); err != nil {
				t.Error(err)
			}
		}()
	}

	// Check Truncate works.
	sz := int64(1e6)
	if err := f.Truncate(sz); err != nil {
		t.Error(err)
		return
	}

	fsz, err := f.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := fsz, sz; g != e {
		t.Error(g, e)
		return
	}

	sz *= 2
	if err := f.Truncate(sz); err != nil {
		t.Error(err)
		return
	}

	fsz, err = f.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := fsz, sz; g != e {
		t.Error(g, e)
		return
	}

	sz = 0
	if err := f.Truncate(sz); err != nil {
		t.Error(err)
		return
	}

	fsz, err = f.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := fsz, sz; g != e {
		t.Error(g, e)
		return
	}

	// Check Truncate(-1) doesn't work.
	sz = -1
	if err := f.Truncate(sz); err == nil {
		t.Error(err)
		return
	}

}

func TestFilerReadAtWriteAt(t *testing.T) {
	testFilerReadAtWriteAt(t, newFileFiler)
	testFilerReadAtWriteAt(t, newOSFileFiler)
	testFilerReadAtWriteAt(t, newMemFiler)
	testFilerReadAtWriteAt(t, nwBitFiler)
	testFilerReadAtWriteAt(t, newRollbackFiler)
}

func testFilerReadAtWriteAt(t *testing.T, nf newFunc) {
	f := nf()
	t.Log(f.Name())
	defer func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	}()

	if _, ok := f.(*RollbackFiler); ok {
		if err := f.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		defer func() {
			if err := f.EndUpdate(); err != nil {
				t.Error(err)
			}
		}()
	}

	const (
		N = 1 << 16
		M = 2e2
	)

	s := make([]byte, N)
	e := make([]byte, N)
	rnd := rand.New(rand.NewSource(42))
	for i := range e {
		s[i] = byte(rnd.Intn(256))
	}
	n2 := 0
	for i := 0; i < M; i++ {
		var from, to int
		for {
			from = rnd.Intn(N)
			to = rnd.Intn(N)
			if from != to {
				break
			}
		}
		if from > to {
			from, to = to, from
		}
		for i := range s[from:to] {
			s[from+i] = byte(rnd.Intn(256))
		}
		copy(e[from:to], s[from:to])
		if to > n2 {
			n2 = to
		}
		n, err := f.WriteAt(s[from:to], int64(from))
		if err != nil {
			t.Error(err)
			return
		}

		if g, e := n, to-from; g != e {
			t.Error(g, e)
			return
		}
	}

	fsz, err := f.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := fsz, int64(n2); g != e {
		t.Error(g, e)
		return
	}

	b := make([]byte, n2)
	for i := 0; i <= M; i++ {
		from := rnd.Intn(n2)
		to := rnd.Intn(n2)
		if from > to {
			from, to = to, from
		}
		if i == M {
			from, to = 0, n2
		}
		n, err := f.ReadAt(b[from:to], int64(from))
		if err != nil && (!fileutil.IsEOF(err) && n != 0) {
			fsz, err = f.Size()
			if err != nil {
				t.Error(err)
				return
			}

			t.Error(fsz, from, to, err)
			return
		}

		if g, e := n, to-from; g != e {
			t.Error(g, e)
			return
		}

		if g, e := b[from:to], e[from:to]; !bytes.Equal(g, e) {
			if x, ok := f.(*MemFiler); ok {
				for i := int64(0); i <= 3; i++ {
					t.Logf("pg %d\n----\n%s", i, hex.Dump(x.m[i][:]))
				}
			}
			t.Errorf(
				"i %d from %d to %d len(g) %d len(e) %d\n---- got ----\n%s\n---- exp ----\n%s",
				i, from, to, len(g), len(e), hex.Dump(g), hex.Dump(e),
			)
			return
		}
	}

	mf, ok := f.(*MemFiler)
	if !ok {
		return
	}

	buf := &bytes.Buffer{}
	if _, err := mf.WriteTo(buf); err != nil {
		t.Error(err)
		return
	}

	if g, e := buf.Bytes(), e[:n2]; !bytes.Equal(g, e) {
		t.Errorf("\nlen %d\n%s\nlen %d\n%s", len(g), hex.Dump(g), len(e), hex.Dump(e))
		return
	}

	if err := mf.Truncate(0); err != nil {
		t.Error(err)
		return
	}

	if _, err := mf.ReadFrom(buf); err != nil {
		t.Error(err)
		return
	}

	roundTrip := make([]byte, n2)
	if n, err := mf.ReadAt(roundTrip, 0); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := roundTrip, e[:n2]; !bytes.Equal(g, e) {
		t.Errorf("\nlen %d\n%s\nlen %d\n%s", len(g), hex.Dump(g), len(e), hex.Dump(e))
		return
	}
}

func TestInnerFiler(t *testing.T) {
	testInnerFiler(t, newFileFiler)
	testInnerFiler(t, newOSFileFiler)
	testInnerFiler(t, newMemFiler)
	testInnerFiler(t, nwBitFiler)
	testInnerFiler(t, newRollbackFiler)
}

func testInnerFiler(t *testing.T, nf newFunc) {
	const (
		HDR_SIZE = 42
		LONG_OFF = 3330
	)
	outer := nf()
	t.Log(outer.Name())
	inner := NewInnerFiler(outer, HDR_SIZE)
	defer func() {
		if err := outer.Close(); err != nil {
			t.Error(err)
		}
	}()

	if _, ok := outer.(*RollbackFiler); ok {
		if err := outer.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		defer func() {
			if err := outer.EndUpdate(); err != nil {
				t.Error(err)
			}
		}()
	}

	b := []byte{2, 5, 11}
	n, err := inner.WriteAt(b, -1)
	if err == nil {
		t.Error("unexpected success")
		return
	}

	n, err = inner.ReadAt(make([]byte, 10), -1)
	if err == nil {
		t.Error("unexpected success")
		return
	}

	n, err = inner.WriteAt(b, 0)
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := n, len(b); g != e {
		t.Error(g, e)
		return
	}

	osz, err := outer.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := osz, int64(HDR_SIZE+3); g != e {
		t.Error(g, e)
		return
	}

	isz, err := inner.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := isz, int64(3); g != e {
		t.Error(g, e)
		return
	}

	rbuf := make([]byte, 3)
	if n, err = outer.ReadAt(rbuf, 0); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := n, len(rbuf); g != e {
		t.Error(g, e)
		return
	}

	if g, e := rbuf, make([]byte, 3); !bytes.Equal(g, e) {
		t.Error(g, e)
	}

	rbuf = make([]byte, 3)
	if n, err = outer.ReadAt(rbuf, HDR_SIZE); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := n, len(rbuf); g != e {
		t.Error(g, e)
		return
	}

	if g, e := rbuf, []byte{2, 5, 11}; !bytes.Equal(g, e) {
		t.Error(g, e)
	}

	rbuf = make([]byte, 3)
	if n, err = inner.ReadAt(rbuf, 0); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := n, len(rbuf); g != e {
		t.Error(g, e)
		return
	}

	if g, e := rbuf, []byte{2, 5, 11}; !bytes.Equal(g, e) {
		t.Error(g, e)
	}

	b = []byte{22, 55, 111}
	if n, err = inner.WriteAt(b, LONG_OFF); err != nil {
		t.Error(err)
		return
	}

	if g, e := n, len(b); g != e {
		t.Error(g, e)
		return
	}

	osz, err = outer.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := osz, int64(HDR_SIZE+LONG_OFF+3); g != e {
		t.Error(g, e)
		return
	}

	isz, err = inner.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := isz, int64(LONG_OFF+3); g != e {
		t.Error(g, e)
		return
	}

	rbuf = make([]byte, 3)
	if n, err = outer.ReadAt(rbuf, HDR_SIZE+LONG_OFF); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := n, len(rbuf); g != e {
		t.Error(g, e)
		return
	}

	if g, e := rbuf, []byte{22, 55, 111}; !bytes.Equal(g, e) {
		t.Error(g, e)
	}

	rbuf = make([]byte, 3)
	if n, err = inner.ReadAt(rbuf, LONG_OFF); err != nil && n == 0 {
		t.Error(err)
		return
	}

	if g, e := n, len(rbuf); g != e {
		t.Error(g, e)
		return
	}

	if g, e := rbuf, []byte{22, 55, 111}; !bytes.Equal(g, e) {
		t.Error(g, e)
		return
	}

	if err = inner.Truncate(1); err != nil {
		t.Error(err)
		return
	}

	isz, err = inner.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := isz, int64(1); g != e {
		t.Error(g, e)
		return
	}

	osz, err = outer.Size()
	if err != nil {
		t.Error(err)
		return
	}

	if g, e := osz, int64(HDR_SIZE+1); g != e {
		t.Error(g, e)
		return
	}
}

func TestFileReadAtHole(t *testing.T) {
	testFileReadAtHole(t, newFileFiler)
	testFileReadAtHole(t, newOSFileFiler)
	testFileReadAtHole(t, newMemFiler)
	testFileReadAtHole(t, nwBitFiler)
	testFileReadAtHole(t, newRollbackFiler)
}

func testFileReadAtHole(t *testing.T, nf newFunc) {
	f := nf()
	t.Log(f.Name())
	defer func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	}()

	if _, ok := f.(*RollbackFiler); ok {
		if err := f.BeginUpdate(); err != nil {
			t.Fatal(err)
		}

		defer func() {
			if err := f.EndUpdate(); err != nil {
				t.Error(err)
			}
		}()
	}

	n, err := f.WriteAt([]byte{1}, 40000)
	if err != nil {
		t.Error(err)
		return
	}

	if n != 1 {
		t.Error(n)
		return
	}

	n, err = f.ReadAt(make([]byte, 1000), 20000)
	if err != nil {
		t.Error(err)
		return
	}

	if n != 1000 {
		t.Error(n)
		return
	}
}

func BenchmarkMemFilerWrSeq(b *testing.B) {
	b.StopTimer()
	buf := make([]byte, filerTestChunkSize)
	for i := range buf {
		buf[i] = byte(rand.Int())
	}
	f := newMemFiler()
	runtime.GC()
	b.StartTimer()
	var ofs int64
	for i := 0; i < b.N; i++ {
		_, err := f.WriteAt(buf, ofs)
		if err != nil {
			b.Fatal(err)
		}

		ofs = (ofs + filerTestChunkSize) % filerTotalSize
	}
}

func BenchmarkMemFilerRdSeq(b *testing.B) {
	b.StopTimer()
	buf := make([]byte, filerTestChunkSize)
	for i := range buf {
		buf[i] = byte(rand.Int())
	}
	f := newMemFiler()
	var ofs int64
	for i := 0; i < b.N; i++ {
		_, err := f.WriteAt(buf, ofs)
		if err != nil {
			b.Fatal(err)
		}

		ofs = (ofs + filerTestChunkSize) % filerTotalSize
	}
	runtime.GC()
	b.StartTimer()
	ofs = 0
	for i := 0; i < b.N; i++ {
		n, err := f.ReadAt(buf, ofs)
		if err != nil && n == 0 {
			b.Fatal(err)
		}

		ofs = (ofs + filerTestChunkSize) % filerTotalSize
	}
}

func BenchmarkMemFilerWrRand(b *testing.B) {
	b.StopTimer()
	rng := rand.New(rand.NewSource(42))
	f := newMemFiler()
	var bytes int64

	var ofs, runs []int
	for i := 0; i < b.N; i++ {
		ofs = append(ofs, rng.Intn(1<<31-1))
		runs = append(runs, rng.Intn(1<<31-1)%(2*pgSize))
	}
	data := make([]byte, 2*pgSize)
	for i := range data {
		data[i] = byte(rng.Int())
	}

	runtime.GC()
	b.StartTimer()
	for i, v := range ofs {
		n := runs[i]
		bytes += int64(n)
		f.WriteAt(data[:n], int64(v))
	}
	b.StopTimer()
}

func BenchmarkMemFilerRdRand(b *testing.B) {
	b.StopTimer()
	rng := rand.New(rand.NewSource(42))
	f := newMemFiler()
	var bytes int64

	var ofs, runs []int
	for i := 0; i < b.N; i++ {
		ofs = append(ofs, rng.Intn(1<<31-1))
		runs = append(runs, rng.Intn(1<<31-1)%(2*pgSize))
	}
	data := make([]byte, 2*pgSize)
	for i := range data {
		data[i] = byte(rng.Int())
	}

	for i, v := range ofs {
		n := runs[i]
		bytes += int64(n)
		f.WriteAt(data[:n], int64(v))
	}

	runtime.GC()
	b.StartTimer()
	for _, v := range ofs {
		f.ReadAt(data, int64(v))
	}
	b.StopTimer()
}
