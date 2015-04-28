// Copyright 2014 The bufs Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufs

import (
	"fmt"
	"path"
	"runtime"
	"testing"
)

var dbg = func(s string, va ...interface{}) {
	_, fn, fl, _ := runtime.Caller(1)
	fmt.Printf("%s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
}

func Test0(t *testing.T) {
	b := New(0)
	defer func() {
		recover()
	}()

	b.Alloc(1)
	t.Fatal("unexpected success")
}

func Test1(t *testing.T) {
	b := New(1)
	expected := false
	defer func() {
		if e := recover(); e != nil && !expected {
			t.Fatal(fmt.Errorf("%v", e))
		}
	}()

	b.Alloc(1)
	expected = true
	b.Alloc(1)
	t.Fatal("unexpected success")
}

func Test2(t *testing.T) {
	b := New(1)
	expected := false
	defer func() {
		if e := recover(); e != nil && !expected {
			t.Fatal(fmt.Errorf("%v", e))
		}
	}()

	b.Alloc(1)
	b.Free()
	b.Alloc(1)
	expected = true
	b.Alloc(1)
	t.Fatal("unexpected success")
}

func Test3(t *testing.T) {
	b := New(1)
	expected := false
	defer func() {
		if e := recover(); e != nil && !expected {
			t.Fatal(fmt.Errorf("%v", e))
		}
	}()

	b.Alloc(1)
	b.Free()
	expected = true
	b.Free()
	t.Fatal("unexpected success")
}

const (
	N       = 1e5
	bufSize = 1 << 12
)

type Foo struct {
	result []byte
}

func NewFoo() *Foo {
	return &Foo{}
}

func (f *Foo) Bar(n int) {
	buf := make([]byte, n)
	sum := 0
	for _, v := range buf {
		sum += int(v)
	}
	f.result = append(f.result, byte(sum))
	f.Qux(n)
}

func (f *Foo) Qux(n int) {
	buf := make([]byte, n)
	sum := 0
	for _, v := range buf {
		sum += int(v)
	}
	f.result = append(f.result, byte(sum))
}

type FooBufs struct {
	buffers Buffers
	result  []byte
}

const maxFooDepth = 2

func NewFooBufs() *FooBufs {
	return &FooBufs{buffers: New(maxFooDepth)}
}

func (f *FooBufs) Bar(n int) {
	buf := f.buffers.Alloc(n)
	defer f.buffers.Free()

	sum := 0
	for _, v := range buf {
		sum += int(v)
	}
	f.result = append(f.result, byte(sum))
	f.Qux(n)
}

func (f *FooBufs) Qux(n int) {
	buf := f.buffers.Alloc(n)
	defer f.buffers.Free()

	sum := 0
	for _, v := range buf {
		sum += int(v)
	}
	f.result = append(f.result, byte(sum))
}

func TestFoo(t *testing.T) {
	foo := NewFoo()
	for i := 0; i < N; i++ {
		foo.Bar(bufSize)
	}
}

func TestFooBufs(t *testing.T) {
	foo := NewFooBufs()
	for i := 0; i < N; i++ {
		foo.Bar(bufSize)
	}
	t.Log("buffers.Stats()", foo.buffers.Stats())
}

func BenchmarkFoo(b *testing.B) {
	b.SetBytes(2 * bufSize)
	foo := NewFoo()
	for i := 0; i < b.N; i++ {
		foo.Bar(bufSize)
	}
}

func BenchmarkFooBufs(b *testing.B) {
	b.SetBytes(2 * bufSize)
	foo := NewFooBufs()
	for i := 0; i < b.N; i++ {
		foo.Bar(bufSize)
	}
}
