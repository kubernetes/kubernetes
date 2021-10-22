// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2019, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package issue503

import (
	"fmt"
	"github.com/gogo/protobuf/proto"
	"math/rand"
	"testing"
)

// Original test case submitted by issue creator
func TestOOMIssue(t *testing.T) {
	// all repeated filed total has 10+10 elements
	a := &Foo{
		Num1: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		Num2: []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	}
	dat, err := a.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	dst := &Foo{}
	if err := dst.Unmarshal(dat); err != nil {
		t.Fatal(err)
	}
	if !a.Equal(dst) {
		t.Fatal("message is not equals")
	}
	fmt.Printf("%v\n", dst.String())
}

func BenchmarkOOMIssue(b *testing.B) {
	popr := rand.New(rand.NewSource(616))
	total := 0
	pops := make([]*Foo, 10000)
	for i := 0; i < 10000; i++ {
		pops[i] = NewPopulatedFoo(popr, false)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dAtA, err := proto.Marshal(pops[i%10000])
		if err != nil {
			panic(err)
		}
		total += len(dAtA)
	}
	b.SetBytes(int64(total / b.N))
}
