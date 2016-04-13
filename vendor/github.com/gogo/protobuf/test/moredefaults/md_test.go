// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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

package moredefaults

import (
	"testing"

	test "github.com/gogo/protobuf/test/example"
)

func TestDefaults(t *testing.T) {
	b := MoreDefaultsB{}
	aa := test.A{}
	a := &MoreDefaultsA{}
	b2 := a.GetB2()
	a2 := a.GetA2()
	if a.GetField1() != 1234 {
		t.Fatalf("Field1 wrong")
	}
	if a.GetField2() != 0 {
		t.Fatalf("Field2 wrong")
	}
	if a.GetB1() != nil {
		t.Fatalf("B1 wrong")
	}
	if b2.GetField1() != b.GetField1() {
		t.Fatalf("B2 wrong")
	}
	if a.GetA1() != nil {
		t.Fatalf("A1 wrong")
	}
	if a2.GetNumber() != aa.GetNumber() {
		t.Fatalf("A2 wrong")
	}
}
