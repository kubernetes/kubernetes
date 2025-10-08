// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2015, The GoGo Authors. All rights reserved.
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

package test

import (
	fast "github.com/gogo/protobuf/vanity/test/fast"
	faster "github.com/gogo/protobuf/vanity/test/faster"
	slick "github.com/gogo/protobuf/vanity/test/slick"
	"testing"
)

func TestFast(t *testing.T) {
	_ = (&fast.A{}).Marshal
	_ = (&fast.A{}).MarshalTo
	_ = (&fast.A{}).Unmarshal
	_ = (&fast.A{}).Size

	_ = (&fast.B{}).Marshal
	_ = (&fast.B{}).MarshalTo
	_ = (&fast.B{}).Unmarshal
	_ = (&fast.B{}).Size
}

func TestFaster(t *testing.T) {
	_ = (&faster.A{}).Marshal
	_ = (&faster.A{}).MarshalTo
	_ = (&faster.A{}).Unmarshal
	_ = (&faster.A{}).Size

	_ = (&faster.A{}).Strings == ""

	_ = (&faster.B{}).Marshal
	_ = (&faster.B{}).MarshalTo
	_ = (&faster.B{}).Unmarshal
	_ = (&faster.B{}).Size

	_ = (&faster.B{}).String_ == nil
	_ = (&faster.B{}).Int64 == 0
	_ = (&faster.B{}).Int32 == nil
	if (&faster.B{}).GetInt32() != 1234 {
		t.Fatalf("expected default")
	}
}

func TestSlick(t *testing.T) {
	_ = (&slick.A{}).Marshal
	_ = (&slick.A{}).MarshalTo
	_ = (&slick.A{}).Unmarshal
	_ = (&slick.A{}).Size

	_ = (&slick.A{}).Strings == ""

	_ = (&slick.A{}).GoString
	_ = (&slick.A{}).String

	_ = (&slick.B{}).Marshal
	_ = (&slick.B{}).MarshalTo
	_ = (&slick.B{}).Unmarshal
	_ = (&slick.B{}).Size

	_ = (&slick.B{}).String_ == nil
	_ = (&slick.B{}).Int64 == 0
	_ = (&slick.B{}).Int32 == nil
	if (&slick.B{}).GetInt32() != 1234 {
		t.Fatalf("expected default")
	}
}
