// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
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

package unrecognizedgroup

import (
	"github.com/gogo/protobuf/proto"
	math_rand "math/rand"
	"testing"
	time "time"
)

func TestNewOld(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	newer := NewPopulatedNewNoGroup(popr, true)
	data1, err := proto.Marshal(newer)
	if err != nil {
		panic(err)
	}
	older := &OldWithGroup{}
	if err = proto.Unmarshal(data1, older); err != nil {
		panic(err)
	}
	data2, err := proto.Marshal(older)
	if err != nil {
		panic(err)
	}
	bluer := &NewNoGroup{}
	if err := proto.Unmarshal(data2, bluer); err != nil {
		panic(err)
	}
	if err := newer.VerboseEqual(bluer); err != nil {
		t.Fatalf("%#v !VerboseProto %#v, since %v", newer, bluer, err)
	}
}

func TestOldNew(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	older := NewPopulatedOldWithGroup(popr, true)
	data1, err := proto.Marshal(older)
	if err != nil {
		panic(err)
	}
	newer := &NewNoGroup{}
	if err = proto.Unmarshal(data1, newer); err != nil {
		panic(err)
	}
	data2, err := proto.Marshal(newer)
	if err != nil {
		panic(err)
	}
	bluer := &OldWithGroup{}
	if err := proto.Unmarshal(data2, bluer); err != nil {
		panic(err)
	}
	if err := older.VerboseEqual(bluer); err != nil {
		t.Fatalf("%#v !VerboseProto %#v, since %v", older, bluer, err)
	}
}

func TestOldNewOldNew(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	older := NewPopulatedOldWithGroup(popr, true)
	data1, err := proto.Marshal(older)
	if err != nil {
		panic(err)
	}
	newer := &NewNoGroup{}
	if err = proto.Unmarshal(data1, newer); err != nil {
		panic(err)
	}
	data2, err := proto.Marshal(newer)
	if err != nil {
		panic(err)
	}
	bluer := &OldWithGroup{}
	if err = proto.Unmarshal(data2, bluer); err != nil {
		panic(err)
	}
	if err = older.VerboseEqual(bluer); err != nil {
		t.Fatalf("%#v !VerboseProto %#v, since %v", older, bluer, err)
	}

	data3, err := proto.Marshal(bluer)
	if err != nil {
		panic(err)
	}
	purple := &NewNoGroup{}
	if err = proto.Unmarshal(data3, purple); err != nil {
		panic(err)
	}
	data4, err := proto.Marshal(purple)
	if err != nil {
		panic(err)
	}
	magenta := &OldWithGroup{}
	if err := proto.Unmarshal(data4, magenta); err != nil {
		panic(err)
	}
	if err := older.VerboseEqual(magenta); err != nil {
		t.Fatalf("%#v !VerboseProto %#v, since %v", older, magenta, err)
	}
}
