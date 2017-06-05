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

package tags

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	math_rand "math/rand"
	"testing"
	"time"
)

type MyJson struct {
	MyField1 string
	MyField2 string
}

func NewPopulatedMyJson(r randyTags) *MyJson {
	this := &MyJson{}
	if r.Intn(10) != 0 {
		this.MyField1 = randStringTags(r)
	}
	if r.Intn(10) != 0 {
		this.MyField2 = randStringTags(r)
	}
	return this
}

func TestJson(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	msg1 := NewPopulatedMyJson(popr)
	data, err := json.Marshal(msg1)
	if err != nil {
		panic(err)
	}
	outside := &Outside{}
	err = json.Unmarshal(data, outside)
	if err != nil {
		panic(err)
	}
	if outside.GetField1() != msg1.MyField1 {
		t.Fatalf("proto field1 %s != %s", outside.GetField1(), msg1.MyField1)
	}
	if outside.GetField2() != msg1.MyField2 {
		t.Fatalf("proto field2 %s != %s", outside.GetField2(), msg1.MyField2)
	}
	data2, err := json.Marshal(outside)
	if err != nil {
		panic(err)
	}
	msg2 := &MyJson{}
	err = json.Unmarshal(data2, msg2)
	if err != nil {
		panic(err)
	}
	if msg2.MyField1 != msg1.MyField1 {
		t.Fatalf("proto field1 %s != %s", msg2.MyField1, msg1.MyField1)
	}
	if msg2.MyField2 != msg1.MyField2 {
		t.Fatalf("proto field2 %s != %s", msg2.MyField2, msg1.MyField2)
	}
}

func TestXml(t *testing.T) {
	s := "<Outside>Field1Value<!--Field2Value--><XXX_unrecognized></XXX_unrecognized></Outside>"
	field1 := "Field1Value"
	field2 := "Field2Value"
	msg1 := &Outside{}
	err := xml.Unmarshal([]byte(s), msg1)
	if err != nil {
		panic(err)
	}
	msg2 := &Outside{
		Inside: &Inside{
			Field1: &field1,
		},
		Field2: &field2,
	}
	if msg1.GetField1() != msg2.GetField1() {
		t.Fatalf("field1 expected %s got %s", msg2.GetField1(), msg1.GetField1())
	}
	if err != nil {
		panic(err)
	}
	data, err := xml.Marshal(msg2)
	if err != nil {
		panic(err)
	}
	if !bytes.Equal(data, []byte(s)) {
		t.Fatalf("expected %s got %s", s, string(data))
	}
}
