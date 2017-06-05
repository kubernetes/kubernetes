// Copyright (c) 2015, Vastech SA (PTY) LTD. All rights reserved.
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

package issue42

import (
	"bytes"
	"github.com/gogo/protobuf/proto"
	math_rand "math/rand"
	"testing"
	time "time"
)

func TestIssue42Order(t *testing.T) {
	unordered := NewPopulatedUnorderedFields(math_rand.New(math_rand.NewSource(time.Now().UnixNano())), false)
	udata, err := proto.Marshal(unordered)
	if err != nil {
		t.Fatal(err)
	}
	ordered := &OrderedFields{}
	if err = proto.Unmarshal(udata, ordered); err != nil {
		t.Fatal(err)
	}
	data, err := proto.Marshal(ordered)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(udata, data) {
		t.Fatalf("expected data to be marshaled in the same order, please sort fields before marshaling")
	}
}
