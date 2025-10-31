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

package defaultcheck

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func testDefaultConflict(t *testing.T, name string) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", name+".proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove(name + ".pb.go"); err != nil {
			t.Error(err)
		}
	}
	t.Logf("received expected error = %v and output = %v", err, string(data))
}

func TestNullableDefault(t *testing.T) {
	testDefaultConflict(t, "nc")
}

func TestNullableExtension(t *testing.T) {
	testDefaultConflict(t, "nx")
}

func TestNullableEnum(t *testing.T) {
	testDefaultConflict(t, "ne")
}

func TestFaceDefault(t *testing.T) {
	testDefaultConflict(t, "df")
}

func TestNoGettersDefault(t *testing.T) {
	testDefaultConflict(t, "dg")
}
