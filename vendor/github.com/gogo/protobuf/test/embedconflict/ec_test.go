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

package embedconflict

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestEmbedConflict(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "ec.proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove("ec.pb.go"); err != nil {
			t.Error(err)
		}
	}
	t.Logf("received expected error = %v and output = %v", err, string(data))
}

func TestEmbedMarshaler(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "em.proto")
	data, err := cmd.CombinedOutput()
	dataStr := string(data)
	t.Logf("received error = %v and output = %v", err, dataStr)
	if !strings.Contains(dataStr, "WARNING: found non-") || !strings.Contains(dataStr, "unsafe_marshaler") {
		t.Errorf("Expected WARNING: found non-[marshaler unsafe_marshaler] C with embedded marshaler D")
	}
	if err = os.Remove("em.pb.go"); err != nil {
		t.Error(err)
	}
}

func TestEmbedExtend(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "ee.proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove("ee.pb.go"); err != nil {
			t.Error(err)
		}
	}
	t.Logf("received expected error = %v and output = %v", err, string(data))
}

func TestCustomName(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "en.proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove("en.pb.go"); err != nil {
			t.Error(err)
		}
	}
	t.Logf("received expected error = %v and output = %v", err, string(data))
}

func TestRepeatedEmbed(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "er.proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove("er.pb.go"); err != nil {
			t.Error(err)
		}
	}
	dataStr := string(data)
	t.Logf("received error = %v and output = %v", err, dataStr)
	warning := "ERROR: found repeated embedded field B in message A"
	if !strings.Contains(dataStr, warning) {
		t.Errorf("Expected " + warning)
	}
}

func TestTakesTooLongToDebug(t *testing.T) {
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", "eb.proto")
	data, err := cmd.CombinedOutput()
	if err == nil && !strings.Contains(string(data), "Plugin failed with status code 1") {
		t.Errorf("Expected error, got: %s", data)
		if err = os.Remove("eb.pb.go"); err != nil {
			t.Error(err)
		}
	}
	dataStr := string(data)
	t.Logf("received error = %v and output = %v", err, dataStr)
	warning := "ERROR: found embedded bytes field"
	if !strings.Contains(dataStr, warning) {
		t.Errorf("Expected " + warning)
	}
}
