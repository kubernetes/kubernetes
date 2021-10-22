// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.10

package dwarf_test

// Stripped-down, simplified version of ../../gosym/pclntab_test.go

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	. "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
)

var (
	pclineTempDir    string
	pclinetestBinary string
)

func dotest(self bool) bool {
	// For now, only works on amd64 platforms.
	if runtime.GOARCH != "amd64" {
		return false
	}
	// Self test reads test binary; only works on Linux or Mac.
	if self {
		if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
			return false
		}
	}
	// Command below expects "sh", so Unix.
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return false
	}
	if pclinetestBinary != "" {
		return true
	}
	var err error
	pclineTempDir, err = ioutil.TempDir("", "pclinetest")
	if err != nil {
		panic(err)
	}
	if strings.Contains(pclineTempDir, " ") {
		panic("unexpected space in tempdir")
	}
	pclinetestBinary = filepath.Join(pclineTempDir, "pclinetest")
	command := fmt.Sprintf("go tool compile -o %s.6 testdata/pclinetest.go && go tool link -H %s -E main -o %s %s.6",
		pclinetestBinary, runtime.GOOS, pclinetestBinary, pclinetestBinary)
	cmd := exec.Command("sh", "-c", command)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		panic(err)
	}
	return true
}

func endtest() {
	if pclineTempDir != "" {
		os.RemoveAll(pclineTempDir)
		pclineTempDir = ""
		pclinetestBinary = ""
	}
}

func TestPCAndLine(t *testing.T) {
	t.Skip("This stopped working in Go 1.12")

	// TODO(jba): go1.9: use subtests
	if !dotest(false) {
		return
	}
	defer endtest()

	data, err := getData(pclinetestBinary)
	if err != nil {
		t.Fatal(err)
	}

	testLineToBreakpointPCs(t, data)
	testPCToLine(t, data)
}

func testPCToLine(t *testing.T, data *Data) {
	entry, err := data.LookupFunction("main.main")
	if err != nil {
		t.Fatal(err)
	}
	pc, ok := entry.Val(AttrLowpc).(uint64)
	if !ok {
		t.Fatal(`DWARF data for function "main" has no PC`)
	}
	for _, tt := range []struct {
		offset, want uint64
	}{
		{0, 19},
		{19, 19},
		{33, 20},
		{97, 22},
		{165, 23},
	} {
		file, line, err := data.PCToLine(pc + tt.offset)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasSuffix(file, "/pclinetest.go") {
			t.Errorf("got %s; want %s", file, "/pclinetest.go")
		}
		if line != tt.want {
			t.Errorf("line for offset %d: got %d; want %d", tt.offset, line, tt.want)
		}
	}
}

func testLineToBreakpointPCs(t *testing.T, data *Data) {
	for _, tt := range []struct {
		line uint64
		want bool
	}{
		{18, false},
		{19, true},
		{20, true},
		{21, false},
		{22, true},
		{23, true},
		{24, false},
	} {
		pcs, err := data.LineToBreakpointPCs("pclinetest.go", uint64(tt.line))
		if err != nil {
			t.Fatal(err)
		}
		if got := len(pcs) > 0; got != tt.want {
			t.Errorf("line %d: got pcs=%t, want pcs=%t", tt.line, got, tt.want)

		}
	}
}
