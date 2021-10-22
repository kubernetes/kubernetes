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

package dwarf_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/elf"
)

var (
	pcspTempDir    string
	pcsptestBinary string
)

func doPCToSPTest(self bool) bool {
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
	if pcsptestBinary != "" {
		return true
	}
	var err error
	pcspTempDir, err = ioutil.TempDir("", "pcsptest")
	if err != nil {
		panic(err)
	}
	if strings.Contains(pcspTempDir, " ") {
		panic("unexpected space in tempdir")
	}
	// This command builds pcsptest from testdata/pcsptest.go.
	pcsptestBinary = filepath.Join(pcspTempDir, "pcsptest")
	command := fmt.Sprintf("go tool compile -o %s.6 testdata/pcsptest.go && go tool link -H %s -o %s %s.6",
		pcsptestBinary, runtime.GOOS, pcsptestBinary, pcsptestBinary)
	cmd := exec.Command("sh", "-c", command)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		panic(err)
	}
	return true
}

func endPCToSPTest() {
	if pcspTempDir != "" {
		os.RemoveAll(pcspTempDir)
		pcspTempDir = ""
		pcsptestBinary = ""
	}
}

func TestPCToSPOffset(t *testing.T) {
	t.Skip("gets a stack layout it doesn't expect")

	if !doPCToSPTest(false) {
		return
	}
	defer endPCToSPTest()

	data, err := getData(pcsptestBinary)
	if err != nil {
		t.Fatal(err)
	}
	entry, err := data.LookupFunction("main.test")
	if err != nil {
		t.Fatal("lookup startPC:", err)
	}
	startPC, ok := entry.Val(dwarf.AttrLowpc).(uint64)
	if !ok {
		t.Fatal(`DWARF data for function "main.test" has no low PC`)
	}
	endPC, ok := entry.Val(dwarf.AttrHighpc).(uint64)
	if !ok {
		t.Fatal(`DWARF data for function "main.test" has no high PC`)
	}

	const addrSize = 8 // TODO: Assumes amd64.
	const argSize = 8  // Defined by int64 arguments in test binary.

	// On 64-bit machines, the first offset must be one address size,
	// for the return PC.
	offset, err := data.PCToSPOffset(startPC)
	if err != nil {
		t.Fatal("startPC:", err)
	}
	if offset != addrSize {
		t.Fatalf("expected %d at start of function; got %d", addrSize, offset)
	}
	// On 64-bit machines, expect some 8s and some 32s. (See the
	// comments in testdata/pcsptest.go.
	// TODO: The test could be stronger, but not much unless we
	// disassemble the binary.
	count := make(map[int64]int)
	for pc := startPC; pc < endPC; pc++ {
		offset, err := data.PCToSPOffset(pc)
		if err != nil {
			t.Fatal("scanning function:", err)
		}
		count[offset]++
	}
	if len(count) != 2 {
		t.Errorf("expected 2 offset values, got %d; counts are: %v", len(count), count)
	}
	if count[addrSize] == 0 {
		t.Errorf("expected some values at offset %d; got %v", addrSize, count)
	}
	if count[addrSize+3*argSize] == 0 {
		t.Errorf("expected some values at offset %d; got %v", addrSize+3*argSize, count)
	}
}

func getData(file string) (*dwarf.Data, error) {
	switch runtime.GOOS {
	case "linux":
		f, err := elf.Open(file)
		if err != nil {
			return nil, err
		}
		dwarf, err := f.DWARF()
		if err != nil {
			return nil, err
		}
		f.Close()
		return dwarf, nil
	}
	panic("unimplemented DWARF for GOOS=" + runtime.GOOS)
}
