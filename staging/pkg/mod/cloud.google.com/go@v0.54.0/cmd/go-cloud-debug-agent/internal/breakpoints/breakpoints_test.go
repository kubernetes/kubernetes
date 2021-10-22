// Copyright 2016 Google LLC
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

package breakpoints

import (
	"testing"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/internal/testutil"
	cd "google.golang.org/api/clouddebugger/v2"
)

var (
	testPC1     uint64 = 0x1234
	testPC2     uint64 = 0x5678
	testPC3     uint64 = 0x3333
	testFile           = "foo.go"
	testLine    uint64 = 42
	testLine2   uint64 = 99
	testLogPC   uint64 = 0x9abc
	testLogLine uint64 = 43
	testBadPC   uint64 = 0xdef0
	testBadLine uint64 = 44
	testBP             = &cd.Breakpoint{
		Action:       "CAPTURE",
		Id:           "TestBreakpoint",
		IsFinalState: false,
		Location:     &cd.SourceLocation{Path: testFile, Line: int64(testLine)},
	}
	testBP2 = &cd.Breakpoint{
		Action:       "CAPTURE",
		Id:           "TestBreakpoint2",
		IsFinalState: false,
		Location:     &cd.SourceLocation{Path: testFile, Line: int64(testLine2)},
	}
	testLogBP = &cd.Breakpoint{
		Action:       "LOG",
		Id:           "TestLogBreakpoint",
		IsFinalState: false,
		Location:     &cd.SourceLocation{Path: testFile, Line: int64(testLogLine)},
	}
	testBadBP = &cd.Breakpoint{
		Action:       "BEEP",
		Id:           "TestBadBreakpoint",
		IsFinalState: false,
		Location:     &cd.SourceLocation{Path: testFile, Line: int64(testBadLine)},
	}
)

func TestBreakpointStore(t *testing.T) {
	p := &Program{breakpointPCs: make(map[uint64]bool)}
	bs := NewBreakpointStore(p)
	checkPCs := func(expected map[uint64]bool) {
		if !testutil.Equal(p.breakpointPCs, expected) {
			t.Errorf("got breakpoint map %v want %v", p.breakpointPCs, expected)
		}
	}
	bs.ProcessBreakpointList([]*cd.Breakpoint{testBP, testBP2, testLogBP, testBadBP})
	checkPCs(map[uint64]bool{
		testPC1:   true,
		testPC2:   true,
		testPC3:   true,
		testLogPC: true,
	})
	for _, test := range []struct {
		pc       uint64
		expected []*cd.Breakpoint
	}{
		{testPC1, []*cd.Breakpoint{testBP}},
		{testPC2, []*cd.Breakpoint{testBP}},
		{testPC3, []*cd.Breakpoint{testBP2}},
		{testLogPC, []*cd.Breakpoint{testLogBP}},
	} {
		if bps := bs.BreakpointsAtPC(test.pc); !testutil.Equal(bps, test.expected) {
			t.Errorf("BreakpointsAtPC(%x): got %v want %v", test.pc, bps, test.expected)
		}
	}
	testBP2.IsFinalState = true
	bs.ProcessBreakpointList([]*cd.Breakpoint{testBP, testBP2, testLogBP, testBadBP})
	checkPCs(map[uint64]bool{
		testPC1:   true,
		testPC2:   true,
		testPC3:   false,
		testLogPC: true,
	})
	bs.RemoveBreakpoint(testBP)
	checkPCs(map[uint64]bool{
		testPC1:   false,
		testPC2:   false,
		testPC3:   false,
		testLogPC: true,
	})
	for _, pc := range []uint64{testPC1, testPC2, testPC3} {
		if bps := bs.BreakpointsAtPC(pc); len(bps) != 0 {
			t.Errorf("BreakpointsAtPC(%x): got %v want []", pc, bps)
		}
	}
	// bs.ErrorBreakpoints should return testBadBP.
	errorBps := bs.ErrorBreakpoints()
	if len(errorBps) != 1 {
		t.Errorf("ErrorBreakpoints: got %d want 1", len(errorBps))
	} else {
		bp := errorBps[0]
		if bp.Id != testBadBP.Id {
			t.Errorf("ErrorBreakpoints: got id %q want 1", bp.Id)
		}
		if bp.Status == nil || !bp.Status.IsError {
			t.Errorf("ErrorBreakpoints: got %v, want error", bp.Status)
		}
	}
	// The error should have been removed by the last call to bs.ErrorBreakpoints.
	errorBps = bs.ErrorBreakpoints()
	if len(errorBps) != 0 {
		t.Errorf("ErrorBreakpoints: got %d want 0", len(errorBps))
	}
	// Even if testBadBP is sent in a new list, it should not be returned again.
	bs.ProcessBreakpointList([]*cd.Breakpoint{testBadBP})
	errorBps = bs.ErrorBreakpoints()
	if len(errorBps) != 0 {
		t.Errorf("ErrorBreakpoints: got %d want 0", len(errorBps))
	}
}

// Program implements the similarly-named interface in x/debug.
// ValueCollector should only call its BreakpointAtLine and DeleteBreakpoints methods.
type Program struct {
	debug.Program
	// breakpointPCs contains the state of code breakpoints -- true if the
	// breakpoint is currently set, false if it has been deleted.
	breakpointPCs map[uint64]bool
}

func (p *Program) BreakpointAtLine(file string, line uint64) ([]uint64, error) {
	var pcs []uint64
	switch {
	case file == testFile && line == testLine:
		pcs = []uint64{testPC1, testPC2}
	case file == testFile && line == testLine2:
		pcs = []uint64{testPC3}
	case file == testFile && line == testLogLine:
		pcs = []uint64{testLogPC}
	default:
		pcs = []uint64{0xbad}
	}
	for _, pc := range pcs {
		p.breakpointPCs[pc] = true
	}
	return pcs, nil
}

func (p *Program) DeleteBreakpoints(pcs []uint64) error {
	for _, pc := range pcs {
		p.breakpointPCs[pc] = false
	}
	return nil
}
