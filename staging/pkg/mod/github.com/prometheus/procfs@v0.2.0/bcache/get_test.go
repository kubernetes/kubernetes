// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bcache

import (
	"math"
	"testing"
)

func TestFSBcacheStats(t *testing.T) {
	bcache, err := NewFS("../fixtures/sys")
	if err != nil {
		t.Fatalf("failed to access bcache fs: %v", err)
	}
	stats, err := bcache.Stats()
	if err != nil {
		t.Fatalf("failed to parse bcache stats: %v", err)
	}

	tests := []struct {
		name   string
		bdevs  int
		caches int
	}{
		{
			name:   "deaddd54-c735-46d5-868e-f331c5fd7c74",
			bdevs:  1,
			caches: 1,
		},
	}

	const expect = 1

	if l := len(stats); l != expect {
		t.Fatalf("unexpected number of bcache stats: %d", l)
	}
	if l := len(tests); l != expect {
		t.Fatalf("unexpected number of tests: %d", l)
	}

	for i, tt := range tests {
		if want, got := tt.name, stats[i].Name; want != got {
			t.Errorf("unexpected stats name:\nwant: %q\nhave: %q", want, got)
		}

		if want, got := tt.bdevs, len(stats[i].Bdevs); want != got {
			t.Errorf("unexpected value allocated:\nwant: %d\nhave: %d", want, got)
		}

		if want, got := tt.caches, len(stats[i].Caches); want != got {
			t.Errorf("unexpected value allocated:\nwant: %d\nhave: %d", want, got)
		}
	}
}

func TestDehumanizeTests(t *testing.T) {
	dehumanizeTests := []struct {
		in      []byte
		out     uint64
		invalid bool
	}{
		{
			in:  []byte("542k"),
			out: 555008,
		},
		{
			in:  []byte("322M"),
			out: 337641472,
		},
		{
			in:  []byte("1.1k"),
			out: 1124,
		},
		{
			in:  []byte("1.9k"),
			out: 1924,
		},
		{
			in:  []byte("1.10k"),
			out: 2024,
		},
		{
			in:      []byte(""),
			out:     0,
			invalid: true,
		},
	}
	for _, tst := range dehumanizeTests {
		got, err := dehumanize(tst.in)
		if tst.invalid && err == nil {
			t.Error("expected an error, but none occurred")
		}
		if !tst.invalid && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if got != tst.out {
			t.Errorf("dehumanize: '%s', want %d, got %d", tst.in, tst.out, got)
		}
	}
}

func TestParsePseudoFloatTests(t *testing.T) {
	parsePseudoFloatTests := []struct {
		in  string
		out float64
	}{
		{
			in:  "1.1",
			out: float64(1.097656),
		},
		{
			in:  "1.9",
			out: float64(1.878906),
		},
		{
			in:  "1.10",
			out: float64(1.976562),
		},
	}
	for _, tst := range parsePseudoFloatTests {
		got, err := parsePseudoFloat(tst.in)
		if err != nil || math.Abs(got-tst.out) > 0.0001 {
			t.Errorf("parsePseudoFloat: %s, want %f, got %f", tst.in, tst.out, got)
		}
	}
}

func TestPriorityStats(t *testing.T) {
	var want = PriorityStats{
		UnusedPercent:   99,
		MetadataPercent: 5,
	}
	var (
		in     string
		gotErr error
		got    PriorityStats
	)
	in = "Metadata:       5%"
	gotErr = parsePriorityStats(in, &got)
	if gotErr != nil || got.MetadataPercent != want.MetadataPercent {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.MetadataPercent, got.MetadataPercent)
	}

	in = "Unused:         99%"
	gotErr = parsePriorityStats(in, &got)
	if gotErr != nil || got.UnusedPercent != want.UnusedPercent {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.UnusedPercent, got.UnusedPercent)
	}
}

func TestWritebackRateDebug(t *testing.T) {
	var want = WritebackRateDebugStats{
		Rate:         1765376,
		Dirty:        21789409280,
		Target:       21894266880,
		Proportional: -1124,
		Integral:     -257624,
		Change:       2648,
		NextIO:       -150773,
	}
	var (
		in     string
		gotErr error
		got    WritebackRateDebugStats
	)
	in = "rate:           1.7M/sec"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Rate != want.Rate {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Rate, got.Rate)
	}

	in = "dirty:           20.3G"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Dirty != want.Dirty {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Dirty, got.Dirty)
	}

	in = "target:           20.4G"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Target != want.Target {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Target, got.Target)
	}

	in = "proportional:           -1.1k"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Proportional != want.Proportional {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Proportional, got.Proportional)
	}

	in = "integral:           -251.6k"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Integral != want.Integral {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Integral, got.Integral)
	}

	in = "change:           2.6k/sec"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.Change != want.Change {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.Change, got.Change)
	}

	in = "next io:           -150773ms"
	gotErr = parseWritebackRateDebug(in, &got)
	if gotErr != nil || got.NextIO != want.NextIO {
		t.Errorf("parsePriorityStats: '%s', want %d, got %d", in, want.NextIO, got.NextIO)
	}
}
