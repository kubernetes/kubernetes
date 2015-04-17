// Copyright 2014 The Prometheus Authors
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

package model

import (
	"runtime"
	"testing"
)

func testLabelsToSignature(t testing.TB) {
	var scenarios = []struct {
		in  map[string]string
		out uint64
	}{
		{
			in:  map[string]string{},
			out: 14695981039346656037,
		},
		{
			in:  map[string]string{"name": "garland, briggs", "fear": "love is not enough"},
			out: 12952432476264840823,
		},
	}

	for i, scenario := range scenarios {
		actual := LabelsToSignature(scenario.in)

		if actual != scenario.out {
			t.Errorf("%d. expected %d, got %d", i, scenario.out, actual)
		}
	}
}

func TestLabelToSignature(t *testing.T) {
	testLabelsToSignature(t)
}

func TestEmptyLabelSignature(t *testing.T) {
	input := []map[string]string{nil, {}}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	alloc := ms.Alloc

	for _, labels := range input {
		LabelsToSignature(labels)
	}

	runtime.ReadMemStats(&ms)

	if got := ms.Alloc; alloc != got {
		t.Fatal("expected LabelsToSignature with empty labels not to perform allocations")
	}
}

func BenchmarkLabelToSignature(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testLabelsToSignature(b)
	}
}

func benchmarkLabelValuesToSignature(b *testing.B, l map[string]string, e uint64) {
	for i := 0; i < b.N; i++ {
		if a := LabelValuesToSignature(l); a != e {
			b.Fatalf("expected signature of %d for %s, got %d", e, l, a)
		}
	}
}

func BenchmarkLabelValuesToSignatureScalar(b *testing.B) {
	benchmarkLabelValuesToSignature(b, nil, 14695981039346656037)
}

func BenchmarkLabelValuesToSignatureSingle(b *testing.B) {
	benchmarkLabelValuesToSignature(b, map[string]string{"first-label": "first-label-value"}, 2653746141194979650)
}

func BenchmarkLabelValuesToSignatureDouble(b *testing.B) {
	benchmarkLabelValuesToSignature(b, map[string]string{"first-label": "first-label-value", "second-label": "second-label-value"}, 8893559499616767364)
}

func BenchmarkLabelValuesToSignatureTriple(b *testing.B) {
	benchmarkLabelValuesToSignature(b, map[string]string{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 1685970066862087833)
}

func benchmarkLabelToSignature(b *testing.B, l map[string]string, e uint64) {
	for i := 0; i < b.N; i++ {
		if a := LabelsToSignature(l); a != e {
			b.Fatalf("expected signature of %d for %s, got %d", e, l, a)
		}
	}
}

func BenchmarkLabelToSignatureScalar(b *testing.B) {
	benchmarkLabelToSignature(b, nil, 14695981039346656037)
}

func BenchmarkLabelToSignatureSingle(b *testing.B) {
	benchmarkLabelToSignature(b, map[string]string{"first-label": "first-label-value"}, 5147259542624943964)
}

func BenchmarkLabelToSignatureDouble(b *testing.B) {
	benchmarkLabelToSignature(b, map[string]string{"first-label": "first-label-value", "second-label": "second-label-value"}, 18269973311206963528)
}

func BenchmarkLabelToSignatureTriple(b *testing.B) {
	benchmarkLabelToSignature(b, map[string]string{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676)
}
