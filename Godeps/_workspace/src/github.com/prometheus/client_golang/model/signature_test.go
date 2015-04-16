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
	"sync"
	"testing"
)

func TestLabelsToSignature(t *testing.T) {
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

func TestMetricToFingerprint(t *testing.T) {
	var scenarios = []struct {
		in  Metric
		out Fingerprint
	}{
		{
			in:  Metric{},
			out: 14695981039346656037,
		},
		{
			in:  Metric{"name": "garland, briggs", "fear": "love is not enough"},
			out: 12952432476264840823,
		},
	}

	for i, scenario := range scenarios {
		actual := metricToFingerprint(scenario.in)

		if actual != scenario.out {
			t.Errorf("%d. expected %d, got %d", i, scenario.out, actual)
		}
	}
}

func TestSignatureForLabels(t *testing.T) {
	var scenarios = []struct {
		in     Metric
		labels LabelNames
		out    uint64
	}{
		{
			in:     Metric{},
			labels: nil,
			out:    14695981039346656037,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: LabelNames{"fear", "name"},
			out:    12952432476264840823,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough", "foo": "bar"},
			labels: LabelNames{"fear", "name"},
			out:    12952432476264840823,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: LabelNames{},
			out:    14695981039346656037,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: nil,
			out:    14695981039346656037,
		},
	}

	for i, scenario := range scenarios {
		actual := SignatureForLabels(scenario.in, scenario.labels)

		if actual != scenario.out {
			t.Errorf("%d. expected %d, got %d", i, scenario.out, actual)
		}
	}
}

func TestSignatureWithoutLabels(t *testing.T) {
	var scenarios = []struct {
		in     Metric
		labels map[LabelName]struct{}
		out    uint64
	}{
		{
			in:     Metric{},
			labels: nil,
			out:    14695981039346656037,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: map[LabelName]struct{}{"fear": struct{}{}, "name": struct{}{}},
			out:    14695981039346656037,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough", "foo": "bar"},
			labels: map[LabelName]struct{}{"foo": struct{}{}},
			out:    12952432476264840823,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: map[LabelName]struct{}{},
			out:    12952432476264840823,
		},
		{
			in:     Metric{"name": "garland, briggs", "fear": "love is not enough"},
			labels: nil,
			out:    12952432476264840823,
		},
	}

	for i, scenario := range scenarios {
		actual := SignatureWithoutLabels(scenario.in, scenario.labels)

		if actual != scenario.out {
			t.Errorf("%d. expected %d, got %d", i, scenario.out, actual)
		}
	}
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

func benchmarkMetricToFingerprint(b *testing.B, m Metric, e Fingerprint) {
	for i := 0; i < b.N; i++ {
		if a := metricToFingerprint(m); a != e {
			b.Fatalf("expected signature of %d for %s, got %d", e, m, a)
		}
	}
}

func BenchmarkMetricToFingerprintScalar(b *testing.B) {
	benchmarkMetricToFingerprint(b, nil, 14695981039346656037)
}

func BenchmarkMetricToFingerprintSingle(b *testing.B) {
	benchmarkMetricToFingerprint(b, Metric{"first-label": "first-label-value"}, 5147259542624943964)
}

func BenchmarkMetricToFingerprintDouble(b *testing.B) {
	benchmarkMetricToFingerprint(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value"}, 18269973311206963528)
}

func BenchmarkMetricToFingerprintTriple(b *testing.B) {
	benchmarkMetricToFingerprint(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676)
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

func benchmarkMetricToFingerprintConc(b *testing.B, m Metric, e Fingerprint, concLevel int) {
	var start, end sync.WaitGroup
	start.Add(1)
	end.Add(concLevel)

	for i := 0; i < concLevel; i++ {
		go func() {
			start.Wait()
			for j := b.N / concLevel; j >= 0; j-- {
				if a := metricToFingerprint(m); a != e {
					b.Fatalf("expected signature of %d for %s, got %d", e, m, a)
				}
			}
			end.Done()
		}()
	}
	b.ResetTimer()
	start.Done()
	end.Wait()
}

func BenchmarkMetricToFingerprintTripleConc1(b *testing.B) {
	benchmarkMetricToFingerprintConc(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676, 1)
}

func BenchmarkMetricToFingerprintTripleConc2(b *testing.B) {
	benchmarkMetricToFingerprintConc(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676, 2)
}

func BenchmarkMetricToFingerprintTripleConc4(b *testing.B) {
	benchmarkMetricToFingerprintConc(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676, 4)
}

func BenchmarkMetricToFingerprintTripleConc8(b *testing.B) {
	benchmarkMetricToFingerprintConc(b, Metric{"first-label": "first-label-value", "second-label": "second-label-value", "third-label": "third-label-value"}, 15738406913934009676, 8)
}
