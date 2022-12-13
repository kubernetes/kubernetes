/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prometheusextension

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// Float64Slice is a slice of float64 that sorts by magnitude
type Float64Slice []float64

func (fs Float64Slice) Len() int { return len(fs) }

func (fs Float64Slice) Less(i, j int) bool { return math.Abs(fs[i]) < math.Abs(fs[j]) }

func (fs Float64Slice) Swap(i, j int) { fs[i], fs[j] = fs[j], fs[i] }

// weightedHistogramSpecFunc returns a WeightedHistogram and the upper bounds
// to expect it to have.
// Every invocation of the same function returns the same histogram.
type weightedHistogramSpecFunc func() (wh WeightedObserver, upperBounds []float64)

// exerciseWeightedHistograms exercises a given collection of WeightedHistograms.
// Each histogram is given by a function that returns it, so that we can test
// that the Vec functions return the same result for the same input.
// For each histogram, with N upper bounds, the exercise provides two 2N+1 values:
// the upper bounds and values halfway between them (extended below the bottom and above
// the top).  For the Jth value, there are J*m1 calls to ObserveWithWeight with m1
// chosen so that m1 * sum[1 <= J <= 2N+1] J is large enough to trigger several
// considerations of spilling from sumHot to sumCold.
// The ObserveWithWeight calls to the various histograms are interleaved to check
// that there is no interference between them.
func exerciseWeightedHistograms(t *testing.T, whSpecs ...weightedHistogramSpecFunc) {
	var whos []weightedHistogramObs
	expectations := []whExerciseExpectation{}
	// Create expectations and specs of calls ot ObserveWithWeight
	for whIdx, whSpec := range whSpecs {
		wh, upperBounds := whSpec()
		numUBs := len(upperBounds)
		numWhos := numUBs*2 + 1
		multSum := (numWhos * (numWhos + 1)) / 2
		m1 := (-10 * initialHotCount) / multSum
		terms := Float64Slice{}
		ee := whExerciseExpectation{wh: wh,
			upperBounds: upperBounds,
			buckets:     make([]uint64, numUBs),
		}
		addWHOs := func(val float64, weight uint64, mult, idx int) {
			multipliedWeight := weight * uint64(mult)
			terms = append(terms, val*float64(multipliedWeight))
			t.Logf("For WH %d, adding obs val=%v, weight=%v, mult=%d, idx=%d", whIdx, val, weight, mult, idx)
			for i := 0; i < mult; i++ {
				whos = append(whos, weightedHistogramObs{whSpec, val, weight})
			}
			for j := idx; j < numUBs; j++ {
				ee.buckets[j] += multipliedWeight
			}
			ee.count += multipliedWeight
		}
		for idx, ub := range upperBounds {
			var val float64
			if idx > 0 {
				val = (upperBounds[idx-1] + ub) / 2
			} else if numUBs > 1 {
				val = (3*ub - upperBounds[1]) / 2
			} else {
				val = ub - 1
			}
			addWHOs(val, (1 << rand.Intn(40)), (2*idx+1)*m1, idx)
			addWHOs(ub, (1 << rand.Intn(40)), (2*idx+2)*m1, idx)
		}
		val := upperBounds[numUBs-1] + 1
		if numUBs > 1 {
			val = (3*upperBounds[numUBs-1] - upperBounds[numUBs-2]) / 2
		}
		addWHOs(val, 1+uint64(rand.Intn(1000000)), (2*numUBs+1)*m1, numUBs)
		sort.Sort(terms)
		for _, term := range terms {
			ee.sum += term
		}
		t.Logf("Adding expectation %#+v", ee)
		expectations = append(expectations, ee)
	}
	// Do the planned calls on ObserveWithWeight, in randomized order
	for len(whos) > 0 {
		var wi weightedHistogramObs
		whos, wi = whosPick(whos)
		wh, _ := wi.whSpec()
		wh.ObserveWithWeight(wi.val, wi.weight)
		// t.Logf("ObserveWithWeight(%v, %v) => %#+v", wi.val, wi.weight, wh)
	}
	// Check expectations
	for idx, ee := range expectations {
		wh := ee.wh
		whAsMetric := wh.(prometheus.Metric)
		var metric dto.Metric
		whAsMetric.Write(&metric)
		actualHist := metric.GetHistogram()
		if actualHist == nil {
			t.Errorf("At idx=%d, Write produced nil Histogram", idx)
		}
		actualCount := actualHist.GetSampleCount()
		if actualCount != ee.count {
			t.Errorf("At idx=%d, expected count %v but got %v", idx, ee.count, actualCount)

		}
		actualBuckets := actualHist.GetBucket()
		if len(ee.buckets) != len(actualBuckets) {
			t.Errorf("At idx=%d, expected %v buckets but got %v", idx, len(ee.buckets), len(actualBuckets))
		}
		for j := 0; j < len(ee.buckets) && j < len(actualBuckets); j++ {
			actualUB := actualBuckets[j].GetUpperBound()
			actualCount := actualBuckets[j].GetCumulativeCount()
			if ee.upperBounds[j] != actualUB {
				t.Errorf("At idx=%d, bucket %d, expected upper bound %v but got %v, err=%v", idx, j, ee.upperBounds[j], actualUB, actualUB-ee.upperBounds[j])
			}
			if ee.buckets[j] != actualCount {
				t.Errorf("At idx=%d, bucket %d expected count %d but got %d", idx, j, ee.buckets[j], actualCount)
			}
		}
		actualSum := actualHist.GetSampleSum()
		num := math.Abs(actualSum - ee.sum)
		den := math.Max(math.Abs(actualSum), math.Abs(ee.sum))
		if num > den/1e14 {
			t.Errorf("At idx=%d, expected sum %v but got %v, err=%v", idx, ee.sum, actualSum, actualSum-ee.sum)
		}
	}
}

// weightedHistogramObs prescribes a call on WeightedHistogram::ObserveWithWeight
type weightedHistogramObs struct {
	whSpec weightedHistogramSpecFunc
	val    float64
	weight uint64
}

// whExerciseExpectation is the expected result from exercising a WeightedHistogram
type whExerciseExpectation struct {
	wh          WeightedObserver
	upperBounds []float64
	buckets     []uint64
	sum         float64
	count       uint64
}

func whosPick(whos []weightedHistogramObs) ([]weightedHistogramObs, weightedHistogramObs) {
	n := len(whos)
	if n < 2 {
		return whos[:0], whos[0]
	}
	idx := rand.Intn(n)
	ans := whos[idx]
	whos[idx] = whos[n-1]
	return whos[:n-1], ans
}

func TestOneWeightedHistogram(t *testing.T) {
	// First, some literal test cases
	for _, testCase := range []struct {
		name        string
		upperBounds []float64
	}{
		{"one bucket", []float64{0.07}},
		{"two buckets", []float64{0.07, 0.13}},
		{"three buckets", []float64{0.07, 0.13, 1e6}},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			wh, err := NewWeightedHistogram(WeightedHistogramOpts{
				Namespace: "testns",
				Subsystem: "testsubsys",
				Name:      "testhist",
				Help:      "Me",
				Buckets:   testCase.upperBounds,
			})
			if err != nil {
				t.Error(err)
			}
			exerciseWeightedHistograms(t, func() (WeightedObserver, []float64) { return wh, testCase.upperBounds })
		})
	}
	// Now, some randomized test cases
	for i := 0; i < 10; i++ {
		name := fmt.Sprintf("random_case_%d", i)
		t.Run(name, func(t *testing.T) {
			nBounds := rand.Intn(10) + 1
			ubs := []float64{}
			var bound float64
			for j := 0; j < nBounds; j++ {
				bound += rand.Float64()
				ubs = append(ubs, bound)
			}
			wh, err := NewWeightedHistogram(WeightedHistogramOpts{
				Namespace:   "testns",
				Subsystem:   "testsubsys",
				Name:        name,
				Help:        "Me",
				Buckets:     ubs,
				ConstLabels: prometheus.Labels{"k0": "v0"},
			})
			if err != nil {
				t.Error(err)
			}
			exerciseWeightedHistograms(t, func() (WeightedObserver, []float64) { return wh, ubs })
		})
	}
}

func TestWeightedHistogramVec(t *testing.T) {
	ubs1 := []float64{0.07, 1.3, 1e6}
	vec1 := NewWeightedHistogramVec(WeightedHistogramOpts{
		Namespace:   "testns",
		Subsystem:   "testsubsys",
		Name:        "vec1",
		Help:        "Me",
		Buckets:     ubs1,
		ConstLabels: prometheus.Labels{"k0": "v0"},
	}, "k1", "k2")
	gen1 := func(lvs ...string) func() (WeightedObserver, []float64) {
		return func() (WeightedObserver, []float64) { return vec1.WithLabelValues(lvs...), ubs1 }
	}
	ubs2 := []float64{-0.03, 0.71, 1e9}
	vec2 := NewWeightedHistogramVec(WeightedHistogramOpts{
		Namespace:   "testns",
		Subsystem:   "testsubsys",
		Name:        "vec2",
		Help:        "Me",
		Buckets:     ubs2,
		ConstLabels: prometheus.Labels{"j0": "u0"},
	}, "j1", "j2")
	gen2 := func(lvs ...string) func() (WeightedObserver, []float64) {
		varLabels := prometheus.Labels{}
		varLabels["j1"] = lvs[0]
		varLabels["j2"] = lvs[1]
		return func() (WeightedObserver, []float64) { return vec2.With(varLabels), ubs2 }
	}
	exerciseWeightedHistograms(t,
		gen1("v11", "v21"),
		gen1("v12", "v21"),
		gen1("v12", "v22"),
		gen2("a", "b"),
		gen2("a", "c"),
		gen2("b", "c"),
	)
}

func BenchmarkWeightedHistogram(b *testing.B) {
	b.StopTimer()
	wh, err := NewWeightedHistogram(WeightedHistogramOpts{
		Namespace: "testns",
		Subsystem: "testsubsys",
		Name:      "testhist",
		Help:      "Me",
		Buckets:   []float64{1, 2, 4, 8, 16},
	})
	if err != nil {
		b.Error(err)
	}
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		wh.ObserveWithWeight(float64(x), uint64(i)%32+1)
		x = (x + i) % 20
	}
}

func BenchmarkHistogram(b *testing.B) {
	b.StopTimer()
	hist := prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "testns",
		Subsystem: "testsubsys",
		Name:      "testhist",
		Help:      "Me",
		Buckets:   []float64{1, 2, 4, 8, 16},
	})
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		hist.Observe(float64(x))
		x = (x + i) % 20
	}
}
