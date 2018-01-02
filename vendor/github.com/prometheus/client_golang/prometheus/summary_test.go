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

package prometheus

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"testing/quick"
	"time"

	dto "github.com/prometheus/client_model/go"
)

func TestSummaryWithDefaultObjectives(t *testing.T) {
	reg := NewRegistry()
	summaryWithDefaultObjectives := NewSummary(SummaryOpts{
		Name: "default_objectives",
		Help: "Test help.",
	})
	if err := reg.Register(summaryWithDefaultObjectives); err != nil {
		t.Error(err)
	}

	m := &dto.Metric{}
	if err := summaryWithDefaultObjectives.Write(m); err != nil {
		t.Error(err)
	}
	if len(m.GetSummary().Quantile) != len(DefObjectives) {
		t.Error("expected default objectives in summary")
	}
}

func TestSummaryWithoutObjectives(t *testing.T) {
	reg := NewRegistry()
	summaryWithEmptyObjectives := NewSummary(SummaryOpts{
		Name:       "empty_objectives",
		Help:       "Test help.",
		Objectives: map[float64]float64{},
	})
	if err := reg.Register(summaryWithEmptyObjectives); err != nil {
		t.Error(err)
	}

	m := &dto.Metric{}
	if err := summaryWithEmptyObjectives.Write(m); err != nil {
		t.Error(err)
	}
	if len(m.GetSummary().Quantile) != 0 {
		t.Error("expected no objectives in summary")
	}
}

func benchmarkSummaryObserve(w int, b *testing.B) {
	b.StopTimer()

	wg := new(sync.WaitGroup)
	wg.Add(w)

	g := new(sync.WaitGroup)
	g.Add(1)

	s := NewSummary(SummaryOpts{})

	for i := 0; i < w; i++ {
		go func() {
			g.Wait()

			for i := 0; i < b.N; i++ {
				s.Observe(float64(i))
			}

			wg.Done()
		}()
	}

	b.StartTimer()
	g.Done()
	wg.Wait()
}

func BenchmarkSummaryObserve1(b *testing.B) {
	benchmarkSummaryObserve(1, b)
}

func BenchmarkSummaryObserve2(b *testing.B) {
	benchmarkSummaryObserve(2, b)
}

func BenchmarkSummaryObserve4(b *testing.B) {
	benchmarkSummaryObserve(4, b)
}

func BenchmarkSummaryObserve8(b *testing.B) {
	benchmarkSummaryObserve(8, b)
}

func benchmarkSummaryWrite(w int, b *testing.B) {
	b.StopTimer()

	wg := new(sync.WaitGroup)
	wg.Add(w)

	g := new(sync.WaitGroup)
	g.Add(1)

	s := NewSummary(SummaryOpts{})

	for i := 0; i < 1000000; i++ {
		s.Observe(float64(i))
	}

	for j := 0; j < w; j++ {
		outs := make([]dto.Metric, b.N)

		go func(o []dto.Metric) {
			g.Wait()

			for i := 0; i < b.N; i++ {
				s.Write(&o[i])
			}

			wg.Done()
		}(outs)
	}

	b.StartTimer()
	g.Done()
	wg.Wait()
}

func BenchmarkSummaryWrite1(b *testing.B) {
	benchmarkSummaryWrite(1, b)
}

func BenchmarkSummaryWrite2(b *testing.B) {
	benchmarkSummaryWrite(2, b)
}

func BenchmarkSummaryWrite4(b *testing.B) {
	benchmarkSummaryWrite(4, b)
}

func BenchmarkSummaryWrite8(b *testing.B) {
	benchmarkSummaryWrite(8, b)
}

func TestSummaryConcurrency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode.")
	}

	rand.Seed(42)

	it := func(n uint32) bool {
		mutations := int(n%1e4 + 1e4)
		concLevel := int(n%5 + 1)
		total := mutations * concLevel

		var start, end sync.WaitGroup
		start.Add(1)
		end.Add(concLevel)

		sum := NewSummary(SummaryOpts{
			Name:       "test_summary",
			Help:       "helpless",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
		})

		allVars := make([]float64, total)
		var sampleSum float64
		for i := 0; i < concLevel; i++ {
			vals := make([]float64, mutations)
			for j := 0; j < mutations; j++ {
				v := rand.NormFloat64()
				vals[j] = v
				allVars[i*mutations+j] = v
				sampleSum += v
			}

			go func(vals []float64) {
				start.Wait()
				for _, v := range vals {
					sum.Observe(v)
				}
				end.Done()
			}(vals)
		}
		sort.Float64s(allVars)
		start.Done()
		end.Wait()

		m := &dto.Metric{}
		sum.Write(m)
		if got, want := int(*m.Summary.SampleCount), total; got != want {
			t.Errorf("got sample count %d, want %d", got, want)
		}
		if got, want := *m.Summary.SampleSum, sampleSum; math.Abs((got-want)/want) > 0.001 {
			t.Errorf("got sample sum %f, want %f", got, want)
		}

		objectives := make([]float64, 0, len(DefObjectives))
		for qu := range DefObjectives {
			objectives = append(objectives, qu)
		}
		sort.Float64s(objectives)

		for i, wantQ := range objectives {
			ε := DefObjectives[wantQ]
			gotQ := *m.Summary.Quantile[i].Quantile
			gotV := *m.Summary.Quantile[i].Value
			min, max := getBounds(allVars, wantQ, ε)
			if gotQ != wantQ {
				t.Errorf("got quantile %f, want %f", gotQ, wantQ)
			}
			if gotV < min || gotV > max {
				t.Errorf("got %f for quantile %f, want [%f,%f]", gotV, gotQ, min, max)
			}
		}
		return true
	}

	if err := quick.Check(it, nil); err != nil {
		t.Error(err)
	}
}

func TestSummaryVecConcurrency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode.")
	}

	rand.Seed(42)

	objectives := make([]float64, 0, len(DefObjectives))
	for qu := range DefObjectives {

		objectives = append(objectives, qu)
	}
	sort.Float64s(objectives)

	it := func(n uint32) bool {
		mutations := int(n%1e4 + 1e4)
		concLevel := int(n%7 + 1)
		vecLength := int(n%3 + 1)

		var start, end sync.WaitGroup
		start.Add(1)
		end.Add(concLevel)

		sum := NewSummaryVec(
			SummaryOpts{
				Name:       "test_summary",
				Help:       "helpless",
				Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
			},
			[]string{"label"},
		)

		allVars := make([][]float64, vecLength)
		sampleSums := make([]float64, vecLength)
		for i := 0; i < concLevel; i++ {
			vals := make([]float64, mutations)
			picks := make([]int, mutations)
			for j := 0; j < mutations; j++ {
				v := rand.NormFloat64()
				vals[j] = v
				pick := rand.Intn(vecLength)
				picks[j] = pick
				allVars[pick] = append(allVars[pick], v)
				sampleSums[pick] += v
			}

			go func(vals []float64) {
				start.Wait()
				for i, v := range vals {
					sum.WithLabelValues(string('A' + picks[i])).Observe(v)
				}
				end.Done()
			}(vals)
		}
		for _, vars := range allVars {
			sort.Float64s(vars)
		}
		start.Done()
		end.Wait()

		for i := 0; i < vecLength; i++ {
			m := &dto.Metric{}
			s := sum.WithLabelValues(string('A' + i))
			s.(Summary).Write(m)
			if got, want := int(*m.Summary.SampleCount), len(allVars[i]); got != want {
				t.Errorf("got sample count %d for label %c, want %d", got, 'A'+i, want)
			}
			if got, want := *m.Summary.SampleSum, sampleSums[i]; math.Abs((got-want)/want) > 0.001 {
				t.Errorf("got sample sum %f for label %c, want %f", got, 'A'+i, want)
			}
			for j, wantQ := range objectives {
				ε := DefObjectives[wantQ]
				gotQ := *m.Summary.Quantile[j].Quantile
				gotV := *m.Summary.Quantile[j].Value
				min, max := getBounds(allVars[i], wantQ, ε)
				if gotQ != wantQ {
					t.Errorf("got quantile %f for label %c, want %f", gotQ, 'A'+i, wantQ)
				}
				if gotV < min || gotV > max {
					t.Errorf("got %f for quantile %f for label %c, want [%f,%f]", gotV, gotQ, 'A'+i, min, max)
				}
			}
		}
		return true
	}

	if err := quick.Check(it, nil); err != nil {
		t.Error(err)
	}
}

func TestSummaryDecay(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode.")
		// More because it depends on timing than because it is particularly long...
	}

	sum := NewSummary(SummaryOpts{
		Name:       "test_summary",
		Help:       "helpless",
		MaxAge:     100 * time.Millisecond,
		Objectives: map[float64]float64{0.1: 0.001},
		AgeBuckets: 10,
	})

	m := &dto.Metric{}
	i := 0
	tick := time.NewTicker(time.Millisecond)
	for range tick.C {
		i++
		sum.Observe(float64(i))
		if i%10 == 0 {
			sum.Write(m)
			if got, want := *m.Summary.Quantile[0].Value, math.Max(float64(i)/10, float64(i-90)); math.Abs(got-want) > 20 {
				t.Errorf("%d. got %f, want %f", i, got, want)
			}
			m.Reset()
		}
		if i >= 1000 {
			break
		}
	}
	tick.Stop()
	// Wait for MaxAge without observations and make sure quantiles are NaN.
	time.Sleep(100 * time.Millisecond)
	sum.Write(m)
	if got := *m.Summary.Quantile[0].Value; !math.IsNaN(got) {
		t.Errorf("got %f, want NaN after expiration", got)
	}
}

func getBounds(vars []float64, q, ε float64) (min, max float64) {
	// TODO(beorn7): This currently tolerates an error of up to 2*ε. The
	// error must be at most ε, but for some reason, it's sometimes slightly
	// higher. That's a bug.
	n := float64(len(vars))
	lower := int((q - 2*ε) * n)
	upper := int(math.Ceil((q + 2*ε) * n))
	min = vars[0]
	if lower > 1 {
		min = vars[lower-1]
	}
	max = vars[len(vars)-1]
	if upper < len(vars) {
		max = vars[upper-1]
	}
	return
}
