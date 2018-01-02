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
	"sync"
	"testing"
)

func BenchmarkCounterWithLabelValues(b *testing.B) {
	m := NewCounterVec(
		CounterOpts{
			Name: "benchmark_counter",
			Help: "A counter to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.WithLabelValues("eins", "zwei", "drei").Inc()
	}
}

func BenchmarkCounterWithLabelValuesConcurrent(b *testing.B) {
	m := NewCounterVec(
		CounterOpts{
			Name: "benchmark_counter",
			Help: "A counter to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	wg := sync.WaitGroup{}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			for j := 0; j < b.N/10; j++ {
				m.WithLabelValues("eins", "zwei", "drei").Inc()
			}
			wg.Done()
		}()
	}
	wg.Wait()
}

func BenchmarkCounterWithMappedLabels(b *testing.B) {
	m := NewCounterVec(
		CounterOpts{
			Name: "benchmark_counter",
			Help: "A counter to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.With(Labels{"two": "zwei", "one": "eins", "three": "drei"}).Inc()
	}
}

func BenchmarkCounterWithPreparedMappedLabels(b *testing.B) {
	m := NewCounterVec(
		CounterOpts{
			Name: "benchmark_counter",
			Help: "A counter to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	labels := Labels{"two": "zwei", "one": "eins", "three": "drei"}
	for i := 0; i < b.N; i++ {
		m.With(labels).Inc()
	}
}

func BenchmarkCounterNoLabels(b *testing.B) {
	m := NewCounter(CounterOpts{
		Name: "benchmark_counter",
		Help: "A counter to benchmark it.",
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Inc()
	}
}

func BenchmarkGaugeWithLabelValues(b *testing.B) {
	m := NewGaugeVec(
		GaugeOpts{
			Name: "benchmark_gauge",
			Help: "A gauge to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.WithLabelValues("eins", "zwei", "drei").Set(3.1415)
	}
}

func BenchmarkGaugeNoLabels(b *testing.B) {
	m := NewGauge(GaugeOpts{
		Name: "benchmark_gauge",
		Help: "A gauge to benchmark it.",
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Set(3.1415)
	}
}

func BenchmarkSummaryWithLabelValues(b *testing.B) {
	m := NewSummaryVec(
		SummaryOpts{
			Name:       "benchmark_summary",
			Help:       "A summary to benchmark it.",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.WithLabelValues("eins", "zwei", "drei").Observe(3.1415)
	}
}

func BenchmarkSummaryNoLabels(b *testing.B) {
	m := NewSummary(SummaryOpts{
		Name:       "benchmark_summary",
		Help:       "A summary to benchmark it.",
		Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
	},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Observe(3.1415)
	}
}

func BenchmarkHistogramWithLabelValues(b *testing.B) {
	m := NewHistogramVec(
		HistogramOpts{
			Name: "benchmark_histogram",
			Help: "A histogram to benchmark it.",
		},
		[]string{"one", "two", "three"},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.WithLabelValues("eins", "zwei", "drei").Observe(3.1415)
	}
}

func BenchmarkHistogramNoLabels(b *testing.B) {
	m := NewHistogram(HistogramOpts{
		Name: "benchmark_histogram",
		Help: "A histogram to benchmark it.",
	},
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Observe(3.1415)
	}
}
