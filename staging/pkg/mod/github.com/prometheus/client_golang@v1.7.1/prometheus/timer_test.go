// Copyright 2016 The Prometheus Authors
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
	"testing"

	dto "github.com/prometheus/client_model/go"
)

func TestTimerObserve(t *testing.T) {
	var (
		his   = NewHistogram(HistogramOpts{Name: "test_histogram"})
		sum   = NewSummary(SummaryOpts{Name: "test_summary"})
		gauge = NewGauge(GaugeOpts{Name: "test_gauge"})
	)

	func() {
		hisTimer := NewTimer(his)
		sumTimer := NewTimer(sum)
		gaugeTimer := NewTimer(ObserverFunc(gauge.Set))
		defer hisTimer.ObserveDuration()
		defer sumTimer.ObserveDuration()
		defer gaugeTimer.ObserveDuration()
	}()

	m := &dto.Metric{}
	his.Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for histogram, got %d", want, got)
	}
	m.Reset()
	sum.Write(m)
	if want, got := uint64(1), m.GetSummary().GetSampleCount(); want != got {
		t.Errorf("want %d observations for summary, got %d", want, got)
	}
	m.Reset()
	gauge.Write(m)
	if got := m.GetGauge().GetValue(); got <= 0 {
		t.Errorf("want value > 0 for gauge, got %f", got)
	}
}

func TestTimerEmpty(t *testing.T) {
	emptyTimer := NewTimer(nil)
	emptyTimer.ObserveDuration()
	// Do nothing, just demonstrate it works without panic.
}

func TestTimerConditionalTiming(t *testing.T) {
	var (
		his = NewHistogram(HistogramOpts{
			Name: "test_histogram",
		})
		timeMe = true
		m      = &dto.Metric{}
	)

	timedFunc := func() {
		timer := NewTimer(ObserverFunc(func(v float64) {
			if timeMe {
				his.Observe(v)
			}
		}))
		defer timer.ObserveDuration()
	}

	timedFunc() // This will time.
	his.Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for histogram, got %d", want, got)
	}

	timeMe = false
	timedFunc() // This will not time again.
	m.Reset()
	his.Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for histogram, got %d", want, got)
	}
}

func TestTimerByOutcome(t *testing.T) {
	var (
		his = NewHistogramVec(
			HistogramOpts{Name: "test_histogram"},
			[]string{"outcome"},
		)
		outcome = "foo"
		m       = &dto.Metric{}
	)

	timedFunc := func() {
		timer := NewTimer(ObserverFunc(func(v float64) {
			his.WithLabelValues(outcome).Observe(v)
		}))
		defer timer.ObserveDuration()

		if outcome == "foo" {
			outcome = "bar"
			return
		}
		outcome = "foo"
	}

	timedFunc()
	his.WithLabelValues("foo").(Histogram).Write(m)
	if want, got := uint64(0), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'foo' histogram, got %d", want, got)
	}
	m.Reset()
	his.WithLabelValues("bar").(Histogram).Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'bar' histogram, got %d", want, got)
	}

	timedFunc()
	m.Reset()
	his.WithLabelValues("foo").(Histogram).Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'foo' histogram, got %d", want, got)
	}
	m.Reset()
	his.WithLabelValues("bar").(Histogram).Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'bar' histogram, got %d", want, got)
	}

	timedFunc()
	m.Reset()
	his.WithLabelValues("foo").(Histogram).Write(m)
	if want, got := uint64(1), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'foo' histogram, got %d", want, got)
	}
	m.Reset()
	his.WithLabelValues("bar").(Histogram).Write(m)
	if want, got := uint64(2), m.GetHistogram().GetSampleCount(); want != got {
		t.Errorf("want %d observations for 'bar' histogram, got %d", want, got)
	}

}
