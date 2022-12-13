/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"testing"
	"time"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	testclock "k8s.io/utils/clock/testing"
)

func TestTimingHistogram(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		*TimingHistogramOpts
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:    "namespace",
				Name:         "metric_test_name",
				Subsystem:    "subsystem",
				Help:         "histogram help message",
				Buckets:      DefBuckets,
				InitialValue: 13,
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "EXPERIMENTAL: [ALPHA] histogram help message",
		},
		{
			desc: "Test deprecated",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.15.0",
				Buckets:           DefBuckets,
				InitialValue:      3,
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "EXPERIMENTAL: [ALPHA] (Deprecated since 1.15.0) histogram help message",
		},
		{
			desc: "Test hidden",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.14.0",
				Buckets:           DefBuckets,
				InitialValue:      5,
			},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "EXPERIMENTAL: histogram help message",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			t0 := time.Now()
			clk := testclock.NewFakePassiveClock(t0)
			c := NewTestableTimingHistogram(clk.Now, test.TimingHistogramOpts)
			registry.MustRegister(c)

			metricChan := make(chan prometheus.Metric)
			go func() {
				c.Collect(metricChan)
				close(metricChan)
			}()
			m1 := <-metricChan
			gm1, ok := m1.(GaugeMetric)
			if !ok || gm1 != c.PrometheusTimingHistogram {
				t.Error("Unexpected metric", m1, c.PrometheusTimingHistogram)
			}
			m2, ok := <-metricChan
			if ok {
				t.Error("Unexpected second metric", m2)
			}

			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)

			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's exercise the metric and check that it still works
			v0 := test.TimingHistogramOpts.InitialValue
			dt1 := time.Nanosecond
			t1 := t0.Add(dt1)
			clk.SetTime(t1)
			var v1 float64 = 10
			c.Set(v1)
			dt2 := time.Hour
			t2 := t1.Add(dt2)
			clk.SetTime(t2)
			var v2 float64 = 1e6
			c.Add(v2 - v1)
			dt3 := time.Microsecond
			t3 := t2.Add(dt3)
			clk.SetTime(t3)
			c.Set(0)
			expectedCount := uint64(dt1 + dt2 + dt3)
			expectedSum := float64(dt1)*v0 + float64(dt2)*v1 + float64(dt3)*v2
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				t.Logf("Considering metric family %s", mf.GetName())
				for _, m := range mf.GetMetric() {
					assert.Equalf(t, expectedCount, m.GetHistogram().GetSampleCount(), "Got %v, want %v as the sample count of metric %s", m.GetHistogram().GetSampleCount(), expectedCount, m.String())
					assert.Equalf(t, expectedSum, m.GetHistogram().GetSampleSum(), "Got %v, want %v as the sample sum of metric %s", m.GetHistogram().GetSampleSum(), expectedSum, m.String())
				}
			}
		})
	}
}

func TestTimingHistogramVec(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		*TimingHistogramOpts
		labels              []string
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:    "namespace",
				Name:         "metric_test_name",
				Subsystem:    "subsystem",
				Help:         "histogram help message",
				Buckets:      DefBuckets,
				InitialValue: 5,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "EXPERIMENTAL: [ALPHA] histogram help message",
		},
		{
			desc: "Test deprecated",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.15.0",
				Buckets:           DefBuckets,
				InitialValue:      13,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "EXPERIMENTAL: [ALPHA] (Deprecated since 1.15.0) histogram help message",
		},
		{
			desc: "Test hidden",
			TimingHistogramOpts: &TimingHistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.14.0",
				Buckets:           DefBuckets,
				InitialValue:      42,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "EXPERIMENTAL: histogram help message",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			t0 := time.Now()
			clk := testclock.NewFakePassiveClock(t0)
			c := NewTestableTimingHistogramVec(clk.Now, test.TimingHistogramOpts, test.labels)
			registry.MustRegister(c)
			var v0 float64 = 3
			cm1, err := c.WithLabelValuesChecked("1", "2")
			if err != nil {
				t.Error(err)
			}
			cm1.Set(v0)

			if test.expectedMetricCount > 0 {
				metricChan := make(chan prometheus.Metric, 2)
				c.Collect(metricChan)
				close(metricChan)
				m1 := <-metricChan
				if m1 != cm1.(prometheus.Metric) {
					t.Error("Unexpected metric", m1, cm1)
				}
				m2, ok := <-metricChan
				if ok {
					t.Error("Unexpected second metric", m2)
				}
			}

			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)
			for _, metric := range ms {
				if metric.GetHelp() != test.expectedHelp {
					assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
				}
			}

			// let's exercise the metric and verify it still works
			c.WithLabelValues("1", "3").Set(v0)
			c.WithLabelValues("2", "3").Set(v0)
			dt1 := time.Nanosecond
			t1 := t0.Add(dt1)
			clk.SetTime(t1)
			c.WithLabelValues("1", "2").Add(5.0)
			c.WithLabelValues("1", "3").Add(5.0)
			c.WithLabelValues("2", "3").Add(5.0)
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				t.Logf("Considering metric family %s", mf.String())
				assert.Equalf(t, 3, len(mf.GetMetric()), "Got %v metrics, wanted 3 as the count for family %#+v", len(mf.GetMetric()), mf)
				for _, m := range mf.GetMetric() {
					expectedCount := uint64(dt1)
					expectedSum := float64(dt1) * v0
					assert.Equalf(t, expectedCount, m.GetHistogram().GetSampleCount(), "Got %v, expected histogram sample count to equal %d for metric %s", m.GetHistogram().GetSampleCount(), expectedCount, m.String())
					assert.Equalf(t, expectedSum, m.GetHistogram().GetSampleSum(), "Got %v, expected histogram sample sum to equal %v for metric %s", m.GetHistogram().GetSampleSum(), expectedSum, m.String())
				}
			}
		})
	}
}

func TestTimingHistogramWithLabelValueAllowList(t *testing.T) {
	labelAllowValues := map[string]string{
		"namespace_subsystem_metric_allowlist_test,label_a": "allowed",
	}
	labels := []string{"label_a", "label_b"}
	opts := &TimingHistogramOpts{
		Namespace:    "namespace",
		Name:         "metric_allowlist_test",
		Subsystem:    "subsystem",
		InitialValue: 7,
	}
	var tests = []struct {
		desc               string
		labelValues        [][]string
		expectMetricValues map[string]uint64
	}{
		{
			desc:        "Test no unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"allowed", "b2"}},
			expectMetricValues: map[string]uint64{
				"allowed b1": 1.0,
				"allowed b2": 1.0,
			},
		},
		{
			desc:        "Test unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"not_allowed", "b1"}},
			expectMetricValues: map[string]uint64{
				"allowed b1":    1.0,
				"unexpected b1": 1.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			SetLabelAllowListFromCLI(labelAllowValues)
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			t0 := time.Now()
			clk := testclock.NewFakePassiveClock(t0)
			c := NewTestableTimingHistogramVec(clk.Now, opts, labels)
			registry.MustRegister(c)
			var v0 float64 = 13
			for _, lv := range test.labelValues {
				c.WithLabelValues(lv...).Set(v0)
			}

			dt1 := 3 * time.Hour
			t1 := t0.Add(dt1)
			clk.SetTime(t1)

			for _, lv := range test.labelValues {
				c.WithLabelValues(lv...).Add(1.0)
			}
			mfs, err := registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range mfs {
				if *mf.Name != BuildFQName(opts.Namespace, opts.Subsystem, opts.Name) {
					continue
				}
				mfMetric := mf.GetMetric()
				t.Logf("Consider metric family %s", mf.GetName())

				for _, m := range mfMetric {
					var aValue, bValue string
					for _, l := range m.Label {
						if *l.Name == "label_a" {
							aValue = *l.Value
						}
						if *l.Name == "label_b" {
							bValue = *l.Value
						}
					}
					labelValuePair := aValue + " " + bValue
					expectedCount, ok := test.expectMetricValues[labelValuePair]
					assert.True(t, ok, "Got unexpected label values, lable_a is %v, label_b is %v", aValue, bValue)
					expectedSum := float64(dt1) * v0 * float64(expectedCount)
					expectedCount *= uint64(dt1)
					actualCount := m.GetHistogram().GetSampleCount()
					actualSum := m.GetHistogram().GetSampleSum()
					assert.Equalf(t, expectedCount, actualCount, "Got %v, wanted %v as the count while setting label_a to %v and label b to %v", actualCount, expectedCount, aValue, bValue)
					assert.Equalf(t, expectedSum, actualSum, "Got %v, wanted %v as the sum while setting label_a to %v and label b to %v", actualSum, expectedSum, aValue, bValue)
				}
			}
		})
	}
}

func BenchmarkTimingHistogram(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	th := NewTestableTimingHistogram(func() time.Time { return now }, &TimingHistogramOpts{
		Namespace:    "testns",
		Subsystem:    "testsubsys",
		Name:         "testhist",
		Help:         "Me",
		Buckets:      []float64{1, 2, 4, 8, 16},
		InitialValue: 3,
	})
	registry := NewKubeRegistry()
	registry.MustRegister(th)
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		th.Set(float64(x))
		x = (x + i) % 23
	}
}

func BenchmarkTimingHistogramVecEltCached(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	hv := NewTestableTimingHistogramVec(func() time.Time { return now }, &TimingHistogramOpts{
		Namespace:    "testns",
		Subsystem:    "testsubsys",
		Name:         "testhist",
		Help:         "Me",
		Buckets:      []float64{1, 2, 4, 8, 16},
		InitialValue: 3,
	},
		[]string{"label1", "label2"})
	registry := NewKubeRegistry()
	registry.MustRegister(hv)
	th, err := hv.WithLabelValuesChecked("v1", "v2")
	if err != nil {
		b.Error(err)
	}
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		th.Set(float64(x))
		x = (x + i) % 23
	}
}

func BenchmarkTimingHistogramVecEltFetched(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	hv := NewTestableTimingHistogramVec(func() time.Time { return now }, &TimingHistogramOpts{
		Namespace:    "testns",
		Subsystem:    "testsubsys",
		Name:         "testhist",
		Help:         "Me",
		Buckets:      []float64{1, 2, 4, 8, 16},
		InitialValue: 3,
	},
		[]string{"label1", "label2"})
	registry := NewKubeRegistry()
	registry.MustRegister(hv)
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		hv.WithLabelValues("v1", "v2").Set(float64(x))
		x = (x + i) % 60
	}
}

func TestUnregisteredVec(t *testing.T) {
	hv := NewTestableTimingHistogramVec(time.Now, &TimingHistogramOpts{
		Namespace:    "testns",
		Subsystem:    "testsubsys",
		Name:         "testhist",
		Help:         "Me",
		Buckets:      []float64{1, 2, 4, 8, 16},
		InitialValue: 3,
	},
		[]string{"label1", "label2"})
	gauge, err := hv.WithLabelValuesChecked("v1", "v2")
	if gauge != noop {
		t.Errorf("Expected noop but got %#+v", gauge)
	}
	if !ErrIsNotRegistered(err) {
		t.Errorf("Expected errNotRegistered but got err=%v", err)
	}
}

func TestBadValues(t *testing.T) {
	hv := NewTestableTimingHistogramVec(time.Now, &TimingHistogramOpts{
		Namespace:    "testns",
		Subsystem:    "testsubsys",
		Name:         "testhist",
		Help:         "Me",
		Buckets:      []float64{1, 2, 4, 8, 16},
		InitialValue: 3,
	},
		[]string{"label1", "label2"})
	registry := NewKubeRegistry()
	registry.MustRegister(hv)
	gauge, err := hv.WithLabelValuesChecked("v1")
	if gauge != noop {
		t.Errorf("Expected noop but got %#+v", gauge)
	}
	if err == nil {
		t.Error("Expected an error but got nil")
	}
	if ErrIsNotRegistered(err) {
		t.Error("Expected an error other than errNotRegistered but got that one")
	}
}
