// Copyright 2019, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metric

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"go.opencensus.io/metric/metricdata"
)

func TestCumulative(t *testing.T) {
	r := NewRegistry()

	f, _ := r.AddFloat64Cumulative("TestCumulative",
		WithLabelKeys("k1", "k2"))
	e, _ := f.GetEntry(metricdata.LabelValue{}, metricdata.LabelValue{})
	e.Inc(5)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	e.Inc(1)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	e.Inc(1)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v2"), metricdata.NewLabelValue("k2v2"))
	e.Inc(1)
	m := r.Read()
	want := []*metricdata.Metric{
		{
			Descriptor: metricdata.Descriptor{
				Name: "TestCumulative",
				LabelKeys: []metricdata.LabelKey{
					{Key: "k1"},
					{Key: "k2"},
				},
				Type: metricdata.TypeCumulativeFloat64,
			},
			TimeSeries: []*metricdata.TimeSeries{
				{
					LabelValues: []metricdata.LabelValue{
						{}, {},
					},
					Points: []metricdata.Point{
						metricdata.NewFloat64Point(time.Time{}, 5),
					},
				},
				{
					LabelValues: []metricdata.LabelValue{
						metricdata.NewLabelValue("k1v1"),
						{},
					},
					Points: []metricdata.Point{
						metricdata.NewFloat64Point(time.Time{}, 2),
					},
				},
				{
					LabelValues: []metricdata.LabelValue{
						metricdata.NewLabelValue("k1v2"),
						metricdata.NewLabelValue("k2v2"),
					},
					Points: []metricdata.Point{
						metricdata.NewFloat64Point(time.Time{}, 1),
					},
				},
			},
		},
	}
	canonicalize(m)
	canonicalize(want)
	if diff := cmp.Diff(m, want, cmp.Comparer(ignoreTimes)); diff != "" {
		t.Errorf("-got +want: %s", diff)
	}
}

func TestCumulativeConstLabel(t *testing.T) {
	r := NewRegistry()

	f, _ := r.AddFloat64Cumulative("TestCumulativeWithConstLabel",
		WithLabelKeys("k1"),
		WithConstLabel(map[metricdata.LabelKey]metricdata.LabelValue{
			{Key: "const"}:  metricdata.NewLabelValue("same"),
			{Key: "const2"}: metricdata.NewLabelValue("same2"),
		}))

	e, _ := f.GetEntry(metricdata.LabelValue{})
	e.Inc(5)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"))
	e.Inc(1)
	m := r.Read()
	want := []*metricdata.Metric{
		{
			Descriptor: metricdata.Descriptor{
				Name: "TestCumulativeWithConstLabel",
				LabelKeys: []metricdata.LabelKey{
					{Key: "const"},
					{Key: "const2"},
					{Key: "k1"}},
				Type: metricdata.TypeCumulativeFloat64,
			},
			TimeSeries: []*metricdata.TimeSeries{
				{
					LabelValues: []metricdata.LabelValue{
						metricdata.NewLabelValue("same"),
						metricdata.NewLabelValue("same2"),
						{}},
					Points: []metricdata.Point{
						metricdata.NewFloat64Point(time.Time{}, 5),
					},
				},
				{
					LabelValues: []metricdata.LabelValue{
						metricdata.NewLabelValue("same"),
						metricdata.NewLabelValue("same2"),
						metricdata.NewLabelValue("k1v1"),
					},
					Points: []metricdata.Point{
						metricdata.NewFloat64Point(time.Time{}, 1),
					},
				},
			},
		},
	}
	canonicalize(m)
	canonicalize(want)
	if diff := cmp.Diff(m, want, cmp.Comparer(ignoreTimes)); diff != "" {
		t.Errorf("-got +want: %s", diff)
	}
}

func TestCumulativeMetricDescriptor(t *testing.T) {
	r := NewRegistry()

	gf, _ := r.AddFloat64Cumulative("float64_gauge")
	compareType(gf.bm.desc.Type, metricdata.TypeCumulativeFloat64, t)
	gi, _ := r.AddInt64Cumulative("int64_gauge")
	compareType(gi.bm.desc.Type, metricdata.TypeCumulativeInt64, t)
	dgf, _ := r.AddFloat64DerivedCumulative("derived_float64_gauge")
	compareType(dgf.bm.desc.Type, metricdata.TypeCumulativeFloat64, t)
	dgi, _ := r.AddInt64DerivedCumulative("derived_int64_gauge")
	compareType(dgi.bm.desc.Type, metricdata.TypeCumulativeInt64, t)
}

func readAndCompareInt64Val(testname string, r *Registry, want int64, t *testing.T) {
	ms := r.Read()
	if got := ms[0].TimeSeries[0].Points[0].Value.(int64); got != want {
		t.Errorf("testname: %s, got = %v, want %v\n", testname, got, want)
	}
}

func TestInt64CumulativeEntry_IncNegative(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64Cumulative("bm")
	e, _ := g.GetEntry()
	e.Inc(5)
	readAndCompareInt64Val("inc", r, 5, t)
	e.Inc(-2)
	readAndCompareInt64Val("inc negative", r, 5, t)
}

func readAndCompareFloat64Val(testname string, r *Registry, want float64, t *testing.T) {
	ms := r.Read()
	if got := ms[0].TimeSeries[0].Points[0].Value.(float64); got != want {
		t.Errorf("testname: %s, got = %v, want %v\n", testname, got, want)
	}
}

func TestFloat64CumulativeEntry_IncNegative(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddFloat64Cumulative("bm")
	e, _ := g.GetEntry()
	e.Inc(5.0)
	readAndCompareFloat64Val("inc", r, 5.0, t)
	e.Inc(-2.0)
	readAndCompareFloat64Val("inc negative", r, 5.0, t)
}

func TestCumulativeWithSameNameDiffType(t *testing.T) {
	r := NewRegistry()
	r.AddInt64Cumulative("bm")
	_, gotErr := r.AddFloat64Cumulative("bm")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
	_, gotErr = r.AddInt64DerivedCumulative("bm")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
	_, gotErr = r.AddFloat64DerivedCumulative("bm")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
}

func TestCumulativeWithLabelMismatch(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64Cumulative("bm", WithLabelKeys("k1"))
	_, gotErr := g.GetEntry(metricdata.NewLabelValue("k1v2"), metricdata.NewLabelValue("k2v2"))
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errKeyValueMismatch)
	}
}

type sysUpTimeInNanoSecs struct {
	size int64
}

func (q *sysUpTimeInNanoSecs) ToInt64() int64 {
	return q.size
}

func TestInt64DerivedCumulativeEntry_Inc(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInNanoSecs{3}
	g, _ := r.AddInt64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	err := g.UpsertEntry(q.ToInt64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(3); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	q.size = 5
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(5); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestInt64DerivedCumulativeEntry_IncWithNilObj(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(nil, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestInt64DerivedCumulativeEntry_IncWithInvalidLabels(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInNanoSecs{3}
	g, _ := r.AddInt64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(q.ToInt64, metricdata.NewLabelValue("k1v1"))
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestInt64DerivedCumulativeEntry_Update(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInNanoSecs{3}
	q2 := &sysUpTimeInNanoSecs{5}
	g, _ := r.AddInt64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	g.UpsertEntry(q.ToInt64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	gotErr := g.UpsertEntry(q2.ToInt64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr != nil {
		t.Errorf("got: %v, want: nil", gotErr)
	}
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(5); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

type sysUpTimeInSeconds struct {
	size float64
}

func (q *sysUpTimeInSeconds) ToFloat64() float64 {
	return q.size
}

func TestFloat64DerivedCumulativeEntry_Inc(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInSeconds{5.0}
	g, _ := r.AddFloat64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	err := g.UpsertEntry(q.ToFloat64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(5.0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	q.size = 7
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(7.0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestFloat64DerivedCumulativeEntry_IncWithNilObj(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddFloat64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(nil, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestFloat64DerivedCumulativeEntry_IncWithInvalidLabels(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInSeconds{3}
	g, _ := r.AddFloat64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(q.ToFloat64, metricdata.NewLabelValue("k1v1"))
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestFloat64DerivedCumulativeEntry_Update(t *testing.T) {
	r := NewRegistry()
	q := &sysUpTimeInSeconds{3.0}
	q2 := &sysUpTimeInSeconds{5.0}
	g, _ := r.AddFloat64DerivedCumulative("bm", WithLabelKeys("k1", "k2"))
	g.UpsertEntry(q.ToFloat64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	gotErr := g.UpsertEntry(q2.ToFloat64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr != nil {
		t.Errorf("got: %v, want: nil", gotErr)
	}
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(5.0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}
