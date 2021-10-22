// Copyright 2018, OpenCensus Authors
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
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"go.opencensus.io/metric/metricdata"
)

func TestGauge(t *testing.T) {
	r := NewRegistry()

	f, _ := r.AddFloat64Gauge("TestGauge",
		WithLabelKeys("k1", "k2"))
	e, _ := f.GetEntry(metricdata.LabelValue{}, metricdata.LabelValue{})
	e.Set(5)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	e.Add(1)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	e.Add(1)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v2"), metricdata.NewLabelValue("k2v2"))
	e.Add(1)
	m := r.Read()
	want := []*metricdata.Metric{
		{
			Descriptor: metricdata.Descriptor{
				Name: "TestGauge",
				LabelKeys: []metricdata.LabelKey{
					{Key: "k1"},
					{Key: "k2"},
				},
				Type: metricdata.TypeGaugeFloat64,
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

func TestGaugeConstLabel(t *testing.T) {
	r := NewRegistry()

	f, _ := r.AddFloat64Gauge("TestGaugeWithConstLabel",
		WithLabelKeys("k1"),
		WithConstLabel(map[metricdata.LabelKey]metricdata.LabelValue{
			{Key: "const"}:  metricdata.NewLabelValue("same"),
			{Key: "const2"}: metricdata.NewLabelValue("same2"),
		}))

	e, _ := f.GetEntry(metricdata.LabelValue{})
	e.Set(5)
	e, _ = f.GetEntry(metricdata.NewLabelValue("k1v1"))
	e.Add(1)
	m := r.Read()
	want := []*metricdata.Metric{
		{
			Descriptor: metricdata.Descriptor{
				Name: "TestGaugeWithConstLabel",
				LabelKeys: []metricdata.LabelKey{
					{Key: "const"},
					{Key: "const2"},
					{Key: "k1"}},
				Type: metricdata.TypeGaugeFloat64,
			},
			TimeSeries: []*metricdata.TimeSeries{
				{
					LabelValues: []metricdata.LabelValue{
						metricdata.NewLabelValue("same"),
						metricdata.NewLabelValue("same2"),
						{},
					},
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

func TestGaugeMetricDescriptor(t *testing.T) {
	r := NewRegistry()

	gf, _ := r.AddFloat64Gauge("float64_gauge")
	compareType(gf.bm.desc.Type, metricdata.TypeGaugeFloat64, t)
	gi, _ := r.AddInt64Gauge("int64_gauge")
	compareType(gi.bm.desc.Type, metricdata.TypeGaugeInt64, t)
	dgf, _ := r.AddFloat64DerivedGauge("derived_float64_gauge")
	compareType(dgf.bm.desc.Type, metricdata.TypeGaugeFloat64, t)
	dgi, _ := r.AddInt64DerivedGauge("derived_int64_gauge")
	compareType(dgi.bm.desc.Type, metricdata.TypeGaugeInt64, t)
}

func compareType(got, want metricdata.Type, t *testing.T) {
	if got != want {
		t.Errorf("metricdata type: got %v, want %v\n", got, want)
	}
}

func TestGaugeMetricOptionDesc(t *testing.T) {
	r := NewRegistry()
	name := "testOptDesc"
	gf, _ := r.AddFloat64Gauge(name, WithDescription("test"))
	want := metricdata.Descriptor{
		Name:        name,
		Description: "test",
		Type:        metricdata.TypeGaugeFloat64,
	}
	got := gf.bm.desc
	if !cmp.Equal(got, want) {
		t.Errorf("metric option description: got %v, want %v\n", got, want)
	}
}

func TestGaugeMetricOptionUnit(t *testing.T) {
	r := NewRegistry()
	name := "testOptUnit"
	gf, _ := r.AddFloat64Gauge(name, WithUnit(metricdata.UnitMilliseconds))
	want := metricdata.Descriptor{
		Name: name,
		Unit: metricdata.UnitMilliseconds,
		Type: metricdata.TypeGaugeFloat64,
	}
	got := gf.bm.desc
	if !cmp.Equal(got, want) {
		t.Errorf("metric descriptor: got %v, want %v\n", got, want)
	}
}

func TestGaugeMetricOptionLabelKeys(t *testing.T) {
	r := NewRegistry()
	name := "testOptUnit"
	gf, _ := r.AddFloat64Gauge(name, WithLabelKeys("k1", "k3"))
	want := metricdata.Descriptor{
		Name: name,
		LabelKeys: []metricdata.LabelKey{
			{Key: "k1"},
			{Key: "k3"},
		},
		Type: metricdata.TypeGaugeFloat64,
	}
	got := gf.bm.desc
	if !cmp.Equal(got, want) {
		t.Errorf("metric descriptor: got %v, want %v\n", got, want)
	}
}

func TestGaugeMetricOptionLabelKeysAndDesc(t *testing.T) {
	r := NewRegistry()
	name := "testOptUnit"
	lks := []metricdata.LabelKey{}
	lks = append(lks, metricdata.LabelKey{Key: "k1", Description: "desc k1"},
		metricdata.LabelKey{Key: "k3", Description: "desc k3"})
	gf, _ := r.AddFloat64Gauge(name, WithLabelKeysAndDescription(lks...))
	want := metricdata.Descriptor{
		Name: name,
		LabelKeys: []metricdata.LabelKey{
			{Key: "k1", Description: "desc k1"},
			{Key: "k3", Description: "desc k3"},
		},
		Type: metricdata.TypeGaugeFloat64,
	}
	got := gf.bm.desc
	if !cmp.Equal(got, want) {
		t.Errorf("metric descriptor: got %v, want %v\n", got, want)
	}
}

func TestGaugeMetricOptionDefault(t *testing.T) {
	r := NewRegistry()
	name := "testOptUnit"
	gf, _ := r.AddFloat64Gauge(name)
	want := metricdata.Descriptor{
		Name: name,
		Type: metricdata.TypeGaugeFloat64,
	}
	got := gf.bm.desc
	if !cmp.Equal(got, want) {
		t.Errorf("metric descriptor: got %v, want %v\n", got, want)
	}
}

func TestFloat64Entry_Add(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddFloat64Gauge("g")
	e, _ := g.GetEntry()
	e.Add(0)
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), 0.0; got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	e, _ = g.GetEntry()
	e.Add(1)
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), 1.0; got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	e, _ = g.GetEntry()
	e.Add(-1)
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), 0.0; got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestFloat64Gauge_Add_NegativeTotals(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddFloat64Gauge("g")
	e, _ := g.GetEntry()
	e.Add(-1.0)
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestInt64GaugeEntry_Add(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64Gauge("g")
	e, _ := g.GetEntry()
	e.Add(0)
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	e, _ = g.GetEntry()
	e.Add(1)
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(1); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestInt64Gauge_Add_NegativeTotals(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64Gauge("g")
	e, _ := g.GetEntry()
	e.Add(-1)
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestGaugeWithSameNameDiffType(t *testing.T) {
	r := NewRegistry()
	r.AddInt64Gauge("g")
	_, gotErr := r.AddFloat64Gauge("g")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
	_, gotErr = r.AddInt64DerivedGauge("g")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
	_, gotErr = r.AddFloat64DerivedGauge("g")
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errMetricExistsWithDiffType)
	}
}

func TestGaugeWithLabelMismatch(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64Gauge("g", WithLabelKeys("k1"))
	_, gotErr := g.GetEntry(metricdata.NewLabelValue("k1v2"), metricdata.NewLabelValue("k2v2"))
	if gotErr == nil {
		t.Errorf("got: nil, want error: %v", errKeyValueMismatch)
	}
}

func TestMapKey(t *testing.T) {
	cases := [][]metricdata.LabelValue{
		{},
		{metricdata.LabelValue{}},
		{metricdata.NewLabelValue("")},
		{metricdata.NewLabelValue("-")},
		{metricdata.NewLabelValue(",")},
		{metricdata.NewLabelValue("v1"), metricdata.NewLabelValue("v2")},
		{metricdata.NewLabelValue("v1"), metricdata.LabelValue{}},
		{metricdata.NewLabelValue("v1"), metricdata.LabelValue{}, metricdata.NewLabelValue(string([]byte{0}))},
		{metricdata.LabelValue{}, metricdata.LabelValue{}},
	}
	for i, tc := range cases {
		t.Run(fmt.Sprintf("case %d", i), func(t *testing.T) {
			g := &baseMetric{
				keys: make([]metricdata.LabelKey, len(tc)),
			}
			mk := g.encodeLabelVals(tc)
			vals := g.decodeLabelVals(mk)
			if diff := cmp.Diff(vals, tc); diff != "" {
				t.Errorf("values differ after serialization -got +want: %s", diff)
			}
		})
	}
}

func TestRaceCondition(t *testing.T) {
	r := NewRegistry()

	// start reader before adding Gauge metric.
	var ms = []*metricdata.Metric{}
	for i := 0; i < 5; i++ {
		go func(k int) {
			for j := 0; j < 5; j++ {
				g, _ := r.AddInt64Gauge(fmt.Sprintf("g%d%d", k, j))
				e, _ := g.GetEntry()
				e.Add(1)
			}
		}(i)
	}
	time.Sleep(1 * time.Second)
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(int64), int64(1); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func ignoreTimes(_, _ time.Time) bool {
	return true
}

func canonicalize(ms []*metricdata.Metric) {
	for _, m := range ms {
		sort.Slice(m.TimeSeries, func(i, j int) bool {
			// sort time series by their label values
			iStr := ""

			for _, label := range m.TimeSeries[i].LabelValues {
				iStr += fmt.Sprintf("%+v", label)
			}

			jStr := ""
			for _, label := range m.TimeSeries[j].LabelValues {
				jStr += fmt.Sprintf("%+v", label)
			}

			return iStr < jStr
		})
	}
}

type queueInt64 struct {
	size int64
}

func (q *queueInt64) ToInt64() int64 {
	return q.size
}

func TestInt64DerivedGaugeEntry_Add(t *testing.T) {
	r := NewRegistry()
	q := &queueInt64{3}
	g, _ := r.AddInt64DerivedGauge("g", WithLabelKeys("k1", "k2"))
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

func TestInt64DerivedGaugeEntry_AddWithNilObj(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddInt64DerivedGauge("g", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(nil, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestInt64DerivedGaugeEntry_AddWithInvalidLabels(t *testing.T) {
	r := NewRegistry()
	q := &queueInt64{3}
	g, _ := r.AddInt64DerivedGauge("g", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(q.ToInt64, metricdata.NewLabelValue("k1v1"))
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestInt64DerivedGaugeEntry_Update(t *testing.T) {
	r := NewRegistry()
	q := &queueInt64{3}
	q2 := &queueInt64{5}
	g, _ := r.AddInt64DerivedGauge("g", WithLabelKeys("k1", "k2"))
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

type queueFloat64 struct {
	size float64
}

func (q *queueFloat64) ToFloat64() float64 {
	return q.size
}

func TestFloat64DerivedGaugeEntry_Add(t *testing.T) {
	r := NewRegistry()
	q := &queueFloat64{5.0}
	g, _ := r.AddFloat64DerivedGauge("g", WithLabelKeys("k1", "k2"))
	err := g.UpsertEntry(q.ToFloat64, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}
	ms := r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(5.0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
	q.size = 5
	ms = r.Read()
	if got, want := ms[0].TimeSeries[0].Points[0].Value.(float64), float64(5.0); got != want {
		t.Errorf("value = %v, want %v", got, want)
	}
}

func TestFloat64DerivedGaugeEntry_AddWithNilObj(t *testing.T) {
	r := NewRegistry()
	g, _ := r.AddFloat64DerivedGauge("g", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(nil, metricdata.NewLabelValue("k1v1"), metricdata.LabelValue{})
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestFloat64DerivedGaugeEntry_AddWithInvalidLabels(t *testing.T) {
	r := NewRegistry()
	q := &queueFloat64{3}
	g, _ := r.AddFloat64DerivedGauge("g", WithLabelKeys("k1", "k2"))
	gotErr := g.UpsertEntry(q.ToFloat64, metricdata.NewLabelValue("k1v1"))
	if gotErr == nil {
		t.Errorf("expected error but got nil")
	}
}

func TestFloat64DerivedGaugeEntry_Update(t *testing.T) {
	r := NewRegistry()
	q := &queueFloat64{3.0}
	q2 := &queueFloat64{5.0}
	g, _ := r.AddFloat64DerivedGauge("g", WithLabelKeys("k1", "k2"))
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
