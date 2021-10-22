// Copyright 2017, OpenCensus Authors
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
//

package view

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"go.opencensus.io/metric/metricdata"

	"go.opencensus.io/stats"
	"go.opencensus.io/tag"
)

func Test_View_MeasureFloat64_AggregationDistribution(t *testing.T) {
	k1 := tag.MustNewKey("k1")
	k2 := tag.MustNewKey("k2")
	k3 := tag.MustNewKey("k3")
	agg1 := Distribution(2)
	m := stats.Int64("Test_View_MeasureFloat64_AggregationDistribution/m1", "", stats.UnitDimensionless)
	view1 := &View{
		TagKeys:     []tag.Key{k1, k2},
		Measure:     m,
		Aggregation: agg1,
	}
	view, err := newViewInternal(view1)
	if err != nil {
		t.Fatal(err)
	}

	type tagString struct {
		k tag.Key
		v string
	}
	type record struct {
		f    float64
		tags []tagString
	}

	type testCase struct {
		label    string
		records  []record
		wantRows []*Row
	}

	tcs := []testCase{
		{
			"1",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k1, "v1"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&DistributionData{
						Count: 2, Min: 1, Max: 5, Mean: 3, SumOfSquaredDev: 8, CountPerBucket: []int64{1, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
			},
		},
		{
			"2",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k2, "v2"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&DistributionData{
						Count: 1, Min: 1, Max: 1, Mean: 1, CountPerBucket: []int64{1, 0}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k2, Value: "v2"}},
					&DistributionData{
						Count: 1, Min: 5, Max: 5, Mean: 5, CountPerBucket: []int64{0, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
			},
		},
		{
			"3",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k1, "v1"}, {k3, "v3"}}},
				{1, []tagString{{k1, "v1 other"}}},
				{5, []tagString{{k2, "v2"}}},
				{5, []tagString{{k1, "v1"}, {k2, "v2"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&DistributionData{
						Count: 2, Min: 1, Max: 5, Mean: 3, SumOfSquaredDev: 8, CountPerBucket: []int64{1, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1 other"}},
					&DistributionData{
						Count: 1, Min: 1, Max: 1, Mean: 1, CountPerBucket: []int64{1, 0}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k2, Value: "v2"}},
					&DistributionData{
						Count: 1, Min: 5, Max: 5, Mean: 5, CountPerBucket: []int64{0, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1"}, {Key: k2, Value: "v2"}},
					&DistributionData{
						Count: 1, Min: 5, Max: 5, Mean: 5, CountPerBucket: []int64{0, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
			},
		},
		{
			"4",
			[]record{
				{1, []tagString{{k1, "v1 is a very long value key"}}},
				{5, []tagString{{k1, "v1 is a very long value key"}, {k3, "v3"}}},
				{1, []tagString{{k1, "v1 is another very long value key"}}},
				{1, []tagString{{k1, "v1 is a very long value key"}, {k2, "v2 is a very long value key"}}},
				{5, []tagString{{k1, "v1 is a very long value key"}, {k2, "v2 is a very long value key"}}},
				{3, []tagString{{k1, "v1 is a very long value key"}, {k2, "v2 is a very long value key"}}},
				{3, []tagString{{k1, "v1 is a very long value key"}, {k2, "v2 is a very long value key"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1 is a very long value key"}},
					&DistributionData{
						Count: 2, Min: 1, Max: 5, Mean: 3, SumOfSquaredDev: 8, CountPerBucket: []int64{1, 1}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1 is another very long value key"}},
					&DistributionData{
						Count: 1, Min: 1, Max: 1, Mean: 1, CountPerBucket: []int64{1, 0}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1 is a very long value key"}, {Key: k2, Value: "v2 is a very long value key"}},
					&DistributionData{
						Count: 4, Min: 1, Max: 5, Mean: 3, SumOfSquaredDev: 2.66666666666667 * 3, CountPerBucket: []int64{1, 3}, bounds: []float64{2}, ExemplarsPerBucket: []*metricdata.Exemplar{nil, nil},
					},
				},
			},
		},
	}

	for _, tc := range tcs {
		view.clearRows()
		view.subscribe()
		for _, r := range tc.records {
			mods := []tag.Mutator{}
			for _, t := range r.tags {
				mods = append(mods, tag.Insert(t.k, t.v))
			}
			ctx, err := tag.New(context.Background(), mods...)
			if err != nil {
				t.Errorf("%v: New = %v", tc.label, err)
			}
			view.addSample(tag.FromContext(ctx), r.f, nil, time.Now())
		}

		gotRows := view.collectedRows()
		for i, got := range gotRows {
			if !containsRow(tc.wantRows, got) {
				t.Errorf("%v-%d: got row %v; want none", tc.label, i, got)
				break
			}
		}

		for i, want := range tc.wantRows {
			if !containsRow(gotRows, want) {
				t.Errorf("%v-%d: got none; want row %v", tc.label, i, want)
				break
			}
		}
	}
}

func Test_View_MeasureFloat64_AggregationSum(t *testing.T) {
	k1 := tag.MustNewKey("k1")
	k2 := tag.MustNewKey("k2")
	k3 := tag.MustNewKey("k3")
	m := stats.Int64("Test_View_MeasureFloat64_AggregationSum/m1", "", stats.UnitDimensionless)
	view, err := newViewInternal(&View{TagKeys: []tag.Key{k1, k2}, Measure: m, Aggregation: Sum()})
	if err != nil {
		t.Fatal(err)
	}

	type tagString struct {
		k tag.Key
		v string
	}
	type record struct {
		f    float64
		tags []tagString
	}

	tcs := []struct {
		label    string
		records  []record
		wantRows []*Row
	}{
		{
			"1",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k1, "v1"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&SumData{Value: 6},
				},
			},
		},
		{
			"2",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k2, "v2"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&SumData{Value: 1},
				},
				{
					[]tag.Tag{{Key: k2, Value: "v2"}},
					&SumData{Value: 5},
				},
			},
		},
		{
			"3",
			[]record{
				{1, []tagString{{k1, "v1"}}},
				{5, []tagString{{k1, "v1"}, {k3, "v3"}}},
				{1, []tagString{{k1, "v1 other"}}},
				{5, []tagString{{k2, "v2"}}},
				{5, []tagString{{k1, "v1"}, {k2, "v2"}}},
			},
			[]*Row{
				{
					[]tag.Tag{{Key: k1, Value: "v1"}},
					&SumData{Value: 6},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1 other"}},
					&SumData{Value: 1},
				},
				{
					[]tag.Tag{{Key: k2, Value: "v2"}},
					&SumData{Value: 5},
				},
				{
					[]tag.Tag{{Key: k1, Value: "v1"}, {Key: k2, Value: "v2"}},
					&SumData{Value: 5},
				},
			},
		},
	}

	for _, tt := range tcs {
		view.clearRows()
		view.subscribe()
		for _, r := range tt.records {
			mods := []tag.Mutator{}
			for _, t := range r.tags {
				mods = append(mods, tag.Insert(t.k, t.v))
			}
			ctx, err := tag.New(context.Background(), mods...)
			if err != nil {
				t.Errorf("%v: New = %v", tt.label, err)
			}
			view.addSample(tag.FromContext(ctx), r.f, nil, time.Now())
		}

		gotRows := view.collectedRows()
		for i, got := range gotRows {
			if !containsRow(tt.wantRows, got) {
				t.Errorf("%v-%d: got row %v; want none", tt.label, i, got)
				break
			}
		}

		for i, want := range tt.wantRows {
			if !containsRow(gotRows, want) {
				t.Errorf("%v-%d: got none; want row %v", tt.label, i, want)
				break
			}
		}
	}
}

func TestCanonicalize(t *testing.T) {
	k1 := tag.MustNewKey("k1")
	k2 := tag.MustNewKey("k2")
	m := stats.Int64("TestCanonicalize/m1", "desc desc", stats.UnitDimensionless)
	v := &View{TagKeys: []tag.Key{k2, k1}, Measure: m, Aggregation: Sum()}
	err := v.canonicalize()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := v.Name, "TestCanonicalize/m1"; got != want {
		t.Errorf("vc.Name = %q; want %q", got, want)
	}
	if got, want := v.Description, "desc desc"; got != want {
		t.Errorf("vc.Description = %q; want %q", got, want)
	}
	if got, want := len(v.TagKeys), 2; got != want {
		t.Errorf("len(vc.TagKeys) = %d; want %d", got, want)
	}
	if got, want := v.TagKeys[0].Name(), "k1"; got != want {
		t.Errorf("vc.TagKeys[0].Name() = %q; want %q", got, want)
	}
}

func TestViewSortedKeys(t *testing.T) {
	k1 := tag.MustNewKey("a")
	k2 := tag.MustNewKey("b")
	k3 := tag.MustNewKey("c")
	ks := []tag.Key{k1, k3, k2}

	m := stats.Int64("TestViewSortedKeys/m1", "", stats.UnitDimensionless)
	Register(&View{
		Name:        "sort_keys",
		Description: "desc sort_keys",
		TagKeys:     ks,
		Measure:     m,
		Aggregation: Sum(),
	})
	// Register normalizes the view by sorting the tag keys, retrieve the normalized view
	v := Find("sort_keys")

	want := []string{"a", "b", "c"}
	vks := v.TagKeys
	if len(vks) != len(want) {
		t.Errorf("Keys = %+v; want %+v", vks, want)
	}

	for i, v := range want {
		if got, want := v, vks[i].Name(); got != want {
			t.Errorf("View name = %q; want %q", got, want)
		}
	}
}

// containsRow returns true if rows contain r.
func containsRow(rows []*Row, r *Row) bool {
	for _, x := range rows {
		if r.Equal(x) {
			return true
		}
	}
	return false
}

func TestRegisterUnregisterParity(t *testing.T) {
	measures := []stats.Measure{
		stats.Int64("ifoo", "iFOO", "iBar"),
		stats.Float64("ffoo", "fFOO", "fBar"),
	}
	aggregations := []*Aggregation{
		Count(),
		Sum(),
		Distribution(1, 2.0, 4.0, 8.0, 16.0),
	}

	for i := 0; i < 10; i++ {
		for _, m := range measures {
			for _, agg := range aggregations {
				v := &View{
					Aggregation: agg,
					Name:        "Lookup here",
					Measure:     m,
				}
				if err := Register(v); err != nil {
					t.Errorf("Iteration #%d:\nMeasure: (%#v)\nAggregation (%#v)\nError: %v", i, m, agg, err)
				}
				Unregister(v)
			}
		}
	}
}

func TestRegisterAfterMeasurement(t *testing.T) {
	// Tests that we can register views after measurements are created and
	// they still take effect.

	m := stats.Int64(t.Name(), "", stats.UnitDimensionless)
	mm := m.M(1)
	ctx := context.Background()

	stats.Record(ctx, mm)
	v := &View{
		Measure:     m,
		Aggregation: Count(),
	}
	if err := Register(v); err != nil {
		t.Fatal(err)
	}

	rows, err := RetrieveData(v.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) > 0 {
		t.Error("View should not have data")
	}

	stats.Record(ctx, mm)

	rows, err = RetrieveData(v.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Error("View should have data")
	}
}

func TestViewRegister_negativeBucketBounds(t *testing.T) {
	m := stats.Int64("TestViewRegister_negativeBucketBounds", "", "")
	v := &View{
		Measure:     m,
		Aggregation: Distribution(-1, 2),
	}
	err := Register(v)
	if err != ErrNegativeBucketBounds {
		t.Errorf("Expected ErrNegativeBucketBounds, got %v", err)
	}
}

func TestViewRegister_sortBuckets(t *testing.T) {
	m := stats.Int64("TestViewRegister_sortBuckets", "", "")
	v := &View{
		Measure:     m,
		Aggregation: Distribution(2, 1),
	}
	err := Register(v)
	if err != nil {
		t.Fatalf("Unexpected err %s", err)
	}
	want := []float64{1, 2}
	if diff := cmp.Diff(v.Aggregation.Buckets, want); diff != "" {
		t.Errorf("buckets differ -got +want: %s", diff)
	}
}

func TestViewRegister_dropZeroBuckets(t *testing.T) {
	m := stats.Int64("TestViewRegister_dropZeroBuckets", "", "")
	v := &View{
		Measure:     m,
		Aggregation: Distribution(2, 0, 1),
	}
	err := Register(v)
	if err != nil {
		t.Fatalf("Unexpected err %s", err)
	}
	want := []float64{1, 2}
	if diff := cmp.Diff(v.Aggregation.Buckets, want); diff != "" {
		t.Errorf("buckets differ -got +want: %s", diff)
	}
}
