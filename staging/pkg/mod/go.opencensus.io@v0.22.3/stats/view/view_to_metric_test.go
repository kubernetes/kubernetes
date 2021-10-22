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
//

package view

import (
	"context"
	"testing"
	"time"

	"encoding/json"

	"github.com/google/go-cmp/cmp"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricexport"
	"go.opencensus.io/stats"
	"go.opencensus.io/tag"
)

type recordValWithTag struct {
	tags  []tag.Tag
	value interface{}
}
type testToMetrics struct {
	vi          *viewInternal
	view        *View
	recordValue []recordValWithTag
	wantMetric  *metricdata.Metric
}

var (
	// tag objects.
	tk1   tag.Key
	tk2   tag.Key
	tk3   tag.Key
	tk1v1 tag.Tag
	tk2v2 tag.Tag
	tags  []tag.Tag

	labelValues      []metricdata.LabelValue
	emptyLabelValues []metricdata.LabelValue

	labelKeys []metricdata.LabelKey

	recordsInt64        []recordValWithTag
	recordsFloat64      []recordValWithTag
	recordsFloat64WoTag []recordValWithTag

	// distribution objects.
	aggDist *Aggregation
	aggCnt  *Aggregation
	aggS    *Aggregation
	aggL    *Aggregation
	buckOpt *metricdata.BucketOptions

	// exemplar objects.
	attachments metricdata.Attachments

	// views and descriptors
	viewTypeFloat64Distribution         *View
	viewTypeInt64Distribution           *View
	viewTypeInt64Count                  *View
	viewTypeFloat64Count                *View
	viewTypeFloat64Sum                  *View
	viewTypeInt64Sum                    *View
	viewTypeFloat64LastValue            *View
	viewTypeInt64LastValue              *View
	viewRecordWithoutLabel              *View
	mdTypeFloat64CumulativeDistribution metricdata.Descriptor
	mdTypeInt64CumulativeDistribution   metricdata.Descriptor
	mdTypeInt64CumulativeCount          metricdata.Descriptor
	mdTypeFloat64CumulativeCount        metricdata.Descriptor
	mdTypeInt64CumulativeSum            metricdata.Descriptor
	mdTypeFloat64CumulativeSum          metricdata.Descriptor
	mdTypeInt64CumulativeLastValue      metricdata.Descriptor
	mdTypeFloat64CumulativeLastValue    metricdata.Descriptor
	mdTypeRecordWithoutLabel            metricdata.Descriptor
)

const (
	nameInt64DistM1        = "viewToMetricTest_Int64_Distribution/m1"
	nameFloat64DistM1      = "viewToMetricTest_Float64_Distribution/m1"
	nameInt64CountM1       = "viewToMetricTest_Int64_Count/m1"
	nameFloat64CountM1     = "viewToMetricTest_Float64_Count/m1"
	nameInt64SumM1         = "viewToMetricTest_Int64_Sum/m1"
	nameFloat64SumM1       = "viewToMetricTest_Float64_Sum/m1"
	nameInt64LastValueM1   = "viewToMetricTest_Int64_LastValue/m1"
	nameFloat64LastValueM1 = "viewToMetricTest_Float64_LastValue/m1"
	nameRecordWithoutLabel = "viewToMetricTest_RecordWithoutLabel/m1"
	v1                     = "v1"
	v2                     = "v2"
)

func init() {
	initTags()
	initAgg()
	initViews()
	initMetricDescriptors()

}

func initTags() {
	tk1 = tag.MustNewKey("k1")
	tk2 = tag.MustNewKey("k2")
	tk3 = tag.MustNewKey("k3")
	tk1v1 = tag.Tag{Key: tk1, Value: v1}
	tk2v2 = tag.Tag{Key: tk2, Value: v2}

	tags = []tag.Tag{tk1v1, tk2v2}
	labelValues = []metricdata.LabelValue{
		{Value: v1, Present: true},
		{Value: v2, Present: true},
	}
	emptyLabelValues = []metricdata.LabelValue{
		{Value: "", Present: false},
		{Value: "", Present: false},
	}
	labelKeys = []metricdata.LabelKey{
		{Key: tk1.Name()},
		{Key: tk2.Name()},
	}

	recordsInt64 = []recordValWithTag{
		{tags: tags, value: int64(2)},
		{tags: tags, value: int64(4)},
	}
	recordsFloat64 = []recordValWithTag{
		{tags: tags, value: float64(1.5)},
		{tags: tags, value: float64(5.4)},
	}
	recordsFloat64WoTag = []recordValWithTag{
		{value: float64(1.5)},
		{value: float64(5.4)},
	}
}

func initAgg() {
	aggDist = Distribution(2.0)
	aggCnt = Count()
	aggS = Sum()
	aggL = LastValue()
	buckOpt = &metricdata.BucketOptions{Bounds: []float64{2.0}}
}

func initViews() {
	// View objects
	viewTypeInt64Distribution = &View{
		Name:        nameInt64DistM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Int64(nameInt64DistM1, "", stats.UnitDimensionless),
		Aggregation: aggDist,
	}
	viewTypeFloat64Distribution = &View{
		Name:        nameFloat64DistM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Float64(nameFloat64DistM1, "", stats.UnitDimensionless),
		Aggregation: aggDist,
	}
	viewTypeInt64Count = &View{
		Name:        nameInt64CountM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Int64(nameInt64CountM1, "", stats.UnitDimensionless),
		Aggregation: aggCnt,
	}
	viewTypeFloat64Count = &View{
		Name:        nameFloat64CountM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Float64(nameFloat64CountM1, "", stats.UnitDimensionless),
		Aggregation: aggCnt,
	}
	viewTypeInt64Sum = &View{
		Name:        nameInt64SumM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Int64(nameInt64SumM1, "", stats.UnitBytes),
		Aggregation: aggS,
	}
	viewTypeFloat64Sum = &View{
		Name:        nameFloat64SumM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Float64(nameFloat64SumM1, "", stats.UnitMilliseconds),
		Aggregation: aggS,
	}
	viewTypeInt64LastValue = &View{
		Name:        nameInt64LastValueM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Int64(nameInt64LastValueM1, "", stats.UnitDimensionless),
		Aggregation: aggL,
	}
	viewTypeFloat64LastValue = &View{
		Name:        nameFloat64LastValueM1,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Float64(nameFloat64LastValueM1, "", stats.UnitDimensionless),
		Aggregation: aggL,
	}
	viewRecordWithoutLabel = &View{
		Name:        nameRecordWithoutLabel,
		TagKeys:     []tag.Key{tk1, tk2},
		Measure:     stats.Float64(nameRecordWithoutLabel, "", stats.UnitDimensionless),
		Aggregation: aggL,
	}
}

func initMetricDescriptors() {
	// Metric objects
	mdTypeFloat64CumulativeDistribution = metricdata.Descriptor{
		Name: nameFloat64DistM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeCumulativeDistribution, LabelKeys: labelKeys,
	}
	mdTypeInt64CumulativeDistribution = metricdata.Descriptor{
		Name: nameInt64DistM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeCumulativeDistribution, LabelKeys: labelKeys,
	}
	mdTypeInt64CumulativeCount = metricdata.Descriptor{
		Name: nameInt64CountM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeCumulativeInt64, LabelKeys: labelKeys,
	}
	mdTypeFloat64CumulativeCount = metricdata.Descriptor{
		Name: nameFloat64CountM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeCumulativeInt64, LabelKeys: labelKeys,
	}
	mdTypeInt64CumulativeSum = metricdata.Descriptor{
		Name: nameInt64SumM1, Description: "", Unit: metricdata.UnitBytes,
		Type: metricdata.TypeCumulativeInt64, LabelKeys: labelKeys,
	}
	mdTypeFloat64CumulativeSum = metricdata.Descriptor{
		Name: nameFloat64SumM1, Description: "", Unit: metricdata.UnitMilliseconds,
		Type: metricdata.TypeCumulativeFloat64, LabelKeys: labelKeys,
	}
	mdTypeInt64CumulativeLastValue = metricdata.Descriptor{
		Name: nameInt64LastValueM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeGaugeInt64, LabelKeys: labelKeys,
	}
	mdTypeFloat64CumulativeLastValue = metricdata.Descriptor{
		Name: nameFloat64LastValueM1, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeGaugeFloat64, LabelKeys: labelKeys,
	}
	mdTypeRecordWithoutLabel = metricdata.Descriptor{
		Name: nameRecordWithoutLabel, Description: "", Unit: metricdata.UnitDimensionless,
		Type: metricdata.TypeGaugeFloat64, LabelKeys: labelKeys,
	}
}

func Test_ViewToMetric(t *testing.T) {
	startTime := time.Now().Add(-60 * time.Second)
	now := time.Now()
	tests := []*testToMetrics{
		{
			view:        viewTypeInt64Distribution,
			recordValue: recordsInt64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeInt64CumulativeDistribution,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						{Value: &metricdata.Distribution{
							Count:                 2,
							Sum:                   6.0,
							SumOfSquaredDeviation: 2,
							BucketOptions:         buckOpt,
							Buckets: []metricdata.Bucket{
								{Count: 0, Exemplar: nil},
								{Count: 2, Exemplar: nil},
							},
						},
							Time: now,
						},
					},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeFloat64Distribution,
			recordValue: recordsFloat64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeFloat64CumulativeDistribution,
				TimeSeries: []*metricdata.TimeSeries{
					{
						Points: []metricdata.Point{
							{
								Value: &metricdata.Distribution{
									Count:                 2,
									Sum:                   6.9,
									SumOfSquaredDeviation: 7.605000000000001,
									BucketOptions:         buckOpt,
									Buckets: []metricdata.Bucket{
										{Count: 1, Exemplar: nil}, // TODO: [rghetia] add exemplar test.
										{Count: 1, Exemplar: nil},
									},
								},
								Time: now,
							},
						},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeInt64Count,
			recordValue: recordsInt64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeInt64CumulativeCount,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewInt64Point(now, 2),
					},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeFloat64Count,
			recordValue: recordsFloat64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeFloat64CumulativeCount,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewInt64Point(now, 2),
					},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeInt64Sum,
			recordValue: recordsInt64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeInt64CumulativeSum,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewInt64Point(now, 6),
					},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeFloat64Sum,
			recordValue: recordsFloat64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeFloat64CumulativeSum,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewFloat64Point(now, 6.9),
					},
						LabelValues: labelValues,
						StartTime:   startTime,
					},
				},
			},
		},
		{
			view:        viewTypeInt64LastValue,
			recordValue: recordsInt64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeInt64CumulativeLastValue,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewInt64Point(now, 4),
					},
						LabelValues: labelValues,
						StartTime:   time.Time{},
					},
				},
			},
		},
		{
			view:        viewTypeFloat64LastValue,
			recordValue: recordsFloat64,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeFloat64CumulativeLastValue,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewFloat64Point(now, 5.4),
					},
						LabelValues: labelValues,
						StartTime:   time.Time{},
					},
				},
			},
		},
		{
			view:        viewRecordWithoutLabel,
			recordValue: recordsFloat64WoTag,
			wantMetric: &metricdata.Metric{
				Descriptor: mdTypeRecordWithoutLabel,
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						metricdata.NewFloat64Point(now, 5.4),
					},
						LabelValues: emptyLabelValues,
						StartTime:   time.Time{},
					},
				},
			},
		},
	}

	wantMetrics := []*metricdata.Metric{}
	for _, tc := range tests {
		tc.vi, _ = defaultWorker.tryRegisterView(tc.view)
		tc.vi.clearRows()
		tc.vi.subscribe()
		wantMetrics = append(wantMetrics, tc.wantMetric)
	}

	for i, tc := range tests {
		for _, r := range tc.recordValue {
			mods := []tag.Mutator{}
			for _, tg := range r.tags {
				mods = append(mods, tag.Insert(tg.Key, tg.Value))
			}
			ctx, err := tag.New(context.Background(), mods...)
			if err != nil {
				t.Errorf("%v: New = %v", tc.view.Name, err)
			}
			var v float64
			switch i := r.value.(type) {
			case float64:
				v = float64(i)
			case int64:
				v = float64(i)
			default:
				t.Errorf("unexpected value type %v", r.tags)
			}
			tc.vi.addSample(tag.FromContext(ctx), v, nil, now)
		}

		gotMetric := viewToMetric(tc.vi, now, startTime)
		if !cmp.Equal(gotMetric, tc.wantMetric) {
			// JSON format is strictly for checking the content when test fails. Do not use JSON
			// format to determine if the two values are same as it doesn't differentiate between
			// int64(2) and float64(2.0)
			t.Errorf("#%d: Unmatched \nGot:\n\t%v\nWant:\n\t%v\nGot Serialized:%s\nWant Serialized:%s\n",
				i, gotMetric, tc.wantMetric, serializeAsJSON(gotMetric), serializeAsJSON(tc.wantMetric))
		}
	}
}

// Test to verify that a metric converted from a view with Aggregation Count should always
// have Dimensionless unit.
func TestUnitConversionForAggCount(t *testing.T) {
	startTime := time.Now().Add(-60 * time.Second)
	now := time.Now()
	tests := []*struct {
		name     string
		vi       *viewInternal
		v        *View
		wantUnit metricdata.Unit
	}{
		{
			name: "View with Count Aggregation on Latency measurement",
			v: &View{
				Name:        "request_count1",
				Measure:     stats.Int64("request_latency", "", stats.UnitMilliseconds),
				Aggregation: aggCnt,
			},
			wantUnit: metricdata.UnitDimensionless,
		},
		{
			name: "View with Count Aggregation on bytes measurement",
			v: &View{
				Name:        "request_count2",
				Measure:     stats.Int64("request_bytes", "", stats.UnitBytes),
				Aggregation: aggCnt,
			},
			wantUnit: metricdata.UnitDimensionless,
		},
		{
			name: "View with aggregation other than Count Aggregation on Latency measurement",
			v: &View{
				Name:        "request_latency",
				Measure:     stats.Int64("request_latency", "", stats.UnitMilliseconds),
				Aggregation: aggSum,
			},
			wantUnit: metricdata.UnitMilliseconds,
		},
	}
	var err error
	for _, tc := range tests {
		tc.vi, err = defaultWorker.tryRegisterView(tc.v)
		if err != nil {
			t.Fatalf("error registering view: %v, err: %v\n", tc.v, err)
		}
		tc.vi.clearRows()
		tc.vi.subscribe()
	}

	for _, tc := range tests {
		tc.vi.addSample(tag.FromContext(context.Background()), 5.0, nil, now)
		gotMetric := viewToMetric(tc.vi, now, startTime)
		gotUnit := gotMetric.Descriptor.Unit
		if !cmp.Equal(gotUnit, tc.wantUnit) {
			t.Errorf("Verify Unit: %s: Got:%v Want:%v", tc.name, gotUnit, tc.wantUnit)
		}
	}
}

type mockExp struct {
	metrics []*metricdata.Metric
}

func (me *mockExp) ExportMetrics(ctx context.Context, metrics []*metricdata.Metric) error {
	me.metrics = append(me.metrics, metrics...)
	return nil
}

var _ metricexport.Exporter = (*mockExp)(nil)

func TestViewToMetric_OutOfOrderWithZeroBuckets(t *testing.T) {
	m := stats.Int64("OutOfOrderWithZeroBuckets", "", "")
	now := time.Now()
	tts := []struct {
		v *View
		m *metricdata.Metric
	}{
		{
			v: &View{
				Name:        m.Name() + "_order1",
				Measure:     m,
				Aggregation: Distribution(10, 0, 2),
			},
			m: &metricdata.Metric{
				Descriptor: metricdata.Descriptor{
					Name:      "OutOfOrderWithZeroBuckets_order1",
					Unit:      metricdata.UnitDimensionless,
					Type:      metricdata.TypeCumulativeDistribution,
					LabelKeys: []metricdata.LabelKey{},
				},
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						{Value: &metricdata.Distribution{
							Count:                 3,
							Sum:                   9.0,
							SumOfSquaredDeviation: 8,
							BucketOptions: &metricdata.BucketOptions{
								Bounds: []float64{2, 10},
							},
							Buckets: []metricdata.Bucket{
								{Count: 1, Exemplar: nil},
								{Count: 2, Exemplar: nil},
								{Count: 0, Exemplar: nil},
							},
						},
							Time: now,
						},
					},
						StartTime:   now,
						LabelValues: []metricdata.LabelValue{},
					},
				},
			},
		},
		{
			v: &View{
				Name:        m.Name() + "_order2",
				Measure:     m,
				Aggregation: Distribution(0, 5, 10),
			},
			m: &metricdata.Metric{
				Descriptor: metricdata.Descriptor{
					Name:      "OutOfOrderWithZeroBuckets_order2",
					Unit:      metricdata.UnitDimensionless,
					Type:      metricdata.TypeCumulativeDistribution,
					LabelKeys: []metricdata.LabelKey{},
				},
				TimeSeries: []*metricdata.TimeSeries{
					{Points: []metricdata.Point{
						{Value: &metricdata.Distribution{
							Count:                 3,
							Sum:                   9.0,
							SumOfSquaredDeviation: 8,
							BucketOptions: &metricdata.BucketOptions{
								Bounds: []float64{5, 10},
							},
							Buckets: []metricdata.Bucket{
								{Count: 2, Exemplar: nil},
								{Count: 1, Exemplar: nil},
								{Count: 0, Exemplar: nil},
							},
						},
							Time: now,
						},
					},
						StartTime:   now,
						LabelValues: []metricdata.LabelValue{},
					},
				},
			},
		},
	}
	for _, tt := range tts {
		err := Register(tt.v)
		if err != nil {
			t.Fatalf("error registering view %v, err: %v", tt.v, err)
		}

	}

	stats.Record(context.Background(), m.M(5), m.M(1), m.M(3))
	time.Sleep(1 * time.Second)

	me := &mockExp{}
	reader := metricexport.NewReader()
	reader.ReadAndExport(me)

	var got *metricdata.Metric
	lookup := func(vname string, metrics []*metricdata.Metric) *metricdata.Metric {
		for _, m := range metrics {
			if m.Descriptor.Name == vname {
				return m
			}
		}
		return nil
	}

	for _, tt := range tts {
		got = lookup(tt.v.Name, me.metrics)
		if got == nil {
			t.Fatalf("metric %s not found in %v\n", tt.v.Name, me.metrics)
		}
		got.TimeSeries[0].Points[0].Time = now
		got.TimeSeries[0].StartTime = now

		want := tt.m
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("buckets differ -got +want: %s \n Serialized got %v\n, Serialized want %v\n", diff, serializeAsJSON(got), serializeAsJSON(want))
		}
	}
}

func serializeAsJSON(v interface{}) string {
	blob, _ := json.MarshalIndent(v, "", "  ")
	return string(blob)
}
