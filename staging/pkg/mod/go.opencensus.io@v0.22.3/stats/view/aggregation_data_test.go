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
//

package view

import (
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opencensus.io/metric/metricdata"
)

func TestDataClone(t *testing.T) {
	agg := &Aggregation{
		Buckets: []float64{1, 2, 3, 4},
	}
	dist := newDistributionData(agg)
	dist.Count = 7
	dist.Max = 11
	dist.Min = 1
	dist.CountPerBucket = []int64{0, 2, 3, 2}
	dist.Mean = 4
	dist.SumOfSquaredDev = 1.2

	tests := []struct {
		name string
		src  AggregationData
	}{
		{
			name: "count data",
			src:  &CountData{Value: 5},
		},
		{
			name: "distribution data",
			src:  dist,
		},
		{
			name: "sum data",
			src:  &SumData{Value: 65.7},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.src.clone()
			if !reflect.DeepEqual(got, tt.src) {
				t.Errorf("AggregationData.clone() = %v, want %v", got, tt.src)
			}
			// TODO(jbd): Make sure that data is deep copied.
			if got == tt.src {
				t.Errorf("AggregationData.clone() returned the same pointer")
			}
		})
	}
}

func TestDistributionData_addSample(t *testing.T) {
	agg := &Aggregation{
		Buckets: []float64{1, 2},
	}
	dd := newDistributionData(agg)
	attachments1 := map[string]interface{}{"key1": "value1"}
	t1 := time.Now()
	dd.addSample(0.5, attachments1, t1)

	e1 := &metricdata.Exemplar{Value: 0.5, Timestamp: t1, Attachments: attachments1}
	want := &DistributionData{
		Count:              1,
		CountPerBucket:     []int64{1, 0, 0},
		ExemplarsPerBucket: []*metricdata.Exemplar{e1, nil, nil},
		Max:                0.5,
		Min:                0.5,
		Mean:               0.5,
		SumOfSquaredDev:    0,
	}
	if diff := cmpDD(dd, want); diff != "" {
		t.Fatalf("Unexpected DistributionData -got +want: %s", diff)
	}

	attachments2 := map[string]interface{}{"key2": "value2"}
	t2 := t1.Add(time.Microsecond)
	dd.addSample(0.7, attachments2, t2)

	// Previous exemplar should be overwritten.
	e2 := &metricdata.Exemplar{Value: 0.7, Timestamp: t2, Attachments: attachments2}
	want = &DistributionData{
		Count:              2,
		CountPerBucket:     []int64{2, 0, 0},
		ExemplarsPerBucket: []*metricdata.Exemplar{e2, nil, nil},
		Max:                0.7,
		Min:                0.5,
		Mean:               0.6,
		SumOfSquaredDev:    0,
	}
	if diff := cmpDD(dd, want); diff != "" {
		t.Fatalf("Unexpected DistributionData -got +want: %s", diff)
	}

	attachments3 := map[string]interface{}{"key3": "value3"}
	t3 := t2.Add(time.Microsecond)
	dd.addSample(1.2, attachments3, t3)

	// e3 is at another bucket. e2 should still be there.
	e3 := &metricdata.Exemplar{Value: 1.2, Timestamp: t3, Attachments: attachments3}
	want = &DistributionData{
		Count:              3,
		CountPerBucket:     []int64{2, 1, 0},
		ExemplarsPerBucket: []*metricdata.Exemplar{e2, e3, nil},
		Max:                1.2,
		Min:                0.5,
		Mean:               0.7999999999999999,
		SumOfSquaredDev:    0,
	}
	if diff := cmpDD(dd, want); diff != "" {
		t.Fatalf("Unexpected DistributionData -got +want: %s", diff)
	}
}

func cmpDD(got, want *DistributionData) string {
	return cmp.Diff(got, want, cmpopts.IgnoreFields(DistributionData{}, "SumOfSquaredDev"), cmpopts.IgnoreUnexported(DistributionData{}))
}
