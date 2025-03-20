/*
Copyright 2017 The Kubernetes Authors.

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

package stats

import (
	"reflect"
	"testing"
	"time"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/utils/ptr"
)

func TestCustomMetrics(t *testing.T) {
	spec := []cadvisorapiv1.MetricSpec{
		{
			Name:   "qos",
			Type:   cadvisorapiv1.MetricGauge,
			Format: cadvisorapiv1.IntType,
			Units:  "per second",
		},
		{
			Name:   "cpuLoad",
			Type:   cadvisorapiv1.MetricCumulative,
			Format: cadvisorapiv1.FloatType,
			Units:  "count",
		},
	}
	timestamp1 := time.Now()
	timestamp2 := time.Now().Add(time.Minute)
	metrics := map[string][]cadvisorapiv1.MetricVal{
		"qos": {
			{
				Timestamp: timestamp1,
				IntValue:  10,
			},
			{
				Timestamp: timestamp2,
				IntValue:  100,
			},
		},
		"cpuLoad": {
			{
				Timestamp:  timestamp1,
				FloatValue: 1.2,
			},
			{
				Timestamp:  timestamp2,
				FloatValue: 2.1,
			},
		},
	}
	cInfo := cadvisorapiv2.ContainerInfo{
		Spec: cadvisorapiv2.ContainerSpec{
			CustomMetrics: spec,
		},
		Stats: []*cadvisorapiv2.ContainerStats{
			{
				CustomMetrics: metrics,
			},
		},
	}
	assert.Contains(t, cadvisorInfoToUserDefinedMetrics(&cInfo),
		statsapi.UserDefinedMetric{
			UserDefinedMetricDescriptor: statsapi.UserDefinedMetricDescriptor{
				Name:  "qos",
				Type:  statsapi.MetricGauge,
				Units: "per second",
			},
			Time:  metav1.NewTime(timestamp2),
			Value: 100,
		},
		statsapi.UserDefinedMetric{
			UserDefinedMetricDescriptor: statsapi.UserDefinedMetricDescriptor{
				Name:  "cpuLoad",
				Type:  statsapi.MetricCumulative,
				Units: "count",
			},
			Time:  metav1.NewTime(timestamp2),
			Value: 2.1,
		})
}

func TestMergeProcessStats(t *testing.T) {
	for _, tc := range []struct {
		desc     string
		first    *statsapi.ProcessStats
		second   *statsapi.ProcessStats
		expected *statsapi.ProcessStats
	}{
		{
			desc:     "both nil",
			first:    nil,
			second:   nil,
			expected: nil,
		},
		{
			desc:     "first non-nil, second not",
			first:    &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
			second:   nil,
			expected: &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
		},
		{
			desc:     "first nil, second non-nil",
			first:    nil,
			second:   &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
			expected: &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
		},
		{
			desc:     "both non nill",
			first:    &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
			second:   &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](100)},
			expected: &statsapi.ProcessStats{ProcessCount: ptr.To[uint64](200)},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			got := mergeProcessStats(tc.first, tc.second)
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Fatalf("Unexpected diff on process stats (-want,+got):\n%s", diff)
			}
		})
	}
}

// TestCadvisorPSIStruct checks the fields in cadvisor PSI structs. If cadvisor
// PSI structs change, the conversion between cadvisor PSI structs and kubelet stats API structs needs to be re-evaluated and updated.
func TestCadvisorPSIStructs(t *testing.T) {
	psiStatsFields := sets.New("Full", "Some")
	s := cadvisorapiv1.PSIStats{}
	st := reflect.TypeOf(s)
	for i := 0; i < st.NumField(); i++ {
		field := st.Field(i)
		if !psiStatsFields.Has(field.Name) {
			t.Errorf("cadvisorapiv1.PSIStats contains unknown field: %s. The conversion between cadvisor PSIStats and kubelet stats API PSIStats needs to be re-evaluated and updated.", field.Name)
		}
	}

	psiDataFields := map[string]reflect.Kind{
		"Total":  reflect.Uint64,
		"Avg10":  reflect.Float64,
		"Avg60":  reflect.Float64,
		"Avg300": reflect.Float64,
	}
	d := cadvisorapiv1.PSIData{}
	dt := reflect.TypeOf(d)
	for i := 0; i < dt.NumField(); i++ {
		field := dt.Field(i)
		wantKind, fieldExist := psiDataFields[field.Name]
		if !fieldExist {
			t.Errorf("cadvisorapiv1.PSIData contains unknown field: %s. The conversion between cadvisor PSIData and kubelet stats API PSIData needs to be re-evaluated and updated.", field.Name)
		}
		if field.Type.Kind() != wantKind {
			t.Errorf("unexpected cadvisorapiv1.PSIStats field %s type, want: %s, got: %s. The conversion between cadvisor PSIStats and kubelet stats API PSIStats needs to be re-evaluated and updated.", field.Name, wantKind, field.Type.Kind())
		}
	}

}
