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
	"testing"
	"time"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
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
