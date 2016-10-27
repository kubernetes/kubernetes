/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/labels"
)

// FakeMetricsClient is a fake implementation of MetricsClient which
// returns either an appropriate response or an error for a given set
// of input arguments
type FakeMetricsClient struct {
	ResourceMetrics map[api.ResourceName]PodResourceInfo
	RawMetrics      map[string]PodMetricsInfo
	ObjectMetrics   map[string]float64

	Timestamp time.Time
}

func NewFakeMetricsClient(timestamp time.Time) *FakeMetricsClient {
	return &FakeMetricsClient{
		ResourceMetrics: map[api.ResourceName]PodResourceInfo{},
		RawMetrics:      map[string]PodMetricsInfo{},
		ObjectMetrics:   map[string]float64{},

		Timestamp: timestamp,
	}
}

// GetResourceMetric gets the given resource metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (f *FakeMetricsClient) GetResourceMetric(resource api.ResourceName, namespace string, selector labels.Selector) (PodResourceInfo, time.Time, error) {
	if entry, present := f.ResourceMetrics[resource]; present {
		return entry, f.Timestamp, nil
	}

	return nil, time.Time{}, fmt.Errorf("no resource %v found for pods in namespace %s matching %s", resource, namespace, selector.String())
}

// GetRawMetric gets the given metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (f *FakeMetricsClient) GetRawMetric(metricName string, namespace string, selector labels.Selector) (PodMetricsInfo, time.Time, error) {
	if entry, present := f.RawMetrics[metricName]; present {
		return entry, f.Timestamp, nil
	}

	return nil, time.Time{}, fmt.Errorf("no metric %s found for pods in namespace %s matching %s", metricName, namespace, selector.String())
}

// GetObjectMetric gets the given metric (and an associated timestamp) for the given
// object in the given namespace
func (f *FakeMetricsClient) GetObjectMetric(metricName string, namespace string, objectRef *autoscaling.CrossVersionObjectReference) (float64, time.Time, error) {
	if entry, present := f.ObjectMetrics[metricName]; present {
		return entry, f.Timestamp, nil
	}

	return 0, time.Time{}, fmt.Errorf("no metric %s found for %s %s/%s ", metricName, objectRef.Kind, namespace, objectRef.Name)
}
