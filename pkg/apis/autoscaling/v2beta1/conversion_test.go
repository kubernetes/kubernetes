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

package v2beta1

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/autoscaling/v2beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

// Testing nil pointer panic uncovered by #70806
// TODO(yue9944882): Test nil/empty conversion across all resource types
func TestNilOrEmptyConversion(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, RegisterConversions(scheme))

	testCases := []struct {
		obj1 interface{}
		obj2 interface{}
	}{
		{
			obj1: &autoscaling.ExternalMetricSource{},
			obj2: &v2beta1.ExternalMetricSource{},
		},
		{
			obj1: &autoscaling.ExternalMetricStatus{},
			obj2: &v2beta1.ExternalMetricStatus{},
		},
		{
			obj1: &autoscaling.PodsMetricSource{},
			obj2: &v2beta1.PodsMetricSource{},
		},
		{
			obj1: &autoscaling.PodsMetricStatus{},
			obj2: &v2beta1.PodsMetricStatus{},
		},
		{
			obj1: &autoscaling.ObjectMetricSource{},
			obj2: &v2beta1.ObjectMetricSource{},
		},
		{
			obj1: &autoscaling.ObjectMetricStatus{},
			obj2: &v2beta1.ObjectMetricStatus{},
		},
		{
			obj1: &autoscaling.ResourceMetricSource{},
			obj2: &v2beta1.ResourceMetricSource{},
		},
		{
			obj1: &autoscaling.ResourceMetricStatus{},
			obj2: &v2beta1.ResourceMetricStatus{},
		},
		{
			obj1: &autoscaling.HorizontalPodAutoscaler{},
			obj2: &v2beta1.HorizontalPodAutoscaler{},
		},
		{
			obj1: &autoscaling.MetricTarget{},
			obj2: &v2beta1.CrossVersionObjectReference{},
		},
	}
	for _, testCase := range testCases {
		assert.NoError(t, scheme.Convert(testCase.obj1, testCase.obj2, nil))
		assert.NoError(t, scheme.Convert(testCase.obj2, testCase.obj1, nil))
	}
}
