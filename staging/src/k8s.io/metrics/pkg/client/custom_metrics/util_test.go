/*
Copyright 2018 The Kubernetes Authors.

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

package custom_metrics

import (
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	cmint "k8s.io/metrics/pkg/apis/custom_metrics"
	cmv1beta1 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
	cmv1beta2 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
)

func TestMetricConverter(t *testing.T) {
	testCases := []struct {
		name     string
		group    schema.GroupVersion
		expected runtime.Object
	}{
		{
			name:  "Use custom metrics v1beta2",
			group: cmv1beta2.SchemeGroupVersion,
			expected: &cmv1beta2.MetricListOptions{
				TypeMeta:            metav1.TypeMeta{Kind: "MetricListOptions", APIVersion: cmv1beta2.SchemeGroupVersion.String()},
				MetricLabelSelector: "foo",
			},
		},
		{
			name:  "Use custom metrics v1beta1",
			group: cmv1beta1.SchemeGroupVersion,
			expected: &cmv1beta1.MetricListOptions{
				TypeMeta:            metav1.TypeMeta{Kind: "MetricListOptions", APIVersion: cmv1beta1.SchemeGroupVersion.String()},
				MetricLabelSelector: "foo",
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			metricConverter := NewMetricConverter()
			opts := &cmint.MetricListOptions{MetricLabelSelector: "foo"}
			res, err := metricConverter.ConvertListOptionsToVersion(opts, test.group)
			require.NoError(t, err)
			require.Equal(t, test.expected, res)
		})
	}
}
