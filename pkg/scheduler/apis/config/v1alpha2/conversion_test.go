/*
Copyright 2020 The Kubernetes Authors.

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

package v1alpha2

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-scheduler/config/v1alpha2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
)

func TestV1alpha2ToConfigKubeSchedulerConfigurationConversion(t *testing.T) {
	cases := []struct {
		name     string
		config   v1alpha2.KubeSchedulerConfiguration
		expected config.KubeSchedulerConfiguration
	}{
		{
			name:     "default conversion v1alpha2 to config",
			config:   v1alpha2.KubeSchedulerConfiguration{},
			expected: config.KubeSchedulerConfiguration{AlgorithmSource: config.SchedulerAlgorithmSource{Provider: pointer.StringPtr(v1alpha2.SchedulerDefaultProviderName)}},
		},
	}

	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out config.KubeSchedulerConfiguration
			if err := scheme.Convert(&tc.config, &out, nil); err != nil {
				t.Errorf("failed to convert: %+v", err)
			}
			if diff := cmp.Diff(tc.expected, out); diff != "" {
				t.Errorf("unexpected conversion (-want, +got):\n%s", diff)
			}
		})
	}
}
