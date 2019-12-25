/*
Copyright 2014 The Kubernetes Authors.

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

package algorithmprovider

import (
	"testing"

	"k8s.io/kubernetes/pkg/scheduler"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

var (
	algorithmProviderNames = []string{
		schedulerapi.SchedulerDefaultProviderName,
	}
)

func TestApplyFeatureGates(t *testing.T) {
	for _, pn := range algorithmProviderNames {
		t.Run(pn, func(t *testing.T) {
			p, err := scheduler.GetAlgorithmProvider(pn)
			if err != nil {
				t.Fatalf("Error retrieving provider: %v", err)
			}

			if !p.PredicateKeys.Has("PodToleratesNodeTaints") {
				t.Fatalf("Failed to find predicate: 'PodToleratesNodeTaints'")
			}
		})
	}

	defer ApplyFeatureGates()()

	for _, pn := range algorithmProviderNames {
		t.Run(pn, func(t *testing.T) {
			p, err := scheduler.GetAlgorithmProvider(pn)
			if err != nil {
				t.Fatalf("Error retrieving '%s' provider: %v", pn, err)
			}

			if !p.PredicateKeys.Has("PodToleratesNodeTaints") {
				t.Fatalf("Failed to find predicate: 'PodToleratesNodeTaints'")
			}
		})
	}
}
