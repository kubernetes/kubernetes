/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package service

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/federation/apis/federation/v1alpha1"
)

func TestGetClusterConditionPredicate(t *testing.T) {
	tests := []struct {
		cluster      v1alpha1.Cluster
		expectAccept bool
		name         string
	}{
		{
			cluster:      v1alpha1.Cluster{},
			expectAccept: false,
			name:         "empty",
		},
		{
			cluster: v1alpha1.Cluster{
				Status: v1alpha1.ClusterStatus{
					Conditions: []v1alpha1.ClusterCondition{
						{Type: v1alpha1.ClusterReady, Status: v1.ConditionTrue},
					},
				},
			},
			expectAccept: true,
			name:         "basic",
		},
		{
			cluster: v1alpha1.Cluster{
				Status: v1alpha1.ClusterStatus{
					Conditions: []v1alpha1.ClusterCondition{
						{Type: v1alpha1.ClusterReady, Status: v1.ConditionFalse},
					},
				},
			},
			expectAccept: false,
			name:         "notready",
		},
	}
	pred := getClusterConditionPredicate()
	for _, test := range tests {
		accept := pred(test.cluster)
		if accept != test.expectAccept {
			t.Errorf("Test failed for %s, expected %v, saw %v", test.name, test.expectAccept, accept)
		}
	}
}
