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
	"sync"
	"testing"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns" // Only for unit testing purposes.
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestGetClusterConditionPredicate(t *testing.T) {
	fakedns, _ := clouddns.NewFakeInterface() // No need to check for unsupported interfaces, as the fake interface supports everything that's required.
	serviceController := ServiceController{
		dns:          fakedns,
		serviceCache: &serviceCache{fedServiceMap: make(map[string]*cachedService)},
		clusterCache: &clusterClientCache{
			rwlock:    sync.Mutex{},
			clientMap: make(map[string]*clusterCache),
		},
		knownClusterSet: make(sets.String),
	}

	tests := []struct {
		cluster           federation.Cluster
		expectAccept      bool
		name              string
		serviceController *ServiceController
	}{
		{
			cluster:           federation.Cluster{},
			expectAccept:      false,
			name:              "empty",
			serviceController: &serviceController,
		},
		{
			cluster: federation.Cluster{
				Status: federation.ClusterStatus{
					Conditions: []federation.ClusterCondition{
						{Type: federation.ClusterReady, Status: api.ConditionTrue},
					},
				},
			},
			expectAccept:      true,
			name:              "basic",
			serviceController: &serviceController,
		},
		{
			cluster: federation.Cluster{
				Status: federation.ClusterStatus{
					Conditions: []federation.ClusterCondition{
						{Type: federation.ClusterReady, Status: api.ConditionFalse},
					},
				},
			},
			expectAccept:      false,
			name:              "notready",
			serviceController: &serviceController,
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
