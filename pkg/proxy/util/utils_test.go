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

package util

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestShouldSkipService(t *testing.T) {
	testCases := []struct {
		service    *api.Service
		svcName    types.NamespacedName
		shouldSkip bool
	}{
		{
			// Cluster IP is None
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: api.ClusterIPNone,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// Cluster IP is empty
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "",
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// ExternalName type service
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeExternalName,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// ClusterIP type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeClusterIP,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
		{
			// NodePort type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeNodePort,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
		{
			// LoadBalancer type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeLoadBalancer,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
	}

	for i := range testCases {
		skip := ShouldSkipService(testCases[i].svcName, testCases[i].service)
		if skip != testCases[i].shouldSkip {
			t.Errorf("case %d: expect %v, got %v", i, testCases[i].shouldSkip, skip)
		}
	}
}

func strToStrPtr(str string) *string {
	tmp := str
	return &tmp
}

func TestSameTopologyDomain(t *testing.T) {
	testCases := []struct {
		topologyKey  string
		topologyMode string
		labels1      map[string]string
		labels2      map[string]string
		expectSame   bool
	}{
		{ // case 0
			topologyMode: string(api.TopologyModeIgnored),
			topologyKey:  "WHAT-EVER",
			labels1:      map[string]string{"a": "b"},
			labels2:      map[string]string{"x": "y"},
			expectSame:   false,
		},
		{ // case 1
			topologyMode: string(api.TopologyModeIgnored),
			expectSame:   false,
		},
		{ // case 2
			topologyMode: string(api.TopologyModePreferred),
			topologyKey:  "k8s.io/hostname",
			labels1:      map[string]string{"k8s.io/hostname": "hostA"},
			labels2:      map[string]string{"k8s.io/hostname": "hostA"},
			expectSame:   true,
		},
		{ // case 3
			topologyMode: string(api.TopologyModePreferred),
			topologyKey:  "k8s.io/hostname",
			labels1:      map[string]string{"k8s.io/hostname": "hostA"},
			labels2:      map[string]string{"k8s.io/hostname": "hostB"},
			expectSame:   false,
		},
		{ // case 4
			topologyMode: string(api.TopologyModePreferred),
			topologyKey:  "k8s.io/hostname",
			labels1:      map[string]string{"k8s.io/hostname": "hostA"},
			expectSame:   false,
		},
		{ // case 5
			topologyMode: string(api.TopologyModeRequired),
			topologyKey:  "k8s.io/region",
			labels1:      map[string]string{"k8s.io/region": "region-1"},
			labels2:      map[string]string{"k8s.io/region": "region-1"},
			expectSame:   true,
		},
		{ // case 6
			topologyMode: string(api.TopologyModeRequired),
			topologyKey:  "k8s.io/region",
			labels1:      map[string]string{"k8s.io/region": "region-1"},
			labels2:      map[string]string{"k8s.io/region": "region-2"},
			expectSame:   false,
		},
		{ // case 7
			topologyMode: string(api.TopologyModeRequired),
			topologyKey:  "k8s.io/region",
			labels2:      map[string]string{"k8s.io/region": "region-2"},
			expectSame:   false,
		},
		{ // case 8
			topologyMode: string(api.TopologyModeRequired),
			topologyKey:  "k8s.io/region",
			labels1:      map[string]string{"k8s.io/hostname": "hostA"},
			labels2:      map[string]string{"k8s.io/region": "region-2"},
			expectSame:   false,
		},
		{ // case 9
			topologyMode: string(api.TopologyModePreferred),
			topologyKey:  "k8s.io/region",
			labels1:      map[string]string{"k8s.io/hostname": "hostA"},
			labels2:      map[string]string{"k8s.io/hostname": "hostA"},
			expectSame:   false,
		},
		{ // case 10
			topologyMode: string(api.TopologyModeRequired),
			topologyKey:  "k8s.io/region",
			expectSame:   false,
		},
	}

	for i := range testCases {
		same := SameTopologyDomain(testCases[i].labels1, testCases[i].labels2, testCases[i].topologyKey, testCases[i].topologyMode)
		if same != testCases[i].expectSame {
			t.Errorf("case %d: expect %v, got %v", i, testCases[i].expectSame, same)
		}
	}
}
