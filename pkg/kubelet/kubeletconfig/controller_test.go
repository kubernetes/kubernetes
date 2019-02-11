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

package kubeletconfig

import (
	"testing"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint/store"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
)

func TestGraduateAssignedToLastKnownGood(t *testing.T) {
	realSource1, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Namespace:        "foo",
			Name:             "1",
			KubeletConfigKey: "kubelet",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	realSource2, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Namespace:        "foo",
			Name:             "2",
			KubeletConfigKey: "kubelet",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		name     string
		assigned checkpoint.RemoteConfigSource
		lkg      checkpoint.RemoteConfigSource
	}{
		{
			name:     "nil lkg to non-nil lkg",
			assigned: realSource1,
			lkg:      nil,
		},
		{
			name:     "non-nil lkg to non-nil lkg",
			assigned: realSource2,
			lkg:      realSource1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			controller := &Controller{
				configStatus:    status.NewNodeConfigStatus(),
				checkpointStore: store.NewFakeStore(),
			}
			controller.checkpointStore.SetLastKnownGood(tc.lkg)
			controller.checkpointStore.SetAssigned(tc.assigned)
			if err := controller.graduateAssignedToLastKnownGood(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
