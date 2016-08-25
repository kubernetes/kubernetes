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

package node

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestNodeOutageTaintAddedRemoved(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: fakeNow,
			Labels: map[string]string{
				unversioned.LabelZoneRegion:        "region1",
				unversioned.LabelZoneFailureDomain: "zone1",
			},
			Annotations: map[string]string{},
		},
	}

	nodeHandler := &FakeNodeHandler{Clientset: fake.NewSimpleClientset()}
	if err := addNodeOutageTaint(nodeHandler, node); err != nil {
		t.Errorf("unexpected error %v", err)
	}
	taints, _ := api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 1 {
		t.Errorf("unexpected taints lengths %v", taints)
	}

	if err := removeNodeOutageTaint(nodeHandler, node); err != nil {
		t.Errorf("unexpected error %v", err)
	}
	taints, _ = api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 0 {
		t.Errorf("taints should be empty %v", taints)
	}
}
