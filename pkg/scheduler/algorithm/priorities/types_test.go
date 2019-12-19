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

package priorities

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// EmptyMetadataProducer should return a no-op MetadataProducer type.
func TestEmptyPriorityMetadataProducer(t *testing.T) {
	fakePod := st.MakePod().Name("p1").Node("node2").Obj()
	fakeLabelSelector := labels.SelectorFromSet(labels.Set{"foo": "bar"})
	fakeNodes := []*v1.Node{st.MakeNode().Name("node1").Obj(), st.MakeNode().Name("node-a").Obj()}

	snapshot := nodeinfosnapshot.NewSnapshot(nodeinfosnapshot.CreateNodeInfoMap([]*v1.Pod{fakePod}, fakeNodes))
	// Test EmptyMetadataProducer
	metadata := EmptyMetadataProducer(fakePod, fakeNodes, snapshot)
	if metadata != nil {
		t.Errorf("failed to produce empty metadata: got %v, expected nil", metadata)
	}
	// Test EmptyControllerLister should return nill
	controllerLister := algorithm.EmptyControllerLister{}
	nilController, nilError := controllerLister.List(fakeLabelSelector)
	if nilController != nil || nilError != nil {
		t.Errorf("failed to produce empty controller lister: got %v, expected nil", nilController)
	}
	// Test GetPodControllers on empty controller lister should return nill
	nilController, nilError = controllerLister.GetPodControllers(fakePod)
	if nilController != nil || nilError != nil {
		t.Errorf("failed to produce empty controller lister: got %v, expected nil", nilController)
	}
	// Test GetPodReplicaSets on empty replica sets should return nill
	replicaSetLister := algorithm.EmptyReplicaSetLister{}
	nilRss, nilErrRss := replicaSetLister.GetPodReplicaSets(fakePod)
	if nilRss != nil || nilErrRss != nil {
		t.Errorf("failed to produce empty replicaSetLister: got %v, expected nil", nilRss)
	}

	// Test GetPodStatefulSets on empty replica sets should return nill
	statefulSetLister := algorithm.EmptyStatefulSetLister{}
	nilSSL, nilErrSSL := statefulSetLister.GetPodStatefulSets(fakePod)
	if nilSSL != nil || nilErrSSL != nil {
		t.Errorf("failed to produce empty statefulSetLister: got %v, expected nil", nilSSL)
	}
}
