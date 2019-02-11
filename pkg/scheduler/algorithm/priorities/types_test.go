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
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// EmptyPriorityMetadataProducer should return a no-op PriorityMetadataProducer type.
func TestEmptyPriorityMetadataProducer(t *testing.T) {
	fakePod := new(v1.Pod)
	fakeLabelSelector := labels.SelectorFromSet(labels.Set{"foo": "bar"})

	nodeNameToInfo := map[string]*schedulernodeinfo.NodeInfo{
		"2": schedulernodeinfo.NewNodeInfo(fakePod),
		"1": schedulernodeinfo.NewNodeInfo(),
	}
	// Test EmptyPriorityMetadataProducer
	metadata := EmptyPriorityMetadataProducer(fakePod, nodeNameToInfo)
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
