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

package algorithm

import (
	"testing"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// EmptyMetadataProducer should returns a no-op MetadataProducer type.
func TestEmptyMetadataProducer(t *testing.T) {
	noResources := v1.PodSpec{
		Containers: []v1.Container{
			{},
		},
	}

	tests := []struct {
		pods          []*v1.Pod
		labels        map[string]string
		labelSelector labels.Selector
	}{
		{
			pods: []*v1.Pod{
				{Spec: noResources}, {Spec: noResources},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			labelSelector: labels.SelectorFromSet(labels.Set{"foo": "bar"}),
		},
	}

	for _, test := range tests {
		nodeNameToInfo := map[string]*schedulercache.NodeInfo{
			"3": schedulercache.NewNodeInfo(test.pods...),
			"2": schedulercache.NewNodeInfo(test.pods[0]),
			"1": schedulercache.NewNodeInfo(),
		}
		// Test EmptyMetadataProducer
		shouldBeNil := EmptyMetadataProducer(test.pods[0], nodeNameToInfo)
		if shouldBeNil != nil {
			t.Errorf("failed to produce empty metadata")
		}
		// Test EmptyControllerLister should return nill
		FakeEmptyControllerLister := EmptyControllerLister{}
		NilController, NilError := FakeEmptyControllerLister.List(test.labelSelector)
		if NilController != nil || NilError != nil {
			t.Errorf("failed to produce empty replica sets")
		}
		// Test GetPodControllers on empty controller lister should return nill
		NilController, NilError = FakeEmptyControllerLister.GetPodControllers(test.pods[0])
		if NilController != nil || NilError != nil {
			t.Errorf("failed to produce empty controller lister")
		}
		// Test GetPodReplicaSets on empty replica sets should return nill
		FakeEmptyReplicaSetLister := EmptyReplicaSetLister{}
		NilRss, NilErrRss := FakeEmptyReplicaSetLister.GetPodReplicaSets(test.pods[0])
		if NilRss != nil || NilErrRss != nil {
			t.Errorf("failed to produce empty replica sets")
		}
	}
}
