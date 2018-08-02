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

package priorities

import (
	"crypto/sha256"
	"reflect"
	"sort"
	"testing"

	"encoding/hex"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/util/parsers"
)

func TestImageLocalityPriority(t *testing.T) {
	test40250 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/250",
			},
		},
	}

	test40300 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/300",
			},
		},
	}

	testMinMax := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/10",
			},
			{
				Image: "gcr.io/2000",
			},
		},
	}

	node403002000 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/40:" + parsers.DefaultImageTag,
					"gcr.io/40:v1",
					"gcr.io/40:v1",
				},
				SizeBytes: int64(40 * mb),
			},
			{
				Names: []string{
					"gcr.io/300:" + parsers.DefaultImageTag,
					"gcr.io/300:v1",
				},
				SizeBytes: int64(300 * mb),
			},
			{
				Names: []string{
					"gcr.io/2000:" + parsers.DefaultImageTag,
				},
				SizeBytes: int64(2000 * mb),
			},
		},
	}

	node25010 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/250:" + parsers.DefaultImageTag,
				},
				SizeBytes: int64(250 * mb),
			},
			{
				Names: []string{
					"gcr.io/10:" + parsers.DefaultImageTag,
					"gcr.io/10:v1",
				},
				SizeBytes: int64(10 * mb),
			},
		},
	}

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			// Pod: gcr.io/40 gcr.io/250

			// Node1
			// Image: gcr.io/40:latest 40MB
			// Score: 0 (40M/2 < 23M, min-threshold)

			// Node2
			// Image: gcr.io/250:latest 250MB
			// Score: 10 * (250M/2 - 23M)/(1000M - 23M) = 1
			pod:          &v1.Pod{Spec: test40250},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 1}},
			name:         "two images spread on two nodes, prefer the larger image one",
		},
		{
			// Pod: gcr.io/40 gcr.io/300

			// Node1
			// Image: gcr.io/40:latest 40MB, gcr.io/300:latest 300MB
			// Score: 10 * ((40M + 300M)/2 - 23M)/(1000M - 23M) = 1

			// Node2
			// Image: not present
			// Score: 0
			pod:          &v1.Pod{Spec: test40300},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 1}, {Host: "machine2", Score: 0}},
			name:         "two images on one node, prefer this node",
		},
		{
			// Pod: gcr.io/2000 gcr.io/10

			// Node1
			// Image: gcr.io/2000:latest 2000MB
			// Score: 10 (2000M/2 >= 1000M, max-threshold)

			// Node2
			// Image: gcr.io/10:latest 10MB
			// Score: 0 (10M/2 < 23M, min-threshold)
			pod:          &v1.Pod{Spec: testMinMax},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: schedulerapi.MaxPriority}, {Host: "machine2", Score: 0}},
			name:         "if exceed limit, use limit",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
			list, err := priorityFunction(ImageLocalityPriorityMap, nil, &priorityMetadata{totalNumNodes: len(test.nodes)})(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			sort.Sort(test.expectedList)
			sort.Sort(list)

			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}

func TestNormalizedImageName(t *testing.T) {
	for _, testCase := range []struct {
		Input  string
		Output string
	}{
		{Input: "root", Output: "root:latest"},
		{Input: "root:tag", Output: "root:tag"},
		{Input: "gcr.io:5000/root", Output: "gcr.io:5000/root:latest"},
		{Input: "root@" + getImageFakeDigest("root"), Output: "root@" + getImageFakeDigest("root")},
	} {
		image := normalizedImageName(testCase.Input)
		if image != testCase.Output {
			t.Errorf("expected image reference: %q, got %q", testCase.Output, image)
		}
	}
}

func makeImageNode(node string, status v1.NodeStatus) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status:     status,
	}
}

func getImageFakeDigest(fakeContent string) string {
	hash := sha256.Sum256([]byte(fakeContent))
	return "sha256:" + hex.EncodeToString(hash[:])
}
