/*
Copyright 2019 The Kubernetes Authors.

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

package imagelocality

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
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
				Image: "gcr.io/4000",
			},
		},
	}

	test300600900 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/300",
			},
			{
				Image: "gcr.io/600",
			},
			{
				Image: "gcr.io/900",
			},
		},
	}

	test3040 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/30",
			},
			{
				Image: "gcr.io/40",
			},
		},
	}

	test30Init300 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/30",
			},
		},
		InitContainers: []v1.Container{
			{Image: "gcr.io/300"},
		},
	}

	node403002000 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/40:latest",
					"gcr.io/40:v1",
					"gcr.io/40:v1",
				},
				SizeBytes: int64(40 * mb),
			},
			{
				Names: []string{
					"gcr.io/300:latest",
					"gcr.io/300:v1",
				},
				SizeBytes: int64(300 * mb),
			},
			{
				Names: []string{
					"gcr.io/2000:latest",
				},
				SizeBytes: int64(2000 * mb),
			},
		},
	}

	node25010 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/250:latest",
				},
				SizeBytes: int64(250 * mb),
			},
			{
				Names: []string{
					"gcr.io/10:latest",
					"gcr.io/10:v1",
				},
				SizeBytes: int64(10 * mb),
			},
		},
	}

	node60040900 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/600:latest",
				},
				SizeBytes: int64(600 * mb),
			},
			{
				Names: []string{
					"gcr.io/40:latest",
				},
				SizeBytes: int64(40 * mb),
			},
			{
				Names: []string{
					"gcr.io/900:latest",
				},
				SizeBytes: int64(900 * mb),
			},
		},
	}

	node300600900 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/300:latest",
				},
				SizeBytes: int64(300 * mb),
			},
			{
				Names: []string{
					"gcr.io/600:latest",
				},
				SizeBytes: int64(600 * mb),
			},
			{
				Names: []string{
					"gcr.io/900:latest",
				},
				SizeBytes: int64(900 * mb),
			},
		},
	}

	node400030 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/4000:latest",
				},
				SizeBytes: int64(4000 * mb),
			},
			{
				Names: []string{
					"gcr.io/30:latest",
				},
				SizeBytes: int64(30 * mb),
			},
		},
	}

	node203040 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/20:latest",
				},
				SizeBytes: int64(20 * mb),
			},
			{
				Names: []string{
					"gcr.io/30:latest",
				},
				SizeBytes: int64(30 * mb),
			},
			{
				Names: []string{
					"gcr.io/40:latest",
				},
				SizeBytes: int64(40 * mb),
			},
		},
	}

	nodeWithNoImages := v1.NodeStatus{}

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			// Pod: gcr.io/40 gcr.io/250

			// Node1
			// Image: gcr.io/40:latest 40MB
			// Score: 0 (40M/2 < 23M, min-threshold)

			// Node2
			// Image: gcr.io/250:latest 250MB
			// Score: 100 * (250M/2 - 23M)/(1000M * 2 - 23M) = 5
			pod:          &v1.Pod{Spec: test40250},
			nodes:        []*v1.Node{makeImageNode("node1", node403002000), makeImageNode("node2", node25010)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 5}},
			name:         "two images spread on two nodes, prefer the larger image one",
		},
		{
			// Pod: gcr.io/40 gcr.io/300

			// Node1
			// Image: gcr.io/40:latest 40MB, gcr.io/300:latest 300MB
			// Score: 100 * ((40M + 300M)/2 - 23M)/(1000M * 2 - 23M) = 7

			// Node2
			// Image: not present
			// Score: 0
			pod:          &v1.Pod{Spec: test40300},
			nodes:        []*v1.Node{makeImageNode("node1", node403002000), makeImageNode("node2", node25010)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 7}, {Name: "node2", Score: 0}},
			name:         "two images on one node, prefer this node",
		},
		{
			// Pod: gcr.io/4000 gcr.io/10

			// Node1
			// Image: gcr.io/4000:latest 2000MB
			// Score: 100 (4000 * 1/2 >= 1000M * 2, max-threshold)

			// Node2
			// Image: gcr.io/10:latest 10MB
			// Score: 0 (10M/2 < 23M, min-threshold)
			pod:          &v1.Pod{Spec: testMinMax},
			nodes:        []*v1.Node{makeImageNode("node1", node400030), makeImageNode("node2", node25010)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: 0}},
			name:         "if exceed limit, use limit",
		},
		{
			// Pod: gcr.io/4000 gcr.io/10

			// Node1
			// Image: gcr.io/4000:latest 4000MB
			// Score: 100 * (4000M/3 - 23M)/(1000M * 2 - 23M) = 66

			// Node2
			// Image: gcr.io/10:latest 10MB
			// Score: 0 (10M*1/3 < 23M, min-threshold)

			// Node3
			// Image:
			// Score: 0
			pod:          &v1.Pod{Spec: testMinMax},
			nodes:        []*v1.Node{makeImageNode("node1", node400030), makeImageNode("node2", node25010), makeImageNode("node3", nodeWithNoImages)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 66}, {Name: "node2", Score: 0}, {Name: "node3", Score: 0}},
			name:         "if exceed limit, use limit (with node which has no images present)",
		},
		{
			// Pod: gcr.io/300 gcr.io/600 gcr.io/900

			// Node1
			// Image: gcr.io/600:latest 600MB, gcr.io/900:latest 900MB
			// Score: 100 * (600M * 2/3 + 900M * 2/3 - 23M) / (1000M * 3 - 23M) = 32

			// Node2
			// Image: gcr.io/300:latest 300MB, gcr.io/600:latest 600MB, gcr.io/900:latest 900MB
			// Score: 100 * (300M * 1/3 + 600M * 2/3 + 900M * 2/3 - 23M) / (1000M *3 - 23M) = 36

			// Node3
			// Image:
			// Score: 0
			pod:          &v1.Pod{Spec: test300600900},
			nodes:        []*v1.Node{makeImageNode("node1", node60040900), makeImageNode("node2", node300600900), makeImageNode("node3", nodeWithNoImages)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 32}, {Name: "node2", Score: 36}, {Name: "node3", Score: 0}},
			name:         "pod with multiple large images, node2 is preferred",
		},
		{
			// Pod: gcr.io/30 gcr.io/40

			// Node1
			// Image: gcr.io/20:latest 20MB, gcr.io/30:latest 30MB, gcr.io/40:latest 40MB
			// Score: 100 * (30M + 40M * 1/2 - 23M) / (1000M * 2 - 23M) = 1

			// Node2
			// Image: 100 * (30M - 23M) / (1000M * 2 - 23M) = 0
			// Score: 0
			pod:          &v1.Pod{Spec: test3040},
			nodes:        []*v1.Node{makeImageNode("node1", node203040), makeImageNode("node2", node400030)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 1}, {Name: "node2", Score: 0}},
			name:         "pod with multiple small images",
		},
		{
			// Pod: gcr.io/30  InitContainers: gcr.io/300

			// Node1
			// Image: gcr.io/40:latest 40MB, gcr.io/300:latest 300MB, gcr.io/2000:latest 2000MB
			// Score: 100 * (300M * 1/2 - 23M) / (1000M * 2 - 23M) = 6

			// Node2
			// Image: gcr.io/20:latest 20MB, gcr.io/30:latest 30MB, gcr.io/40:latest 40MB
			// Score: 100 * (30M * 1/2  - 23M) / (1000M * 2 - 23M) = 0
			pod:          &v1.Pod{Spec: test30Init300},
			nodes:        []*v1.Node{makeImageNode("node1", node403002000), makeImageNode("node2", node203040)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 6}, {Name: "node2", Score: 0}},
			name:         "include InitContainers: two images spread on two nodes, prefer the larger image one",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			snapshot := cache.NewSnapshot(nil, test.nodes)
			state := framework.NewCycleState()
			fh, _ := runtime.NewFramework(ctx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))

			p, err := New(ctx, nil, fh)
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				nodeName := n.ObjectMeta.Name
				score, status := p.(framework.ScorePlugin).Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			if diff := cmp.Diff(test.expectedList, gotList); diff != "" {
				t.Errorf("Unexpected node score list (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestNormalizedImageName(t *testing.T) {
	for _, testCase := range []struct {
		Name   string
		Input  string
		Output string
	}{
		{Name: "add :latest postfix 1", Input: "root", Output: "root:latest"},
		{Name: "add :latest postfix 2", Input: "gcr.io:5000/root", Output: "gcr.io:5000/root:latest"},
		{Name: "keep it as is 1", Input: "root:tag", Output: "root:tag"},
		{Name: "keep it as is 2", Input: "root@" + getImageFakeDigest("root"), Output: "root@" + getImageFakeDigest("root")},
	} {
		t.Run(testCase.Name, func(t *testing.T) {
			image := normalizedImageName(testCase.Input)
			if image != testCase.Output {
				t.Errorf("expected image reference: %q, got %q", testCase.Output, image)
			}
		})
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
