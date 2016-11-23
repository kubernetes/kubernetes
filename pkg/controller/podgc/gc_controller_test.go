/*
Copyright 2015 The Kubernetes Authors.

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

package podgc

import (
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/controller/node/testutil"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

type FakeController struct{}

func (*FakeController) Run(<-chan struct{}) {}

func (*FakeController) HasSynced() bool {
	return true
}

func TestGCTerminated(t *testing.T) {
	type nameToPhase struct {
		name  string
		phase v1.PodPhase
	}

	testCases := []struct {
		pods            []nameToPhase
		threshold       int
		deletedPodNames sets.String
	}{
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold: 0,
			// threshold = 0 disables terminated pod deletion
			deletedPodNames: sets.NewString(),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a", "b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodRunning},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       5,
			deletedPodNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{*testutil.NewNode("node")}})
		gcc := NewFromClient(client, test.threshold)
		deletedPodNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deletePod = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedPodNames = append(deletedPodNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			gcc.podStore.Indexer.Add(&v1.Pod{
				ObjectMeta: v1.ObjectMeta{Name: pod.name, CreationTimestamp: metav1.Time{Time: creationTime}},
				Status:     v1.PodStatus{Phase: pod.phase},
				Spec:       v1.PodSpec{NodeName: "node"},
			})
		}

		gcc.podController = &FakeController{}

		gcc.gc()

		pass := true
		for _, pod := range deletedPodNames {
			if !test.deletedPodNames.Has(pod) {
				pass = false
			}
		}
		if len(deletedPodNames) != len(test.deletedPodNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedPodNames, deletedPodNames)
		}
	}
}

func TestGCOrphaned(t *testing.T) {
	type nameToPhase struct {
		name  string
		phase v1.PodPhase
	}

	testCases := []struct {
		pods            []nameToPhase
		threshold       int
		deletedPodNames sets.String
	}{
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       0,
			deletedPodNames: sets.NewString("a", "b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodRunning},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a"),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset()
		gcc := NewFromClient(client, test.threshold)
		deletedPodNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deletePod = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedPodNames = append(deletedPodNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			gcc.podStore.Indexer.Add(&v1.Pod{
				ObjectMeta: v1.ObjectMeta{Name: pod.name, CreationTimestamp: metav1.Time{Time: creationTime}},
				Status:     v1.PodStatus{Phase: pod.phase},
				Spec:       v1.PodSpec{NodeName: "node"},
			})
		}

		gcc.podController = &FakeController{}

		pods, err := gcc.podStore.List(labels.Everything())
		if err != nil {
			t.Errorf("Error while listing all Pods: %v", err)
			return
		}
		gcc.gcOrphaned(pods)

		pass := true
		for _, pod := range deletedPodNames {
			if !test.deletedPodNames.Has(pod) {
				pass = false
			}
		}
		if len(deletedPodNames) != len(test.deletedPodNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedPodNames, deletedPodNames)
		}
	}
}

func TestGCUnscheduledTerminating(t *testing.T) {
	type nameToPhase struct {
		name              string
		phase             v1.PodPhase
		deletionTimeStamp *metav1.Time
		nodeName          string
	}

	testCases := []struct {
		name            string
		pods            []nameToPhase
		deletedPodNames sets.String
	}{
		{
			name: "Unscheduled pod in any phase must be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
			},
			deletedPodNames: sets.NewString("a", "b", "c"),
		},
		{
			name: "Scheduled pod in any phase must not be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: nil, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: nil, nodeName: "node"},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: "node"},
			},
			deletedPodNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset()
		gcc := NewFromClient(client, -1)
		deletedPodNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deletePod = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedPodNames = append(deletedPodNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			gcc.podStore.Indexer.Add(&v1.Pod{
				ObjectMeta: v1.ObjectMeta{Name: pod.name, CreationTimestamp: metav1.Time{Time: creationTime},
					DeletionTimestamp: pod.deletionTimeStamp},
				Status: v1.PodStatus{Phase: pod.phase},
				Spec:   v1.PodSpec{NodeName: pod.nodeName},
			})
		}

		gcc.podController = &FakeController{}

		pods, err := gcc.podStore.List(labels.Everything())
		if err != nil {
			t.Errorf("Error while listing all Pods: %v", err)
			return
		}
		gcc.gcUnscheduledTerminating(pods)

		pass := true
		for _, pod := range deletedPodNames {
			if !test.deletedPodNames.Has(pod) {
				pass = false
			}
		}
		if len(deletedPodNames) != len(test.deletedPodNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v, test: %v", i, test.deletedPodNames, deletedPodNames, test.name)
		}
	}
}
