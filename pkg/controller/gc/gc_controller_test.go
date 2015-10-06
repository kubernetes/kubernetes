/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package gc

import (
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

type FakePodControl struct {
	podSpec       []api.PodTemplateSpec
	deletePodName []string
	lock          sync.Mutex
	err           error
}

func (f *FakePodControl) CreatePods(namespace string, spec *api.PodTemplateSpec, object runtime.Object) error {
	panic("unimplemented")
}

func (f *FakePodControl) CreatePodsOnNode(nodeName, namespace string, spec *api.PodTemplateSpec, object runtime.Object) error {
	panic("unimplemented")
}

func (f *FakePodControl) DeletePod(namespace string, podName string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.err != nil {
		return f.err
	}
	f.deletePodName = append(f.deletePodName, podName)
	return nil
}
func (f *FakePodControl) clear() {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.deletePodName = []string{}
	f.podSpec = []api.PodTemplateSpec{}
}

func TestGC(t *testing.T) {
	type nameToPhase struct {
		name  string
		phase api.PodPhase
	}

	testCases := []struct {
		pods            []nameToPhase
		threshold       int
		deletedPodNames sets.String
	}{
		{
			pods: []nameToPhase{
				{name: "a", phase: api.PodFailed},
				{name: "b", phase: api.PodSucceeded},
			},
			threshold:       0,
			deletedPodNames: sets.NewString("a", "b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: api.PodFailed},
				{name: "b", phase: api.PodSucceeded},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: api.PodFailed},
				{name: "b", phase: api.PodSucceeded},
			},
			threshold:       5,
			deletedPodNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := testclient.NewSimpleFake()
		gcc := New(client, controller.NoResyncPeriodFunc, test.threshold)
		fake := &FakePodControl{}
		gcc.podControl = fake

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			gcc.podStore.Store.Add(&api.Pod{
				ObjectMeta: api.ObjectMeta{Name: pod.name, CreationTimestamp: unversioned.Time{creationTime}},
				Status:     api.PodStatus{Phase: pod.phase},
			})
		}

		gcc.gc()

		pass := true
		for _, pod := range fake.deletePodName {
			if !test.deletedPodNames.Has(pod) {
				pass = false
			}
		}
		if len(fake.deletePodName) != len(test.deletedPodNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedPodNames, fake.deletePodName)
		}
	}
}

func TestTerminatedPodSelectorCompiles(t *testing.T) {
	compileTerminatedPodSelector()
}
