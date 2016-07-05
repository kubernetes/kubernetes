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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/sets"
)

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
		client := fake.NewSimpleClientset()
		gcc := New(client, controller.NoResyncPeriodFunc, test.threshold)
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
			gcc.podStore.Indexer.Add(&api.Pod{
				ObjectMeta: api.ObjectMeta{Name: pod.name, CreationTimestamp: unversioned.Time{Time: creationTime}},
				Status:     api.PodStatus{Phase: pod.phase},
			})
		}

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
