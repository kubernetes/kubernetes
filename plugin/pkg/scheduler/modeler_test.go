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

package scheduler

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/labels"
)

type nn struct {
	namespace, name string
}

type names []nn

func (ids names) list() []*api.Pod {
	out := make([]*api.Pod, 0, len(ids))
	for _, id := range ids {
		out = append(out, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Namespace: id.namespace,
				Name:      id.name,
			},
		})
	}
	return out
}

func (ids names) has(pod *api.Pod) bool {
	for _, id := range ids {
		if pod.Namespace == id.namespace && pod.Name == id.name {
			return true
		}
	}
	return false
}

func TestModeler(t *testing.T) {
	table := []struct {
		queuedPods    []*api.Pod
		scheduledPods []*api.Pod
		assumedPods   []*api.Pod
		expectPods    names
	}{
		{
			queuedPods:    names{}.list(),
			scheduledPods: names{{"default", "foo"}, {"custom", "foo"}}.list(),
			assumedPods:   names{{"default", "foo"}}.list(),
			expectPods:    names{{"default", "foo"}, {"custom", "foo"}},
		}, {
			queuedPods:    names{}.list(),
			scheduledPods: names{{"default", "foo"}}.list(),
			assumedPods:   names{{"default", "foo"}, {"custom", "foo"}}.list(),
			expectPods:    names{{"default", "foo"}, {"custom", "foo"}},
		}, {
			queuedPods:    names{{"custom", "foo"}}.list(),
			scheduledPods: names{{"default", "foo"}}.list(),
			assumedPods:   names{{"default", "foo"}, {"custom", "foo"}}.list(),
			expectPods:    names{{"default", "foo"}},
		},
	}

	for _, item := range table {
		q := &cache.StoreToPodLister{cache.NewStore(cache.MetaNamespaceKeyFunc)}
		for _, pod := range item.queuedPods {
			q.Store.Add(pod)
		}
		s := &cache.StoreToPodLister{cache.NewStore(cache.MetaNamespaceKeyFunc)}
		for _, pod := range item.scheduledPods {
			s.Store.Add(pod)
		}
		m := NewSimpleModeler(q, s)
		for _, pod := range item.assumedPods {
			m.AssumePod(pod)
		}

		list, err := m.PodLister().List(labels.Everything())
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		found := 0
		for _, pod := range list {
			if item.expectPods.has(pod) {
				found++
			} else {
				t.Errorf("found unexpected pod %#v", pod)
			}
		}
		if e, a := item.expectPods, found; len(e) != a {
			t.Errorf("Expected pods:\n%+v\nFound pods:\n%s\n", podNames(e.list()), podNames(list))
		}
	}
}
