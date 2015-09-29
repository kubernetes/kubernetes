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

package cache

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestStoreToNodeLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	ids := sets.NewString("foo", "bar", "baz")
	for id := range ids {
		store.Add(&api.Node{ObjectMeta: api.ObjectMeta{Name: id}})
	}
	sml := StoreToNodeLister{store}

	gotNodes, err := sml.List()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	got := make([]string, len(gotNodes.Items))
	for ix := range gotNodes.Items {
		got[ix] = gotNodes.Items[ix].Name
	}
	if !ids.HasAll(got...) || len(got) != len(ids) {
		t.Errorf("Expected %v, got %v", ids, got)
	}
}

func TestStoreToReplicationControllerLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	lister := StoreToReplicationControllerLister{store}
	testCases := []struct {
		inRCs      []*api.ReplicationController
		list       func() ([]api.ReplicationController, error)
		outRCNames sets.String
		expectErr  bool
	}{
		// Basic listing with all labels and no selectors
		{
			inRCs: []*api.ReplicationController{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}},
			},
			list: func() ([]api.ReplicationController, error) {
				return lister.List()
			},
			outRCNames: sets.NewString("basic"),
		},
		// No pod labels
		{
			inRCs: []*api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
					Spec: api.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "baz"},
					},
				},
			},
			list: func() ([]api.ReplicationController, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{Name: "pod1", Namespace: "ns"},
				}
				return lister.GetPodControllers(pod)
			},
			outRCNames: sets.NewString(),
			expectErr:  true,
		},
		// No RC selectors
		{
			inRCs: []*api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
				},
			},
			list: func() ([]api.ReplicationController, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod1",
						Namespace: "ns",
						Labels:    map[string]string{"foo": "bar"},
					},
				}
				return lister.GetPodControllers(pod)
			},
			outRCNames: sets.NewString(),
			expectErr:  true,
		},
		// Matching labels to selectors and namespace
		{
			inRCs: []*api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Spec: api.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: api.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			list: func() ([]api.ReplicationController, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod1",
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "ns",
					},
				}
				return lister.GetPodControllers(pod)
			},
			outRCNames: sets.NewString("bar"),
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRCs {
			store.Add(r)
		}

		gotControllers, err := c.list()
		if err != nil && c.expectErr {
			continue
		} else if c.expectErr {
			t.Fatalf("Expected error, got none")
		} else if err != nil {
			t.Fatalf("Unexpected error %#v", err)
		}
		gotNames := make([]string, len(gotControllers))
		for ix := range gotControllers {
			gotNames[ix] = gotControllers[ix].Name
		}
		if !c.outRCNames.HasAll(gotNames...) || len(gotNames) != len(c.outRCNames) {
			t.Errorf("Unexpected got controllers %+v expected %+v", gotNames, c.outRCNames)
		}
	}
}

func TestStoreToDaemonSetLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	lister := StoreToDaemonSetLister{store}
	testCases := []struct {
		inDSs             []*experimental.DaemonSet
		list              func() ([]experimental.DaemonSet, error)
		outDaemonSetNames sets.String
		expectErr         bool
	}{
		// Basic listing
		{
			inDSs: []*experimental.DaemonSet{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}},
			},
			list: func() ([]experimental.DaemonSet, error) {
				return lister.List()
			},
			outDaemonSetNames: sets.NewString("basic"),
		},
		// Listing multiple daemon sets
		{
			inDSs: []*experimental.DaemonSet{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}},
				{ObjectMeta: api.ObjectMeta{Name: "complex"}},
				{ObjectMeta: api.ObjectMeta{Name: "complex2"}},
			},
			list: func() ([]experimental.DaemonSet, error) {
				return lister.List()
			},
			outDaemonSetNames: sets.NewString("basic", "complex", "complex2"),
		},
		// No pod labels
		{
			inDSs: []*experimental.DaemonSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
					Spec: experimental.DaemonSetSpec{
						Selector: map[string]string{"foo": "baz"},
					},
				},
			},
			list: func() ([]experimental.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{Name: "pod1", Namespace: "ns"},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString(),
			expectErr:         true,
		},
		// No DS selectors
		{
			inDSs: []*experimental.DaemonSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
				},
			},
			list: func() ([]experimental.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod1",
						Namespace: "ns",
						Labels:    map[string]string{"foo": "bar"},
					},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString(),
			expectErr:         true,
		},
		// Matching labels to selectors and namespace
		{
			inDSs: []*experimental.DaemonSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Spec: experimental.DaemonSetSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: experimental.DaemonSetSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			list: func() ([]experimental.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod1",
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "ns",
					},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString("bar"),
		},
	}
	for _, c := range testCases {
		for _, r := range c.inDSs {
			store.Add(r)
		}

		daemonSets, err := c.list()
		if err != nil && c.expectErr {
			continue
		} else if c.expectErr {
			t.Fatalf("Expected error, got none")
		} else if err != nil {
			t.Fatalf("Unexpected error %#v", err)
		}
		daemonSetNames := make([]string, len(daemonSets))
		for ix := range daemonSets {
			daemonSetNames[ix] = daemonSets[ix].Name
		}
		if !c.outDaemonSetNames.HasAll(daemonSetNames...) || len(daemonSetNames) != len(c.outDaemonSetNames) {
			t.Errorf("Unexpected got controllers %+v expected %+v", daemonSetNames, c.outDaemonSetNames)
		}
	}
}

func TestStoreToPodLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	ids := []string{"foo", "bar", "baz"}
	for _, id := range ids {
		store.Add(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   id,
				Labels: map[string]string{"name": id},
			},
		})
	}
	spl := StoreToPodLister{store}

	for _, id := range ids {
		got, err := spl.List(labels.Set{"name": id}.AsSelector())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := 1, len(got); e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}
		if e, a := id, got[0].Name; e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}

		exists, err := spl.Exists(&api.Pod{ObjectMeta: api.ObjectMeta{Name: id}})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !exists {
			t.Errorf("exists returned false for %v", id)
		}
	}

	exists, err := spl.Exists(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "qux"}})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if exists {
		t.Errorf("Unexpected pod exists")
	}
}

func TestStoreToServiceLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
		},
	})
	store.Add(&api.Service{ObjectMeta: api.ObjectMeta{Name: "bar"}})
	ssl := StoreToServiceLister{store}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   "foopod",
			Labels: map[string]string{"role": "foo"},
		},
	}

	services, err := ssl.GetPodServices(pod)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(services) != 1 {
		t.Fatalf("Expected 1 service, got %v", len(services))
	}
	if e, a := "foo", services[0].Name; e != a {
		t.Errorf("Expected service %q, got %q", e, a)
	}
}
