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

package daemon

import (
	"testing"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestDaemonSetUpdatesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 2
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 1, 0)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesWhenNewPosIsNotReady(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	// new pods are not ready numUnavailable == maxUnavailable
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesAllOldPodsNotReady(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	// all old pods are unavailable so should be removed
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 5, 0)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesNoTemplateChanged(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	// template is not changed no pod should be removed
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestGetUnavailableNumbers(t *testing.T) {
	cases := []struct {
		name           string
		Manager        *daemonSetsController
		ds             *apps.DaemonSet
		nodeToPods     map[string][]*v1.Pod
		maxUnavailable int
		numUnavailable int
		Err            error
	}{
		{
			name: "No nodes",
			Manager: func() *daemonSetsController {
				manager, _, _, err := newTestController()
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				return manager
			}(),
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(0)
				ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods:     make(map[string][]*v1.Pod),
			maxUnavailable: 0,
			numUnavailable: 0,
		},
		{
			name: "Two nodes with ready pods",
			Manager: func() *daemonSetsController {
				manager, _, _, err := newTestController()
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(1)
				ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
				markPodReady(pod0)
				markPodReady(pod1)
				mapping["node-0"] = []*v1.Pod{pod0}
				mapping["node-1"] = []*v1.Pod{pod1}
				return mapping
			}(),
			maxUnavailable: 1,
			numUnavailable: 0,
		},
		{
			name: "Two nodes, one node without pods",
			Manager: func() *daemonSetsController {
				manager, _, _, err := newTestController()
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(0)
				ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				markPodReady(pod0)
				mapping["node-0"] = []*v1.Pod{pod0}
				return mapping
			}(),
			maxUnavailable: 0,
			numUnavailable: 1,
		},
		{
			name: "Two nodes with pods, MaxUnavailable in percents",
			Manager: func() *daemonSetsController {
				manager, _, _, err := newTestController()
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromString("50%")
				ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
				markPodReady(pod0)
				markPodReady(pod1)
				mapping["node-0"] = []*v1.Pod{pod0}
				mapping["node-1"] = []*v1.Pod{pod1}
				return mapping
			}(),
			maxUnavailable: 1,
			numUnavailable: 0,
		},
		{
			name: "Two nodes with pods, MaxUnavailable in percents, pod terminating",
			Manager: func() *daemonSetsController {
				manager, _, _, err := newTestController()
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromString("50%")
				ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
				now := metav1.Now()
				markPodReady(pod0)
				markPodReady(pod1)
				pod1.DeletionTimestamp = &now
				mapping["node-0"] = []*v1.Pod{pod0}
				mapping["node-1"] = []*v1.Pod{pod1}
				return mapping
			}(),
			maxUnavailable: 1,
			numUnavailable: 1,
		},
	}

	for _, c := range cases {
		c.Manager.dsStore.Add(c.ds)
		maxUnavailable, numUnavailable, err := c.Manager.getUnavailableNumbers(c.ds, c.nodeToPods)
		if err != nil && c.Err != nil {
			if c.Err != err {
				t.Errorf("Test case: %s. Expected error: %v but got: %v", c.name, c.Err, err)
			}
		} else if err != nil {
			t.Errorf("Test case: %s. Unexpected error: %v", c.name, err)
		} else if maxUnavailable != c.maxUnavailable || numUnavailable != c.numUnavailable {
			t.Errorf("Test case: %s. Wrong values. maxUnavailable: %d, expected: %d, numUnavailable: %d. expected: %d", c.name, maxUnavailable, c.maxUnavailable, numUnavailable, c.numUnavailable)
		}
	}
}
