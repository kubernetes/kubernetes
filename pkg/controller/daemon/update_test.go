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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestDaemonSetUpdatesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 2
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
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

func TestDaemonSetUpdatesWhenNewPodIsNotReady(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
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
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
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
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
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
		ds             *extensions.DaemonSet
		nodeToPods     map[string][]*v1.Pod
		maxUnavailable int
		numUnavailable int
		Err            error
	}{
		{
			name: "No nodes",
			Manager: func() *daemonSetsController {
				manager, _, _ := newTestController()
				return manager
			}(),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(0)
				ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
				return ds
			}(),
			nodeToPods:     make(map[string][]*v1.Pod),
			maxUnavailable: 0,
			numUnavailable: 0,
		},
		{
			name: "Two nodes with ready pods",
			Manager: func() *daemonSetsController {
				manager, _, _ := newTestController()
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(1)
				ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
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
				manager, _, _ := newTestController()
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromInt(0)
				ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
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
				manager, _, _ := newTestController()
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			}(),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("x")
				intStr := intstr.FromString("50%")
				ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
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

func TestDaemonSetSurgingUpdatesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxSurge := 2
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxSurge)
	ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
	ds.Spec.TemplateGeneration++
	manager.dsStore.Update(ds)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxSurge, 0, 0)
	clearExpectations(t, manager, ds, podControl)
	markPodsReady(podControl.podStore)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxSurge, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxSurge, 0, 0)
	clearExpectations(t, manager, ds, podControl)
	markPodsReady(podControl.podStore)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxSurge, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0, 0)
	clearExpectations(t, manager, ds, podControl)
	markPodsReady(podControl.podStore)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 1, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestPruneSurgingDaemonPods(t *testing.T) {
	manager, _, _ := newTestController()
	pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
	pod1 := newPod("pod-1", "node-0", simpleDaemonSetLabel, nil)
	newVersionHash := "new-version-hash"
	newVersionLabels := map[string]string{"name": "simple-daemon", "type": "production", extensions.DefaultDaemonSetUniqueLabelKey: newVersionHash}
	podNew := newPod("pod-new", "node-0", newVersionLabels, nil)
	for _, c := range []struct {
		name string
		ds   *extensions.DaemonSet
		hash string
		pods []*v1.Pod
		want []*v1.Pod
	}{{
		name: "No pods",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		pods: nil,
		want: nil,
	}, {
		name: "One pod",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		pods: []*v1.Pod{pod0},
		want: []*v1.Pod{pod0},
	}, {
		name: "One pod, wrong generation",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		hash: newVersionHash,
		pods: []*v1.Pod{pod0},
		want: []*v1.Pod{pod0},
	}, {
		name: "Two pods, both wrong generation",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			ds.Spec.TemplateGeneration = 1234
			return ds
		}(),
		hash: newVersionHash,
		pods: []*v1.Pod{pod0, pod1},
		want: []*v1.Pod{pod0, pod1},
	}, {
		name: "Two pods",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			ds.Spec.TemplateGeneration = 1234
			return ds
		}(),
		hash: newVersionHash,
		pods: []*v1.Pod{pod0, podNew},
		want: []*v1.Pod{podNew},
	}, {
		name: "Three pods",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(0)
			ds.Spec.UpdateStrategy.Type = extensions.SurgingRollingUpdateDaemonSetStrategyType
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			ds.Spec.TemplateGeneration = 1234
			return ds
		}(),
		hash: newVersionHash,
		pods: []*v1.Pod{pod0, podNew, pod1},
		want: []*v1.Pod{podNew},
	}, {
		name: "Non-surging strategy",
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			ds.Spec.UpdateStrategy.Type = extensions.OnDeleteDaemonSetStrategyType
			return ds
		}(),
		pods: []*v1.Pod{pod0, pod1},
		want: []*v1.Pod{pod0, pod1},
	}} {
		got := manager.pruneSurgingDaemonPods(c.ds, c.pods, c.hash)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Test case %q: got %d pods, want: %d pods", c.name, len(got), len(c.want))
		}
	}
}

func TestGetSurgeNumbers(t *testing.T) {
	for _, c := range []struct {
		name       string
		manager    *daemonSetsController
		ds         *extensions.DaemonSet
		nodeToPods map[string][]*v1.Pod
		maxSurge   int
		numSurge   int
	}{{
		name: "No nodes",
		manager: func() *daemonSetsController {
			manager, _, _ := newTestController()
			return manager
		}(),
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(2)
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		nodeToPods: make(map[string][]*v1.Pod),
		maxSurge:   2,
		numSurge:   0,
	}, {
		name: "Two nodes with pods",
		manager: func() *daemonSetsController {
			manager, _, _ := newTestController()
			addNodes(manager.nodeStore, 0, 2, nil)
			return manager
		}(),
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(1)
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		nodeToPods: func() map[string][]*v1.Pod {
			mapping := make(map[string][]*v1.Pod)
			pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
			pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
			mapping["node-0"] = []*v1.Pod{pod0}
			mapping["node-1"] = []*v1.Pod{pod1}
			return mapping
		}(),
		maxSurge: 1,
		numSurge: 0,
	}, {
		name: "Two nodes with pods, one surging",
		manager: func() *daemonSetsController {
			manager, _, _ := newTestController()
			addNodes(manager.nodeStore, 0, 2, nil)
			return manager
		}(),
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(1)
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		nodeToPods: func() map[string][]*v1.Pod {
			mapping := make(map[string][]*v1.Pod)
			pod0a := newPod("pod-0a", "node-0", simpleDaemonSetLabel, nil)
			pod0b := newPod("pod-0b", "node-0", simpleDaemonSetLabel, nil)
			pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
			mapping["node-0"] = []*v1.Pod{pod0a, pod0b}
			mapping["node-1"] = []*v1.Pod{pod1}
			return mapping
		}(),
		maxSurge: 1,
		numSurge: 1,
	}, {
		name: "Two nodes, one surging, one missing a pod",
		manager: func() *daemonSetsController {
			manager, _, _ := newTestController()
			addNodes(manager.nodeStore, 0, 2, nil)
			return manager
		}(),
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromInt(1)
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		nodeToPods: func() map[string][]*v1.Pod {
			mapping := make(map[string][]*v1.Pod)
			pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
			pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
			mapping["node-0"] = []*v1.Pod{pod0, pod1}
			mapping["node-1"] = []*v1.Pod{}
			return mapping
		}(),
		maxSurge: 1,
		numSurge: 1,
	}, {
		name: "Two nodes with pods, one surging, MaxSurge in percents",
		manager: func() *daemonSetsController {
			manager, _, _ := newTestController()
			addNodes(manager.nodeStore, 0, 2, nil)
			return manager
		}(),
		ds: func() *extensions.DaemonSet {
			ds := newDaemonSet("x")
			intStr := intstr.FromString("50%")
			ds.Spec.UpdateStrategy.SurgingRollingUpdate = &extensions.SurgingRollingUpdateDaemonSet{MaxSurge: &intStr}
			return ds
		}(),
		nodeToPods: func() map[string][]*v1.Pod {
			mapping := make(map[string][]*v1.Pod)
			pod0a := newPod("pod-0a", "node-0", simpleDaemonSetLabel, nil)
			pod0b := newPod("pod-0b", "node-0", simpleDaemonSetLabel, nil)
			pod1 := newPod("pod-1", "node-1", simpleDaemonSetLabel, nil)
			mapping["node-0"] = []*v1.Pod{pod0a, pod0b}
			mapping["node-1"] = []*v1.Pod{pod1}
			return mapping
		}(),
		maxSurge: 1,
		numSurge: 1,
	}} {
		c.manager.dsStore.Add(c.ds)
		maxSurge, numSurge, err := c.manager.getSurgeNumbers(c.ds, c.nodeToPods)
		if err != nil {
			t.Errorf("Test case: %s. got error %v, want: nil", c.name, err)
		} else if maxSurge != c.maxSurge || numSurge != c.numSurge {
			t.Errorf("Test case: %s. Wrong values. maxSurge: %d, expected: %d, numSurge: %d. expected: %d", c.name, maxSurge, c.maxSurge, numSurge, c.numSurge)
		}
	}
}
