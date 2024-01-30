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
	"context"
	"reflect"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon/util"
	testingclock "k8s.io/utils/clock/testing"
)

func TestDaemonSetUpdatesPods(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 2
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt32(int32(maxUnavailable))
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 1, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 1, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesPodsWithMaxSurge(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	// surge is thhe controlling amount
	maxSurge := 2
	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromInt32(int32(maxSurge)))
	manager.dsStore.Update(ds)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, 0, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 5%maxSurge, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 5%maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
}

func TestDaemonSetUpdatesPodsNotMatchTainstWithMaxSurge(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	ds := newDaemonSet("foo")
	maxSurge := 1
	ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromInt(maxSurge))
	tolerations := []v1.Toleration{
		{Key: "node-role.kubernetes.io/control-plane", Operator: v1.TolerationOpExists},
	}
	setDaemonSetToleration(ds, tolerations)
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	err = manager.dsStore.Add(ds)
	if err != nil {
		t.Fatal(err)
	}

	// Add five nodes and taint to one node
	addNodes(manager.nodeStore, 0, 5, nil)
	taints := []v1.Taint{
		{Key: "node-role.kubernetes.io/control-plane", Effect: v1.TaintEffectNoSchedule},
	}
	node := newNode("node-0", nil)
	setNodeTaint(node, taints)
	err = manager.nodeStore.Update(node)
	if err != nil {
		t.Fatal(err)
	}

	// Create DaemonSet with toleration
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	// RollingUpdate DaemonSet without toleration
	ds.Spec.Template.Spec.Tolerations = nil
	err = manager.dsStore.Update(ds)
	if err != nil {
		t.Fatal(err)
	}

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, 1, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxSurge, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, maxSurge, 0)
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
}

func TestDaemonSetUpdatesWhenNewPodIsNotReady(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	err = manager.dsStore.Add(ds)
	if err != nil {
		t.Fatal(err)
	}
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt32(int32(maxUnavailable))
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	err = manager.dsStore.Update(ds)
	if err != nil {
		t.Fatal(err)
	}

	// new pods are not ready numUnavailable == maxUnavailable
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, maxUnavailable, 0)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, maxUnavailable, 0, 0)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesAllOldPodsNotReady(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	err = manager.dsStore.Add(ds)
	if err != nil {
		t.Fatal(err)
	}
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt32(int32(maxUnavailable))
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	err = manager.dsStore.Update(ds)
	if err != nil {
		t.Fatal(err)
	}

	// all old pods are unavailable so should be removed
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 5, 0)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesAllOldPodsNotReadyMaxSurge(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	maxSurge := 3
	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromInt32(int32(maxSurge)))
	manager.dsStore.Update(ds)

	// all old pods are unavailable so should be surged
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(100, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	// waiting for pods to go ready, old pods are deleted
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(200, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 5, 0)

	setPodReadiness(t, manager, true, 5, func(_ *v1.Pod) bool { return true })
	ds.Spec.MinReadySeconds = 15
	ds.Spec.Template.Spec.Containers[0].Image = "foo3/bar3"
	manager.dsStore.Update(ds)

	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(300, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 3, 0, 0)

	hash, err := currentDSHash(manager, ds)
	if err != nil {
		t.Fatal(err)
	}
	currentPods := podsByNodeMatchingHash(manager, hash)
	// mark two updated pods as ready at time 300
	setPodReadiness(t, manager, true, 2, func(pod *v1.Pod) bool {
		return pod.Labels[apps.ControllerRevisionHashLabelKey] == hash
	})
	// mark one of the old pods that is on a node without an updated pod as unready
	setPodReadiness(t, manager, false, 1, func(pod *v1.Pod) bool {
		nodeName, err := util.GetTargetNodeName(pod)
		if err != nil {
			t.Fatal(err)
		}
		return pod.Labels[apps.ControllerRevisionHashLabelKey] != hash && len(currentPods[nodeName]) == 0
	})

	// the new pods should still be considered waiting to hit min readiness, so one pod should be created to replace
	// the deleted old pod
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(310, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 1, 0, 0)

	// the new pods are now considered available, so delete the old pods
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(320, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 1, 3, 0)

	// mark all updated pods as ready at time 320
	currentPods = podsByNodeMatchingHash(manager, hash)
	setPodReadiness(t, manager, true, 3, func(pod *v1.Pod) bool {
		return pod.Labels[apps.ControllerRevisionHashLabelKey] == hash
	})

	// the new pods are now considered available, so delete the old pods
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(340, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 2, 0)

	// controller has completed upgrade
	manager.failedPodsBackoff.Clock = testingclock.NewFakeClock(time.Unix(350, 0))
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
}

func podsByNodeMatchingHash(dsc *daemonSetsController, hash string) map[string][]string {
	byNode := make(map[string][]string)
	for _, obj := range dsc.podStore.List() {
		pod := obj.(*v1.Pod)
		if pod.Labels[apps.ControllerRevisionHashLabelKey] != hash {
			continue
		}
		nodeName, err := util.GetTargetNodeName(pod)
		if err != nil {
			panic(err)
		}
		byNode[nodeName] = append(byNode[nodeName], pod.Name)
	}
	return byNode
}

func setPodReadiness(t *testing.T, dsc *daemonSetsController, ready bool, count int, fn func(*v1.Pod) bool) {
	t.Helper()
	logger, _ := ktesting.NewTestContext(t)
	for _, obj := range dsc.podStore.List() {
		if count <= 0 {
			break
		}
		pod := obj.(*v1.Pod)
		if pod.DeletionTimestamp != nil {
			continue
		}
		if podutil.IsPodReady(pod) == ready {
			continue
		}
		if !fn(pod) {
			continue
		}
		condition := v1.PodCondition{Type: v1.PodReady}
		if ready {
			condition.Status = v1.ConditionTrue
		} else {
			condition.Status = v1.ConditionFalse
		}
		if !podutil.UpdatePodCondition(&pod.Status, &condition) {
			t.Fatal("failed to update pod")
		}
		// TODO: workaround UpdatePodCondition calling time.Now() directly
		setCondition := podutil.GetPodReadyCondition(pod.Status)
		setCondition.LastTransitionTime.Time = dsc.failedPodsBackoff.Clock.Now()
		logger.Info("marked pod ready", "pod", pod.Name, "ready", ready)
		count--
	}
	if count > 0 {
		t.Fatalf("could not mark %d pods ready=%t", count, ready)
	}
}

func currentDSHash(dsc *daemonSetsController, ds *apps.DaemonSet) (string, error) {
	// Construct histories of the DaemonSet, and get the hash of current history
	cur, _, err := dsc.constructHistory(context.TODO(), ds)
	if err != nil {
		return "", err
	}
	return cur.Labels[apps.DefaultDaemonSetUniqueLabelKey], nil

}

func TestDaemonSetUpdatesNoTemplateChanged(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ds := newDaemonSet("foo")
	manager, podControl, _, err := newTestController(ctx, ds)
	if err != nil {
		t.Fatalf("error creating DaemonSets controller: %v", err)
	}
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	expectSyncDaemonSets(t, manager, ds, podControl, 5, 0, 0)

	ds.Spec.UpdateStrategy.Type = apps.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt32(int32(maxUnavailable))
	ds.Spec.UpdateStrategy.RollingUpdate = &apps.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	// template is not changed no pod should be removed
	clearExpectations(t, manager, ds, podControl)
	expectSyncDaemonSets(t, manager, ds, podControl, 0, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func newUpdateSurge(value intstr.IntOrString) apps.DaemonSetUpdateStrategy {
	zero := intstr.FromInt32(0)
	return apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &zero,
			MaxSurge:       &value,
		},
	}
}

func newUpdateUnavailable(value intstr.IntOrString) apps.DaemonSetUpdateStrategy {
	return apps.DaemonSetUpdateStrategy{
		Type: apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{
			MaxUnavailable: &value,
		},
	}
}

func TestGetUnavailableNumbers(t *testing.T) {
	cases := []struct {
		name                   string
		ManagerFunc            func(ctx context.Context) *daemonSetsController
		ds                     *apps.DaemonSet
		nodeToPods             map[string][]*v1.Pod
		maxSurge               int
		maxUnavailable         int
		desiredNumberScheduled int
		emptyNodes             int
		Err                    error
	}{
		{
			name: "No nodes",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateUnavailable(intstr.FromInt32(0))
				return ds
			}(),
			nodeToPods:     make(map[string][]*v1.Pod),
			maxUnavailable: 0,
			emptyNodes:     0,
		},
		{
			name: "Two nodes with ready pods",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateUnavailable(intstr.FromInt32(1))
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
			maxUnavailable:         1,
			desiredNumberScheduled: 2,
			emptyNodes:             0,
		},
		{
			name: "Two nodes, one node without pods",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateUnavailable(intstr.FromInt32(0))
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				markPodReady(pod0)
				mapping["node-0"] = []*v1.Pod{pod0}
				return mapping
			}(),
			maxUnavailable:         1,
			desiredNumberScheduled: 2,
			emptyNodes:             1,
		},
		{
			name: "Two nodes, one node without pods, surge",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromInt32(0))
				return ds
			}(),
			nodeToPods: func() map[string][]*v1.Pod {
				mapping := make(map[string][]*v1.Pod)
				pod0 := newPod("pod-0", "node-0", simpleDaemonSetLabel, nil)
				markPodReady(pod0)
				mapping["node-0"] = []*v1.Pod{pod0}
				return mapping
			}(),
			maxUnavailable:         1,
			desiredNumberScheduled: 2,
			emptyNodes:             1,
		},
		{
			name: "Two nodes with pods, MaxUnavailable in percents",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateUnavailable(intstr.FromString("50%"))
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
			maxUnavailable:         1,
			desiredNumberScheduled: 2,
			emptyNodes:             0,
		},
		{
			name: "Two nodes with pods, MaxUnavailable in percents, surge",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromString("50%"))
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
			maxSurge:               1,
			maxUnavailable:         0,
			desiredNumberScheduled: 2,
			emptyNodes:             0,
		},
		{
			name: "Two nodes with pods, MaxUnavailable is 100%, surge",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 2, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateSurge(intstr.FromString("100%"))
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
			maxSurge:               2,
			maxUnavailable:         0,
			desiredNumberScheduled: 2,
			emptyNodes:             0,
		},
		{
			name: "Two nodes with pods, MaxUnavailable in percents, pod terminating",
			ManagerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				addNodes(manager.nodeStore, 0, 3, nil)
				return manager
			},
			ds: func() *apps.DaemonSet {
				ds := newDaemonSet("x")
				ds.Spec.UpdateStrategy = newUpdateUnavailable(intstr.FromString("50%"))
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
			maxUnavailable:         2,
			desiredNumberScheduled: 3,
			emptyNodes:             1,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			manager := c.ManagerFunc(ctx)
			manager.dsStore.Add(c.ds)
			nodeList, err := manager.nodeLister.List(labels.Everything())
			if err != nil {
				t.Fatalf("error listing nodes: %v", err)
			}
			maxSurge, maxUnavailable, desiredNumberScheduled, err := manager.updatedDesiredNodeCounts(ctx, c.ds, nodeList, c.nodeToPods)
			if err != nil && c.Err != nil {
				if c.Err != err {
					t.Fatalf("Expected error: %v but got: %v", c.Err, err)
				}
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if maxSurge != c.maxSurge || maxUnavailable != c.maxUnavailable || desiredNumberScheduled != c.desiredNumberScheduled {
				t.Errorf("Wrong values. maxSurge: %d, expected %d, maxUnavailable: %d, expected: %d, desiredNumberScheduled: %d, expected: %d", maxSurge, c.maxSurge, maxUnavailable, c.maxUnavailable, desiredNumberScheduled, c.desiredNumberScheduled)
			}
			var emptyNodes int
			for _, pods := range c.nodeToPods {
				if len(pods) == 0 {
					emptyNodes++
				}
			}
			if emptyNodes != c.emptyNodes {
				t.Errorf("expected numEmpty to be %d, was %d", c.emptyNodes, emptyNodes)
			}
		})
	}
}

func TestControlledHistories(t *testing.T) {
	ds1 := newDaemonSet("ds1")
	crOfDs1 := newControllerRevision(ds1.GetName()+"-x1", ds1.GetNamespace(), ds1.Spec.Template.Labels,
		[]metav1.OwnerReference{*metav1.NewControllerRef(ds1, controllerKind)})
	orphanCrInSameNsWithDs1 := newControllerRevision(ds1.GetName()+"-x2", ds1.GetNamespace(), ds1.Spec.Template.Labels, nil)
	orphanCrNotInSameNsWithDs1 := newControllerRevision(ds1.GetName()+"-x3", ds1.GetNamespace()+"-other", ds1.Spec.Template.Labels, nil)
	cases := []struct {
		name                      string
		managerFunc               func(ctx context.Context) *daemonSetsController
		historyCRAll              []*apps.ControllerRevision
		expectControllerRevisions []*apps.ControllerRevision
	}{
		{
			name: "controller revision in the same namespace",
			managerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx, ds1, crOfDs1, orphanCrInSameNsWithDs1)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				manager.dsStore.Add(ds1)
				return manager
			},
			historyCRAll:              []*apps.ControllerRevision{crOfDs1, orphanCrInSameNsWithDs1},
			expectControllerRevisions: []*apps.ControllerRevision{crOfDs1, orphanCrInSameNsWithDs1},
		},
		{
			name: "Skip adopting the controller revision in namespace other than the one in which DS lives",
			managerFunc: func(ctx context.Context) *daemonSetsController {
				manager, _, _, err := newTestController(ctx, ds1, orphanCrNotInSameNsWithDs1)
				if err != nil {
					t.Fatalf("error creating DaemonSets controller: %v", err)
				}
				manager.dsStore.Add(ds1)
				return manager
			},
			historyCRAll:              []*apps.ControllerRevision{orphanCrNotInSameNsWithDs1},
			expectControllerRevisions: []*apps.ControllerRevision{},
		},
	}
	for _, c := range cases {
		_, ctx := ktesting.NewTestContext(t)
		manager := c.managerFunc(ctx)
		for _, h := range c.historyCRAll {
			manager.historyStore.Add(h)
		}
		crList, err := manager.controlledHistories(context.TODO(), ds1)
		if err != nil {
			t.Fatalf("Test case: %s. Unexpected error: %v", c.name, err)
		}
		if len(crList) != len(c.expectControllerRevisions) {
			t.Errorf("Test case: %s, expect controllerrevision count %d but got:%d",
				c.name, len(c.expectControllerRevisions), len(crList))
		} else {
			// check controller revisions match
			for _, cr := range crList {
				found := false
				for _, expectCr := range c.expectControllerRevisions {
					if reflect.DeepEqual(cr, expectCr) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Test case: %s, controllerrevision %v not expected",
						c.name, cr)
				}
			}
			t.Logf("Test case: %s done", c.name)
		}
	}
}
