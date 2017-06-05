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

	"k8s.io/apimachinery/pkg/util/intstr"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestDaemonSetUpdatesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 2
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
	manager.dsStore.Update(ds)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 1)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
	markPodsReady(podControl.podStore)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesWhenNewPosIsNotReady(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)
	markPodsReady(podControl.podStore)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
	manager.dsStore.Update(ds)

	// new pods are not ready numUnavailable == maxUnavailable
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, maxUnavailable)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, maxUnavailable, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesAllOldPodsNotReady(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)

	ds.Spec.Template.Spec.Containers[0].Image = "foo2/bar2"
	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	ds.Spec.TemplateGeneration++
	manager.dsStore.Update(ds)

	// all old pods are unavailable so should be removed
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 5)
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)

	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}

func TestDaemonSetUpdatesNoTemplateChanged(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	maxUnavailable := 3
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)

	ds.Spec.UpdateStrategy.Type = extensions.RollingUpdateDaemonSetStrategyType
	intStr := intstr.FromInt(maxUnavailable)
	ds.Spec.UpdateStrategy.RollingUpdate = &extensions.RollingUpdateDaemonSet{MaxUnavailable: &intStr}
	manager.dsStore.Update(ds)

	// template is not changed no pod should be removed
	clearExpectations(t, manager, ds, podControl)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	clearExpectations(t, manager, ds, podControl)
}
