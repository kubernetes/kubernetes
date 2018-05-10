/*
Copyright 2018 The Kubernetes Authors.

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

package daemonset

import (
	"fmt"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	appstyped "k8s.io/client-go/kubernetes/typed/apps/v1"
	corev1typed "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
)

const (
	pollInterval = 100 * time.Millisecond
	pollTimeout  = 60 * time.Second
)

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

func newDaemonSet(name, namespace string) *apps.DaemonSet {
	two := int32(2)
	return &apps.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: apps.DaemonSetSpec{
			RevisionHistoryLimit: &two,
			Selector:             &metav1.LabelSelector{MatchLabels: testLabels()},
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.OnDeleteDaemonSetStrategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: testLabels(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "foo", Image: "bar"}},
				},
			},
		},
	}
}

func newRollbackStrategy() *apps.DaemonSetUpdateStrategy {
	one := intstr.FromInt(1)
	return &apps.DaemonSetUpdateStrategy{
		Type:          apps.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &apps.RollingUpdateDaemonSet{MaxUnavailable: &one},
	}
}

func newOnDeleteStrategy() *apps.DaemonSetUpdateStrategy {
	return &apps.DaemonSetUpdateStrategy{
		Type: apps.OnDeleteDaemonSetStrategyType,
	}
}

func updateStrategies() []*apps.DaemonSetUpdateStrategy {
	return []*apps.DaemonSetUpdateStrategy{newOnDeleteStrategy(), newRollbackStrategy()}
}

func allocatableResources(memory, cpu string) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceMemory: resource.MustParse(memory),
		v1.ResourceCPU:    resource.MustParse(cpu),
		v1.ResourcePods:   resource.MustParse("100"),
	}
}

func resourcePodSpec(nodeName, memory, cpu string) v1.PodSpec {
	return v1.PodSpec{
		NodeName: nodeName,
		Containers: []v1.Container{
			{
				Name:  "foo",
				Image: "bar",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse(memory),
						v1.ResourceCPU:    resource.MustParse(cpu),
					},
				},
			},
		},
	}
}

func newNode(name string, label map[string]string) *v1.Node {
	return &v1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Status: v1.NodeStatus{
			Conditions:  []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{v1.ResourcePods: resource.MustParse("100")},
		},
	}
}

func addNodes(nodeClient corev1typed.NodeInterface, startIndex, numNodes int, label map[string]string, t *testing.T) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		_, err := nodeClient.Create(newNode(fmt.Sprintf("node-%d", i), label))
		if err != nil {
			t.Fatalf("failed to create node: %v", err)
		}
	}
}

func validateDaemonSetPodsAndMarkReady(
	podClient corev1typed.PodInterface,
	podInformer cache.SharedIndexInformer,
	numberPods int,
	t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) != numberPods {
			return false, nil
		}

		for _, object := range objects {
			pod := object.(*v1.Pod)

			ownerReferences := pod.ObjectMeta.OwnerReferences
			if len(ownerReferences) != 1 {
				return false, fmt.Errorf("pod %s has %d OwnerReferences, expected only 1", pod.Name, len(ownerReferences))
			}
			controllerRef := ownerReferences[0]
			if got, want := controllerRef.Kind, "DaemonSet"; got != want {
				t.Errorf("controllerRef.Kind = %q, want %q", got, want)
			}
			if controllerRef.Controller == nil || *controllerRef.Controller != true {
				t.Errorf("controllerRef.Controller is not set to true")
			}

			if !podutil.IsPodReady(pod) {
				podCopy := pod.DeepCopy()
				podCopy.Status = v1.PodStatus{
					Phase:      v1.PodRunning,
					Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
				}
				_, err := podClient.UpdateStatus(podCopy)
				if err != nil {
					return false, err
				}
			}
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateDaemonSetStatus(
	dsClient appstyped.DaemonSetInterface,
	dsName string,
	dsNamespace string,
	expectedNumberReady int32,
	t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		ds, err := dsClient.Get(dsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ds.Status.NumberReady == expectedNumberReady, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func validateFailedPlacementEvent(eventClient corev1typed.EventInterface, t *testing.T) {
	if err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		eventList, err := eventClient.List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(eventList.Items) == 0 {
			return false, nil
		}
		if len(eventList.Items) > 1 {
			t.Errorf("expected 1 event got %d", len(eventList.Items))
		}
		event := eventList.Items[0]
		if event.Type != v1.EventTypeWarning {
			t.Errorf("event type expected %s got %s", v1.EventTypeWarning, event.Type)
		}
		if event.Reason != daemon.FailedPlacementReason {
			t.Errorf("event reason expected %s got %s", daemon.FailedPlacementReason, event.Reason)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}
