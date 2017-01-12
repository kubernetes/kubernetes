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

package daemon

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	simpleDaemonSetLabel  = map[string]string{"name": "simple-daemon", "type": "production"}
	simpleDaemonSetLabel2 = map[string]string{"name": "simple-daemon", "type": "test"}
	simpleNodeLabel       = map[string]string{"color": "blue", "speed": "fast"}
	simpleNodeLabel2      = map[string]string{"color": "red", "speed": "fast"}
	alwaysReady           = func() bool { return true }
)

func getKey(ds *extensions.DaemonSet, t *testing.T) string {
	if key, err := controller.KeyFunc(ds); err != nil {
		t.Errorf("Unexpected error getting key for ds %v: %v", ds.Name, err)
		return ""
	} else {
		return key
	}
}

func newDaemonSet(name string) *extensions.DaemonSet {
	return &extensions.DaemonSet{
		TypeMeta: metav1.TypeMeta{APIVersion: testapi.Extensions.GroupVersion().String()},
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: simpleDaemonSetLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: v1.TerminationMessagePathDefault,
							ImagePullPolicy:        v1.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					DNSPolicy: v1.DNSDefault,
				},
			},
		},
	}
}

func newNode(name string, label map[string]string) *v1.Node {
	return &v1.Node{
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: v1.NamespaceDefault,
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
			Allocatable: v1.ResourceList{
				v1.ResourcePods: resource.MustParse("100"),
			},
		},
	}
}

func addNodes(nodeStore cache.Store, startIndex, numNodes int, label map[string]string) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		nodeStore.Add(newNode(fmt.Sprintf("node-%d", i), label))
	}
}

func newPod(podName string, nodeName string, label map[string]string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: v1.ObjectMeta{
			GenerateName: podName,
			Labels:       label,
			Namespace:    v1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Containers: []v1.Container{
				{
					Image: "foo/bar",
					TerminationMessagePath: v1.TerminationMessagePathDefault,
					ImagePullPolicy:        v1.PullIfNotPresent,
					SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
			DNSPolicy: v1.DNSDefault,
		},
	}
	v1.GenerateName(v1.SimpleNameGenerator, &pod.ObjectMeta)
	return pod
}

func addPods(podStore cache.Store, nodeName string, label map[string]string, number int) {
	for i := 0; i < number; i++ {
		podStore.Add(newPod(fmt.Sprintf("%s-", nodeName), nodeName, label))
	}
}

func newTestController(initialObjects ...runtime.Object) (*DaemonSetsController, *controller.FakePodControl, *fake.Clientset) {
	clientset := fake.NewSimpleClientset(initialObjects...)
	informerFactory := informers.NewSharedInformerFactory(clientset, nil, controller.NoResyncPeriodFunc())

	manager := NewDaemonSetsController(informerFactory.DaemonSets(), informerFactory.Pods(), informerFactory.Nodes(), clientset, 0)

	manager.podStoreSynced = alwaysReady
	manager.nodeStoreSynced = alwaysReady
	manager.dsStoreSynced = alwaysReady
	podControl := &controller.FakePodControl{}
	manager.podControl = podControl
	return manager, podControl, clientset
}

func validateSyncDaemonSets(t *testing.T, fakePodControl *controller.FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.Templates) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.DeletePodName))
	}
}

func syncAndValidateDaemonSets(t *testing.T, manager *DaemonSetsController, ds *extensions.DaemonSet, podControl *controller.FakePodControl, expectedCreates, expectedDeletes int) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		t.Errorf("Could not get key for daemon.")
	}
	manager.syncHandler(key)
	validateSyncDaemonSets(t, podControl, expectedCreates, expectedDeletes)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	manager, _, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 1, nil)
	ds := newDaemonSet("foo")
	// DeletedFinalStateUnknown should queue the embedded DS if found.
	manager.deleteDaemonset(cache.DeletedFinalStateUnknown{Key: "foo", Obj: ds})
	enqueuedKey, _ := manager.queue.Get()
	if enqueuedKey.(string) != "default/foo" {
		t.Errorf("expected delete of DeletedFinalStateUnknown to enqueue the daemonset but found: %#v", enqueuedKey)
	}
}

// DaemonSets without node selectors should launch pods on every node.
func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)
}

// DaemonSets should do nothing if there aren't any nodes
func TestNoNodesDoesNothing(t *testing.T) {
	manager, podControl, _ := newTestController()
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSets without node selectors should launch on a single node in a
// single node cluster.
func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	manager, podControl, _ := newTestController()
	manager.nodeStore.Add(newNode("only-node", nil))
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should place onto NotReady nodes
func TestNotReadNodeDaemonDoesNotLaunchPod(t *testing.T) {
	manager, podControl, _ := newTestController()
	node := newNode("not-ready", nil)
	node.Status.Conditions = []v1.NodeCondition{
		{Type: v1.NodeReady, Status: v1.ConditionFalse},
	}
	manager.nodeStore.Add(node)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should not place onto OutOfDisk nodes
func TestOutOfDiskNodeDaemonDoesNotLaunchPod(t *testing.T) {
	manager, podControl, _ := newTestController()
	node := newNode("not-enough-disk", nil)
	node.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}
	manager.nodeStore.Add(node)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

func resourcePodSpec(nodeName, memory, cpu string) v1.PodSpec {
	return v1.PodSpec{
		NodeName: nodeName,
		Containers: []v1.Container{{
			Resources: v1.ResourceRequirements{
				Requests: allocatableResources(memory, cpu),
			},
		}},
	}
}

func allocatableResources(memory, cpu string) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceMemory: resource.MustParse(memory),
		v1.ResourceCPU:    resource.MustParse(cpu),
		v1.ResourcePods:   resource.MustParse("100"),
	}
}

// DaemonSets should not place onto nodes with insufficient free resource
func TestInsufficentCapacityNodeDaemonDoesNotLaunchPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	manager, podControl, _ := newTestController()
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSets should not unschedule a daemonset pod from a node with insufficient free resource
func TestInsufficentCapacityNodeDaemonDoesNotUnscheduleRunningPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	podSpec.NodeName = "too-much-mem"
	manager, podControl, _ := newTestController()
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

func TestSufficentCapacityWithTerminatedPodsDaemonLaunchesPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	manager, podControl, _ := newTestController()
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec:   podSpec,
		Status: v1.PodStatus{Phase: v1.PodSucceeded},
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should place onto nodes with sufficient free resource
func TestSufficentCapacityNodeDaemonLaunchesPod(t *testing.T) {
	podSpec := resourcePodSpec("not-too-much-mem", "75M", "75m")
	manager, podControl, _ := newTestController()
	node := newNode("not-too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("200M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets not take any actions when being deleted
func TestDontDoAnythingIfBeingDeleted(t *testing.T) {
	podSpec := resourcePodSpec("not-too-much-mem", "75M", "75m")
	manager, podControl, _ := newTestController()
	node := newNode("not-too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("200M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	now := metav1.Now()
	ds.DeletionTimestamp = &now
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSets should not place onto nodes that would cause port conflicts
func TestPortConflictNodeDaemonDoesNotLaunchPod(t *testing.T) {
	podSpec := v1.PodSpec{
		NodeName: "port-conflict",
		Containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 666,
			}},
		}},
	}
	manager, podControl, _ := newTestController()
	node := newNode("port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec,
	})

	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// Test that if the node is already scheduled with a pod using a host port
// but belonging to the same daemonset, we don't delete that pod
//
// Issue: https://github.com/kubernetes/kubernetes/issues/22309
func TestPortConflictWithSameDaemonPodDoesNotDeletePod(t *testing.T) {
	podSpec := v1.PodSpec{
		NodeName: "port-conflict",
		Containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 666,
			}},
		}},
	}
	manager, podControl, _ := newTestController()
	node := newNode("port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Labels:    simpleDaemonSetLabel,
			Namespace: v1.NamespaceDefault,
		},
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSets should place onto nodes that would not cause port conflicts
func TestNoPortConflictNodeDaemonLaunchesPod(t *testing.T) {
	podSpec1 := v1.PodSpec{
		NodeName: "no-port-conflict",
		Containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 6661,
			}},
		}},
	}
	podSpec2 := v1.PodSpec{
		NodeName: "no-port-conflict",
		Containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 6662,
			}},
		}},
	}
	manager, podControl, _ := newTestController()
	node := newNode("no-port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Indexer.Add(&v1.Pod{
		Spec: podSpec1,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec2
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSetController should not sync DaemonSets with empty pod selectors.
//
// issue https://github.com/kubernetes/kubernetes/pull/23223
func TestPodIsNotDeletedByDaemonsetWithEmptyLabelSelector(t *testing.T) {
	manager, podControl, _ := newTestController()
	manager.nodeStore.Store.Add(newNode("node1", nil))
	// Create pod not controlled by a daemonset.
	manager.podStore.Indexer.Add(&v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Labels:    map[string]string{"bang": "boom"},
			Namespace: v1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			NodeName: "node1",
		},
	})

	// Create a misconfigured DaemonSet. An empty pod selector is invalid but could happen
	// if we upgrade and make a backwards incompatible change.
	//
	// The node selector matches no nodes which mimics the behavior of kubectl delete.
	//
	// The DaemonSet should not schedule pods and should not delete scheduled pods in
	// this case even though it's empty pod selector matches all pods. The DaemonSetController
	// should detect this misconfiguration and choose not to sync the DaemonSet. We should
	// not observe a deletion of the pod on node1.
	ds := newDaemonSet("foo")
	ls := metav1.LabelSelector{}
	ds.Spec.Selector = &ls
	ds.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// Controller should not create pods on nodes which have daemon pods, and should remove excess pods from nodes that have extra pods.
func TestDealsWithExistingPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Indexer, "node-2", simpleDaemonSetLabel, 2)
	addPods(manager.podStore.Indexer, "node-3", simpleDaemonSetLabel, 5)
	addPods(manager.podStore.Indexer, "node-4", simpleDaemonSetLabel2, 2)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 2, 5)
}

// Daemon with node selector should launch pods on nodes matching selector.
func TestSelectorDaemonLaunchesPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 3, 0)
}

// Daemon with node selector should delete pods from nodes that do not satisfy selector.
func TestSelectorDaemonDeletesUnselectedPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Indexer, "node-0", simpleDaemonSetLabel2, 2)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel, 3)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel2, 1)
	addPods(manager.podStore.Indexer, "node-4", simpleDaemonSetLabel, 1)
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 5, 4)
}

// DaemonSet with node selector should launch pods on nodes matching selector, but also deal with existing pods on nodes.
func TestSelectorDaemonDealsWithExistingPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Indexer, "node-0", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel, 3)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel2, 2)
	addPods(manager.podStore.Indexer, "node-2", simpleDaemonSetLabel, 4)
	addPods(manager.podStore.Indexer, "node-6", simpleDaemonSetLabel, 13)
	addPods(manager.podStore.Indexer, "node-7", simpleDaemonSetLabel2, 4)
	addPods(manager.podStore.Indexer, "node-9", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Indexer, "node-9", simpleDaemonSetLabel2, 1)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 3, 20)
}

// DaemonSet with node selector which does not match any node labels should not launch pods.
func TestBadSelectorDaemonDoesNothing(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel2
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSet with node name should launch pod on node with corresponding name.
func TestNameDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet with node name that does not exist should not launch pods.
func TestBadNameDaemonSetDoesNothing(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-10"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSet with node selector, and node name, matching a node, should launch a pod on the node.
func TestNameAndSelectorDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-6"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet with node selector that matches some nodes, and node name that matches a different node, should do nothing.
func TestInconsistentNameSelectorDaemonSetDoesNothing(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// Daemon with node affinity should launch pods on nodes matching affinity.
func TestNodeAffinityDaemonLaunchesPods(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "color",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{simpleNodeLabel["color"]},
							},
						},
					},
				},
			},
		},
	}
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 3, 0)
}

func TestNumberReadyStatus(t *testing.T) {
	daemon := newDaemonSet("foo")
	manager, podControl, clientset := newTestController()
	var updated *extensions.DaemonSet
	clientset.PrependReactor("update", "daemonsets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if action.GetSubresource() != "status" {
			return false, nil, nil
		}
		if u, ok := action.(core.UpdateAction); ok {
			updated = u.GetObject().(*extensions.DaemonSet)
		}
		return false, nil, nil
	})
	addNodes(manager.nodeStore.Store, 0, 2, simpleNodeLabel)
	addPods(manager.podStore.Indexer, "node-0", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Indexer, "node-1", simpleDaemonSetLabel, 1)
	manager.dsStore.Add(daemon)

	syncAndValidateDaemonSets(t, manager, daemon, podControl, 0, 0)
	if updated.Status.NumberReady != 0 {
		t.Errorf("Wrong daemon %s status: %v", updated.Name, updated.Status)
	}

	selector, _ := metav1.LabelSelectorAsSelector(daemon.Spec.Selector)
	daemonPods, _ := manager.podStore.Pods(daemon.Namespace).List(selector)
	for _, pod := range daemonPods {
		condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
		pod.Status.Conditions = append(pod.Status.Conditions, condition)
	}

	syncAndValidateDaemonSets(t, manager, daemon, podControl, 0, 0)
	if updated.Status.NumberReady != 2 {
		t.Errorf("Wrong daemon %s status: %v", updated.Name, updated.Status)
	}
}

func TestObservedGeneration(t *testing.T) {
	daemon := newDaemonSet("foo")
	daemon.Generation = 1
	manager, podControl, clientset := newTestController()
	var updated *extensions.DaemonSet
	clientset.PrependReactor("update", "daemonsets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if action.GetSubresource() != "status" {
			return false, nil, nil
		}
		if u, ok := action.(core.UpdateAction); ok {
			updated = u.GetObject().(*extensions.DaemonSet)
		}
		return false, nil, nil
	})

	addNodes(manager.nodeStore.Store, 0, 1, simpleNodeLabel)
	addPods(manager.podStore.Indexer, "node-0", simpleDaemonSetLabel, 1)
	manager.dsStore.Add(daemon)

	syncAndValidateDaemonSets(t, manager, daemon, podControl, 0, 0)
	if updated.Status.ObservedGeneration != daemon.Generation {
		t.Errorf("Wrong ObservedGeneration for daemon %s in status. Expected %d, got %d", updated.Name, daemon.Generation, updated.Status.ObservedGeneration)
	}
}

func TestNodeShouldRunDaemonPod(t *testing.T) {
	cases := []struct {
		podsOnNode                                       []*v1.Pod
		ds                                               *extensions.DaemonSet
		wantToRun, shouldSchedule, shouldContinueRunning bool
		err                                              error
	}{
		{
			ds: &extensions.DaemonSet{
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{
							Labels: simpleDaemonSetLabel,
						},
						Spec: resourcePodSpec("", "50M", "0.5"),
					},
				},
			},
			wantToRun:             true,
			shouldSchedule:        true,
			shouldContinueRunning: true,
		},
		{
			ds: &extensions.DaemonSet{
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{
							Labels: simpleDaemonSetLabel,
						},
						Spec: resourcePodSpec("", "200M", "0.5"),
					},
				},
			},
			wantToRun:             true,
			shouldSchedule:        false,
			shouldContinueRunning: true,
		},
		{
			ds: &extensions.DaemonSet{
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{
							Labels: simpleDaemonSetLabel,
						},
						Spec: resourcePodSpec("other-node", "50M", "0.5"),
					},
				},
			},
			wantToRun:             false,
			shouldSchedule:        false,
			shouldContinueRunning: false,
		},
		{
			podsOnNode: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Ports: []v1.ContainerPort{{
								HostPort: 666,
							}},
						}},
					},
				},
			},
			ds: &extensions.DaemonSet{
				Spec: extensions.DaemonSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{
							Labels: simpleDaemonSetLabel,
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{{
								Ports: []v1.ContainerPort{{
									HostPort: 666,
								}},
							}},
						},
					},
				},
			},
			wantToRun:             false,
			shouldSchedule:        false,
			shouldContinueRunning: false,
		},
	}

	for i, c := range cases {
		node := newNode("test-node", nil)
		node.Status.Allocatable = allocatableResources("100M", "1")
		manager, _, _ := newTestController()
		manager.nodeStore.Store.Add(node)
		for _, p := range c.podsOnNode {
			manager.podStore.Indexer.Add(p)
			p.Spec.NodeName = "test-node"
		}
		wantToRun, shouldSchedule, shouldContinueRunning, err := manager.nodeShouldRunDaemonPod(node, c.ds)

		if wantToRun != c.wantToRun {
			t.Errorf("[%v] expected wantToRun: %v, got: %v", i, c.wantToRun, wantToRun)
		}
		if shouldSchedule != c.shouldSchedule {
			t.Errorf("[%v] expected shouldSchedule: %v, got: %v", i, c.shouldSchedule, shouldSchedule)
		}
		if shouldContinueRunning != c.shouldContinueRunning {
			t.Errorf("[%v] expected shouldContinueRunning: %v, got: %v", i, c.shouldContinueRunning, shouldContinueRunning)
		}
		if err != c.err {
			t.Errorf("[%v] expected err: %v, got: %v", i, c.err, err)
		}
	}
}
