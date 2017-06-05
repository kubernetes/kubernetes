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
	"reflect"
	"sort"
	"strconv"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	simpleDaemonSetLabel  = map[string]string{"name": "simple-daemon", "type": "production"}
	simpleDaemonSetLabel2 = map[string]string{"name": "simple-daemon", "type": "test"}
	simpleNodeLabel       = map[string]string{"color": "blue", "speed": "fast"}
	simpleNodeLabel2      = map[string]string{"color": "red", "speed": "fast"}
	alwaysReady           = func() bool { return true }
)

var (
	noScheduleTolerations = []v1.Toleration{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}
	noScheduleTaints      = []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}
)

var (
	nodeNotReady = []v1.Taint{{
		Key:       metav1.TaintNodeNotReady,
		Effect:    v1.TaintEffectNoExecute,
		TimeAdded: metav1.Now(),
	}}

	nodeUnreachable = []v1.Taint{{
		Key:       metav1.TaintNodeUnreachable,
		Effect:    v1.TaintEffectNoExecute,
		TimeAdded: metav1.Now(),
	}}
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
		ObjectMeta: metav1.ObjectMeta{
			UID:       uuid.NewUUID(),
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: simpleDaemonSetLabel},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
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
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
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

func newPod(podName string, nodeName string, label map[string]string, ds *extensions.DaemonSet) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName,
			Labels:       label,
			Namespace:    metav1.NamespaceDefault,
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
	pod.Name = names.SimpleNameGenerator.GenerateName(podName)
	if ds != nil {
		pod.OwnerReferences = []metav1.OwnerReference{*newControllerRef(ds)}
	}
	return pod
}

func addPods(podStore cache.Store, nodeName string, label map[string]string, ds *extensions.DaemonSet, number int) {
	for i := 0; i < number; i++ {
		podStore.Add(newPod(fmt.Sprintf("%s-", nodeName), nodeName, label, ds))
	}
}

func addFailedPods(podStore cache.Store, nodeName string, label map[string]string, ds *extensions.DaemonSet, number int) {
	for i := 0; i < number; i++ {
		pod := newPod(fmt.Sprintf("%s-", nodeName), nodeName, label, ds)
		pod.Status = v1.PodStatus{Phase: v1.PodFailed}
		podStore.Add(pod)
	}
}

type fakePodControl struct {
	sync.Mutex
	*controller.FakePodControl
	podStore cache.Store
	podIDMap map[string]*v1.Pod
}

func newFakePodControl() *fakePodControl {
	podIDMap := make(map[string]*v1.Pod)
	return &fakePodControl{
		FakePodControl: &controller.FakePodControl{},
		podIDMap:       podIDMap}
}

func (f *fakePodControl) CreatePodsOnNode(nodeName, namespace string, template *v1.PodTemplateSpec, object runtime.Object, controllerRef *metav1.OwnerReference) error {
	f.Lock()
	defer f.Unlock()
	if err := f.FakePodControl.CreatePodsOnNode(nodeName, namespace, template, object, controllerRef); err != nil {
		return fmt.Errorf("failed to create pod on node %q", nodeName)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels:       template.Labels,
			Namespace:    namespace,
			GenerateName: fmt.Sprintf("%s-", nodeName),
		},
	}

	if err := api.Scheme.Convert(&template.Spec, &pod.Spec, nil); err != nil {
		return fmt.Errorf("unable to convert pod template: %v", err)
	}
	if len(nodeName) != 0 {
		pod.Spec.NodeName = nodeName
	}
	pod.Name = names.SimpleNameGenerator.GenerateName(fmt.Sprintf("%s-", nodeName))

	f.podStore.Update(pod)
	f.podIDMap[pod.Name] = pod
	return nil
}

func (f *fakePodControl) DeletePod(namespace string, podID string, object runtime.Object) error {
	f.Lock()
	defer f.Unlock()
	if err := f.FakePodControl.DeletePod(namespace, podID, object); err != nil {
		return fmt.Errorf("failed to delete pod %q", podID)
	}
	pod, ok := f.podIDMap[podID]
	if !ok {
		return fmt.Errorf("pod %q does not exist", podID)
	}
	f.podStore.Delete(pod)
	delete(f.podIDMap, podID)
	return nil
}

type daemonSetsController struct {
	*DaemonSetsController

	dsStore   cache.Store
	podStore  cache.Store
	nodeStore cache.Store
}

func newTestController(initialObjects ...runtime.Object) (*daemonSetsController, *fakePodControl, *fake.Clientset) {
	clientset := fake.NewSimpleClientset(initialObjects...)
	informerFactory := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())

	manager := NewDaemonSetsController(
		informerFactory.Extensions().V1beta1().DaemonSets(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Nodes(),
		clientset,
	)
	manager.eventRecorder = record.NewFakeRecorder(100)

	manager.podStoreSynced = alwaysReady
	manager.nodeStoreSynced = alwaysReady
	manager.dsStoreSynced = alwaysReady
	podControl := newFakePodControl()
	manager.podControl = podControl
	podControl.podStore = informerFactory.Core().V1().Pods().Informer().GetStore()

	return &daemonSetsController{
		manager,
		informerFactory.Extensions().V1beta1().DaemonSets().Informer().GetStore(),
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Nodes().Informer().GetStore(),
	}, podControl, clientset
}

func validateSyncDaemonSets(t *testing.T, fakePodControl *fakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.Templates) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.DeletePodName))
	}
	// Every Pod created should have a ControllerRef.
	if got, want := len(fakePodControl.ControllerRefs), expectedCreates; got != want {
		t.Errorf("len(ControllerRefs) = %v, want %v", got, want)
	}
	// Make sure the ControllerRefs are correct.
	for _, controllerRef := range fakePodControl.ControllerRefs {
		if got, want := controllerRef.APIVersion, "extensions/v1beta1"; got != want {
			t.Errorf("controllerRef.APIVersion = %q, want %q", got, want)
		}
		if got, want := controllerRef.Kind, "DaemonSet"; got != want {
			t.Errorf("controllerRef.Kind = %q, want %q", got, want)
		}
		if controllerRef.Controller == nil || *controllerRef.Controller != true {
			t.Errorf("controllerRef.Controller is not set to true")
		}
	}
}

func syncAndValidateDaemonSets(t *testing.T, manager *daemonSetsController, ds *extensions.DaemonSet, podControl *fakePodControl, expectedCreates, expectedDeletes int) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		t.Errorf("Could not get key for daemon.")
	}
	manager.syncHandler(key)
	validateSyncDaemonSets(t, podControl, expectedCreates, expectedDeletes)
}

// clearExpectations copies the FakePodControl to PodStore and clears the create and delete expectations.
func clearExpectations(t *testing.T, manager *daemonSetsController, ds *extensions.DaemonSet, fakePodControl *fakePodControl) {
	fakePodControl.Clear()

	key, err := controller.KeyFunc(ds)
	if err != nil {
		t.Errorf("Could not get key for daemon.")
		return
	}
	manager.expectations.DeleteExpectations(key)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	manager, _, _ := newTestController()
	addNodes(manager.nodeStore, 0, 1, nil)
	ds := newDaemonSet("foo")
	// DeletedFinalStateUnknown should queue the embedded DS if found.
	manager.deleteDaemonset(cache.DeletedFinalStateUnknown{Key: "foo", Obj: ds})
	enqueuedKey, _ := manager.queue.Get()
	if enqueuedKey.(string) != "default/foo" {
		t.Errorf("expected delete of DeletedFinalStateUnknown to enqueue the daemonset but found: %#v", enqueuedKey)
	}
}

func markPodsReady(store cache.Store) {
	// mark pods as ready
	for _, obj := range store.List() {
		pod := obj.(*v1.Pod)
		condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
		v1.UpdatePodCondition(&pod.Status, &condition)
	}
}

// DaemonSets without node selectors should launch pods on every node.
func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)
}

func TestSimpleDaemonSetUpdatesStatusAfterLaunchingPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, clientset := newTestController(ds)

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

	manager.dsStore.Add(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)

	// Make sure the single sync() updated Status already for the change made
	// during the manage() phase.
	if got, want := updated.Status.CurrentNumberScheduled, int32(5); got != want {
		t.Errorf("Status.CurrentNumberScheduled = %v, want %v", got, want)
	}
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
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	manager.nodeStore.Add(newNode("only-node", nil))
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should place onto NotReady nodes
func TestNotReadNodeDaemonDoesNotLaunchPod(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	node := newNode("not-ready", nil)
	node.Status.Conditions = []v1.NodeCondition{
		{Type: v1.NodeReady, Status: v1.ConditionFalse},
	}
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should not place onto OutOfDisk nodes
func TestOutOfDiskNodeDaemonDoesNotLaunchPod(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	node := newNode("not-enough-disk", nil)
	node.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}
	manager.nodeStore.Add(node)
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
func TestInsufficientCapacityNodeDaemonDoesNotLaunchPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager, podControl, _ := newTestController(ds)
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSets should not unschedule a daemonset pod from a node with insufficient free resource
func TestInsufficentCapacityNodeDaemonDoesNotUnscheduleRunningPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	podSpec.NodeName = "too-much-mem"
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager, podControl, _ := newTestController(ds)
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

func TestSufficientCapacityWithTerminatedPodsDaemonLaunchesPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager, podControl, _ := newTestController(ds)
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec:   podSpec,
		Status: v1.PodStatus{Phase: v1.PodSucceeded},
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should place onto nodes with sufficient free resource
func TestSufficientCapacityNodeDaemonLaunchesPod(t *testing.T) {
	podSpec := resourcePodSpec("not-too-much-mem", "75M", "75m")
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager, podControl, _ := newTestController(ds)
	node := newNode("not-too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("200M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet should launch a pod on a node with taint NetworkUnavailable condition.
func TestNetworkUnavailableNodeDaemonLaunchesPod(t *testing.T) {
	ds := newDaemonSet("simple")
	manager, podControl, _ := newTestController(ds)

	node := newNode("network-unavailable", nil)
	node.Status.Conditions = []v1.NodeCondition{
		{Type: v1.NodeNetworkUnavailable, Status: v1.ConditionTrue},
	}
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets not take any actions when being deleted
func TestDontDoAnythingIfBeingDeleted(t *testing.T) {
	podSpec := resourcePodSpec("not-too-much-mem", "75M", "75m")
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	now := metav1.Now()
	ds.DeletionTimestamp = &now
	manager, podControl, _ := newTestController(ds)
	node := newNode("not-too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("200M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

func TestDontDoAnythingIfBeingDeletedRace(t *testing.T) {
	// Bare client says it IS deleted.
	ds := newDaemonSet("foo")
	now := metav1.Now()
	ds.DeletionTimestamp = &now
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 5, nil)

	// Lister (cache) says it's NOT deleted.
	ds2 := *ds
	ds2.DeletionTimestamp = nil
	manager.dsStore.Add(&ds2)

	// The existence of a matching orphan should block all actions in this state.
	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, nil)
	manager.podStore.Add(pod)

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
	manager.podStore.Add(&v1.Pod{
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
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	manager.podStore.Add(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          simpleDaemonSetLabel,
			Namespace:       metav1.NamespaceDefault,
			OwnerReferences: []metav1.OwnerReference{*newControllerRef(ds)},
		},
		Spec: podSpec,
	})
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
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec2
	manager, podControl, _ := newTestController(ds)
	node := newNode("no-port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec1,
	})
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSetController should not sync DaemonSets with empty pod selectors.
//
// issue https://github.com/kubernetes/kubernetes/pull/23223
func TestPodIsNotDeletedByDaemonsetWithEmptyLabelSelector(t *testing.T) {
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

	manager, podControl, _ := newTestController(ds)
	manager.nodeStore.Add(newNode("node1", nil))
	// Create pod not controlled by a daemonset.
	manager.podStore.Add(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels:    map[string]string{"bang": "boom"},
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			NodeName: "node1",
		},
	})
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// Controller should not create pods on nodes which have daemon pods, and should remove excess pods from nodes that have extra pods.
func TestDealsWithExistingPods(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, _ := newTestController(ds)
	manager.dsStore.Add(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel, ds, 1)
	addPods(manager.podStore, "node-2", simpleDaemonSetLabel, ds, 2)
	addPods(manager.podStore, "node-3", simpleDaemonSetLabel, ds, 5)
	addPods(manager.podStore, "node-4", simpleDaemonSetLabel2, ds, 2)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 2, 5)
}

// Daemon with node selector should launch pods on nodes matching selector.
func TestSelectorDaemonLaunchesPods(t *testing.T) {
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager, podControl, _ := newTestController(daemon)
	addNodes(manager.nodeStore, 0, 4, nil)
	addNodes(manager.nodeStore, 4, 3, simpleNodeLabel)
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 3, 0)
}

// Daemon with node selector should delete pods from nodes that do not satisfy selector.
func TestSelectorDaemonDeletesUnselectedPods(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager, podControl, _ := newTestController(ds)
	manager.dsStore.Add(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	addNodes(manager.nodeStore, 5, 5, simpleNodeLabel)
	addPods(manager.podStore, "node-0", simpleDaemonSetLabel2, ds, 2)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel, ds, 3)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel2, ds, 1)
	addPods(manager.podStore, "node-4", simpleDaemonSetLabel, ds, 1)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 4)
}

// DaemonSet with node selector should launch pods on nodes matching selector, but also deal with existing pods on nodes.
func TestSelectorDaemonDealsWithExistingPods(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager, podControl, _ := newTestController(ds)
	manager.dsStore.Add(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	addNodes(manager.nodeStore, 5, 5, simpleNodeLabel)
	addPods(manager.podStore, "node-0", simpleDaemonSetLabel, ds, 1)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel, ds, 3)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel2, ds, 2)
	addPods(manager.podStore, "node-2", simpleDaemonSetLabel, ds, 4)
	addPods(manager.podStore, "node-6", simpleDaemonSetLabel, ds, 13)
	addPods(manager.podStore, "node-7", simpleDaemonSetLabel2, ds, 4)
	addPods(manager.podStore, "node-9", simpleDaemonSetLabel, ds, 1)
	addPods(manager.podStore, "node-9", simpleDaemonSetLabel2, ds, 1)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 3, 20)
}

// DaemonSet with node selector which does not match any node labels should not launch pods.
func TestBadSelectorDaemonDoesNothing(t *testing.T) {
	manager, podControl, _ := newTestController()
	addNodes(manager.nodeStore, 0, 4, nil)
	addNodes(manager.nodeStore, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel2
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSet with node name should launch pod on node with corresponding name.
func TestNameDaemonSetLaunchesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet with node name that does not exist should not launch pods.
func TestBadNameDaemonSetDoesNothing(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-10"
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 5, nil)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSet with node selector, and node name, matching a node, should launch a pod on the node.
func TestNameAndSelectorDaemonSetLaunchesPods(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-6"
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 4, nil)
	addNodes(manager.nodeStore, 4, 3, simpleNodeLabel)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet with node selector that matches some nodes, and node name that matches a different node, should do nothing.
func TestInconsistentNameSelectorDaemonSetDoesNothing(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager, podControl, _ := newTestController(ds)
	addNodes(manager.nodeStore, 0, 4, nil)
	addNodes(manager.nodeStore, 4, 3, simpleNodeLabel)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// Daemon with node affinity should launch pods on nodes matching affinity.
func TestNodeAffinityDaemonLaunchesPods(t *testing.T) {
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

	manager, podControl, _ := newTestController(daemon)
	addNodes(manager.nodeStore, 0, 4, nil)
	addNodes(manager.nodeStore, 4, 3, simpleNodeLabel)
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 3, 0)
}

func TestNumberReadyStatus(t *testing.T) {
	ds := newDaemonSet("foo")
	manager, podControl, clientset := newTestController(ds)
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
	addNodes(manager.nodeStore, 0, 2, simpleNodeLabel)
	addPods(manager.podStore, "node-0", simpleDaemonSetLabel, ds, 1)
	addPods(manager.podStore, "node-1", simpleDaemonSetLabel, ds, 1)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	if updated.Status.NumberReady != 0 {
		t.Errorf("Wrong daemon %s status: %v", updated.Name, updated.Status)
	}

	selector, _ := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	daemonPods, _ := manager.podLister.Pods(ds.Namespace).List(selector)
	for _, pod := range daemonPods {
		condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
		pod.Status.Conditions = append(pod.Status.Conditions, condition)
	}

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	if updated.Status.NumberReady != 2 {
		t.Errorf("Wrong daemon %s status: %v", updated.Name, updated.Status)
	}
}

func TestObservedGeneration(t *testing.T) {
	ds := newDaemonSet("foo")
	ds.Generation = 1
	manager, podControl, clientset := newTestController(ds)
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

	addNodes(manager.nodeStore, 0, 1, simpleNodeLabel)
	addPods(manager.podStore, "node-0", simpleDaemonSetLabel, ds, 1)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	if updated.Status.ObservedGeneration != ds.Generation {
		t.Errorf("Wrong ObservedGeneration for daemon %s in status. Expected %d, got %d", updated.Name, ds.Generation, updated.Status.ObservedGeneration)
	}
}

// DaemonSet controller should kill all failed pods and create at most 1 pod on every node.
func TestDaemonKillFailedPods(t *testing.T) {
	tests := []struct {
		numFailedPods, numNormalPods, expectedCreates, expectedDeletes int
		test                                                           string
	}{
		{numFailedPods: 0, numNormalPods: 1, expectedCreates: 0, expectedDeletes: 0, test: "normal (do nothing)"},
		{numFailedPods: 0, numNormalPods: 0, expectedCreates: 1, expectedDeletes: 0, test: "no pods (create 1)"},
		{numFailedPods: 1, numNormalPods: 0, expectedCreates: 0, expectedDeletes: 1, test: "1 failed pod (kill 1), 0 normal pod (create 0; will create in the next sync)"},
		{numFailedPods: 1, numNormalPods: 3, expectedCreates: 0, expectedDeletes: 3, test: "1 failed pod (kill 1), 3 normal pods (kill 2)"},
		{numFailedPods: 2, numNormalPods: 1, expectedCreates: 0, expectedDeletes: 2, test: "2 failed pods (kill 2), 1 normal pod"},
	}

	for _, test := range tests {
		t.Logf("test case: %s\n", test.test)
		ds := newDaemonSet("foo")
		manager, podControl, _ := newTestController(ds)
		manager.dsStore.Add(ds)
		addNodes(manager.nodeStore, 0, 1, nil)
		addFailedPods(manager.podStore, "node-0", simpleDaemonSetLabel, ds, test.numFailedPods)
		addPods(manager.podStore, "node-0", simpleDaemonSetLabel, ds, test.numNormalPods)
		syncAndValidateDaemonSets(t, manager, ds, podControl, test.expectedCreates, test.expectedDeletes)
	}
}

// DaemonSet should not launch a pod on a tainted node when the pod doesn't tolerate that taint.
func TestTaintedNodeDaemonDoesNotLaunchUntoleratePod(t *testing.T) {
	ds := newDaemonSet("untolerate")
	manager, podControl, _ := newTestController(ds)

	node := newNode("tainted", nil)
	setNodeTaint(node, noScheduleTaints)
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// DaemonSet should launch a pod on a tainted node when the pod can tolerate that taint.
func TestTaintedNodeDaemonLaunchesToleratePod(t *testing.T) {
	ds := newDaemonSet("tolerate")
	setDaemonSetToleration(ds, noScheduleTolerations)
	manager, podControl, _ := newTestController(ds)

	node := newNode("tainted", nil)
	setNodeTaint(node, noScheduleTaints)
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet should launch a pod on a not ready node with taint notReady:NoExecute.
func TestNotReadyNodeDaemonLaunchesPod(t *testing.T) {
	ds := newDaemonSet("simple")
	manager, podControl, _ := newTestController(ds)

	node := newNode("tainted", nil)
	setNodeTaint(node, nodeNotReady)
	node.Status.Conditions = []v1.NodeCondition{
		{Type: v1.NodeReady, Status: v1.ConditionFalse},
	}
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet should launch a pod on an unreachable node with taint unreachable:NoExecute.
func TestUnreachableNodeDaemonLaunchesPod(t *testing.T) {
	ds := newDaemonSet("simple")
	manager, podControl, _ := newTestController(ds)

	node := newNode("tainted", nil)
	setNodeTaint(node, nodeUnreachable)
	node.Status.Conditions = []v1.NodeCondition{
		{Type: v1.NodeReady, Status: v1.ConditionUnknown},
	}
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet should launch a pod on an untainted node when the pod has tolerations.
func TestNodeDaemonLaunchesToleratePod(t *testing.T) {
	ds := newDaemonSet("tolerate")
	setDaemonSetToleration(ds, noScheduleTolerations)
	manager, podControl, _ := newTestController(ds)

	node := newNode("untainted", nil)
	manager.nodeStore.Add(node)
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

func setNodeTaint(node *v1.Node, taints []v1.Taint) {
	node.Spec.Taints = taints
}

func setDaemonSetToleration(ds *extensions.DaemonSet, tolerations []v1.Toleration) {
	ds.Spec.Template.Spec.Tolerations = tolerations
}

// DaemonSet should launch a critical pod even when the node is OutOfDisk.
func TestOutOfDiskNodeDaemonLaunchesCriticalPod(t *testing.T) {
	ds := newDaemonSet("critical")
	setDaemonSetCritical(ds)
	manager, podControl, _ := newTestController(ds)

	node := newNode("not-enough-disk", nil)
	node.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}
	manager.nodeStore.Add(node)

	// Without enabling critical pod annotation feature gate, we shouldn't create critical pod
	utilfeature.DefaultFeatureGate.Set("ExperimentalCriticalPodAnnotation=False")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)

	// Enabling critical pod annotation feature gate should create critical pod
	utilfeature.DefaultFeatureGate.Set("ExperimentalCriticalPodAnnotation=True")
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSet should launch a critical pod even when the node has insufficient free resource.
func TestInsufficientCapacityNodeDaemonLaunchesCriticalPod(t *testing.T) {
	podSpec := resourcePodSpec("too-much-mem", "75M", "75m")
	ds := newDaemonSet("critical")
	ds.Spec.Template.Spec = podSpec
	setDaemonSetCritical(ds)

	manager, podControl, _ := newTestController(ds)
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = allocatableResources("100M", "200m")
	manager.nodeStore.Add(node)
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})

	// Without enabling critical pod annotation feature gate, we shouldn't create critical pod
	utilfeature.DefaultFeatureGate.Set("ExperimentalCriticalPodAnnotation=False")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)

	// Enabling critical pod annotation feature gate should create critical pod
	utilfeature.DefaultFeatureGate.Set("ExperimentalCriticalPodAnnotation=True")
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// DaemonSets should NOT launch a critical pod when there are port conflicts.
func TestPortConflictNodeDaemonDoesNotLaunchCriticalPod(t *testing.T) {
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
	manager.podStore.Add(&v1.Pod{
		Spec: podSpec,
	})

	utilfeature.DefaultFeatureGate.Set("ExperimentalCriticalPodAnnotation=True")
	ds := newDaemonSet("critical")
	ds.Spec.Template.Spec = podSpec
	setDaemonSetCritical(ds)
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

func setDaemonSetCritical(ds *extensions.DaemonSet) {
	ds.Namespace = api.NamespaceSystem
	if ds.Spec.Template.ObjectMeta.Annotations == nil {
		ds.Spec.Template.ObjectMeta.Annotations = make(map[string]string)
	}
	ds.Spec.Template.ObjectMeta.Annotations[kubelettypes.CriticalPodAnnotationKey] = ""
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
						ObjectMeta: metav1.ObjectMeta{
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
						ObjectMeta: metav1.ObjectMeta{
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
						ObjectMeta: metav1.ObjectMeta{
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
						ObjectMeta: metav1.ObjectMeta{
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
		manager.nodeStore.Add(node)
		for _, p := range c.podsOnNode {
			manager.podStore.Add(p)
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

// DaemonSets should be resynced when node labels or taints changed
func TestUpdateNode(t *testing.T) {
	var enqueued bool

	cases := []struct {
		test          string
		newNode       *v1.Node
		oldNode       *v1.Node
		ds            *extensions.DaemonSet
		shouldEnqueue bool
	}{
		{
			test:    "Nothing changed, should not enqueue",
			oldNode: newNode("node1", nil),
			newNode: newNode("node1", nil),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("ds")
				ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
				return ds
			}(),
			shouldEnqueue: false,
		},
		{
			test:    "Node labels changed",
			oldNode: newNode("node1", nil),
			newNode: newNode("node1", simpleNodeLabel),
			ds: func() *extensions.DaemonSet {
				ds := newDaemonSet("ds")
				ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
				return ds
			}(),
			shouldEnqueue: true,
		},
		{
			test: "Node taints changed",
			oldNode: func() *v1.Node {
				node := newNode("node1", nil)
				setNodeTaint(node, noScheduleTaints)
				return node
			}(),
			newNode:       newNode("node1", nil),
			ds:            newDaemonSet("ds"),
			shouldEnqueue: true,
		},
	}
	for _, c := range cases {
		manager, podControl, _ := newTestController()
		manager.nodeStore.Add(c.oldNode)
		manager.dsStore.Add(c.ds)
		syncAndValidateDaemonSets(t, manager, c.ds, podControl, 0, 0)

		manager.enqueueDaemonSet = func(ds *extensions.DaemonSet) {
			if ds.Name == "ds" {
				enqueued = true
			}
		}

		enqueued = false
		manager.updateNode(c.oldNode, c.newNode)
		if enqueued != c.shouldEnqueue {
			t.Errorf("Test case: '%s', expected: %t, got: %t", c.test, c.shouldEnqueue, enqueued)
		}
	}
}

func TestGetNodesToDaemonPods(t *testing.T) {
	ds := newDaemonSet("foo")
	ds2 := newDaemonSet("foo2")
	manager, _, _ := newTestController(ds, ds2)
	manager.dsStore.Add(ds)
	manager.dsStore.Add(ds2)
	addNodes(manager.nodeStore, 0, 2, nil)

	// These pods should be returned.
	wantedPods := []*v1.Pod{
		newPod("matching-owned-0-", "node-0", simpleDaemonSetLabel, ds),
		newPod("matching-orphan-0-", "node-0", simpleDaemonSetLabel, nil),
		newPod("matching-owned-1-", "node-1", simpleDaemonSetLabel, ds),
		newPod("matching-orphan-1-", "node-1", simpleDaemonSetLabel, nil),
	}
	failedPod := newPod("matching-owned-failed-pod-1-", "node-1", simpleDaemonSetLabel, ds)
	failedPod.Status = v1.PodStatus{Phase: v1.PodFailed}
	wantedPods = append(wantedPods, failedPod)
	for _, pod := range wantedPods {
		manager.podStore.Add(pod)
	}

	// These pods should be ignored.
	ignoredPods := []*v1.Pod{
		newPod("non-matching-owned-0-", "node-0", simpleDaemonSetLabel2, ds),
		newPod("non-matching-orphan-1-", "node-1", simpleDaemonSetLabel2, nil),
		newPod("matching-owned-by-other-0-", "node-0", simpleDaemonSetLabel, ds2),
	}
	for _, pod := range ignoredPods {
		manager.podStore.Add(pod)
	}

	nodesToDaemonPods, err := manager.getNodesToDaemonPods(ds)
	if err != nil {
		t.Fatalf("getNodesToDaemonPods() error: %v", err)
	}
	gotPods := map[string]bool{}
	for node, pods := range nodesToDaemonPods {
		for _, pod := range pods {
			if pod.Spec.NodeName != node {
				t.Errorf("pod %v grouped into %v but belongs in %v", pod.Name, node, pod.Spec.NodeName)
			}
			gotPods[pod.Name] = true
		}
	}
	for _, pod := range wantedPods {
		if !gotPods[pod.Name] {
			t.Errorf("expected pod %v but didn't get it", pod.Name)
		}
		delete(gotPods, pod.Name)
	}
	for podName := range gotPods {
		t.Errorf("unexpected pod %v was returned", podName)
	}
}

func TestAddPod(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod1 := newPod("pod1-", "node-0", simpleDaemonSetLabel, ds1)
	manager.addPod(pod1)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(ds1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	pod2 := newPod("pod2-", "node-0", simpleDaemonSetLabel, ds2)
	manager.addPod(pod2)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(ds2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestAddPodOrphan(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	ds3 := newDaemonSet("foo3")
	ds3.Spec.Selector.MatchLabels = simpleDaemonSetLabel2
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)
	manager.dsStore.Add(ds3)

	// Make pod an orphan. Expect matching sets to be queued.
	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, nil)
	manager.addPod(pod)
	if got, want := manager.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	if got, want := getQueuedKeys(manager.queue), []string{"default/foo1", "default/foo2"}; !reflect.DeepEqual(got, want) {
		t.Errorf("getQueuedKeys() = %v, want %v", got, want)
	}
}

func TestUpdatePod(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod1 := newPod("pod1-", "node-0", simpleDaemonSetLabel, ds1)
	prev := *pod1
	bumpResourceVersion(pod1)
	manager.updatePod(&prev, pod1)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(ds1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	pod2 := newPod("pod2-", "node-0", simpleDaemonSetLabel, ds2)
	prev = *pod2
	bumpResourceVersion(pod2)
	manager.updatePod(&prev, pod2)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(ds2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestUpdatePodOrphanSameLabels(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, nil)
	prev := *pod
	bumpResourceVersion(pod)
	manager.updatePod(&prev, pod)
	if got, want := manager.queue.Len(), 0; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdatePodOrphanWithNewLabels(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, nil)
	prev := *pod
	prev.Labels = map[string]string{"foo2": "bar2"}
	bumpResourceVersion(pod)
	manager.updatePod(&prev, pod)
	if got, want := manager.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	if got, want := getQueuedKeys(manager.queue), []string{"default/foo1", "default/foo2"}; !reflect.DeepEqual(got, want) {
		t.Errorf("getQueuedKeys() = %v, want %v", got, want)
	}
}

func TestUpdatePodChangeControllerRef(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, ds1)
	prev := *pod
	prev.OwnerReferences = []metav1.OwnerReference{*newControllerRef(ds2)}
	bumpResourceVersion(pod)
	manager.updatePod(&prev, pod)
	if got, want := manager.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdatePodControllerRefRemoved(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, ds1)
	prev := *pod
	pod.OwnerReferences = nil
	bumpResourceVersion(pod)
	manager.updatePod(&prev, pod)
	if got, want := manager.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestDeletePod(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)

	pod1 := newPod("pod1-", "node-0", simpleDaemonSetLabel, ds1)
	manager.deletePod(pod1)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(ds1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	pod2 := newPod("pod2-", "node-0", simpleDaemonSetLabel, ds2)
	manager.deletePod(pod2)
	if got, want := manager.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = manager.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(ds2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestDeletePodOrphan(t *testing.T) {
	manager, _, _ := newTestController()
	ds1 := newDaemonSet("foo1")
	ds2 := newDaemonSet("foo2")
	ds3 := newDaemonSet("foo3")
	ds3.Spec.Selector.MatchLabels = simpleDaemonSetLabel2
	manager.dsStore.Add(ds1)
	manager.dsStore.Add(ds2)
	manager.dsStore.Add(ds3)

	pod := newPod("pod1-", "node-0", simpleDaemonSetLabel, nil)
	manager.deletePod(pod)
	if got, want := manager.queue.Len(), 0; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func bumpResourceVersion(obj metav1.Object) {
	ver, _ := strconv.ParseInt(obj.GetResourceVersion(), 10, 32)
	obj.SetResourceVersion(strconv.FormatInt(ver+1, 10))
}

// getQueuedKeys returns a sorted list of keys in the queue.
// It can be used to quickly check that multiple keys are in there.
func getQueuedKeys(queue workqueue.RateLimitingInterface) []string {
	var keys []string
	count := queue.Len()
	for i := 0; i < count; i++ {
		key, done := queue.Get()
		if done {
			return keys
		}
		keys = append(keys, key.(string))
	}
	sort.Strings(keys)
	return keys
}
