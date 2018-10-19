/*
Copyright 2016 The Kubernetes Authors.

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

package disruption

import (
	"fmt"
	"reflect"
	"runtime/debug"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/Azure/go-autorest/autorest/to"
)

type pdbStates map[string]policy.PodDisruptionBudget

var alwaysReady = func() bool { return true }

func (ps *pdbStates) Set(pdb *policy.PodDisruptionBudget) error {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		return err
	}
	(*ps)[key] = *pdb.DeepCopy()
	return nil
}

func (ps *pdbStates) Get(key string) policy.PodDisruptionBudget {
	return (*ps)[key]
}

func (ps *pdbStates) VerifyPdbStatus(t *testing.T, key string, disruptionsAllowed, currentHealthy, desiredHealthy, expectedPods int32,
	disruptedPodMap map[string]metav1.Time) {
	actualPDB := ps.Get(key)
	expectedStatus := policy.PodDisruptionBudgetStatus{
		PodDisruptionsAllowed: disruptionsAllowed,
		CurrentHealthy:        currentHealthy,
		DesiredHealthy:        desiredHealthy,
		ExpectedPods:          expectedPods,
		DisruptedPods:         disruptedPodMap,
		ObservedGeneration:    actualPDB.Generation,
	}
	actualStatus := actualPDB.Status
	if !reflect.DeepEqual(actualStatus, expectedStatus) {
		debug.PrintStack()
		t.Fatalf("PDB %q status mismatch.  Expected %+v but got %+v.", key, expectedStatus, actualStatus)
	}
}

func (ps *pdbStates) VerifyDisruptionAllowed(t *testing.T, key string, disruptionsAllowed int32) {
	pdb := ps.Get(key)
	if pdb.Status.PodDisruptionsAllowed != disruptionsAllowed {
		debug.PrintStack()
		t.Fatalf("PodDisruptionAllowed mismatch for PDB %q.  Expected %v but got %v.", key, disruptionsAllowed, pdb.Status.PodDisruptionsAllowed)
	}
}

type disruptionController struct {
	*DisruptionController

	podStore cache.Store
	pdbStore cache.Store
	rcStore  cache.Store
	rsStore  cache.Store
	dStore   cache.Store
	ssStore  cache.Store
}

func newFakeDisruptionController() (*disruptionController, *pdbStates) {
	ps := &pdbStates{}

	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())

	dc := NewDisruptionController(
		informerFactory.Core().V1().Pods(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Extensions().V1beta1().Deployments(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		nil,
	)
	dc.getUpdater = func() updater { return ps.Set }
	dc.podListerSynced = alwaysReady
	dc.pdbListerSynced = alwaysReady
	dc.rcListerSynced = alwaysReady
	dc.rsListerSynced = alwaysReady
	dc.dListerSynced = alwaysReady
	dc.ssListerSynced = alwaysReady

	return &disruptionController{
		dc,
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets().Informer().GetStore(),
		informerFactory.Core().V1().ReplicationControllers().Informer().GetStore(),
		informerFactory.Extensions().V1beta1().ReplicaSets().Informer().GetStore(),
		informerFactory.Extensions().V1beta1().Deployments().Informer().GetStore(),
		informerFactory.Apps().V1beta1().StatefulSets().Informer().GetStore(),
	}, ps
}

func fooBar() map[string]string {
	return map[string]string{"foo": "bar"}
}

func newSel(labels map[string]string) *metav1.LabelSelector {
	return &metav1.LabelSelector{MatchLabels: labels}
}

func newSelFooBar() *metav1.LabelSelector {
	return newSel(map[string]string{"foo": "bar"})
}

func newMinAvailablePodDisruptionBudget(t *testing.T, minAvailable intstr.IntOrString) (*policy.PodDisruptionBudget, string) {

	pdb := &policy.PodDisruptionBudget{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
			Selector:     newSelFooBar(),
		},
	}

	pdbName, err := controller.KeyFunc(pdb)
	if err != nil {
		t.Fatalf("Unexpected error naming pdb %q: %v", pdb.Name, err)
	}

	return pdb, pdbName
}

func newMaxUnavailablePodDisruptionBudget(t *testing.T, maxUnavailable intstr.IntOrString) (*policy.PodDisruptionBudget, string) {
	pdb := &policy.PodDisruptionBudget{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MaxUnavailable: &maxUnavailable,
			Selector:       newSelFooBar(),
		},
	}

	pdbName, err := controller.KeyFunc(pdb)
	if err != nil {
		t.Fatalf("Unexpected error naming pdb %q: %v", pdb.Name, err)
	}

	return pdb, pdbName
}

func updatePodOwnerToRc(t *testing.T, pod *v1.Pod, rc *v1.ReplicationController) {
	var controllerReference metav1.OwnerReference
	var trueVar = true
	controllerReference = metav1.OwnerReference{UID: rc.UID, APIVersion: controllerKindRC.GroupVersion().String(), Kind: controllerKindRC.Kind, Name: rc.Name, Controller: &trueVar}
	pod.OwnerReferences = append(pod.OwnerReferences, controllerReference)
}

func updatePodOwnerToRs(t *testing.T, pod *v1.Pod, rs *extensions.ReplicaSet) {
	var controllerReference metav1.OwnerReference
	var trueVar = true
	controllerReference = metav1.OwnerReference{UID: rs.UID, APIVersion: controllerKindRS.GroupVersion().String(), Kind: controllerKindRS.Kind, Name: rs.Name, Controller: &trueVar}
	pod.OwnerReferences = append(pod.OwnerReferences, controllerReference)
}

//	pod, podName := newPod(t, name)
func updatePodOwnerToSs(t *testing.T, pod *v1.Pod, ss *apps.StatefulSet) {
	var controllerReference metav1.OwnerReference
	var trueVar = true
	controllerReference = metav1.OwnerReference{UID: ss.UID, APIVersion: controllerKindSS.GroupVersion().String(), Kind: controllerKindSS.Kind, Name: ss.Name, Controller: &trueVar}
	pod.OwnerReferences = append(pod.OwnerReferences, controllerReference)
}

func newPod(t *testing.T, name string) (*v1.Pod, string) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Annotations:     make(map[string]string),
			Name:            name,
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: v1.PodSpec{},
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{Type: v1.PodReady, Status: v1.ConditionTrue},
			},
		},
	}

	podName, err := controller.KeyFunc(pod)
	if err != nil {
		t.Fatalf("Unexpected error naming pod %q: %v", pod.Name, err)
	}

	return pod, podName
}

func newReplicationController(t *testing.T, size int32) (*v1.ReplicationController, string) {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &size,
			Selector: fooBar(),
		},
	}

	rcName, err := controller.KeyFunc(rc)
	if err != nil {
		t.Fatalf("Unexpected error naming RC %q", rc.Name)
	}

	return rc, rcName
}

func newDeployment(t *testing.T, size int32) (*extensions.Deployment, string) {
	d := &extensions.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: extensions.DeploymentSpec{
			Replicas: &size,
			Selector: newSelFooBar(),
		},
	}

	dName, err := controller.KeyFunc(d)
	if err != nil {
		t.Fatalf("Unexpected error naming Deployment %q: %v", d.Name, err)
	}

	return d, dName
}

func newReplicaSet(t *testing.T, size int32) (*extensions.ReplicaSet, string) {
	rs := &extensions.ReplicaSet{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: &size,
			Selector: newSelFooBar(),
		},
	}

	rsName, err := controller.KeyFunc(rs)
	if err != nil {
		t.Fatalf("Unexpected error naming ReplicaSet %q: %v", rs.Name, err)
	}

	return rs, rsName
}

func newStatefulSet(t *testing.T, size int32) (*apps.StatefulSet, string) {
	ss := &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: apps.StatefulSetSpec{
			Replicas: &size,
			Selector: newSelFooBar(),
		},
	}

	ssName, err := controller.KeyFunc(ss)
	if err != nil {
		t.Fatalf("Unexpected error naming StatefulSet %q: %v", ss.Name, err)
	}

	return ss, ssName
}

func update(t *testing.T, store cache.Store, obj interface{}) {
	if err := store.Update(obj); err != nil {
		t.Fatalf("Could not add %+v to %+v: %v", obj, store, err)
	}
}

func add(t *testing.T, store cache.Store, obj interface{}) {
	if err := store.Add(obj); err != nil {
		t.Fatalf("Could not add %+v to %+v: %v", obj, store, err)
	}
}

// Create one with no selector.  Verify it matches 0 pods.
func TestNoSelector(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt(3))
	pdb.Spec.Selector = &metav1.LabelSelector{}
	pod, _ := newPod(t, "yo-yo-yo")

	add(t, dc.pdbStore, pdb)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 3, 0, map[string]metav1.Time{})

	add(t, dc.podStore, pod)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 3, 0, map[string]metav1.Time{})
}

// Verify that available/expected counts go up as we add pods, then verify that
// available count goes down when we make a pod unavailable.
func TestUnavailable(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt(3))
	add(t, dc.pdbStore, pdb)
	dc.sync(pdbName)

	// Add three pods, verifying that the counts go up at each step.
	pods := []*v1.Pod{}
	for i := int32(0); i < 4; i++ {
		ps.VerifyPdbStatus(t, pdbName, 0, i, 3, i, map[string]metav1.Time{})
		pod, _ := newPod(t, fmt.Sprintf("yo-yo-yo %d", i))
		pods = append(pods, pod)
		add(t, dc.podStore, pod)
		dc.sync(pdbName)
	}
	ps.VerifyPdbStatus(t, pdbName, 1, 4, 3, 4, map[string]metav1.Time{})

	// Now set one pod as unavailable
	pods[0].Status.Conditions = []v1.PodCondition{}
	update(t, dc.podStore, pods[0])
	dc.sync(pdbName)

	// Verify expected update
	ps.VerifyPdbStatus(t, pdbName, 0, 3, 3, 4, map[string]metav1.Time{})
}

// Verify that an integer MaxUnavailable won't
// allow a disruption for pods with no controller.
func TestIntegerMaxUnavailable(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromInt(1))
	add(t, dc.pdbStore, pdb)
	dc.sync(pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "naked")
	add(t, dc.podStore, pod)
	dc.sync(pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, 0)
}

// Verify that an integer MaxUnavailable will recompute allowed disruptions when the scale of
// the selected pod's controller is modified.
func TestIntegerMaxUnavailableWithScaling(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromInt(2))
	add(t, dc.pdbStore, pdb)

	rs, _ := newReplicaSet(t, 7)
	add(t, dc.rsStore, rs)

	pod, _ := newPod(t, "pod")
	updatePodOwnerToRs(t, pod, rs)
	add(t, dc.podStore, pod)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 5, 7, map[string]metav1.Time{})

	// Update scale of ReplicaSet and check PDB
	rs.Spec.Replicas = to.Int32Ptr(5)
	update(t, dc.rsStore, rs)

	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 3, 5, map[string]metav1.Time{})
}

// Create a pod  with no controller, and verify that a PDB with a percentage
// specified won't allow a disruption.
func TestNakedPod(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	dc.sync(pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "naked")
	add(t, dc.podStore, pod)
	dc.sync(pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, 0)
}

// Verify that we count the scale of a ReplicaSet even when it has no Deployment.
func TestReplicaSet(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("20%"))
	add(t, dc.pdbStore, pdb)

	rs, _ := newReplicaSet(t, 10)
	add(t, dc.rsStore, rs)

	pod, _ := newPod(t, "pod")
	updatePodOwnerToRs(t, pod, rs)
	add(t, dc.podStore, pod)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 2, 10, map[string]metav1.Time{})
}

// Verify that multiple controllers doesn't allow the PDB to be set true.
func TestMultipleControllers(t *testing.T) {
	const podCount = 2

	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("1%"))
	add(t, dc.pdbStore, pdb)

	pods := []*v1.Pod{}
	for i := 0; i < podCount; i++ {
		pod, _ := newPod(t, fmt.Sprintf("pod %d", i))
		pods = append(pods, pod)
		add(t, dc.podStore, pod)
	}

	dc.sync(pdbName)

	// No controllers yet => no disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	rc, _ := newReplicationController(t, 1)
	rc.Name = "rc 1"
	for i := 0; i < podCount; i++ {
		updatePodOwnerToRc(t, pods[i], rc)
	}
	add(t, dc.rcStore, rc)
	dc.sync(pdbName)
	// One RC and 200%>1% healthy => disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, 1)

	rc, _ = newReplicationController(t, 1)
	rc.Name = "rc 2"
	for i := 0; i < podCount; i++ {
		updatePodOwnerToRc(t, pods[i], rc)
	}
	add(t, dc.rcStore, rc)
	dc.sync(pdbName)

	// 100%>1% healthy BUT two RCs => no disruption allowed
	// TODO: Find out if this assert is still needed
	//ps.VerifyDisruptionAllowed(t, pdbName, 0)
}

func TestReplicationController(t *testing.T) {
	// The budget in this test matches foo=bar, but the RC and its pods match
	// {foo=bar, baz=quux}.  Later, when we add a rogue pod with only a foo=bar
	// label, it will match the budget but have no controllers, which should
	// trigger the controller to set PodDisruptionAllowed to false.
	labels := map[string]string{
		"foo": "bar",
		"baz": "quux",
	}

	dc, ps := newFakeDisruptionController()

	// 34% should round up to 2
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("34%"))
	add(t, dc.pdbStore, pdb)
	rc, _ := newReplicationController(t, 3)
	rc.Spec.Selector = labels
	add(t, dc.rcStore, rc)
	dc.sync(pdbName)

	// It starts out at 0 expected because, with no pods, the PDB doesn't know
	// about the RC.  This is a known bug.  TODO(mml): file issue
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})

	pods := []*v1.Pod{}

	for i := int32(0); i < 3; i++ {
		pod, _ := newPod(t, fmt.Sprintf("foobar %d", i))
		updatePodOwnerToRc(t, pod, rc)
		pods = append(pods, pod)
		pod.Labels = labels
		add(t, dc.podStore, pod)
		dc.sync(pdbName)
		if i < 2 {
			ps.VerifyPdbStatus(t, pdbName, 0, i+1, 2, 3, map[string]metav1.Time{})
		} else {
			ps.VerifyPdbStatus(t, pdbName, 1, 3, 2, 3, map[string]metav1.Time{})
		}
	}

	rogue, _ := newPod(t, "rogue")
	add(t, dc.podStore, rogue)
	dc.sync(pdbName)
	ps.VerifyDisruptionAllowed(t, pdbName, 0)
}

func TestStatefulSetController(t *testing.T) {
	labels := map[string]string{
		"foo": "bar",
		"baz": "quux",
	}

	dc, ps := newFakeDisruptionController()

	// 34% should round up to 2
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("34%"))
	add(t, dc.pdbStore, pdb)
	ss, _ := newStatefulSet(t, 3)
	add(t, dc.ssStore, ss)
	dc.sync(pdbName)

	// It starts out at 0 expected because, with no pods, the PDB doesn't know
	// about the SS.  This is a known bug.  TODO(mml): file issue
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})

	pods := []*v1.Pod{}

	for i := int32(0); i < 3; i++ {
		pod, _ := newPod(t, fmt.Sprintf("foobar %d", i))
		updatePodOwnerToSs(t, pod, ss)
		pods = append(pods, pod)
		pod.Labels = labels
		add(t, dc.podStore, pod)
		dc.sync(pdbName)
		if i < 2 {
			ps.VerifyPdbStatus(t, pdbName, 0, i+1, 2, 3, map[string]metav1.Time{})
		} else {
			ps.VerifyPdbStatus(t, pdbName, 1, 3, 2, 3, map[string]metav1.Time{})
		}
	}
}

func TestTwoControllers(t *testing.T) {
	// Most of this test is in verifying intermediate cases as we define the
	// three controllers and create the pods.
	rcLabels := map[string]string{
		"foo": "bar",
		"baz": "quux",
	}
	dLabels := map[string]string{
		"foo": "bar",
		"baz": "quuux",
	}
	dc, ps := newFakeDisruptionController()

	// These constants are related, but I avoid calculating the correct values in
	// code.  If you update a parameter here, recalculate the correct values for
	// all of them.  Further down in the test, we use these to control loops, and
	// that level of logic is enough complexity for me.
	const collectionSize int32 = 11 // How big each collection is
	const minimumOne int32 = 4      // integer minimum with one controller
	const minimumTwo int32 = 7      // integer minimum with two controllers

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	rc, _ := newReplicationController(t, collectionSize)
	rc.Spec.Selector = rcLabels
	add(t, dc.rcStore, rc)
	dc.sync(pdbName)

	ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})

	pods := []*v1.Pod{}

	unavailablePods := collectionSize - minimumOne - 1
	for i := int32(1); i <= collectionSize; i++ {
		pod, _ := newPod(t, fmt.Sprintf("quux %d", i))
		updatePodOwnerToRc(t, pod, rc)
		pods = append(pods, pod)
		pod.Labels = rcLabels
		if i <= unavailablePods {
			pod.Status.Conditions = []v1.PodCondition{}
		}
		add(t, dc.podStore, pod)
		dc.sync(pdbName)
		if i <= unavailablePods {
			ps.VerifyPdbStatus(t, pdbName, 0, 0, minimumOne, collectionSize, map[string]metav1.Time{})
		} else if i-unavailablePods <= minimumOne {
			ps.VerifyPdbStatus(t, pdbName, 0, i-unavailablePods, minimumOne, collectionSize, map[string]metav1.Time{})
		} else {
			ps.VerifyPdbStatus(t, pdbName, 1, i-unavailablePods, minimumOne, collectionSize, map[string]metav1.Time{})
		}
	}

	d, _ := newDeployment(t, collectionSize)
	d.Spec.Selector = newSel(dLabels)
	add(t, dc.dStore, d)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, minimumOne+1, minimumOne, collectionSize, map[string]metav1.Time{})

	rs, _ := newReplicaSet(t, collectionSize)
	rs.Spec.Selector = newSel(dLabels)
	rs.Labels = dLabels
	add(t, dc.rsStore, rs)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, minimumOne+1, minimumOne, collectionSize, map[string]metav1.Time{})

	// By the end of this loop, the number of ready pods should be N+2 (hence minimumTwo+2).
	unavailablePods = 2*collectionSize - (minimumTwo + 2) - unavailablePods
	for i := int32(1); i <= collectionSize; i++ {
		pod, _ := newPod(t, fmt.Sprintf("quuux %d", i))
		updatePodOwnerToRs(t, pod, rs)
		pods = append(pods, pod)
		pod.Labels = dLabels
		if i <= unavailablePods {
			pod.Status.Conditions = []v1.PodCondition{}
		}
		add(t, dc.podStore, pod)
		dc.sync(pdbName)
		if i <= unavailablePods {
			ps.VerifyPdbStatus(t, pdbName, 0, minimumOne+1, minimumTwo, 2*collectionSize, map[string]metav1.Time{})
		} else if i-unavailablePods <= minimumTwo-(minimumOne+1) {
			ps.VerifyPdbStatus(t, pdbName, 0, (minimumOne+1)+(i-unavailablePods), minimumTwo, 2*collectionSize, map[string]metav1.Time{})
		} else {
			ps.VerifyPdbStatus(t, pdbName, i-unavailablePods-(minimumTwo-(minimumOne+1)),
				(minimumOne+1)+(i-unavailablePods), minimumTwo, 2*collectionSize, map[string]metav1.Time{})
		}
	}

	// Now we verify we can bring down 1 pod and a disruption is still permitted,
	// but if we bring down two, it's not.  Then we make the pod ready again and
	// verify that a disruption is permitted again.
	ps.VerifyPdbStatus(t, pdbName, 2, 2+minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})
	pods[collectionSize-1].Status.Conditions = []v1.PodCondition{}
	update(t, dc.podStore, pods[collectionSize-1])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, 1+minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})

	pods[collectionSize-2].Status.Conditions = []v1.PodCondition{}
	update(t, dc.podStore, pods[collectionSize-2])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})

	pods[collectionSize-1].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
	update(t, dc.podStore, pods[collectionSize-1])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, 1+minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})
}

// Test pdb doesn't exist
func TestPDBNotExist(t *testing.T) {
	dc, _ := newFakeDisruptionController()
	pdb, _ := newMinAvailablePodDisruptionBudget(t, intstr.FromString("67%"))
	add(t, dc.pdbStore, pdb)
	if err := dc.sync("notExist"); err != nil {
		t.Errorf("Unexpected error: %v, expect nil", err)
	}
}

func TestUpdateDisruptedPods(t *testing.T) {
	dc, ps := newFakeDisruptionController()
	dc.recheckQueue = workqueue.NewNamedDelayingQueue("pdb-queue")
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt(1))
	currentTime := time.Now()
	pdb.Status.DisruptedPods = map[string]metav1.Time{
		"p1":       {Time: currentTime},                       // Should be removed, pod deletion started.
		"p2":       {Time: currentTime.Add(-5 * time.Minute)}, // Should be removed, expired.
		"p3":       {Time: currentTime},                       // Should remain, pod untouched.
		"notthere": {Time: currentTime},                       // Should be removed, pod deleted.
	}
	add(t, dc.pdbStore, pdb)

	pod1, _ := newPod(t, "p1")
	pod1.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	pod2, _ := newPod(t, "p2")
	pod3, _ := newPod(t, "p3")

	add(t, dc.podStore, pod1)
	add(t, dc.podStore, pod2)
	add(t, dc.podStore, pod3)

	dc.sync(pdbName)

	ps.VerifyPdbStatus(t, pdbName, 0, 1, 1, 3, map[string]metav1.Time{"p3": {Time: currentTime}})
}
