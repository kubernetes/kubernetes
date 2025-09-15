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
	"context"
	"fmt"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	apps "k8s.io/api/apps/v1"
	autoscalingapi "k8s.io/api/autoscaling/v1"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	discoveryfake "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	scalefake "k8s.io/client-go/scale/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/utils/ktesting"
	clocktesting "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

type pdbStates map[string]policy.PodDisruptionBudget

var alwaysReady = func() bool { return true }

func (ps *pdbStates) Set(ctx context.Context, pdb *policy.PodDisruptionBudget) error {
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

func (ps *pdbStates) VerifyPdbStatus(t *testing.T, key string, disruptionsAllowed, currentHealthy, desiredHealthy, expectedPods int32, disruptedPodMap map[string]metav1.Time) {
	t.Helper()
	actualPDB := ps.Get(key)
	actualConditions := actualPDB.Status.Conditions
	actualPDB.Status.Conditions = nil
	expectedStatus := policy.PodDisruptionBudgetStatus{
		DisruptionsAllowed: disruptionsAllowed,
		CurrentHealthy:     currentHealthy,
		DesiredHealthy:     desiredHealthy,
		ExpectedPods:       expectedPods,
		DisruptedPods:      disruptedPodMap,
		ObservedGeneration: actualPDB.Generation,
	}
	actualStatus := actualPDB.Status
	if diff := cmp.Diff(expectedStatus, actualStatus, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("PDB %q status mismatch (-want,+got):\n%s", key, diff)
	}

	cond := apimeta.FindStatusCondition(actualConditions, policy.DisruptionAllowedCondition)
	if cond == nil {
		t.Fatalf("Expected condition %q, but didn't find it", policy.DisruptionAllowedCondition)
	}
	if disruptionsAllowed > 0 {
		if cond.Status != metav1.ConditionTrue {
			t.Fatalf("Expected condition %q to have status %q, but was %q",
				policy.DisruptionAllowedCondition, metav1.ConditionTrue, cond.Status)
		}
	} else {
		if cond.Status != metav1.ConditionFalse {
			t.Fatalf("Expected condition %q to have status %q, but was %q",
				policy.DisruptionAllowedCondition, metav1.ConditionFalse, cond.Status)
		}
	}
}

func (ps *pdbStates) VerifyDisruptionAllowed(t *testing.T, key string, disruptionsAllowed int32) {
	pdb := ps.Get(key)
	if pdb.Status.DisruptionsAllowed != disruptionsAllowed {
		debug.PrintStack()
		t.Fatalf("PodDisruptionAllowed mismatch for PDB %q.  Expected %v but got %v.", key, disruptionsAllowed, pdb.Status.DisruptionsAllowed)
	}
}

func (ps *pdbStates) VerifyNoStatusError(t *testing.T, key string) {
	pdb := ps.Get(key)
	for _, condition := range pdb.Status.Conditions {
		if strings.Contains(condition.Message, "found no controller ref") && condition.Reason == policy.SyncFailedReason {
			t.Fatalf("PodDisruption Controller should not error when unmanaged pods are found but it failed for %q", key)
		}
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

	coreClient      *fake.Clientset
	scaleClient     *scalefake.FakeScaleClient
	discoveryClient *discoveryfake.FakeDiscovery
	informerFactory informers.SharedInformerFactory
}

var customGVK = schema.GroupVersionKind{
	Group:   "custom.k8s.io",
	Version: "v1",
	Kind:    "customresource",
}

func newFakeDisruptionController(ctx context.Context) (*disruptionController, *pdbStates) {
	return newFakeDisruptionControllerWithTime(ctx, time.Now())
}

func newFakeDisruptionControllerWithTime(ctx context.Context, now time.Time) (*disruptionController, *pdbStates) {
	ps := &pdbStates{}

	coreClient := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(coreClient, controller.NoResyncPeriodFunc())

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(customGVK, &v1.Service{})
	fakeScaleClient := &scalefake.FakeScaleClient{}
	fakeDiscovery := &discoveryfake.FakeDiscovery{
		Fake: &core.Fake{},
	}
	fakeClock := clocktesting.NewFakeClock(now)

	dc := NewDisruptionControllerInternal(
		ctx,
		informerFactory.Core().V1().Pods(),
		informerFactory.Policy().V1().PodDisruptionBudgets(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Apps().V1().ReplicaSets(),
		informerFactory.Apps().V1().Deployments(),
		informerFactory.Apps().V1().StatefulSets(),
		coreClient,
		testrestmapper.TestOnlyStaticRESTMapper(scheme),
		fakeScaleClient,
		fakeDiscovery,
		fakeClock,
		stalePodDisruptionTimeout,
	)
	dc.getUpdater = func() updater { return ps.Set }
	dc.podListerSynced = alwaysReady
	dc.pdbListerSynced = alwaysReady
	dc.rcListerSynced = alwaysReady
	dc.rsListerSynced = alwaysReady
	dc.dListerSynced = alwaysReady
	dc.ssListerSynced = alwaysReady
	dc.recorder = record.NewFakeRecorder(100)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	return &disruptionController{
		dc,
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Policy().V1().PodDisruptionBudgets().Informer().GetStore(),
		informerFactory.Core().V1().ReplicationControllers().Informer().GetStore(),
		informerFactory.Apps().V1().ReplicaSets().Informer().GetStore(),
		informerFactory.Apps().V1().Deployments().Informer().GetStore(),
		informerFactory.Apps().V1().StatefulSets().Informer().GetStore(),
		coreClient,
		fakeScaleClient,
		fakeDiscovery,
		informerFactory,
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

func updatePodOwnerToRs(t *testing.T, pod *v1.Pod, rs *apps.ReplicaSet) {
	var controllerReference metav1.OwnerReference
	var trueVar = true
	controllerReference = metav1.OwnerReference{UID: rs.UID, APIVersion: controllerKindRS.GroupVersion().String(), Kind: controllerKindRS.Kind, Name: rs.Name, Controller: &trueVar}
	pod.OwnerReferences = append(pod.OwnerReferences, controllerReference)
}

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

func newDeployment(t *testing.T, size int32) (*apps.Deployment, string) {
	d := &apps.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: apps.DeploymentSpec{
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

func newReplicaSet(t *testing.T, size int32) (*apps.ReplicaSet, string) {
	rs := &apps.ReplicaSet{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: apps.ReplicaSetSpec{
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

// Create one with no selector.  Verify it matches all pods
func TestNoSelector(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(3))
	pdb.Spec.Selector = &metav1.LabelSelector{}
	pod, _ := newPod(t, "yo-yo-yo")

	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 3, 0, map[string]metav1.Time{})

	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 3, 1, map[string]metav1.Time{})
}

// Verify that available/expected counts go up as we add pods, then verify that
// available count goes down when we make a pod unavailable.
func TestUnavailable(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(3))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)

	// Add three pods, verifying that the counts go up at each step.
	pods := []*v1.Pod{}
	for i := int32(0); i < 4; i++ {
		ps.VerifyPdbStatus(t, pdbName, 0, i, 3, i, map[string]metav1.Time{})
		pod, _ := newPod(t, fmt.Sprintf("yo-yo-yo %d", i))
		pods = append(pods, pod)
		add(t, dc.podStore, pod)
		dc.sync(ctx, pdbName)
	}
	ps.VerifyPdbStatus(t, pdbName, 1, 4, 3, 4, map[string]metav1.Time{})

	// Now set one pod as unavailable
	pods[0].Status.Conditions = []v1.PodCondition{}
	update(t, dc.podStore, pods[0])
	dc.sync(ctx, pdbName)

	// Verify expected update
	ps.VerifyPdbStatus(t, pdbName, 0, 3, 3, 4, map[string]metav1.Time{})
}

// Verify that an integer MaxUnavailable won't
// allow a disruption for pods with no controller.
func TestIntegerMaxUnavailable(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromInt32(1))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "naked")
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, 0)
	verifyEventEmitted(t, dc, "UnmanagedPods")

}

// Verify that an integer MaxUnavailable will recompute allowed disruptions when the scale of
// the selected pod's controller is modified.
func TestIntegerMaxUnavailableWithScaling(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromInt32(2))
	add(t, dc.pdbStore, pdb)

	rs, _ := newReplicaSet(t, 7)
	add(t, dc.rsStore, rs)

	pod, _ := newPod(t, "pod")
	updatePodOwnerToRs(t, pod, rs)
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 5, 7, map[string]metav1.Time{})

	// Update scale of ReplicaSet and check PDB
	rs.Spec.Replicas = ptr.To[int32](5)
	update(t, dc.rsStore, rs)

	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 3, 5, map[string]metav1.Time{})
}

// Verify that an percentage MaxUnavailable will recompute allowed disruptions when the scale of
// the selected pod's controller is modified.
func TestPercentageMaxUnavailableWithScaling(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromString("30%"))
	add(t, dc.pdbStore, pdb)

	rs, _ := newReplicaSet(t, 7)
	add(t, dc.rsStore, rs)

	pod, _ := newPod(t, "pod")
	updatePodOwnerToRs(t, pod, rs)
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 4, 7, map[string]metav1.Time{})

	// Update scale of ReplicaSet and check PDB
	rs.Spec.Replicas = ptr.To[int32](3)
	update(t, dc.rsStore, rs)

	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 2, 3, map[string]metav1.Time{})
}

// Create a pod  with no controller, and verify that a PDB with a percentage
// specified won't allow a disruption.
func TestNakedPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "naked")
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, 0)
	verifyEventEmitted(t, dc, "UnmanagedPods")
}

// Create a pod  with unsupported controller, and verify that a PDB with a percentage
// specified won't allow a disruption.
func TestUnsupportedControllerPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "naked")
	isController := true
	pod.OwnerReferences = append(pod.OwnerReferences, metav1.OwnerReference{
		APIVersion: "apps.test.io/v1",
		Kind:       "TestWorkload",
		Name:       "fake-controller",
		UID:        "b7329742-8daa-493a-8881-6ca07139172b",
		Controller: &isController,
	})

	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, 0)
	verifyEventEmitted(t, dc, "CalculateExpectedPodCountFailed")
}

// Verify that disruption controller is not erroring when unmanaged pods are found
func TestStatusForUnmanagedPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "unmanaged")
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	ps.VerifyNoStatusError(t, pdbName)
	verifyEventEmitted(t, dc, "UnmanagedPods")
}

// Check if the unmanaged pods are correctly collected or not
func TestTotalUnmanagedPods(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbStore, pdb)
	dc.sync(ctx, pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	pod, _ := newPod(t, "unmanaged")
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	var pods []*v1.Pod
	pods = append(pods, pod)
	_, unmanagedPods, _ := dc.getExpectedScale(ctx, pdb, pods)
	if len(unmanagedPods) != 1 {
		t.Fatalf("expected one pod to be unmanaged pod but found %d", len(unmanagedPods))
	}
	ps.VerifyNoStatusError(t, pdbName)
	verifyEventEmitted(t, dc, "UnmanagedPods")
}

// Verify that we count the scale of a ReplicaSet even when it has no Deployment.
func TestReplicaSet(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("20%"))
	add(t, dc.pdbStore, pdb)

	rs, _ := newReplicaSet(t, 10)
	add(t, dc.rsStore, rs)
	pod, _ := newPod(t, "pod")
	updatePodOwnerToRs(t, pod, rs)
	add(t, dc.podStore, pod)
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, 1, 2, 10, map[string]metav1.Time{})
}

func TestScaleResource(t *testing.T) {
	customResourceUID := uuid.NewUUID()
	replicas := int32(10)
	pods := int32(4)
	maxUnavailable := int32(5)

	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	dc.scaleClient.AddReactor("get", "customresources", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &autoscalingapi.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: metav1.NamespaceDefault,
				UID:       customResourceUID,
			},
			Spec: autoscalingapi.ScaleSpec{
				Replicas: replicas,
			},
		}
		return true, obj, nil
	})

	pdb, pdbName := newMaxUnavailablePodDisruptionBudget(t, intstr.FromInt32(maxUnavailable))
	add(t, dc.pdbStore, pdb)

	trueVal := true
	for i := 0; i < int(pods); i++ {
		pod, _ := newPod(t, fmt.Sprintf("pod-%d", i))
		pod.SetOwnerReferences([]metav1.OwnerReference{
			{
				Kind:       customGVK.Kind,
				APIVersion: customGVK.GroupVersion().String(),
				Controller: &trueVal,
				UID:        customResourceUID,
			},
		})
		add(t, dc.podStore, pod)
	}
	dc.sync(ctx, pdbName)
	disruptionsAllowed := int32(0)
	if replicas-pods < maxUnavailable {
		disruptionsAllowed = maxUnavailable - (replicas - pods)
	}
	ps.VerifyPdbStatus(t, pdbName, disruptionsAllowed, pods, replicas-maxUnavailable, replicas, map[string]metav1.Time{})
}

func TestScaleFinderNoResource(t *testing.T) {
	resourceName := "customresources"
	testCases := map[string]struct {
		apiResources []metav1.APIResource
		expectError  bool
	}{
		"resource implements scale": {
			apiResources: []metav1.APIResource{
				{
					Kind: customGVK.Kind,
					Name: resourceName + "/status",
				},
				{
					Kind:    "Scale",
					Group:   autoscalingapi.GroupName,
					Version: "v1",
					Name:    resourceName + "/scale",
				},
				{
					Kind: customGVK.Kind,
					Name: resourceName,
				},
			},
			expectError: false,
		},
		"resource implements unsupported data format for scale subresource": {
			apiResources: []metav1.APIResource{
				{
					Kind: customGVK.Kind,
					Name: resourceName,
				},
				{
					Kind: customGVK.Kind,
					Name: resourceName + "/scale",
				},
			},
			expectError: true,
		},
		"resource does not implement scale": {
			apiResources: []metav1.APIResource{
				{
					Kind: customGVK.Kind,
					Name: resourceName,
				},
			},
			expectError: true,
		},
	}

	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			customResourceUID := uuid.NewUUID()

			_, ctx := ktesting.NewTestContext(t)
			dc, _ := newFakeDisruptionController(ctx)

			dc.scaleClient.AddReactor("get", resourceName, func(action core.Action) (handled bool, ret runtime.Object, err error) {
				gr := schema.GroupResource{
					Group:    customGVK.Group,
					Resource: resourceName,
				}
				return true, nil, errors.NewNotFound(gr, "name")
			})
			dc.discoveryClient.Resources = []*metav1.APIResourceList{
				{
					GroupVersion: customGVK.GroupVersion().String(),
					APIResources: tc.apiResources,
				},
			}

			trueVal := true
			ownerRef := &metav1.OwnerReference{
				Kind:       customGVK.Kind,
				APIVersion: customGVK.GroupVersion().String(),
				Controller: &trueVal,
				UID:        customResourceUID,
			}

			_, err := dc.getScaleController(ctx, ownerRef, "default")

			if tc.expectError && err == nil {
				t.Error("expected error, but didn't get one")
			}

			if !tc.expectError && err != nil {
				t.Errorf("did not expect error, but got %v", err)
			}
		})
	}
}

// Verify that multiple controllers doesn't allow the PDB to be set true.
func TestMultipleControllers(t *testing.T) {
	const podCount = 2
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("1%"))
	add(t, dc.pdbStore, pdb)

	pods := []*v1.Pod{}
	for i := 0; i < podCount; i++ {
		pod, _ := newPod(t, fmt.Sprintf("pod %d", i))
		pods = append(pods, pod)
		add(t, dc.podStore, pod)
	}
	dc.sync(ctx, pdbName)

	// No controllers yet => no disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, 0)

	rc, _ := newReplicationController(t, 1)
	rc.Name = "rc 1"
	for i := 0; i < podCount; i++ {
		updatePodOwnerToRc(t, pods[i], rc)
	}
	add(t, dc.rcStore, rc)
	dc.sync(ctx, pdbName)
	// One RC and 200%>1% healthy => disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, 1)

	rc, _ = newReplicationController(t, 1)
	rc.Name = "rc 2"
	for i := 0; i < podCount; i++ {
		updatePodOwnerToRc(t, pods[i], rc)
	}
	add(t, dc.rcStore, rc)
	dc.sync(ctx, pdbName)

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
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	// 34% should round up to 2
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("34%"))
	add(t, dc.pdbStore, pdb)
	rc, _ := newReplicationController(t, 3)
	rc.Spec.Selector = labels
	add(t, dc.rcStore, rc)
	dc.sync(ctx, pdbName)

	// It starts out at 0 expected because, with no pods, the PDB doesn't know
	// about the RC.  This is a known bug.  TODO(mml): file issue
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})

	for i := int32(0); i < 3; i++ {
		pod, _ := newPod(t, fmt.Sprintf("foobar %d", i))
		updatePodOwnerToRc(t, pod, rc)
		pod.Labels = labels
		add(t, dc.podStore, pod)
		dc.sync(ctx, pdbName)
		if i < 2 {
			ps.VerifyPdbStatus(t, pdbName, 0, i+1, 2, 3, map[string]metav1.Time{})
		} else {
			ps.VerifyPdbStatus(t, pdbName, 1, 3, 2, 3, map[string]metav1.Time{})
		}
	}

	rogue, _ := newPod(t, "rogue")
	add(t, dc.podStore, rogue)
	dc.sync(ctx, pdbName)
	ps.VerifyDisruptionAllowed(t, pdbName, 2)
}

func TestStatefulSetController(t *testing.T) {
	labels := map[string]string{
		"foo": "bar",
		"baz": "quux",
	}

	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	// 34% should round up to 2
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromString("34%"))
	add(t, dc.pdbStore, pdb)
	ss, _ := newStatefulSet(t, 3)
	add(t, dc.ssStore, ss)
	dc.sync(ctx, pdbName)

	// It starts out at 0 expected because, with no pods, the PDB doesn't know
	// about the SS.  This is a known bug.  TODO(mml): file issue
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})

	for i := int32(0); i < 3; i++ {
		pod, _ := newPod(t, fmt.Sprintf("foobar %d", i))
		updatePodOwnerToSs(t, pod, ss)
		pod.Labels = labels
		add(t, dc.podStore, pod)
		dc.sync(ctx, pdbName)
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
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

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
	dc.sync(ctx, pdbName)

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
		dc.sync(ctx, pdbName)
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
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, minimumOne+1, minimumOne, collectionSize, map[string]metav1.Time{})

	rs, _ := newReplicaSet(t, collectionSize)
	rs.Spec.Selector = newSel(dLabels)
	rs.Labels = dLabels
	add(t, dc.rsStore, rs)
	dc.sync(ctx, pdbName)
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
		dc.sync(ctx, pdbName)
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
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, 1+minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})

	pods[collectionSize-2].Status.Conditions = []v1.PodCondition{}
	update(t, dc.podStore, pods[collectionSize-2])
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 0, minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})

	pods[collectionSize-1].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
	update(t, dc.podStore, pods[collectionSize-1])
	dc.sync(ctx, pdbName)
	ps.VerifyPdbStatus(t, pdbName, 1, 1+minimumTwo, minimumTwo, 2*collectionSize, map[string]metav1.Time{})
}

// Test pdb doesn't exist
func TestPDBNotExist(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, _ := newFakeDisruptionController(ctx)
	pdb, _ := newMinAvailablePodDisruptionBudget(t, intstr.FromString("67%"))
	add(t, dc.pdbStore, pdb)
	if err := dc.sync(ctx, "notExist"); err != nil {
		t.Errorf("Unexpected error: %v, expect nil", err)
	}
}

func TestUpdateDisruptedPods(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)
	dc.recheckQueue = workqueue.NewTypedDelayingQueueWithConfig(workqueue.TypedDelayingQueueConfig[string]{Name: "pdb_queue"})
	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(1))
	currentTime := dc.clock.Now()
	pdb.Status.DisruptedPods = map[string]metav1.Time{
		"p1":       {Time: currentTime},                       // Should be removed, pod deletion started.
		"p2":       {Time: currentTime.Add(-3 * time.Minute)}, // Should be removed, expired.
		"p3":       {Time: currentTime.Add(-time.Minute)},     // Should remain, pod untouched.
		"notthere": {Time: currentTime},                       // Should be removed, pod deleted.
	}
	add(t, dc.pdbStore, pdb)

	pod1, _ := newPod(t, "p1")
	pod1.DeletionTimestamp = &metav1.Time{Time: dc.clock.Now()}
	pod2, _ := newPod(t, "p2")
	pod3, _ := newPod(t, "p3")

	add(t, dc.podStore, pod1)
	add(t, dc.podStore, pod2)
	add(t, dc.podStore, pod3)

	dc.sync(ctx, pdbName)

	ps.VerifyPdbStatus(t, pdbName, 0, 1, 1, 3, map[string]metav1.Time{"p3": {Time: currentTime.Add(-time.Minute)}})
}

func TestBasicFinderFunctions(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, _ := newFakeDisruptionController(ctx)

	rs, _ := newReplicaSet(t, 10)
	add(t, dc.rsStore, rs)
	rc, _ := newReplicationController(t, 12)
	add(t, dc.rcStore, rc)
	ss, _ := newStatefulSet(t, 14)
	add(t, dc.ssStore, ss)

	testCases := map[string]struct {
		finderFunc    podControllerFinder
		apiVersion    string
		kind          string
		name          string
		uid           types.UID
		findsScale    bool
		expectedScale int32
	}{
		"replicaset controller with apps group": {
			finderFunc:    dc.getPodReplicaSet,
			apiVersion:    "apps/v1",
			kind:          controllerKindRS.Kind,
			name:          rs.Name,
			uid:           rs.UID,
			findsScale:    true,
			expectedScale: 10,
		},
		"replicaset controller with invalid group": {
			finderFunc: dc.getPodReplicaSet,
			apiVersion: "invalid/v1",
			kind:       controllerKindRS.Kind,
			name:       rs.Name,
			uid:        rs.UID,
			findsScale: false,
		},
		"replicationcontroller with empty group": {
			finderFunc:    dc.getPodReplicationController,
			apiVersion:    "/v1",
			kind:          controllerKindRC.Kind,
			name:          rc.Name,
			uid:           rc.UID,
			findsScale:    true,
			expectedScale: 12,
		},
		"replicationcontroller with invalid group": {
			finderFunc: dc.getPodReplicationController,
			apiVersion: "apps/v1",
			kind:       controllerKindRC.Kind,
			name:       rc.Name,
			uid:        rc.UID,
			findsScale: false,
		},
		"statefulset controller with extensions group": {
			finderFunc:    dc.getPodStatefulSet,
			apiVersion:    "apps/v1",
			kind:          controllerKindSS.Kind,
			name:          ss.Name,
			uid:           ss.UID,
			findsScale:    true,
			expectedScale: 14,
		},
		"statefulset controller with invalid kind": {
			finderFunc: dc.getPodStatefulSet,
			apiVersion: "apps/v1",
			kind:       controllerKindRS.Kind,
			name:       ss.Name,
			uid:        ss.UID,
			findsScale: false,
		},
	}

	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			controllerRef := &metav1.OwnerReference{
				APIVersion: tc.apiVersion,
				Kind:       tc.kind,
				Name:       tc.name,
				UID:        tc.uid,
			}

			controllerAndScale, _ := tc.finderFunc(ctx, controllerRef, metav1.NamespaceDefault)

			if controllerAndScale == nil {
				if tc.findsScale {
					t.Error("Expected scale, but got nil")
				}
				return
			}

			if got, want := controllerAndScale.scale, tc.expectedScale; got != want {
				t.Errorf("Expected scale %d, but got %d", want, got)
			}

			if got, want := controllerAndScale.UID, tc.uid; got != want {
				t.Errorf("Expected uid %s, but got %s", want, got)
			}
		})
	}
}

func TestDeploymentFinderFunction(t *testing.T) {
	labels := map[string]string{
		"foo": "bar",
	}

	testCases := map[string]struct {
		rsApiVersion  string
		rsKind        string
		depApiVersion string
		depKind       string
		findsScale    bool
		expectedScale int32
	}{
		"happy path": {
			rsApiVersion:  "apps/v1",
			rsKind:        controllerKindRS.Kind,
			depApiVersion: "extensions/v1",
			depKind:       controllerKindDep.Kind,
			findsScale:    true,
			expectedScale: 10,
		},
		"invalid rs apiVersion": {
			rsApiVersion:  "invalid/v1",
			rsKind:        controllerKindRS.Kind,
			depApiVersion: "apps/v1",
			depKind:       controllerKindDep.Kind,
			findsScale:    false,
		},
		"invalid rs kind": {
			rsApiVersion:  "apps/v1",
			rsKind:        "InvalidKind",
			depApiVersion: "apps/v1",
			depKind:       controllerKindDep.Kind,
			findsScale:    false,
		},
		"invalid deployment apiVersion": {
			rsApiVersion:  "extensions/v1",
			rsKind:        controllerKindRS.Kind,
			depApiVersion: "deployment/v1",
			depKind:       controllerKindDep.Kind,
			findsScale:    false,
		},
		"invalid deployment kind": {
			rsApiVersion:  "apps/v1",
			rsKind:        controllerKindRS.Kind,
			depApiVersion: "extensions/v1",
			depKind:       "InvalidKind",
			findsScale:    false,
		},
	}

	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			dc, _ := newFakeDisruptionController(ctx)

			dep, _ := newDeployment(t, 10)
			dep.Spec.Selector = newSel(labels)
			add(t, dc.dStore, dep)

			rs, _ := newReplicaSet(t, 5)
			rs.Labels = labels
			trueVal := true
			rs.OwnerReferences = append(rs.OwnerReferences, metav1.OwnerReference{
				APIVersion: tc.depApiVersion,
				Kind:       tc.depKind,
				Name:       dep.Name,
				UID:        dep.UID,
				Controller: &trueVal,
			})
			add(t, dc.rsStore, rs)

			controllerRef := &metav1.OwnerReference{
				APIVersion: tc.rsApiVersion,
				Kind:       tc.rsKind,
				Name:       rs.Name,
				UID:        rs.UID,
			}

			controllerAndScale, _ := dc.getPodDeployment(ctx, controllerRef, metav1.NamespaceDefault)

			if controllerAndScale == nil {
				if tc.findsScale {
					t.Error("Expected scale, but got nil")
				}
				return
			}

			if got, want := controllerAndScale.scale, tc.expectedScale; got != want {
				t.Errorf("Expected scale %d, but got %d", want, got)
			}

			if got, want := controllerAndScale.UID, dep.UID; got != want {
				t.Errorf("Expected uid %s, but got %s", want, got)
			}
		})
	}
}

// This test checks that the disruption controller does not write stale data to
// a PDB status during race conditions with the eviction handler. Specifically,
// failed updates due to ResourceVersion conflict should not cause a stale value
// of DisruptionsAllowed to be written.
//
// In this test, DisruptionsAllowed starts at 2.
// (A) We will delete 1 pod and trigger DisruptionController to set
// DisruptionsAllowed to 1.
// (B) As the DisruptionController attempts this write, we will evict the
// remaining 2 pods and update DisruptionsAllowed to 0. (The real eviction
// handler would allow this because it still sees DisruptionsAllowed=2.)
// (C) If the DisruptionController writes DisruptionsAllowed=1 despite the
// resource conflict error, then there is a bug.
func TestUpdatePDBStatusRetries(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, _ := newFakeDisruptionController(ctx)
	// Inject the production code over our fake impl
	dc.getUpdater = func() updater { return dc.writePdbStatus }
	// Create a PDB and 3 pods that match it.
	pdb, pdbKey := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(1))
	pdb, err := dc.coreClient.PolicyV1().PodDisruptionBudgets(pdb.Namespace).Create(ctx, pdb, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create PDB: %v", err)
	}
	podNames := []string{"moe", "larry", "curly"}
	for _, name := range podNames {
		pod, _ := newPod(t, name)
		_, err := dc.coreClient.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
	}

	// Block until the fake clientset writes are observable in the informer caches.
	// FUN FACT: This guarantees that the informer caches have updated, but it does
	// not guarantee that informer event handlers have completed. Fortunately,
	// DisruptionController does most of its logic by reading from informer
	// listers, so this guarantee is sufficient.
	if err := waitForCacheCount(dc.pdbStore, 1); err != nil {
		t.Fatalf("Failed to verify PDB in informer cache: %v", err)
	}
	if err := waitForCacheCount(dc.podStore, len(podNames)); err != nil {
		t.Fatalf("Failed to verify pods in informer cache: %v", err)
	}

	// Sync DisruptionController once to update PDB status.
	if err := dc.sync(ctx, pdbKey); err != nil {
		t.Fatalf("Failed initial sync: %v", err)
	}

	// Evict simulates the visible effects of eviction in our fake client.
	evict := func(podNames ...string) {
		// These GVRs are copied from the generated fake code because they are not exported.
		var (
			podsResource                 = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
			poddisruptionbudgetsResource = schema.GroupVersionResource{Group: "policy", Version: "v1", Resource: "poddisruptionbudgets"}
		)

		// Bypass the coreClient.Fake and write directly to the ObjectTracker, because
		// this helper will be called while the Fake is holding a lock.
		obj, err := dc.coreClient.Tracker().Get(poddisruptionbudgetsResource, pdb.Namespace, pdb.Name)
		if err != nil {
			t.Fatalf("Failed to get PDB: %v", err)
		}
		updatedPDB := obj.(*policy.PodDisruptionBudget)
		// Each eviction,
		// - decrements DisruptionsAllowed
		// - adds the pod to DisruptedPods
		// - deletes the pod
		updatedPDB.Status.DisruptionsAllowed -= int32(len(podNames))
		updatedPDB.Status.DisruptedPods = make(map[string]metav1.Time)
		for _, name := range podNames {
			updatedPDB.Status.DisruptedPods[name] = metav1.NewTime(dc.clock.Now())
		}
		if err := dc.coreClient.Tracker().Update(poddisruptionbudgetsResource, updatedPDB, updatedPDB.Namespace); err != nil {
			t.Fatalf("Eviction (PDB update) failed: %v", err)
		}
		for _, name := range podNames {
			if err := dc.coreClient.Tracker().Delete(podsResource, "default", name); err != nil {
				t.Fatalf("Eviction (pod delete) failed: %v", err)
			}
		}
	}

	// The fake kube client does not update ResourceVersion or check for conflicts.
	// Instead, we add a reactor that returns a conflict error on the first PDB
	// update and success after that.
	var failOnce sync.Once
	dc.coreClient.Fake.PrependReactor("update", "poddisruptionbudgets", func(a core.Action) (handled bool, obj runtime.Object, err error) {
		failOnce.Do(func() {
			// (B) Evict two pods and fail this update.
			evict(podNames[1], podNames[2])
			handled = true
			err = errors.NewConflict(a.GetResource().GroupResource(), pdb.Name, fmt.Errorf("conflict"))
		})
		return handled, obj, err
	})

	// (A) Delete one pod
	if err := dc.coreClient.CoreV1().Pods("default").Delete(ctx, podNames[0], metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := waitForCacheCount(dc.podStore, len(podNames)-1); err != nil {
		t.Fatalf("Failed to verify pods in informer cache: %v", err)
	}

	// The sync() function should either write a correct status which takes the
	// evictions into account, or re-queue the PDB for another sync (by returning
	// an error)
	if err := dc.sync(ctx, pdbKey); err != nil {
		t.Logf("sync() returned with error: %v", err)
	} else {
		t.Logf("sync() returned with no error")
	}

	// (C) Whether or not sync() returned an error, the PDB status should reflect
	// the evictions that took place.
	finalPDB, err := dc.coreClient.PolicyV1().PodDisruptionBudgets("default").Get(ctx, pdb.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get PDB: %v", err)
	}
	if expected, actual := int32(0), finalPDB.Status.DisruptionsAllowed; expected != actual {
		t.Errorf("DisruptionsAllowed should be %d, got %d", expected, actual)
	}
}

func TestInvalidSelectors(t *testing.T) {
	testCases := map[string]struct {
		labelSelector *metav1.LabelSelector
	}{
		"illegal value key": {
			labelSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"k8s.io/too/many/slashes": "value",
				},
			},
		},
		"illegal operator": {
			labelSelector: &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "foo",
						Operator: metav1.LabelSelectorOperator("illegal"),
						Values:   []string{"bar"},
					},
				},
			},
		},
	}

	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			dc, ps := newFakeDisruptionController(ctx)

			pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(3))
			pdb.Spec.Selector = tc.labelSelector

			add(t, dc.pdbStore, pdb)
			dc.sync(ctx, pdbName)
			ps.VerifyPdbStatus(t, pdbName, 0, 0, 0, 0, map[string]metav1.Time{})
		})
	}
}

func TestStalePodDisruption(t *testing.T) {
	now := time.Now()
	cases := map[string]struct {
		pod            *v1.Pod
		timePassed     time.Duration
		wantConditions []v1.PodCondition
	}{
		"stale pod disruption": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				Status: v1.PodStatus{
					Conditions: []v1.PodCondition{
						{
							Type:               v1.DisruptionTarget,
							Status:             v1.ConditionTrue,
							LastTransitionTime: metav1.Time{Time: now},
						},
					},
				},
			},
			timePassed: 2*time.Minute + time.Second,
			wantConditions: []v1.PodCondition{
				{
					Type:   v1.DisruptionTarget,
					Status: v1.ConditionFalse,
				},
			},
		},
		"pod disruption in progress": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				Status: v1.PodStatus{
					Conditions: []v1.PodCondition{
						{
							Type:               v1.DisruptionTarget,
							Status:             v1.ConditionTrue,
							LastTransitionTime: metav1.Time{Time: now},
						},
					},
				},
			},
			timePassed: 2*time.Minute - time.Second,
			wantConditions: []v1.PodCondition{
				{
					Type:   v1.DisruptionTarget,
					Status: v1.ConditionTrue,
				},
			},
		},
		"pod disruption actuated": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foo",
					Namespace:         metav1.NamespaceDefault,
					DeletionTimestamp: &metav1.Time{Time: now},
				},
				Status: v1.PodStatus{
					Conditions: []v1.PodCondition{
						{
							Type:               v1.DisruptionTarget,
							Status:             v1.ConditionTrue,
							LastTransitionTime: metav1.Time{Time: now},
						},
					},
				},
			},
			timePassed: 2*time.Minute + time.Second,
			wantConditions: []v1.PodCondition{
				{
					Type:   v1.DisruptionTarget,
					Status: v1.ConditionTrue,
				},
			},
		},
		"no pod disruption": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foo",
					Namespace:         metav1.NamespaceDefault,
					DeletionTimestamp: &metav1.Time{Time: now},
				},
			},
			timePassed: 2*time.Minute + time.Second,
		},
		"pod disruption cleared": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foo",
					Namespace:         metav1.NamespaceDefault,
					DeletionTimestamp: &metav1.Time{Time: now},
				},
				Status: v1.PodStatus{
					Conditions: []v1.PodCondition{
						{
							Type:   v1.DisruptionTarget,
							Status: v1.ConditionFalse,
						},
					},
				},
			},
			timePassed: 2*time.Minute + time.Second,
			wantConditions: []v1.PodCondition{
				{
					Type:   v1.DisruptionTarget,
					Status: v1.ConditionFalse,
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			dc, _ := newFakeDisruptionControllerWithTime(tCtx, now)
			go dc.Run(tCtx)
			if _, err := dc.coreClient.CoreV1().Pods(tc.pod.Namespace).Create(tCtx, tc.pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod: %v", err)
			}
			dc.clock.Sleep(tc.timePassed)
			if err := dc.informerFactory.Core().V1().Pods().Informer().GetIndexer().Add(tc.pod); err != nil {
				t.Fatalf("Failed adding pod to indexer: %v", err)
			}
			diff := ""
			if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
				pod, err := dc.kubeClient.CoreV1().Pods(tc.pod.Namespace).Get(tCtx, tc.pod.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed getting updated pod: %v", err)
				}
				diff = cmp.Diff(tc.wantConditions, pod.Status.Conditions, cmpopts.IgnoreFields(v1.PodCondition{}, "LastTransitionTime"))
				return diff == "", nil
			}); err != nil {
				t.Fatalf("Failed waiting for worker to sync: %v, (-want,+got):\n%s", err, diff)
			}
		})
	}
}

func TestKeepExistingPDBConditionDuringSync(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	dc, ps := newFakeDisruptionController(ctx)

	pdb, pdbName := newMinAvailablePodDisruptionBudget(t, intstr.FromInt32(3))
	pdb.Spec.Selector = &metav1.LabelSelector{}

	pdb.Status.Conditions = append(pdb.Status.Conditions, metav1.Condition{
		Type:               "ExistingTestCondition",
		Status:             metav1.ConditionTrue,
		Message:            "This is a test condition",
		Reason:             "Test",
		LastTransitionTime: metav1.Now(),
	})

	add(t, dc.pdbStore, pdb)
	if err := dc.sync(ctx, pdbName); err != nil {
		t.Fatalf("Failed to sync PDB: %v", err)
	}
	ps.VerifyPdbStatus(t, pdbName, 0, 0, 3, 0, map[string]metav1.Time{})

	actualPDB := ps.Get(pdbName)
	condition := apimeta.FindStatusCondition(actualPDB.Status.Conditions, "ExistingTestCondition")
	if len(actualPDB.Status.Conditions) != 2 {
		t.Fatalf("Expected 2 conditions, but got %d", len(actualPDB.Status.Conditions))
	}
	if condition == nil {
		t.Fatalf("Expected ExistingTestCondition condition, but didn't find it")
	}
}

// waitForCacheCount blocks until the given cache store has the desired number
// of items in it. This will return an error if the condition is not met after a
// 10 second timeout.
func waitForCacheCount(store cache.Store, n int) error {
	return wait.Poll(10*time.Millisecond, 10*time.Second, func() (bool, error) {
		return len(store.List()) == n, nil
	})
}

func verifyEventEmitted(t *testing.T, dc *disruptionController, expectedEvent string) {
	ticker := time.NewTicker(500 * time.Millisecond)
	for {
		select {
		case e := <-dc.recorder.(*record.FakeRecorder).Events:
			if strings.Contains(e, expectedEvent) {
				return
			}
		case <-ticker.C:
			t.Fatalf("Timed out: expected event not generated: %v", expectedEvent)
		}
	}
}
