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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
)

type pdbStates map[string]policy.PodDisruptionBudget

func (ps *pdbStates) Set(pdb *policy.PodDisruptionBudget) error {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		return err
	}
	obj, err := api.Scheme.DeepCopy(*pdb)
	if err != nil {
		return err
	}
	(*ps)[key] = obj.(policy.PodDisruptionBudget)

	return nil
}

func (ps *pdbStates) Get(key string) policy.PodDisruptionBudget {
	return (*ps)[key]
}

func (ps *pdbStates) VerifyPdbStatus(t *testing.T, key string, disruptionAllowed bool, currentHealthy, desiredHealthy, expectedPods int32) {
	expectedStatus := policy.PodDisruptionBudgetStatus{
		PodDisruptionAllowed: disruptionAllowed,
		CurrentHealthy:       currentHealthy,
		DesiredHealthy:       desiredHealthy,
		ExpectedPods:         expectedPods,
	}
	actualStatus := ps.Get(key).Status
	if !reflect.DeepEqual(actualStatus, expectedStatus) {
		t.Fatalf("PDB %q status mismatch.  Expected %+v but got %+v.", key, expectedStatus, actualStatus)
	}
}

func (ps *pdbStates) VerifyDisruptionAllowed(t *testing.T, key string, disruptionAllowed bool) {
	pdb := ps.Get(key)
	if pdb.Status.PodDisruptionAllowed != disruptionAllowed {
		t.Fatalf("PodDisruptionAllowed mismatch for PDB %q.  Expected %v but got %v.", key, disruptionAllowed, pdb.Status.PodDisruptionAllowed)
	}
}

func newFakeDisruptionController() (*DisruptionController, *pdbStates) {
	ps := &pdbStates{}

	dc := &DisruptionController{
		pdbLister:  cache.StoreToPodDisruptionBudgetLister{Store: cache.NewStore(controller.KeyFunc)},
		podLister:  cache.StoreToPodLister{Indexer: cache.NewIndexer(controller.KeyFunc, cache.Indexers{})},
		rcLister:   cache.StoreToReplicationControllerLister{Indexer: cache.NewIndexer(controller.KeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})},
		rsLister:   cache.StoreToReplicaSetLister{Store: cache.NewStore(controller.KeyFunc)},
		dLister:    cache.StoreToDeploymentLister{Store: cache.NewStore(controller.KeyFunc)},
		getUpdater: func() updater { return ps.Set },
	}

	return dc, ps
}

func fooBar() map[string]string {
	return map[string]string{"foo": "bar"}
}

func newSel(labels map[string]string) *unversioned.LabelSelector {
	return &unversioned.LabelSelector{MatchLabels: labels}
}

func newSelFooBar() *unversioned.LabelSelector {
	return newSel(map[string]string{"foo": "bar"})
}

func newPodDisruptionBudget(t *testing.T, minAvailable intstr.IntOrString) (*policy.PodDisruptionBudget, string) {

	pdb := &policy.PodDisruptionBudget{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: minAvailable,
			Selector:     newSelFooBar(),
		},
	}

	pdbName, err := controller.KeyFunc(pdb)
	if err != nil {
		t.Fatalf("Unexpected error naming pdb %q: %v", pdb.Name, err)
	}

	return pdb, pdbName
}

func newPod(t *testing.T, name string) (*api.Pod, string) {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Annotations:     make(map[string]string),
			Name:            name,
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: api.PodSpec{},
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{Type: api.PodReady, Status: api.ConditionTrue},
			},
		},
	}

	podName, err := controller.KeyFunc(pod)
	if err != nil {
		t.Fatalf("Unexpected error naming pod %q: %v", pod.Name, err)
	}

	return pod, podName
}

func newReplicationController(t *testing.T, size int32) (*api.ReplicationController, string) {
	rc := &api.ReplicationController{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: size,
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
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: extensions.DeploymentSpec{
			Replicas: size,
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
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
			Labels:          fooBar(),
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: size,
			Selector: newSelFooBar(),
		},
	}

	rsName, err := controller.KeyFunc(rs)
	if err != nil {
		t.Fatalf("Unexpected error naming ReplicaSet %q: %v", rs.Name, err)
	}

	return rs, rsName
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

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromInt(3))
	pdb.Spec.Selector = &unversioned.LabelSelector{}
	pod, _ := newPod(t, "yo-yo-yo")

	add(t, dc.pdbLister.Store, pdb)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, false, 0, 3, 0)

	add(t, dc.podLister.Indexer, pod)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, false, 0, 3, 0)
}

// Verify that available/expected counts go up as we add pods, then verify that
// available count goes down when we make a pod unavailable.
func TestUnavailable(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromInt(3))
	add(t, dc.pdbLister.Store, pdb)
	dc.sync(pdbName)

	// Add three pods, verifying that the counts go up at each step.
	pods := []*api.Pod{}
	for i := int32(0); i < 3; i++ {
		ps.VerifyPdbStatus(t, pdbName, false, i, 3, i)
		pod, _ := newPod(t, fmt.Sprintf("yo-yo-yo %d", i))
		pods = append(pods, pod)
		add(t, dc.podLister.Indexer, pod)
		dc.sync(pdbName)
	}
	ps.VerifyPdbStatus(t, pdbName, true, 3, 3, 3)

	// Now set one pod as unavailable
	pods[0].Status.Conditions = []api.PodCondition{}
	update(t, dc.podLister.Indexer, pods[0])
	dc.sync(pdbName)

	// Verify expected update
	ps.VerifyPdbStatus(t, pdbName, false, 2, 3, 3)
}

// Create a pod  with no controller, and verify that a PDB with a percentage
// specified won't allow a disruption.
func TestNakedPod(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbLister.Store, pdb)
	dc.sync(pdbName)
	// This verifies that when a PDB has 0 pods, disruptions are not allowed.
	ps.VerifyDisruptionAllowed(t, pdbName, false)

	pod, _ := newPod(t, "naked")
	add(t, dc.podLister.Indexer, pod)
	dc.sync(pdbName)

	ps.VerifyDisruptionAllowed(t, pdbName, false)
}

// Verify that we count the scale of a ReplicaSet even when it has no Deployment.
func TestReplicaSet(t *testing.T) {
	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromString("20%"))
	add(t, dc.pdbLister.Store, pdb)

	rs, _ := newReplicaSet(t, 10)
	add(t, dc.rsLister.Store, rs)

	pod, _ := newPod(t, "pod")
	add(t, dc.podLister.Indexer, pod)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, false, 1, 2, 10)
}

// Verify that multiple controllers doesn't allow the PDB to be set true.
func TestMultipleControllers(t *testing.T) {
	const rcCount = 2
	const podCount = 2

	dc, ps := newFakeDisruptionController()

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromString("1%"))
	add(t, dc.pdbLister.Store, pdb)

	for i := 0; i < podCount; i++ {
		pod, _ := newPod(t, fmt.Sprintf("pod %d", i))
		add(t, dc.podLister.Indexer, pod)
	}
	dc.sync(pdbName)

	// No controllers yet => no disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, false)

	rc, _ := newReplicationController(t, 1)
	rc.Name = "rc 1"
	add(t, dc.rcLister.Indexer, rc)
	dc.sync(pdbName)

	// One RC and 200%>1% healthy => disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, true)

	rc, _ = newReplicationController(t, 1)
	rc.Name = "rc 2"
	add(t, dc.rcLister.Indexer, rc)
	dc.sync(pdbName)

	// 100%>1% healthy BUT two RCs => no disruption allowed
	ps.VerifyDisruptionAllowed(t, pdbName, false)
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

	// 67% should round up to 3
	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromString("67%"))
	add(t, dc.pdbLister.Store, pdb)
	rc, _ := newReplicationController(t, 3)
	rc.Spec.Selector = labels
	add(t, dc.rcLister.Indexer, rc)
	dc.sync(pdbName)
	// It starts out at 0 expected because, with no pods, the PDB doesn't know
	// about the RC.  This is a known bug.  TODO(mml): file issue
	ps.VerifyPdbStatus(t, pdbName, false, 0, 0, 0)

	pods := []*api.Pod{}

	for i := int32(0); i < 3; i++ {
		pod, _ := newPod(t, fmt.Sprintf("foobar %d", i))
		pods = append(pods, pod)
		pod.Labels = labels
		add(t, dc.podLister.Indexer, pod)
		dc.sync(pdbName)
		if i < 2 {
			ps.VerifyPdbStatus(t, pdbName, false, i+1, 3, 3)
		} else {
			ps.VerifyPdbStatus(t, pdbName, true, 3, 3, 3)
		}
	}

	rogue, _ := newPod(t, "rogue")
	add(t, dc.podLister.Indexer, rogue)
	dc.sync(pdbName)
	ps.VerifyDisruptionAllowed(t, pdbName, false)
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

	pdb, pdbName := newPodDisruptionBudget(t, intstr.FromString("28%"))
	add(t, dc.pdbLister.Store, pdb)
	rc, _ := newReplicationController(t, 11)
	rc.Spec.Selector = rcLabels
	add(t, dc.rcLister.Indexer, rc)
	dc.sync(pdbName)

	ps.VerifyPdbStatus(t, pdbName, false, 0, 0, 0)

	pods := []*api.Pod{}

	for i := int32(0); i < 11; i++ {
		pod, _ := newPod(t, fmt.Sprintf("quux %d", i))
		pods = append(pods, pod)
		pod.Labels = rcLabels
		if i < 7 {
			pod.Status.Conditions = []api.PodCondition{}
		}
		add(t, dc.podLister.Indexer, pod)
		dc.sync(pdbName)
		if i < 7 {
			ps.VerifyPdbStatus(t, pdbName, false, 0, 4, 11)
		} else if i < 10 {
			ps.VerifyPdbStatus(t, pdbName, false, i-6, 4, 11)
		} else {
			ps.VerifyPdbStatus(t, pdbName, true, 4, 4, 11)
		}
	}

	d, _ := newDeployment(t, 11)
	d.Spec.Selector = newSel(dLabels)
	add(t, dc.dLister.Store, d)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, true, 4, 4, 11)

	rs, _ := newReplicaSet(t, 11)
	rs.Spec.Selector = newSel(dLabels)
	rs.Labels = dLabels
	add(t, dc.rsLister.Store, rs)
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, true, 4, 4, 11)

	for i := int32(0); i < 11; i++ {
		pod, _ := newPod(t, fmt.Sprintf("quuux %d", i))
		pods = append(pods, pod)
		pod.Labels = dLabels
		if i < 7 {
			pod.Status.Conditions = []api.PodCondition{}
		}
		add(t, dc.podLister.Indexer, pod)
		dc.sync(pdbName)
		if i < 7 {
			ps.VerifyPdbStatus(t, pdbName, false, 4, 7, 22)
		} else if i < 9 {
			ps.VerifyPdbStatus(t, pdbName, false, 4+i-6, 7, 22)
		} else {
			ps.VerifyPdbStatus(t, pdbName, true, 4+i-6, 7, 22)
		}
	}

	// Now we verify we can bring down 1 pod and a disruption is still permitted,
	// but if we bring down two, it's not.  Then we make the pod ready again and
	// verify that a disruption is permitted again.
	ps.VerifyPdbStatus(t, pdbName, true, 8, 7, 22)
	pods[10].Status.Conditions = []api.PodCondition{}
	update(t, dc.podLister.Indexer, pods[10])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, true, 7, 7, 22)

	pods[9].Status.Conditions = []api.PodCondition{}
	update(t, dc.podLister.Indexer, pods[9])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, false, 6, 7, 22)

	pods[10].Status.Conditions = []api.PodCondition{{Type: api.PodReady, Status: api.ConditionTrue}}
	update(t, dc.podLister.Indexer, pods[10])
	dc.sync(pdbName)
	ps.VerifyPdbStatus(t, pdbName, true, 7, 7, 22)
}
