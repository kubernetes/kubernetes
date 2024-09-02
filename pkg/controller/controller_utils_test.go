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

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http/httptest"
	"sort"
	"sync"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/pointer"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// NewFakeControllerExpectationsLookup creates a fake store for PodExpectations.
func NewFakeControllerExpectationsLookup(ttl time.Duration) (*ControllerExpectations, *testingclock.FakeClock) {
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := testingclock.NewFakeClock(fakeTime)
	ttlPolicy := &cache.TTLPolicy{TTL: ttl, Clock: fakeClock}
	ttlStore := cache.NewFakeExpirationStore(
		ExpKeyFunc, nil, ttlPolicy, fakeClock)
	return &ControllerExpectations{ttlStore}, fakeClock
}

func newReplicationController(replicas int) *v1.ReplicationController {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: pointer.Int32(int32(replicas)),
			Selector: map[string]string{"foo": "bar"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image:                  "foo/bar",
							TerminationMessagePath: v1.TerminationMessagePathDefault,
							ImagePullPolicy:        v1.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					RestartPolicy: v1.RestartPolicyAlways,
					DNSPolicy:     v1.DNSDefault,
					NodeSelector: map[string]string{
						"baz": "blah",
					},
				},
			},
		},
	}
	return rc
}

// create count pods with the given phase for the given rc (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status v1.PodPhase, rc *v1.ReplicationController) *v1.PodList {
	pods := []v1.Pod{}
	for i := 0; i < count; i++ {
		newPod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    rc.Spec.Selector,
				Namespace: rc.Namespace,
			},
			Status: v1.PodStatus{Phase: status},
		}
		if store != nil {
			store.Add(&newPod)
		}
		pods = append(pods, newPod)
	}
	return &v1.PodList{
		Items: pods,
	}
}

func newReplicaSet(name string, replicas int, rsUuid types.UID) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			UID:             rsUuid,
			Name:            name,
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: apps.ReplicaSetSpec{
			Replicas: pointer.Int32(int32(replicas)),
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image:                  "foo/bar",
							TerminationMessagePath: v1.TerminationMessagePathDefault,
							ImagePullPolicy:        v1.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					RestartPolicy: v1.RestartPolicyAlways,
					DNSPolicy:     v1.DNSDefault,
					NodeSelector: map[string]string{
						"baz": "blah",
					},
				},
			},
		},
	}
}

func TestControllerExpectations(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	ttl := 30 * time.Second
	e, fakeClock := NewFakeControllerExpectationsLookup(ttl)
	// In practice we can't really have add and delete expectations since we only either create or
	// delete replicas in one rc pass, and the rc goes to sleep soon after until the expectations are
	// either fulfilled or timeout.
	adds, dels := 10, 30
	rc := newReplicationController(1)

	// RC fires off adds and deletes at apiserver, then sets expectations
	rcKey, err := KeyFunc(rc)
	require.NoError(t, err, "Couldn't get key for object %#v: %v", rc, err)

	e.SetExpectations(logger, rcKey, adds, dels)
	var wg sync.WaitGroup
	for i := 0; i < adds+1; i++ {
		wg.Add(1)
		go func() {
			// In prod this can happen either because of a failed create by the rc
			// or after having observed a create via informer
			e.CreationObserved(logger, rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	// There are still delete expectations
	assert.False(t, e.SatisfiedExpectations(logger, rcKey), "Rc will sync before expectations are met")

	for i := 0; i < dels+1; i++ {
		wg.Add(1)
		go func() {
			e.DeletionObserved(logger, rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	tests := []struct {
		name                      string
		expectationsToSet         []int
		expireExpectations        bool
		wantPodExpectations       []int64
		wantExpectationsSatisfied bool
	}{
		{
			name:                      "Expectations have been surpassed",
			expireExpectations:        false,
			wantPodExpectations:       []int64{int64(-1), int64(-1)},
			wantExpectationsSatisfied: true,
		},
		{
			name:                      "Old expectations are cleared because of ttl",
			expectationsToSet:         []int{1, 2},
			expireExpectations:        true,
			wantPodExpectations:       []int64{int64(1), int64(2)},
			wantExpectationsSatisfied: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if len(test.expectationsToSet) > 0 {
				e.SetExpectations(logger, rcKey, test.expectationsToSet[0], test.expectationsToSet[1])
			}
			podExp, exists, err := e.GetExpectations(rcKey)
			require.NoError(t, err, "Could not get expectations for rc, exists %v and err %v", exists, err)
			assert.True(t, exists, "Could not get expectations for rc, exists %v and err %v", exists, err)

			add, del := podExp.GetExpectations()
			assert.Equal(t, test.wantPodExpectations[0], add, "Unexpected pod expectations %#v", podExp)
			assert.Equal(t, test.wantPodExpectations[1], del, "Unexpected pod expectations %#v", podExp)
			assert.Equal(t, test.wantExpectationsSatisfied, e.SatisfiedExpectations(logger, rcKey), "Expectations are met but the rc will not sync")

			if test.expireExpectations {
				fakeClock.Step(ttl + 1)
				assert.True(t, e.SatisfiedExpectations(logger, rcKey), "Expectations should have expired but didn't")
			}
		})
	}
}

func TestUIDExpectations(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	uidExp := NewUIDTrackingControllerExpectations(NewControllerExpectations())
	type test struct {
		name        string
		numReplicas int
	}

	shuffleTests := func(tests []test) {
		for i := range tests {
			j := rand.Intn(i + 1)
			tests[i], tests[j] = tests[j], tests[i]
		}
	}

	getRcDataFrom := func(test test) (string, []string) {
		rc := newReplicationController(test.numReplicas)

		rcName := fmt.Sprintf("rc-%v", test.numReplicas)
		rc.Name = rcName
		rc.Spec.Selector[rcName] = rcName

		podList := newPodList(nil, 5, v1.PodRunning, rc)
		rcKey, err := KeyFunc(rc)
		if err != nil {
			t.Fatalf("Couldn't get key for object %#v: %v", rc, err)
		}

		rcPodNames := []string{}
		for i := range podList.Items {
			p := &podList.Items[i]
			p.Name = fmt.Sprintf("%v-%v", p.Name, rc.Name)
			rcPodNames = append(rcPodNames, PodKey(p))
		}
		uidExp.ExpectDeletions(logger, rcKey, rcPodNames)

		return rcKey, rcPodNames
	}

	tests := []test{
		{name: "Replication controller with 2 replicas", numReplicas: 2},
		{name: "Replication controller with 1 replica", numReplicas: 1},
		{name: "Replication controller with no replicas", numReplicas: 0},
		{name: "Replication controller with 5 replicas", numReplicas: 5},
	}

	shuffleTests(tests)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			rcKey, rcPodNames := getRcDataFrom(test)
			assert.False(t, uidExp.SatisfiedExpectations(logger, rcKey),
				"Controller %v satisfied expectations before deletion", rcKey)

			for _, p := range rcPodNames {
				uidExp.DeletionObserved(logger, rcKey, p)
			}

			assert.True(t, uidExp.SatisfiedExpectations(logger, rcKey),
				"Controller %v didn't satisfy expectations after deletion", rcKey)

			uidExp.DeleteExpectations(logger, rcKey)

			assert.Nil(t, uidExp.GetUIDs(rcKey),
				"Failed to delete uid expectations for %v", rcKey)
		})
	}
}

func TestCreatePodsWithGenerateName(t *testing.T) {
	ns := metav1.NamespaceDefault
	generateName := "hello-"
	controllerSpec := newReplicationController(1)
	controllerRef := metav1.NewControllerRef(controllerSpec, v1.SchemeGroupVersion.WithKind("ReplicationController"))

	type test struct {
		name            string
		podCreationFunc func(podControl RealPodControl) error
		wantPod         *v1.Pod
	}
	var tests = []test{
		{
			name: "Create pod",
			podCreationFunc: func(podControl RealPodControl) error {
				return podControl.CreatePods(context.TODO(), ns, controllerSpec.Spec.Template, controllerSpec, controllerRef)
			},
			wantPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels:       controllerSpec.Spec.Template.Labels,
					GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
				},
				Spec: controllerSpec.Spec.Template.Spec,
			},
		},
		{
			name: "Create pod with generate name",
			podCreationFunc: func(podControl RealPodControl) error {
				// Make sure createReplica sends a POST to the apiserver with a pod from the controllers pod template
				return podControl.CreatePodsWithGenerateName(context.TODO(), ns, controllerSpec.Spec.Template, controllerSpec, controllerRef, generateName)
			},
			wantPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels:          controllerSpec.Spec.Template.Labels,
					GenerateName:    generateName,
					OwnerReferences: []metav1.OwnerReference{*controllerRef},
				},
				Spec: controllerSpec.Spec.Template.Spec,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "empty_pod"}})
			fakeHandler := utiltesting.FakeHandler{
				StatusCode:   200,
				ResponseBody: string(body),
			}
			testServer := httptest.NewServer(&fakeHandler)
			defer testServer.Close()
			clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

			podControl := RealPodControl{
				KubeClient: clientset,
				Recorder:   &record.FakeRecorder{},
			}

			err := test.podCreationFunc(podControl)
			require.NoError(t, err, "unexpected error: %v", err)

			fakeHandler.ValidateRequest(t, "/api/v1/namespaces/default/pods", "POST", nil)
			var actualPod = &v1.Pod{}
			err = json.Unmarshal([]byte(fakeHandler.RequestBody), actualPod)
			require.NoError(t, err, "unexpected error: %v", err)
			assert.True(t, apiequality.Semantic.DeepDerivative(test.wantPod, actualPod),
				"Body: %s", fakeHandler.RequestBody)
		})
	}
}

func TestDeletePodsAllowsMissing(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	podControl := RealPodControl{
		KubeClient: fakeClient,
		Recorder:   &record.FakeRecorder{},
	}

	controllerSpec := newReplicationController(1)

	err := podControl.DeletePod(context.TODO(), "namespace-name", "podName", controllerSpec)
	assert.True(t, apierrors.IsNotFound(err))
}

func TestCountTerminatingPods(t *testing.T) {
	now := metav1.Now()

	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, 7, v1.PodRunning, rc)
	podList.Items[0].Status.Phase = v1.PodSucceeded
	podList.Items[1].Status.Phase = v1.PodFailed
	podList.Items[2].Status.Phase = v1.PodPending
	podList.Items[2].SetDeletionTimestamp(&now)
	podList.Items[3].Status.Phase = v1.PodRunning
	podList.Items[3].SetDeletionTimestamp(&now)
	var podPointers []*v1.Pod
	for i := range podList.Items {
		podPointers = append(podPointers, &podList.Items[i])
	}

	terminatingPods := CountTerminatingPods(podPointers)

	assert.Equal(t, int32(2), terminatingPods)

	terminatingList := FilterTerminatingPods(podPointers)
	assert.Len(t, terminatingList, int(2))
}

func TestActivePodFiltering(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	type podData struct {
		podName  string
		podPhase v1.PodPhase
	}

	type test struct {
		name         string
		pods         []podData
		wantPodNames []string
	}

	tests := []test{
		{
			name: "Filters active pods",
			pods: []podData{
				{podName: "pod-1", podPhase: v1.PodSucceeded},
				{podName: "pod-2", podPhase: v1.PodFailed},
				{podName: "pod-3"},
				{podName: "pod-4"},
				{podName: "pod-5"},
			},
			wantPodNames: []string{"pod-3", "pod-4", "pod-5"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
			rc := newReplicationController(0)
			podList := newPodList(nil, 5, v1.PodRunning, rc)
			for idx, testPod := range test.pods {
				podList.Items[idx].Name = testPod.podName
				podList.Items[idx].Status.Phase = testPod.podPhase
			}

			var podPointers []*v1.Pod
			for i := range podList.Items {
				podPointers = append(podPointers, &podList.Items[i])
			}
			got := FilterActivePods(logger, podPointers)
			gotNames := sets.NewString()
			for _, pod := range got {
				gotNames.Insert(pod.Name)
			}

			if diff := cmp.Diff(test.wantPodNames, gotNames.List()); diff != "" {
				t.Errorf("Active pod names (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestSortingActivePods(t *testing.T) {
	now := metav1.Now()
	then := metav1.Time{Time: now.AddDate(0, -1, 0)}

	tests := []struct {
		name      string
		pods      []v1.Pod
		wantOrder []string
	}{
		{
			name: "Sorts by active pod",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "unscheduled"},
					Spec:       v1.PodSpec{NodeName: ""},
					Status:     v1.PodStatus{Phase: v1.PodPending},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "scheduledButPending"},
					Spec:       v1.PodSpec{NodeName: "bar"},
					Status:     v1.PodStatus{Phase: v1.PodPending},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "unknownPhase"},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status:     v1.PodStatus{Phase: v1.PodUnknown},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "runningButNotReady"},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "runningNoLastTransitionTime"},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status: v1.PodStatus{
						Phase:             v1.PodRunning,
						Conditions:        []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
						ContainerStatuses: []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "runningWithLastTransitionTime"},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status: v1.PodStatus{
						Phase:             v1.PodRunning,
						Conditions:        []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: now}},
						ContainerStatuses: []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "runningLongerTime"},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status: v1.PodStatus{
						Phase:             v1.PodRunning,
						Conditions:        []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}},
						ContainerStatuses: []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "lowerContainerRestartCount", CreationTimestamp: now},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status: v1.PodStatus{
						Phase:             v1.PodRunning,
						Conditions:        []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}},
						ContainerStatuses: []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "oldest", CreationTimestamp: then},
					Spec:       v1.PodSpec{NodeName: "foo"},
					Status: v1.PodStatus{
						Phase:             v1.PodRunning,
						Conditions:        []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}},
						ContainerStatuses: []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}},
					},
				},
			},
			wantOrder: []string{
				"unscheduled",
				"scheduledButPending",
				"unknownPhase",
				"runningButNotReady",
				"runningNoLastTransitionTime",
				"runningWithLastTransitionTime",
				"runningLongerTime",
				"lowerContainerRestartCount",
				"oldest",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			numPods := len(test.pods)

			for i := 0; i < 20; i++ {
				idx := rand.Perm(numPods)
				randomizedPods := make([]*v1.Pod, numPods)
				for j := 0; j < numPods; j++ {
					randomizedPods[j] = &test.pods[idx[j]]
				}

				sort.Sort(ActivePods(randomizedPods))
				gotOrder := make([]string, len(randomizedPods))
				for i := range randomizedPods {
					gotOrder[i] = randomizedPods[i].Name
				}

				if diff := cmp.Diff(test.wantOrder, gotOrder); diff != "" {
					t.Errorf("Sorted active pod names (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestSortingActivePodsWithRanks(t *testing.T) {
	now := metav1.Now()
	then1Month := metav1.Time{Time: now.AddDate(0, -1, 0)}
	then2Hours := metav1.Time{Time: now.Add(-2 * time.Hour)}
	then5Hours := metav1.Time{Time: now.Add(-5 * time.Hour)}
	then8Hours := metav1.Time{Time: now.Add(-8 * time.Hour)}
	zeroTime := metav1.Time{}
	pod := func(podName, nodeName string, phase v1.PodPhase, ready bool, restarts int32, readySince metav1.Time, created metav1.Time, annotations map[string]string) *v1.Pod {
		var conditions []v1.PodCondition
		var containerStatuses []v1.ContainerStatus
		if ready {
			conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: readySince}}
			containerStatuses = []v1.ContainerStatus{{RestartCount: restarts}}
		}
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				CreationTimestamp: created,
				Name:              podName,
				Annotations:       annotations,
			},
			Spec: v1.PodSpec{NodeName: nodeName},
			Status: v1.PodStatus{
				Conditions:        conditions,
				ContainerStatuses: containerStatuses,
				Phase:             phase,
			},
		}
	}
	var (
		unscheduledPod                      = pod("unscheduled", "", v1.PodPending, false, 0, zeroTime, zeroTime, nil)
		scheduledPendingPod                 = pod("pending", "node", v1.PodPending, false, 0, zeroTime, zeroTime, nil)
		unknownPhasePod                     = pod("unknown-phase", "node", v1.PodUnknown, false, 0, zeroTime, zeroTime, nil)
		runningNotReadyPod                  = pod("not-ready", "node", v1.PodRunning, false, 0, zeroTime, zeroTime, nil)
		runningReadyNoLastTransitionTimePod = pod("ready-no-last-transition-time", "node", v1.PodRunning, true, 0, zeroTime, zeroTime, nil)
		runningReadyNow                     = pod("ready-now", "node", v1.PodRunning, true, 0, now, now, nil)
		runningReadyThen                    = pod("ready-then", "node", v1.PodRunning, true, 0, then1Month, then1Month, nil)
		runningReadyNowHighRestarts         = pod("ready-high-restarts", "node", v1.PodRunning, true, 9001, now, now, nil)
		runningReadyNowCreatedThen          = pod("ready-now-created-then", "node", v1.PodRunning, true, 0, now, then1Month, nil)
		lowPodDeletionCost                  = pod("low-deletion-cost", "node", v1.PodRunning, true, 0, now, then1Month, map[string]string{core.PodDeletionCost: "10"})
		highPodDeletionCost                 = pod("high-deletion-cost", "node", v1.PodRunning, true, 0, now, then1Month, map[string]string{core.PodDeletionCost: "100"})
		unscheduled5Hours                   = pod("unscheduled-5-hours", "", v1.PodPending, false, 0, then5Hours, then5Hours, nil)
		unscheduled8Hours                   = pod("unscheduled-10-hours", "", v1.PodPending, false, 0, then8Hours, then8Hours, nil)
		ready2Hours                         = pod("ready-2-hours", "", v1.PodRunning, true, 0, then2Hours, then1Month, nil)
		ready5Hours                         = pod("ready-5-hours", "", v1.PodRunning, true, 0, then5Hours, then1Month, nil)
		ready10Hours                        = pod("ready-10-hours", "", v1.PodRunning, true, 0, then8Hours, then1Month, nil)
	)
	equalityTests := []struct {
		p1                          *v1.Pod
		p2                          *v1.Pod
		disableLogarithmicScaleDown bool
	}{
		{p1: unscheduledPod},
		{p1: scheduledPendingPod},
		{p1: unknownPhasePod},
		{p1: runningNotReadyPod},
		{p1: runningReadyNowCreatedThen},
		{p1: runningReadyNow},
		{p1: runningReadyThen},
		{p1: runningReadyNowHighRestarts},
		{p1: runningReadyNowCreatedThen},
		{p1: unscheduled5Hours, p2: unscheduled8Hours},
		{p1: ready5Hours, p2: ready10Hours},
	}
	for i, test := range equalityTests {
		t.Run(fmt.Sprintf("Equality tests %d", i), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LogarithmicScaleDown, !test.disableLogarithmicScaleDown)
			if test.p2 == nil {
				test.p2 = test.p1
			}
			podsWithRanks := ActivePodsWithRanks{
				Pods: []*v1.Pod{test.p1, test.p2},
				Rank: []int{1, 1},
				Now:  now,
			}
			if podsWithRanks.Less(0, 1) || podsWithRanks.Less(1, 0) {
				t.Errorf("expected pod %q to be equivalent to %q", test.p1.Name, test.p2.Name)
			}
		})
	}

	type podWithRank struct {
		pod  *v1.Pod
		rank int
	}
	inequalityTests := []struct {
		lesser, greater             podWithRank
		disablePodDeletioncost      bool
		disableLogarithmicScaleDown bool
	}{
		{lesser: podWithRank{unscheduledPod, 1}, greater: podWithRank{scheduledPendingPod, 2}},
		{lesser: podWithRank{unscheduledPod, 2}, greater: podWithRank{scheduledPendingPod, 1}},
		{lesser: podWithRank{scheduledPendingPod, 1}, greater: podWithRank{unknownPhasePod, 2}},
		{lesser: podWithRank{unknownPhasePod, 1}, greater: podWithRank{runningNotReadyPod, 2}},
		{lesser: podWithRank{runningNotReadyPod, 1}, greater: podWithRank{runningReadyNoLastTransitionTimePod, 1}},
		{lesser: podWithRank{runningReadyNoLastTransitionTimePod, 1}, greater: podWithRank{runningReadyNow, 1}},
		{lesser: podWithRank{runningReadyNow, 2}, greater: podWithRank{runningReadyNoLastTransitionTimePod, 1}},
		{lesser: podWithRank{runningReadyNow, 1}, greater: podWithRank{runningReadyThen, 1}},
		{lesser: podWithRank{runningReadyNow, 2}, greater: podWithRank{runningReadyThen, 1}},
		{lesser: podWithRank{runningReadyNowHighRestarts, 1}, greater: podWithRank{runningReadyNow, 1}},
		{lesser: podWithRank{runningReadyNow, 2}, greater: podWithRank{runningReadyNowHighRestarts, 1}},
		{lesser: podWithRank{runningReadyNow, 1}, greater: podWithRank{runningReadyNowCreatedThen, 1}},
		{lesser: podWithRank{runningReadyNowCreatedThen, 2}, greater: podWithRank{runningReadyNow, 1}},
		{lesser: podWithRank{lowPodDeletionCost, 2}, greater: podWithRank{highPodDeletionCost, 1}},
		{lesser: podWithRank{highPodDeletionCost, 2}, greater: podWithRank{lowPodDeletionCost, 1}, disablePodDeletioncost: true},
		{lesser: podWithRank{ready2Hours, 1}, greater: podWithRank{ready5Hours, 1}},
	}

	for i, test := range inequalityTests {
		t.Run(fmt.Sprintf("Inequality tests %d", i), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDeletionCost, !test.disablePodDeletioncost)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LogarithmicScaleDown, !test.disableLogarithmicScaleDown)

			podsWithRanks := ActivePodsWithRanks{
				Pods: []*v1.Pod{test.lesser.pod, test.greater.pod},
				Rank: []int{test.lesser.rank, test.greater.rank},
				Now:  now,
			}
			if !podsWithRanks.Less(0, 1) {
				t.Errorf("expected pod %q with rank %v to be less than %q with rank %v", podsWithRanks.Pods[0].Name, podsWithRanks.Rank[0], podsWithRanks.Pods[1].Name, podsWithRanks.Rank[1])
			}
			if podsWithRanks.Less(1, 0) {
				t.Errorf("expected pod %q with rank %v not to be less than %v with rank %v", podsWithRanks.Pods[1].Name, podsWithRanks.Rank[1], podsWithRanks.Pods[0].Name, podsWithRanks.Rank[0])
			}
		})
	}
}

func TestActiveReplicaSetsFiltering(t *testing.T) {

	rsUuid := uuid.NewUUID()
	tests := []struct {
		name            string
		replicaSets     []*apps.ReplicaSet
		wantReplicaSets []*apps.ReplicaSet
	}{
		{
			name: "Filters active replica sets",
			replicaSets: []*apps.ReplicaSet{
				newReplicaSet("zero", 0, rsUuid),
				nil,
				newReplicaSet("foo", 1, rsUuid),
				newReplicaSet("bar", 2, rsUuid),
			},
			wantReplicaSets: []*apps.ReplicaSet{
				newReplicaSet("foo", 1, rsUuid),
				newReplicaSet("bar", 2, rsUuid),
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotReplicaSets := FilterActiveReplicaSets(test.replicaSets)

			if diff := cmp.Diff(test.wantReplicaSets, gotReplicaSets); diff != "" {
				t.Errorf("Active replica set names (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestComputeHash(t *testing.T) {
	collisionCount := int32(1)
	otherCollisionCount := int32(2)
	maxCollisionCount := int32(math.MaxInt32)
	tests := []struct {
		name                string
		template            *v1.PodTemplateSpec
		collisionCount      *int32
		otherCollisionCount *int32
	}{
		{
			name:                "simple",
			template:            &v1.PodTemplateSpec{},
			collisionCount:      &collisionCount,
			otherCollisionCount: &otherCollisionCount,
		},
		{
			name:                "using math.MaxInt64",
			template:            &v1.PodTemplateSpec{},
			collisionCount:      nil,
			otherCollisionCount: &maxCollisionCount,
		},
	}

	for _, test := range tests {
		hash := ComputeHash(test.template, test.collisionCount)
		otherHash := ComputeHash(test.template, test.otherCollisionCount)

		assert.NotEqual(t, hash, otherHash, "expected different hashes but got the same: %d", hash)
	}
}

func TestRemoveTaintOffNode(t *testing.T) {
	tests := []struct {
		name           string
		nodeHandler    *testutil.FakeNodeHandler
		nodeName       string
		taintsToRemove []*v1.Taint
		expectedTaints []v1.Taint
		requestCount   int
	}{
		{
			name: "remove one taint from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
			},
			requestCount: 4,
		},
		{
			name: "remove multiple taints from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
								{Key: "key3", Value: "value3", Effect: "NoSchedule"},
								{Key: "key4", Value: "value4", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			requestCount: 4,
		},
		{
			name: "remove no-exist taints from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
		{
			name: "remove taint from node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: nil,
			requestCount:   2,
		},
		{
			name: "remove empty taint list from node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToRemove: []*v1.Taint{},
			expectedTaints: nil,
			requestCount:   2,
		},
		{
			name: "remove empty taint list from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToRemove: []*v1.Taint{},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
	}
	for _, test := range tests {
		node, _ := test.nodeHandler.Get(context.TODO(), test.nodeName, metav1.GetOptions{})
		err := RemoveTaintOffNode(context.TODO(), test.nodeHandler, test.nodeName, node, test.taintsToRemove...)
		require.NoError(t, err, "%s: RemoveTaintOffNode() error = %v", test.name, err)

		node, _ = test.nodeHandler.Get(context.TODO(), test.nodeName, metav1.GetOptions{})
		assert.EqualValues(t, test.expectedTaints, node.Spec.Taints,
			"%s: failed to remove taint off node: expected %+v, got %+v",
			test.name, test.expectedTaints, node.Spec.Taints)

		assert.Equal(t, test.requestCount, test.nodeHandler.RequestCount,
			"%s: unexpected request count: expected %+v, got %+v",
			test.name, test.requestCount, test.nodeHandler.RequestCount)
	}
}

func TestAddOrUpdateTaintOnNode(t *testing.T) {
	tests := []struct {
		name           string
		nodeHandler    *testutil.FakeNodeHandler
		nodeName       string
		taintsToAdd    []*v1.Taint
		expectedTaints []v1.Taint
		requestCount   int
		expectedErr    error
	}{
		{
			name: "add one taint on node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 3,
		},
		{
			name: "add multiple taints to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			requestCount: 3,
		},
		{
			name: "add exist taints to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
		{
			name: "add taint to node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			requestCount: 3,
		},
		{
			name: "add empty taint list to node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToAdd:    []*v1.Taint{},
			expectedTaints: nil,
			requestCount:   1,
		},
		{
			name: "add empty taint list to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:    "node1",
			taintsToAdd: []*v1.Taint{},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 1,
		},
		{
			name: "add taint to changed node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:            "node1",
							ResourceVersion: "1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
				AsyncCalls: []func(*testutil.FakeNodeHandler){func(m *testutil.FakeNodeHandler) {
					if len(m.UpdatedNodes) == 0 {
						m.UpdatedNodes = append(m.UpdatedNodes, &v1.Node{
							ObjectMeta: metav1.ObjectMeta{
								Name:            "node1",
								ResourceVersion: "2",
							},
							Spec: v1.NodeSpec{
								Taints: []v1.Taint{},
							}})
					}
				}},
			},
			nodeName:    "node1",
			taintsToAdd: []*v1.Taint{{Key: "key2", Value: "value2", Effect: "NoExecute"}},
			expectedTaints: []v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 5,
		},
		{
			name: "add taint to non-exist node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:            "node1",
							ResourceVersion: "1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:    "node2",
			taintsToAdd: []*v1.Taint{{Key: "key2", Value: "value2", Effect: "NoExecute"}},
			expectedErr: apierrors.NewNotFound(schema.GroupResource{Resource: "nodes"}, "node2"),
		},
	}
	for _, test := range tests {
		err := AddOrUpdateTaintOnNode(context.TODO(), test.nodeHandler, test.nodeName, test.taintsToAdd...)
		if test.expectedErr != nil {
			assert.Equal(t, test.expectedErr, err, "AddOrUpdateTaintOnNode get unexpected error")
			continue
		}
		require.NoError(t, err, "%s: AddOrUpdateTaintOnNode() error = %v", test.name, err)

		node, _ := test.nodeHandler.Get(context.TODO(), test.nodeName, metav1.GetOptions{})
		assert.EqualValues(t, test.expectedTaints, node.Spec.Taints,
			"%s: failed to add taint to node: expected %+v, got %+v",
			test.name, test.expectedTaints, node.Spec.Taints)

		assert.Equal(t, test.requestCount, test.nodeHandler.RequestCount,
			"%s: unexpected request count: expected %+v, got %+v",
			test.name, test.requestCount, test.nodeHandler.RequestCount)
	}
}
