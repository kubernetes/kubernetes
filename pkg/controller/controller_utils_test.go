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
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http/httptest"
	"sort"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/securitycontext"

	"github.com/stretchr/testify/assert"
)

// NewFakeControllerExpectationsLookup creates a fake store for PodExpectations.
func NewFakeControllerExpectationsLookup(ttl time.Duration) (*ControllerExpectations, *clock.FakeClock) {
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := clock.NewFakeClock(fakeTime)
	ttlPolicy := &cache.TTLPolicy{Ttl: ttl, Clock: fakeClock}
	ttlStore := cache.NewFakeExpirationStore(
		ExpKeyFunc, nil, ttlPolicy, fakeClock)
	return &ControllerExpectations{ttlStore}, fakeClock
}

func newReplicationController(replicas int) *v1.ReplicationController {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{APIVersion: legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
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
							Image: "foo/bar",
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

func newReplicaSet(name string, replicas int) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		TypeMeta: metav1.TypeMeta{APIVersion: legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            name,
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
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
							Image: "foo/bar",
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
	ttl := 30 * time.Second
	e, fakeClock := NewFakeControllerExpectationsLookup(ttl)
	// In practice we can't really have add and delete expectations since we only either create or
	// delete replicas in one rc pass, and the rc goes to sleep soon after until the expectations are
	// either fulfilled or timeout.
	adds, dels := 10, 30
	rc := newReplicationController(1)

	// RC fires off adds and deletes at apiserver, then sets expectations
	rcKey, err := KeyFunc(rc)
	assert.NoError(t, err, "Couldn't get key for object %#v: %v", rc, err)

	e.SetExpectations(rcKey, adds, dels)
	var wg sync.WaitGroup
	for i := 0; i < adds+1; i++ {
		wg.Add(1)
		go func() {
			// In prod this can happen either because of a failed create by the rc
			// or after having observed a create via informer
			e.CreationObserved(rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	// There are still delete expectations
	assert.False(t, e.SatisfiedExpectations(rcKey), "Rc will sync before expectations are met")

	for i := 0; i < dels+1; i++ {
		wg.Add(1)
		go func() {
			e.DeletionObserved(rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	// Expectations have been surpassed
	podExp, exists, err := e.GetExpectations(rcKey)
	assert.NoError(t, err, "Could not get expectations for rc, exists %v and err %v", exists, err)
	assert.True(t, exists, "Could not get expectations for rc, exists %v and err %v", exists, err)

	add, del := podExp.GetExpectations()
	assert.Equal(t, int64(-1), add, "Unexpected pod expectations %#v", podExp)
	assert.Equal(t, int64(-1), del, "Unexpected pod expectations %#v", podExp)
	assert.True(t, e.SatisfiedExpectations(rcKey), "Expectations are met but the rc will not sync")

	// Next round of rc sync, old expectations are cleared
	e.SetExpectations(rcKey, 1, 2)
	podExp, exists, err = e.GetExpectations(rcKey)
	assert.NoError(t, err, "Could not get expectations for rc, exists %v and err %v", exists, err)
	assert.True(t, exists, "Could not get expectations for rc, exists %v and err %v", exists, err)
	add, del = podExp.GetExpectations()

	assert.Equal(t, int64(1), add, "Unexpected pod expectations %#v", podExp)
	assert.Equal(t, int64(2), del, "Unexpected pod expectations %#v", podExp)

	// Expectations have expired because of ttl
	fakeClock.Step(ttl + 1)
	assert.True(t, e.SatisfiedExpectations(rcKey),
		"Expectations should have expired but didn't")
}

func TestUIDExpectations(t *testing.T) {
	uidExp := NewUIDTrackingControllerExpectations(NewControllerExpectations())
	rcList := []*v1.ReplicationController{
		newReplicationController(2),
		newReplicationController(1),
		newReplicationController(0),
		newReplicationController(5),
	}
	rcToPods := map[string][]string{}
	rcKeys := []string{}
	for i := range rcList {
		rc := rcList[i]
		rcName := fmt.Sprintf("rc-%v", i)
		rc.Name = rcName
		rc.Spec.Selector[rcName] = rcName
		podList := newPodList(nil, 5, v1.PodRunning, rc)
		rcKey, err := KeyFunc(rc)
		if err != nil {
			t.Fatalf("Couldn't get key for object %#v: %v", rc, err)
		}
		rcKeys = append(rcKeys, rcKey)
		rcPodNames := []string{}
		for i := range podList.Items {
			p := &podList.Items[i]
			p.Name = fmt.Sprintf("%v-%v", p.Name, rc.Name)
			rcPodNames = append(rcPodNames, PodKey(p))
		}
		rcToPods[rcKey] = rcPodNames
		uidExp.ExpectDeletions(rcKey, rcPodNames)
	}
	for i := range rcKeys {
		j := rand.Intn(i + 1)
		rcKeys[i], rcKeys[j] = rcKeys[j], rcKeys[i]
	}
	for _, rcKey := range rcKeys {
		assert.False(t, uidExp.SatisfiedExpectations(rcKey),
			"Controller %v satisfied expectations before deletion", rcKey)

		for _, p := range rcToPods[rcKey] {
			uidExp.DeletionObserved(rcKey, p)
		}

		assert.True(t, uidExp.SatisfiedExpectations(rcKey),
			"Controller %v didn't satisfy expectations after deletion", rcKey)

		uidExp.DeleteExpectations(rcKey)

		assert.Nil(t, uidExp.GetUIDs(rcKey),
			"Failed to delete uid expectations for %v", rcKey)
	}
}

func TestCreatePods(t *testing.T) {
	ns := metav1.NamespaceDefault
	body := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "empty_pod"}})
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	podControl := RealPodControl{
		KubeClient: clientset,
		Recorder:   &record.FakeRecorder{},
	}

	controllerSpec := newReplicationController(1)

	// Make sure createReplica sends a POST to the apiserver with a pod from the controllers pod template
	err := podControl.CreatePods(ns, controllerSpec.Spec.Template, controllerSpec)
	assert.NoError(t, err, "unexpected error: %v", err)

	expectedPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels:       controllerSpec.Spec.Template.Labels,
			GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
		},
		Spec: controllerSpec.Spec.Template.Spec,
	}
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath("pods", metav1.NamespaceDefault, ""), "POST", nil)
	var actualPod = &v1.Pod{}
	err = json.Unmarshal([]byte(fakeHandler.RequestBody), actualPod)
	assert.NoError(t, err, "unexpected error: %v", err)
	assert.True(t, apiequality.Semantic.DeepDerivative(&expectedPod, actualPod),
		"Body: %s", fakeHandler.RequestBody)
}

func TestActivePodFiltering(t *testing.T) {
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, 5, v1.PodRunning, rc)
	podList.Items[0].Status.Phase = v1.PodSucceeded
	podList.Items[1].Status.Phase = v1.PodFailed
	expectedNames := sets.NewString()
	for _, pod := range podList.Items[2:] {
		expectedNames.Insert(pod.Name)
	}

	var podPointers []*v1.Pod
	for i := range podList.Items {
		podPointers = append(podPointers, &podList.Items[i])
	}
	got := FilterActivePods(podPointers)
	gotNames := sets.NewString()
	for _, pod := range got {
		gotNames.Insert(pod.Name)
	}

	assert.Equal(t, 0, expectedNames.Difference(gotNames).Len(),
		"expected %v, got %v", expectedNames.List(), gotNames.List())
	assert.Equal(t, 0, gotNames.Difference(expectedNames).Len(),
		"expected %v, got %v", expectedNames.List(), gotNames.List())
}

func TestSortingActivePods(t *testing.T) {
	numPods := 9
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, numPods, v1.PodRunning, rc)

	pods := make([]*v1.Pod, len(podList.Items))
	for i := range podList.Items {
		pods[i] = &podList.Items[i]
	}
	// pods[0] is not scheduled yet.
	pods[0].Spec.NodeName = ""
	pods[0].Status.Phase = v1.PodPending
	// pods[1] is scheduled but pending.
	pods[1].Spec.NodeName = "bar"
	pods[1].Status.Phase = v1.PodPending
	// pods[2] is unknown.
	pods[2].Spec.NodeName = "foo"
	pods[2].Status.Phase = v1.PodUnknown
	// pods[3] is running but not ready.
	pods[3].Spec.NodeName = "foo"
	pods[3].Status.Phase = v1.PodRunning
	// pods[4] is running and ready but without LastTransitionTime.
	now := metav1.Now()
	pods[4].Spec.NodeName = "foo"
	pods[4].Status.Phase = v1.PodRunning
	pods[4].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
	pods[4].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[5] is running and ready and with LastTransitionTime.
	pods[5].Spec.NodeName = "foo"
	pods[5].Status.Phase = v1.PodRunning
	pods[5].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: now}}
	pods[5].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[6] is running ready for a longer time than pods[5].
	then := metav1.Time{Time: now.AddDate(0, -1, 0)}
	pods[6].Spec.NodeName = "foo"
	pods[6].Status.Phase = v1.PodRunning
	pods[6].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[6].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[7] has lower container restart count than pods[6].
	pods[7].Spec.NodeName = "foo"
	pods[7].Status.Phase = v1.PodRunning
	pods[7].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[7].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}}
	pods[7].CreationTimestamp = now
	// pods[8] is older than pods[7].
	pods[8].Spec.NodeName = "foo"
	pods[8].Status.Phase = v1.PodRunning
	pods[8].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[8].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}}
	pods[8].CreationTimestamp = then

	getOrder := func(pods []*v1.Pod) []string {
		names := make([]string, len(pods))
		for i := range pods {
			names[i] = pods[i].Name
		}
		return names
	}

	expected := getOrder(pods)

	for i := 0; i < 20; i++ {
		idx := rand.Perm(numPods)
		randomizedPods := make([]*v1.Pod, numPods)
		for j := 0; j < numPods; j++ {
			randomizedPods[j] = pods[idx[j]]
		}
		sort.Sort(ActivePods(randomizedPods))
		actual := getOrder(randomizedPods)

		assert.EqualValues(t, expected, actual, "expected %v, got %v", expected, actual)
	}
}

func TestActiveReplicaSetsFiltering(t *testing.T) {
	var replicaSets []*extensions.ReplicaSet
	replicaSets = append(replicaSets, newReplicaSet("zero", 0))
	replicaSets = append(replicaSets, nil)
	replicaSets = append(replicaSets, newReplicaSet("foo", 1))
	replicaSets = append(replicaSets, newReplicaSet("bar", 2))
	expectedNames := sets.NewString()
	for _, rs := range replicaSets[2:] {
		expectedNames.Insert(rs.Name)
	}

	got := FilterActiveReplicaSets(replicaSets)
	gotNames := sets.NewString()
	for _, rs := range got {
		gotNames.Insert(rs.Name)
	}

	assert.Equal(t, 0, expectedNames.Difference(gotNames).Len(),
		"expected %v, got %v", expectedNames.List(), gotNames.List())
	assert.Equal(t, 0, gotNames.Difference(expectedNames).Len(),
		"expected %v, got %v", expectedNames.List(), gotNames.List())
}

func int64P(num int64) *int64 {
	return &num
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
		node, _ := test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
		err := RemoveTaintOffNode(test.nodeHandler, test.nodeName, node, test.taintsToRemove...)
		assert.NoError(t, err, "%s: RemoveTaintOffNode() error = %v", test.name, err)

		node, _ = test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
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
	}
	for _, test := range tests {
		err := AddOrUpdateTaintOnNode(test.nodeHandler, test.nodeName, test.taintsToAdd...)
		assert.NoError(t, err, "%s: AddOrUpdateTaintOnNode() error = %v", test.name, err)

		node, _ := test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
		assert.EqualValues(t, test.expectedTaints, node.Spec.Taints,
			"%s: failed to add taint to node: expected %+v, got %+v",
			test.name, test.expectedTaints, node.Spec.Taints)

		assert.Equal(t, test.requestCount, test.nodeHandler.RequestCount,
			"%s: unexpected request count: expected %+v, got %+v",
			test.name, test.requestCount, test.nodeHandler.RequestCount)
	}
}
