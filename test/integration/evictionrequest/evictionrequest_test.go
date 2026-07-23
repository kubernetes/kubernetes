/*
Copyright The Kubernetes Authors.

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

package evictionrequest

//import (
//	"context"
//	"fmt"
//	"testing"
//	"time"
//
//	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
//	v1 "k8s.io/api/core/v1"
//	apierrors "k8s.io/apimachinery/pkg/api/errors"
//	"k8s.io/apimachinery/pkg/api/meta"
//	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
//	apimachinerytypes "k8s.io/apimachinery/pkg/types"
//	"k8s.io/apimachinery/pkg/util/wait"
//	utilfeature "k8s.io/apiserver/pkg/util/feature"
//	coordinationapply "k8s.io/client-go/applyconfigurations/coordination/v1alpha1"
//	"k8s.io/client-go/informers"
//	clientset "k8s.io/client-go/kubernetes"
//	restclient "k8s.io/client-go/rest"
//	featuregatetesting "k8s.io/component-base/featuregate/testing"
//	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
//	"k8s.io/kubernetes/pkg/controller/evictionrequest"
//	"k8s.io/kubernetes/pkg/features"
//	"k8s.io/kubernetes/test/integration/framework"
//	"k8s.io/kubernetes/test/utils/ktesting"
//	"k8s.io/utils/ptr"
//)
//
//func setup(ctx context.Context, t *testing.T, extraServerFlags ...string) (
//	kubeapiservertesting.TearDownFunc,
//	*evictionrequest.EvictionRequestController,
//	informers.SharedInformerFactory,
//	clientset.Interface,
//) {
//	t.Helper()
//
//	// Enable the EvictionRequestAPI feature gate (alpha) and the coordination.k8s.io/v1alpha1 API group.
//	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)
//
//	serverFlags := append(framework.DefaultTestServerFlags(),
//		"--feature-gates=EvictionRequestAPI=true",
//		"--runtime-config=coordination.k8s.io/v1alpha1=true",
//	)
//	serverFlags = append(serverFlags, extraServerFlags...)
//
//	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
//		serverFlags,
//		framework.SharedEtcd(),
//	)
//
//	config := restclient.CopyConfig(server.ClientConfig)
//	cs, err := clientset.NewForConfig(config)
//	if err != nil {
//		t.Fatalf("Error creating clientset: %v", err)
//	}
//
//	resyncPeriod := 12 * time.Hour
//	inf := informers.NewSharedInformerFactory(
//		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "evictionrequest-informers")),
//		resyncPeriod,
//	)
//
//	controllerClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "evictionrequest-controller"))
//
//	c, err := evictionrequest.NewController(
//		ctx,
//		inf.Coordination().V1alpha1().EvictionRequests(),
//		inf.Core().V1().Pods(),
//		controllerClient,
//		"evictionrequest-controller",
//	)
//	if err != nil {
//		t.Fatalf("Error creating eviction request controller: %v", err)
//	}
//
//	return server.TearDownFn, c, inf, cs
//}
//
//// newTestPod creates a running pod for testing.
//func newTestPod(name string) *v1.Pod {
//	return &v1.Pod{
//		ObjectMeta: metav1.ObjectMeta{
//			Name:   name,
//			Labels: map[string]string{"app": "test"},
//		},
//		Spec: v1.PodSpec{
//			Containers: []v1.Container{
//				{Name: "test", Image: "registry.k8s.io/pause:3.9"},
//			},
//		},
//	}
//}
//
//// newTestEvictionRequest creates an EvictionRequest targeting the given pod.
//func newTestEvictionRequest(pod *v1.Pod) *coordinationv1alpha1.EvictionRequest {
//	return &coordinationv1alpha1.EvictionRequest{
//		ObjectMeta: metav1.ObjectMeta{
//			Name:      string(pod.UID),
//			Namespace: pod.Namespace,
//		},
//		Spec: coordinationv1alpha1.EvictionRequestSpec{
//			Target: coordinationv1alpha1.EvictionTarget{
//				Pod: &coordinationv1alpha1.PodReference{
//					Name: pod.Name,
//					UID:  pod.UID,
//				},
//			},
//			Requesters: []coordinationv1alpha1.Requester{
//				{Name: "test-requester.example.com/default", Intent: coordinationv1alpha1.RequesterIntentEviction},
//			},
//		},
//	}
//}
//
//// waitForEvictionRequestCondition polls until the EvictionRequest has the expected condition or times out.
//func waitForEvictionRequestCondition(
//	ctx context.Context,
//	t *testing.T,
//	cs clientset.Interface,
//	namespace, name string,
//	conditionType string,
//	expectedStatus metav1.ConditionStatus,
//) *coordinationv1alpha1.EvictionRequest {
//	t.Helper()
//	var er *coordinationv1alpha1.EvictionRequest
//	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
//		var err error
//		er, err = cs.CoordinationV1alpha1().EvictionRequests(namespace).Get(ctx, name, metav1.GetOptions{})
//		if err != nil {
//			return false, nil
//		}
//		cond := meta.FindStatusCondition(er.Status.Conditions, conditionType)
//		if cond == nil {
//			return false, nil
//		}
//		return cond.Status == expectedStatus, nil
//	}); err != nil {
//		t.Fatalf("timed out waiting for EvictionRequest %s/%s to have condition %s=%s: %v",
//			namespace, name, conditionType, expectedStatus, err)
//	}
//	return er
//}
//
//// waitForEvictionRequestStatus polls until the status check function returns true.
//func waitForEvictionRequestStatus(
//	ctx context.Context,
//	t *testing.T,
//	cs clientset.Interface,
//	namespace, name string,
//	check func(*coordinationv1alpha1.EvictionRequest) bool,
//	description string,
//) *coordinationv1alpha1.EvictionRequest {
//	t.Helper()
//	var er *coordinationv1alpha1.EvictionRequest
//	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
//		var err error
//		er, err = cs.CoordinationV1alpha1().EvictionRequests(namespace).Get(ctx, name, metav1.GetOptions{})
//		if err != nil {
//			return false, nil
//		}
//		return check(er), nil
//	}); err != nil {
//		t.Fatalf("timed out waiting for EvictionRequest %s/%s: %s: %v",
//			namespace, name, description, err)
//	}
//	return er
//}
//
//// createPodAndWait creates a pod and waits for it to be persisted, returning the pod with its UID.
//func createPodAndWait(ctx context.Context, t *testing.T, cs clientset.Interface, namespace string, pod *v1.Pod) *v1.Pod {
//	t.Helper()
//	created, err := cs.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
//	}
//	return created
//}
//
//// TestValidation_PodNotFound verifies that an EvictionRequest targeting a non-existent pod
//// gets a Failed condition.
//func TestValidation_PodNotFound(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-not-found", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create an EvictionRequest targeting a pod that doesn't exist.
//	// EvictionRequest names and pod UIDs must be RFC 4122 UUIDs.
//	nonexistentUID := "00000000-0000-0000-0000-000000000001"
//	er := &coordinationv1alpha1.EvictionRequest{
//		ObjectMeta: metav1.ObjectMeta{
//			Name:      nonexistentUID,
//			Namespace: ns.Name,
//		},
//		Spec: coordinationv1alpha1.EvictionRequestSpec{
//			Target: coordinationv1alpha1.EvictionTarget{
//				Pod: &coordinationv1alpha1.PodReference{
//					Name: "nonexistent-pod",
//					UID:  apimachinerytypes.UID(nonexistentUID),
//				},
//			},
//			Requesters: []coordinationv1alpha1.Requester{
//				{Name: "test-requester.example.com/default", Intent: coordinationv1alpha1.RequesterIntentEviction},
//			},
//		},
//	}
//
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for Failed condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, nonexistentUID, string(coordinationv1alpha1.EvictionRequestConditionFailed), metav1.ConditionTrue)
//	cond := meta.FindStatusCondition(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid), cond.Reason)
//	}
//}
//
//// TestValidation_UIDMismatch verifies that an EvictionRequest with the wrong pod UID
//// gets a Failed condition.
//func TestValidation_UIDMismatch(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-uid-mismatch", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create a pod
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	// Create EvictionRequest with wrong UID.
//	// EvictionRequest names and pod UIDs must be RFC 4122 UUIDs.
//	wrongUID := "00000000-0000-0000-0000-000000000002"
//	er := &coordinationv1alpha1.EvictionRequest{
//		ObjectMeta: metav1.ObjectMeta{
//			Name:      wrongUID,
//			Namespace: ns.Name,
//		},
//		Spec: coordinationv1alpha1.EvictionRequestSpec{
//			Target: coordinationv1alpha1.EvictionTarget{
//				Pod: &coordinationv1alpha1.PodReference{
//					Name: pod.Name,
//					UID:  apimachinerytypes.UID(wrongUID),
//				},
//			},
//			Requesters: []coordinationv1alpha1.Requester{
//				{Name: "test-requester.example.com/default", Intent: coordinationv1alpha1.RequesterIntentEviction},
//			},
//		},
//	}
//
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for Failed condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, wrongUID, string(coordinationv1alpha1.EvictionRequestConditionFailed), metav1.ConditionTrue)
//	cond := meta.FindStatusCondition(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid), cond.Reason)
//	}
//}
//
//// hasActiveResponder returns true if any TargetResponder has Active state.
//func hasActiveResponder(targetResponders []coordinationv1alpha1.TargetResponder) bool {
//	for _, tr := range targetResponders {
//		if tr.State == coordinationv1alpha1.ResponderStateActive {
//			return true
//		}
//	}
//	return false
//}
//
//// activeResponderName returns the name of the active responder, or "" if none.
//func activeResponderName(targetResponders []coordinationv1alpha1.TargetResponder) string {
//	for _, tr := range targetResponders {
//		if tr.State == coordinationv1alpha1.ResponderStateActive {
//			return tr.Name
//		}
//	}
//	return ""
//}
//
//// findTargetResponderState returns the state of the named responder, or "" if not found.
//func findTargetResponderState(targetResponders []coordinationv1alpha1.TargetResponder, name string) coordinationv1alpha1.ResponderStateType {
//	for _, tr := range targetResponders {
//		if tr.Name == name {
//			return tr.State
//		}
//	}
//	return ""
//}
//
//// findResponderStatus finds the status for a given responder name.
//func findResponderStatus(statuses []coordinationv1alpha1.ResponderStatus, name string) *coordinationv1alpha1.ResponderStatus {
//	for i := range statuses {
//		if statuses[i].Name == name {
//			return &statuses[i]
//		}
//	}
//	return nil
//}
//
//// TestInitializeTargetResponders verifies that the controller initializes target responders
//// from the pod's EvictionResponders list plus the default imperative responder, and that
//// matching ResponderStatus entries are created in the same order.
//func TestInitializeTargetResponders(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-responders", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod with two custom eviction responders
//	pod := newTestPod("test-pod")
//	pod.Spec.EvictionResponders = []v1.EvictionResponder{
//		{Name: "first.example.com/handler"},
//		{Name: "second.example.com/handler"},
//	}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	expectedNames := []string{"first.example.com/handler", "second.example.com/handler", string(coordinationv1alpha1.EvictionResponderImperativeEviction)}
//
//	// Wait for target responders and responder statuses to be initialized
//	updated := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return len(er.Status.TargetResponders) == 3 && len(er.Status.Responders) == 3
//		},
//		"waiting for 3 target responders and 3 responder statuses",
//	)
//
//	// Verify target responders order
//	for i, expected := range expectedNames {
//		if updated.Status.TargetResponders[i].Name != expected {
//			t.Errorf("targetResponders[%d]: expected %q, got %q", i, expected, updated.Status.TargetResponders[i].Name)
//		}
//	}
//
//	// Verify responder statuses match in the same order
//	for i, expected := range expectedNames {
//		if updated.Status.Responders[i].Name != expected {
//			t.Errorf("responders[%d]: expected name %q, got %q", i, expected, updated.Status.Responders[i].Name)
//		}
//	}
//}
//
//// TestSelectFirstResponder verifies that the controller selects the first responder,
//// initializes its StartTime, and that a pod with no custom responders gets only the
//// imperative responder as its target.
//func TestSelectFirstResponder(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-select-first", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod (no custom responders, so only imperative will be used)
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for active responder to be set
//	updated := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return hasActiveResponder(er.Status.TargetResponders)
//		},
//		"waiting for active responder",
//	)
//
//	// Verify only the imperative responder is present as target
//	if len(updated.Status.TargetResponders) != 1 {
//		t.Fatalf("expected 1 target responder, got %d", len(updated.Status.TargetResponders))
//	}
//	if updated.Status.TargetResponders[0].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
//		t.Errorf("expected only imperative target responder, got %s", updated.Status.TargetResponders[0].Name)
//	}
//
//	if updated.Status.TargetResponders[0].State != coordinationv1alpha1.ResponderStateActive {
//		t.Errorf("expected responder state to be Active, got %s", updated.Status.TargetResponders[0].State)
//	}
//
//	// Verify start time was initialized (not heartbeat — controller sets StartTime,
//	// responder is responsible for setting HeartbeatTime)
//	if len(updated.Status.Responders) == 0 {
//		t.Fatal("expected responder status to be initialized")
//	}
//	if updated.Status.Responders[0].StartTime == nil {
//		t.Error("expected start time to be initialized for active responder")
//	}
//	if updated.Status.Responders[0].HeartbeatTime != nil {
//		t.Error("expected heartbeat time to not be set by controller")
//	}
//}
//
//// TestPodDeleted_Evicted verifies that when a target pod is deleted, the EvictionRequest
//// gets the Evicted condition.
//func TestPodDeleted_Evicted(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-deleted", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for controller to process the EvictionRequest (active responder set)
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return er.Status.ObservedGeneration != nil && *er.Status.ObservedGeneration > 0
//		},
//		"waiting for initial processing",
//	)
//
//	// Delete the pod
//	err = cs.CoreV1().Pods(ns.Name).Delete(tCtx, pod.Name, metav1.DeleteOptions{
//		GracePeriodSeconds: new(int64), // immediate deletion
//	})
//	if err != nil {
//		t.Fatalf("Failed to delete pod: %v", err)
//	}
//
//	// Wait for Evicted condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
//		"Evicted", metav1.ConditionTrue)
//
//	cond := meta.FindStatusCondition(updated.Status.Conditions, "Evicted")
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonPodDeleted) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonPodDeleted), cond.Reason)
//	}
//
//	// Should NOT have Failed condition
//	if meta.IsStatusConditionTrue(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed)) {
//		t.Error("should not have Failed condition when pod is deleted after validation")
//	}
//}
//
//// TestEmptyRequesters_RejectedByAPI verifies that an EvictionRequest with no requesters
//// is rejected at creation time by the API validation.
//func TestEmptyRequesters_RejectedByAPI(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-no-requesters", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	// Create EvictionRequest with no requesters
//	er := newTestEvictionRequest(pod)
//	er.Spec.Requesters = nil
//
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err == nil {
//		t.Fatal("expected creation to fail with no requesters")
//	}
//	if !apierrors.IsInvalid(err) {
//		t.Errorf("expected Invalid error, got: %v", err)
//	}
//}
//
//// TestLabelSync verifies that pod labels are synced to the EvictionRequest.
//func TestLabelSync(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-label-sync", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod with labels
//	pod := newTestPod("test-pod")
//	pod.Labels = map[string]string{
//		"app":     "myapp",
//		"version": "v1",
//	}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for labels to be synced
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return er.Labels != nil &&
//				er.Labels["app"] == "myapp" &&
//				er.Labels["version"] == "v1"
//		},
//		"waiting for pod labels to sync to EvictionRequest",
//	)
//}
//
//// TestLabelSyncUpdate verifies that label changes on the pod are propagated to the EvictionRequest.
//func TestLabelSyncUpdate(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-label-update", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod
//	pod := newTestPod("test-pod")
//	pod.Labels = map[string]string{"app": "v1"}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for initial label sync
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return er.Labels != nil && er.Labels["app"] == "v1"
//		},
//		"waiting for initial label sync",
//	)
//
//	// Update pod labels
//	pod, err = cs.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
//	if err != nil {
//		t.Fatalf("Failed to get pod: %v", err)
//	}
//	pod.Labels["app"] = "v2"
//	_, err = cs.CoreV1().Pods(ns.Name).Update(tCtx, pod, metav1.UpdateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to update pod labels: %v", err)
//	}
//
//	// Wait for updated labels to sync
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return er.Labels != nil && er.Labels["app"] == "v2"
//		},
//		"waiting for updated label sync",
//	)
//}
//
//// TestMultipleEvictionRequests verifies that the controller can handle multiple
//// EvictionRequests concurrently.
//func TestMultipleEvictionRequests(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-multiple", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 2) // Run with 2 workers
//
//	numPods := 5
//	pods := make([]*v1.Pod, numPods)
//	for i := range numPods {
//		pods[i] = createPodAndWait(tCtx, t, cs, ns.Name, newTestPod(fmt.Sprintf("test-pod-%d", i)))
//	}
//
//	// Create EvictionRequests for all pods
//	for _, pod := range pods {
//		er := newTestEvictionRequest(pod)
//		_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//		if err != nil {
//			t.Fatalf("Failed to create EvictionRequest for %s: %v", pod.Name, err)
//		}
//	}
//
//	// Wait for all EvictionRequests to be processed (active responder set)
//	for _, pod := range pods {
//		waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//			func(er *coordinationv1alpha1.EvictionRequest) bool {
//				return hasActiveResponder(er.Status.TargetResponders)
//			},
//			fmt.Sprintf("waiting for EvictionRequest for %s to have active responder", pod.Name),
//		)
//	}
//}
//
//// TestValidation_WorkloadRef verifies that pods with WorkloadRef are rejected.
//func TestValidation_WorkloadRef(t *testing.T) {
//	tCtx := ktesting.Init(t)
//
//	// WorkloadRef requires the GenericWorkload feature gate
//	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
//
//	closeFn, c, inf, cs := setup(tCtx, t, "--feature-gates=EvictionRequestAPI=true,GenericWorkload=true")
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-workloadref", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create pod with WorkloadRef
//	pod := newTestPod("test-pod")
//	pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("my-podgroup")}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	// Create EvictionRequest
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for Failed condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
//		string(coordinationv1alpha1.EvictionRequestConditionFailed), metav1.ConditionTrue)
//
//	cond := meta.FindStatusCondition(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid), cond.Reason)
//	}
//}
//
//// TestTerminalStateIdempotent verifies that re-syncing an already-terminal
//// EvictionRequest does not re-process or overwrite existing conditions.
//func TestTerminalStateIdempotent(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-terminal-idempotent", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	// Create an EvictionRequest targeting a nonexistent pod to get a Failed condition
//	nonexistentUID := "00000000-0000-0000-0000-000000000099"
//	er := &coordinationv1alpha1.EvictionRequest{
//		ObjectMeta: metav1.ObjectMeta{
//			Name:      nonexistentUID,
//			Namespace: ns.Name,
//		},
//		Spec: coordinationv1alpha1.EvictionRequestSpec{
//			Target: coordinationv1alpha1.EvictionTarget{
//				Pod: &coordinationv1alpha1.PodReference{
//					Name: "nonexistent-pod",
//					UID:  apimachinerytypes.UID(nonexistentUID),
//				},
//			},
//			Requesters: []coordinationv1alpha1.Requester{
//				{Name: "test-requester.example.com/default", Intent: coordinationv1alpha1.RequesterIntentEviction},
//			},
//		},
//	}
//
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for Failed condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, nonexistentUID, string(coordinationv1alpha1.EvictionRequestConditionFailed), metav1.ConditionTrue)
//	observedGenBefore := updated.Status.ObservedGeneration
//
//	// Trigger a re-sync by touching a label on the EvictionRequest
//	updated.Labels = map[string]string{"trigger": "resync"}
//	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Update(tCtx, updated, metav1.UpdateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to update EvictionRequest: %v", err)
//	}
//
//	// Wait for the label update to be visible, confirming the controller had a
//	// chance to process the re-sync triggered by the EvictionRequest update event.
//	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, nonexistentUID,
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return er.Labels != nil && er.Labels["trigger"] == "resync"
//		},
//		"waiting for label update to be visible",
//	)
//
//	cond := meta.FindStatusCondition(final.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
//	if cond == nil || cond.Status != metav1.ConditionTrue {
//		t.Error("expected Failed condition to remain True after re-sync")
//	}
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid) {
//		t.Errorf("expected reason to remain %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid), cond.Reason)
//	}
//
//	// ObservedGeneration should not have been bumped by re-processing
//	if *final.Status.ObservedGeneration != *observedGenBefore {
//		t.Errorf("expected ObservedGeneration to remain %d after re-sync of terminal state, got %d",
//			*observedGenBefore, *final.Status.ObservedGeneration)
//	}
//}
//
//// TestPodTerminal_Evicted verifies that when a target pod is already in a terminal phase
//// (Succeeded or Failed), the EvictionRequest gets the Evicted condition with TargetTerminal reason.
//func TestPodTerminal_Evicted(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-terminal", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	// Mark pod as terminal (Succeeded) before creating the EvictionRequest
//	pod, err := cs.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
//	if err != nil {
//		t.Fatalf("Failed to get pod: %v", err)
//	}
//	pod.Status.Phase = v1.PodSucceeded
//	pod, err = cs.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to update pod status: %v", err)
//	}
//
//	// Create EvictionRequest targeting the terminal pod
//	er := newTestEvictionRequest(pod)
//	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for Evicted condition with TargetTerminal reason
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
//		"Evicted", metav1.ConditionTrue)
//
//	cond := meta.FindStatusCondition(updated.Status.Conditions, "Evicted")
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonPodTerminal) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonPodTerminal), cond.Reason)
//	}
//
//	if meta.IsStatusConditionTrue(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed)) {
//		t.Error("should not have Failed condition when pod is terminal")
//	}
//}
//
//// TestResponderStatusPreservedOnAdvancement verifies that when the controller advances
//// to the next responder, the completed responder's status fields are preserved.
//func TestResponderStatusPreservedOnAdvancement(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-status-preserved", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	pod := newTestPod("test-pod")
//	pod.Spec.EvictionResponders = []v1.EvictionResponder{
//		{Name: "first.example.com/handler"},
//	}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for first responder to become active
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return activeResponderName(er.Status.TargetResponders) == "first.example.com/handler"
//		},
//		"waiting for first responder to become active",
//	)
//
//	// Simulate responder completion via SSA with the responder's own field manager.
//	// The responder must include startTime (set by the controller) to prevent SSA
//	// from removing it, since the immutability validation would reject that.
//	now := metav1.Now()
//	current, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
//	if err != nil {
//		t.Fatalf("Failed to get EvictionRequest: %v", err)
//	}
//	var startTime metav1.Time
//	for _, rs := range current.Status.Responders {
//		if rs.Name == "first.example.com/handler" && rs.StartTime != nil {
//			startTime = *rs.StartTime
//			break
//		}
//	}
//	responderApply := coordinationapply.EvictionRequest(string(pod.UID), ns.Name).
//		WithStatus(coordinationapply.EvictionRequestStatus().
//			WithResponders(
//				coordinationapply.ResponderStatus().
//					WithName("first.example.com/handler").
//					WithStartTime(startTime).
//					WithHeartbeatTime(now).
//					WithCompletionTime(now).
//					WithMessage("eviction completed successfully"),
//			),
//		)
//	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).
//		ApplyStatus(tCtx, responderApply, metav1.ApplyOptions{
//			FieldManager: "first.example.com/handler",
//			Force:        true,
//		})
//	if err != nil {
//		t.Fatalf("Failed to apply responder status: %v", err)
//	}
//
//	// Wait for controller to advance to imperative responder
//	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return activeResponderName(er.Status.TargetResponders) == string(coordinationv1alpha1.EvictionResponderImperativeEviction)
//		},
//		"waiting for advancement to imperative responder",
//	)
//
//	// Verify the completed responder's status fields are preserved
//	firstStatus := findResponderStatus(final.Status.Responders, "first.example.com/handler")
//	if firstStatus == nil {
//		t.Fatal("first responder status not found after advancement")
//	}
//	if firstStatus.CompletionTime == nil {
//		t.Error("expected CompletionTime to be preserved")
//	}
//	if firstStatus.HeartbeatTime == nil {
//		t.Error("expected HeartbeatTime to be preserved")
//	}
//	if firstStatus.Message != "eviction completed successfully" {
//		t.Errorf("expected Message to be preserved, got %q", firstStatus.Message)
//	}
//
//	// Verify it was moved to Completed state
//	state := findTargetResponderState(final.Status.TargetResponders, "first.example.com/handler")
//	if state != coordinationv1alpha1.ResponderStateCompleted {
//		t.Errorf("expected first responder to be Completed, got %s", state)
//	}
//}
//
//// TestAllRespondersProcessed verifies that after all responders complete,
//// all are in terminal state with no active responder.
//func TestAllRespondersProcessed(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-all-processed", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	pod := newTestPod("test-pod")
//	pod.Spec.EvictionResponders = []v1.EvictionResponder{
//		{Name: "first.example.com/handler"},
//	}
//	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)
//
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// completeActiveResponder simulates the active responder completing.
//	completeActiveResponder := func(responderName string) {
//		t.Helper()
//		waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//			func(er *coordinationv1alpha1.EvictionRequest) bool {
//				return activeResponderName(er.Status.TargetResponders) == responderName
//			},
//			fmt.Sprintf("waiting for %s to become active", responderName),
//		)
//
//		now := metav1.Now()
//		current, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
//		if err != nil {
//			t.Fatalf("Failed to get EvictionRequest: %v", err)
//		}
//		for i := range current.Status.Responders {
//			if current.Status.Responders[i].Name == responderName {
//				current.Status.Responders[i].HeartbeatTime = &now
//				current.Status.Responders[i].CompletionTime = &now
//				current.Status.Responders[i].Message = "done"
//				break
//			}
//		}
//		_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).UpdateStatus(tCtx, current, metav1.UpdateOptions{})
//		if err != nil {
//			t.Fatalf("Failed to update EvictionRequest status: %v", err)
//		}
//	}
//
//	completeActiveResponder("first.example.com/handler")
//	completeActiveResponder(string(coordinationv1alpha1.EvictionResponderImperativeEviction))
//
//	// Wait for all responders to reach terminal state
//	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			completed := 0
//			for _, tr := range er.Status.TargetResponders {
//				if tr.State == coordinationv1alpha1.ResponderStateCompleted {
//					completed++
//				}
//			}
//			return completed == 2 && !hasActiveResponder(er.Status.TargetResponders)
//		},
//		"waiting for all responders to be processed",
//	)
//
//	expectedCompleted := map[string]bool{
//		"first.example.com/handler":                                      false,
//		string(coordinationv1alpha1.EvictionResponderImperativeEviction): false,
//	}
//	for _, tr := range final.Status.TargetResponders {
//		if tr.State == coordinationv1alpha1.ResponderStateCompleted {
//			expectedCompleted[tr.Name] = true
//		}
//	}
//	for name, found := range expectedCompleted {
//		if !found {
//			t.Errorf("expected %s to be Completed", name)
//		}
//	}
//}
//
//// TestRequestersWithdrawnDuringProcessing verifies that withdrawing all requesters
//// during active processing causes the EvictionRequest to be Failed.
//func TestRequestersWithdrawnDuringProcessing(t *testing.T) {
//	tCtx := ktesting.Init(t)
//	closeFn, c, inf, cs := setup(tCtx, t)
//	defer closeFn()
//
//	ns := framework.CreateNamespaceOrDie(cs, "evreq-requesters-removed", t)
//	defer framework.DeleteNamespaceOrDie(cs, ns, t)
//	defer tCtx.Cancel("test has completed")
//
//	inf.Start(tCtx.Done())
//	go c.Run(tCtx, 1)
//
//	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))
//
//	er := newTestEvictionRequest(pod)
//	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to create EvictionRequest: %v", err)
//	}
//
//	// Wait for controller to process (active responder set)
//	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
//		func(er *coordinationv1alpha1.EvictionRequest) bool {
//			return hasActiveResponder(er.Status.TargetResponders)
//		},
//		"waiting for active responder",
//	)
//
//	// Withdraw all requesters (cancellation signal)
//	current, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
//	if err != nil {
//		t.Fatalf("Failed to get EvictionRequest: %v", err)
//	}
//	for i := range current.Spec.Requesters {
//		current.Spec.Requesters[i].Intent = coordinationv1alpha1.RequesterIntentWithdrawn
//	}
//	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Update(tCtx, current, metav1.UpdateOptions{})
//	if err != nil {
//		t.Fatalf("Failed to update EvictionRequest: %v", err)
//	}
//
//	// Wait for Failed condition
//	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
//		string(coordinationv1alpha1.EvictionRequestConditionFailed), metav1.ConditionTrue)
//
//	cond := meta.FindStatusCondition(updated.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
//	if cond.Reason != string(coordinationv1alpha1.EvictionRequestConditionReasonCanceledDueToNoRequesters) {
//		t.Errorf("expected reason %s, got %s", string(coordinationv1alpha1.EvictionRequestConditionReasonCanceledDueToNoRequesters), cond.Reason)
//	}
//
//	if meta.IsStatusConditionTrue(updated.Status.Conditions, "Evicted") {
//		t.Error("should not have Evicted condition when requesters are withdrawn")
//	}
//}
