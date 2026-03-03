/*
Copyright 2026 The Kubernetes Authors.

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

import (
	"context"
	"fmt"
	"testing"
	"time"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/evictionrequest"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

func setup(ctx context.Context, t *testing.T, extraServerFlags ...string) (
	kubeapiservertesting.TearDownFunc,
	*evictionrequest.EvictionRequestController,
	informers.SharedInformerFactory,
	clientset.Interface,
) {
	t.Helper()

	// Enable the EvictionRequestAPI feature gate (alpha) and the coordination.k8s.io/v1alpha1 API group.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)

	serverFlags := append(framework.DefaultTestServerFlags(),
		"--feature-gates=EvictionRequestAPI=true",
		"--runtime-config=coordination.k8s.io/v1alpha1=true",
	)
	serverFlags = append(serverFlags, extraServerFlags...)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		serverFlags,
		framework.SharedEtcd(),
	)

	config := restclient.CopyConfig(server.ClientConfig)
	cs, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	inf := informers.NewSharedInformerFactory(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "evictionrequest-informers")),
		resyncPeriod,
	)

	controllerClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "evictionrequest-controller"))

	c, err := evictionrequest.NewController(
		ctx,
		inf.Coordination().V1alpha1().EvictionRequests(),
		inf.Core().V1().Pods(),
		controllerClient,
		"evictionrequest-controller",
	)
	if err != nil {
		t.Fatalf("Error creating eviction request controller: %v", err)
	}

	return server.TearDownFn, c, inf, cs
}

// newTestPod creates a running pod for testing.
func newTestPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"app": "test"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "test", Image: "registry.k8s.io/pause:3.9"},
			},
		},
	}
}

// newTestEvictionRequest creates an EvictionRequest targeting the given pod.
func newTestEvictionRequest(pod *v1.Pod) *coordinationv1alpha1.EvictionRequest {
	return &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      string(pod.UID),
			Namespace: pod.Namespace,
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target: coordinationv1alpha1.EvictionTarget{
				Pod: &coordinationv1alpha1.LocalTargetReference{
					Name: pod.Name,
					UID:  string(pod.UID),
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}
}

// waitForEvictionRequestCondition polls until the EvictionRequest has the expected condition or times out.
func waitForEvictionRequestCondition(
	ctx context.Context,
	t *testing.T,
	cs clientset.Interface,
	namespace, name string,
	conditionType string,
	expectedStatus metav1.ConditionStatus,
) *coordinationv1alpha1.EvictionRequest {
	t.Helper()
	var er *coordinationv1alpha1.EvictionRequest
	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		var err error
		er, err = cs.CoordinationV1alpha1().EvictionRequests(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		cond := meta.FindStatusCondition(er.Status.Conditions, conditionType)
		if cond == nil {
			return false, nil
		}
		return cond.Status == expectedStatus, nil
	}); err != nil {
		t.Fatalf("timed out waiting for EvictionRequest %s/%s to have condition %s=%s: %v",
			namespace, name, conditionType, expectedStatus, err)
	}
	return er
}

// waitForEvictionRequestStatus polls until the status check function returns true.
func waitForEvictionRequestStatus(
	ctx context.Context,
	t *testing.T,
	cs clientset.Interface,
	namespace, name string,
	check func(*coordinationv1alpha1.EvictionRequest) bool,
	description string,
) *coordinationv1alpha1.EvictionRequest {
	t.Helper()
	var er *coordinationv1alpha1.EvictionRequest
	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		var err error
		er, err = cs.CoordinationV1alpha1().EvictionRequests(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return check(er), nil
	}); err != nil {
		t.Fatalf("timed out waiting for EvictionRequest %s/%s: %s: %v",
			namespace, name, description, err)
	}
	return er
}

// createPodAndWait creates a pod and waits for it to be persisted, returning the pod with its UID.
func createPodAndWait(ctx context.Context, t *testing.T, cs clientset.Interface, namespace string, pod *v1.Pod) *v1.Pod {
	t.Helper()
	created, err := cs.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}
	return created
}

// TestValidation_PodNotFound verifies that an EvictionRequest targeting a non-existent pod
// gets a Canceled condition.
func TestValidation_PodNotFound(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-not-found", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create an EvictionRequest targeting a pod that doesn't exist.
	// EvictionRequest names and pod UIDs must be RFC 4122 UUIDs.
	nonexistentUID := "00000000-0000-0000-0000-000000000001"
	er := &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nonexistentUID,
			Namespace: ns.Name,
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target: coordinationv1alpha1.EvictionTarget{
				Pod: &coordinationv1alpha1.LocalTargetReference{
					Name: "nonexistent-pod",
					UID:  nonexistentUID,
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}

	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for Canceled condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, nonexistentUID, "Canceled", metav1.ConditionTrue)
	cond := meta.FindStatusCondition(updated.Status.Conditions, "Canceled")
	if cond.Reason != evictionrequest.ValidationFailedReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.ValidationFailedReason, cond.Reason)
	}
}

// TestValidation_UIDMismatch verifies that an EvictionRequest with the wrong pod UID
// gets a Canceled condition.
func TestValidation_UIDMismatch(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-uid-mismatch", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create a pod
	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	// Create EvictionRequest with wrong UID.
	// EvictionRequest names and pod UIDs must be RFC 4122 UUIDs.
	wrongUID := "00000000-0000-0000-0000-000000000002"
	er := &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      wrongUID,
			Namespace: ns.Name,
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target: coordinationv1alpha1.EvictionTarget{
				Pod: &coordinationv1alpha1.LocalTargetReference{
					Name: pod.Name,
					UID:  wrongUID,
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}

	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for Canceled condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, wrongUID, "Canceled", metav1.ConditionTrue)
	cond := meta.FindStatusCondition(updated.Status.Conditions, "Canceled")
	if cond.Reason != evictionrequest.ValidationFailedReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.ValidationFailedReason, cond.Reason)
	}
}

// TestInitializeTargetInterceptors verifies that the controller initializes target interceptors
// from the pod's EvictionInterceptors list plus the default imperative interceptor, and that
// matching InterceptorStatus entries are created in the same order.
func TestInitializeTargetInterceptors(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-interceptors", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod with two custom eviction interceptors
	pod := newTestPod("test-pod")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: "first.example.com"},
		{Name: "second.example.com"},
	}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	expectedNames := []string{"first.example.com", "second.example.com", evictionrequest.ImperativeEvictionInterceptor}

	// Wait for target interceptors and interceptor statuses to be initialized
	updated := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.TargetInterceptors) == 3 && len(er.Status.Interceptors) == 3
		},
		"waiting for 3 target interceptors and 3 interceptor statuses",
	)

	// Verify target interceptors order
	for i, expected := range expectedNames {
		if updated.Status.TargetInterceptors[i].Name != expected {
			t.Errorf("targetInterceptors[%d]: expected %q, got %q", i, expected, updated.Status.TargetInterceptors[i].Name)
		}
	}

	// Verify interceptor statuses match in the same order
	for i, expected := range expectedNames {
		if updated.Status.Interceptors[i].Name != expected {
			t.Errorf("interceptors[%d]: expected name %q, got %q", i, expected, updated.Status.Interceptors[i].Name)
		}
	}
}

// TestSelectFirstInterceptor verifies that the controller selects the first interceptor,
// initializes its StartTime, and that a pod with no custom interceptors gets only the
// imperative interceptor as its target.
func TestSelectFirstInterceptor(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-select-first", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod (no custom interceptors, so only imperative will be used)
	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for active interceptor to be set
	updated := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.ActiveInterceptors) == 1
		},
		"waiting for active interceptor",
	)

	// Verify only the imperative interceptor is present as target
	if len(updated.Status.TargetInterceptors) != 1 {
		t.Fatalf("expected 1 target interceptor, got %d", len(updated.Status.TargetInterceptors))
	}
	if updated.Status.TargetInterceptors[0].Name != evictionrequest.ImperativeEvictionInterceptor {
		t.Errorf("expected only imperative target interceptor, got %s", updated.Status.TargetInterceptors[0].Name)
	}

	if updated.Status.ActiveInterceptors[0] != evictionrequest.ImperativeEvictionInterceptor {
		t.Errorf("expected active interceptor to be imperative, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify start time was initialized (not heartbeat — controller sets StartTime,
	// interceptor is responsible for setting HeartbeatTime)
	if len(updated.Status.Interceptors) == 0 {
		t.Fatal("expected interceptor status to be initialized")
	}
	if updated.Status.Interceptors[0].StartTime == nil {
		t.Error("expected start time to be initialized for active interceptor")
	}
	if updated.Status.Interceptors[0].HeartbeatTime != nil {
		t.Error("expected heartbeat time to not be set by controller")
	}
}

// TestPodDeleted_Evicted verifies that when a target pod is deleted, the EvictionRequest
// gets the Evicted condition.
func TestPodDeleted_Evicted(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-deleted", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod
	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for controller to process the EvictionRequest (active interceptor set)
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return er.Status.ObservedGeneration > 0
		},
		"waiting for initial processing",
	)

	// Delete the pod
	err = cs.CoreV1().Pods(ns.Name).Delete(tCtx, pod.Name, metav1.DeleteOptions{
		GracePeriodSeconds: new(int64), // immediate deletion
	})
	if err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}

	// Wait for Evicted condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
		"Evicted", metav1.ConditionTrue)

	cond := meta.FindStatusCondition(updated.Status.Conditions, "Evicted")
	if cond.Reason != evictionrequest.TargetDeletedReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.TargetDeletedReason, cond.Reason)
	}

	// Should NOT have Canceled condition
	if meta.IsStatusConditionTrue(updated.Status.Conditions, "Canceled") {
		t.Error("should not have Canceled condition when pod is deleted after validation")
	}
}

// TestEmptyRequesters_RejectedByAPI verifies that an EvictionRequest with no requesters
// is rejected at creation time by the API validation.
func TestEmptyRequesters_RejectedByAPI(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-no-requesters", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod
	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	// Create EvictionRequest with no requesters
	er := newTestEvictionRequest(pod)
	er.Spec.Requesters = nil

	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("expected creation to fail with no requesters")
	}
	if !apierrors.IsInvalid(err) {
		t.Errorf("expected Invalid error, got: %v", err)
	}
}

// TestLabelSync verifies that pod labels are synced to the EvictionRequest.
func TestLabelSync(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-label-sync", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod with labels
	pod := newTestPod("test-pod")
	pod.Labels = map[string]string{
		"app":     "myapp",
		"version": "v1",
	}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for labels to be synced
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return er.Labels != nil &&
				er.Labels["app"] == "myapp" &&
				er.Labels["version"] == "v1"
		},
		"waiting for pod labels to sync to EvictionRequest",
	)
}

// TestLabelSyncUpdate verifies that label changes on the pod are propagated to the EvictionRequest.
func TestLabelSyncUpdate(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-label-update", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod
	pod := newTestPod("test-pod")
	pod.Labels = map[string]string{"app": "v1"}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for initial label sync
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return er.Labels != nil && er.Labels["app"] == "v1"
		},
		"waiting for initial label sync",
	)

	// Update pod labels
	pod, err = cs.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod: %v", err)
	}
	pod.Labels["app"] = "v2"
	_, err = cs.CoreV1().Pods(ns.Name).Update(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod labels: %v", err)
	}

	// Wait for updated labels to sync
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return er.Labels != nil && er.Labels["app"] == "v2"
		},
		"waiting for updated label sync",
	)
}

// TestMultipleEvictionRequests verifies that the controller can handle multiple
// EvictionRequests concurrently.
func TestMultipleEvictionRequests(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-multiple", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 2) // Run with 2 workers

	numPods := 5
	pods := make([]*v1.Pod, numPods)
	for i := range numPods {
		pods[i] = createPodAndWait(tCtx, t, cs, ns.Name, newTestPod(fmt.Sprintf("test-pod-%d", i)))
	}

	// Create EvictionRequests for all pods
	for _, pod := range pods {
		er := newTestEvictionRequest(pod)
		_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create EvictionRequest for %s: %v", pod.Name, err)
		}
	}

	// Wait for all EvictionRequests to be processed (active interceptor set)
	for _, pod := range pods {
		waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
			func(er *coordinationv1alpha1.EvictionRequest) bool {
				return len(er.Status.ActiveInterceptors) == 1
			},
			fmt.Sprintf("waiting for EvictionRequest for %s to have active interceptor", pod.Name),
		)
	}
}

// TestValidation_WorkloadRef verifies that pods with WorkloadRef are rejected.
func TestValidation_WorkloadRef(t *testing.T) {
	tCtx := ktesting.Init(t)

	// WorkloadRef requires the GenericWorkload feature gate
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	closeFn, c, inf, cs := setup(tCtx, t, "--feature-gates=EvictionRequestAPI=true,GenericWorkload=true")
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-workloadref", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create pod with WorkloadRef
	pod := newTestPod("test-pod")
	pod.Spec.WorkloadRef = &v1.WorkloadReference{Name: "my-workload", PodGroup: "my-group"}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	// Create EvictionRequest
	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for Canceled condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
		"Canceled", metav1.ConditionTrue)

	cond := meta.FindStatusCondition(updated.Status.Conditions, "Canceled")
	if cond.Reason != evictionrequest.ValidationFailedReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.ValidationFailedReason, cond.Reason)
	}
}

// TestTerminalStateIdempotent verifies that re-syncing an already-terminal
// EvictionRequest does not re-process or overwrite existing conditions.
func TestTerminalStateIdempotent(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-terminal-idempotent", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	// Create an EvictionRequest targeting a nonexistent pod to get a Canceled condition
	nonexistentUID := "00000000-0000-0000-0000-000000000099"
	er := &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nonexistentUID,
			Namespace: ns.Name,
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target: coordinationv1alpha1.EvictionTarget{
				Pod: &coordinationv1alpha1.LocalTargetReference{
					Name: "nonexistent-pod",
					UID:  nonexistentUID,
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}

	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for Canceled condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, nonexistentUID, "Canceled", metav1.ConditionTrue)
	observedGenBefore := updated.Status.ObservedGeneration

	// Trigger a re-sync by touching a label on the EvictionRequest
	updated.Labels = map[string]string{"trigger": "resync"}
	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Update(tCtx, updated, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update EvictionRequest: %v", err)
	}

	// Wait for the label update to be visible, confirming the controller had a
	// chance to process the re-sync triggered by the EvictionRequest update event.
	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, nonexistentUID,
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return er.Labels != nil && er.Labels["trigger"] == "resync"
		},
		"waiting for label update to be visible",
	)

	cond := meta.FindStatusCondition(final.Status.Conditions, "Canceled")
	if cond == nil || cond.Status != metav1.ConditionTrue {
		t.Error("expected Canceled condition to remain True after re-sync")
	}
	if cond.Reason != evictionrequest.ValidationFailedReason {
		t.Errorf("expected reason to remain %s, got %s", evictionrequest.ValidationFailedReason, cond.Reason)
	}

	// ObservedGeneration should not have been bumped by re-processing
	if final.Status.ObservedGeneration != observedGenBefore {
		t.Errorf("expected ObservedGeneration to remain %d after re-sync of terminal state, got %d",
			observedGenBefore, final.Status.ObservedGeneration)
	}
}

// TestPodTerminal_Evicted verifies that when a target pod is already in a terminal phase
// (Succeeded or Failed), the EvictionRequest gets the Evicted condition with TargetTerminal reason.
func TestPodTerminal_Evicted(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-pod-terminal", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	// Mark pod as terminal (Succeeded) before creating the EvictionRequest
	pod, err := cs.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod: %v", err)
	}
	pod.Status.Phase = v1.PodSucceeded
	pod, err = cs.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod status: %v", err)
	}

	// Create EvictionRequest targeting the terminal pod
	er := newTestEvictionRequest(pod)
	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for Evicted condition with TargetTerminal reason
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
		"Evicted", metav1.ConditionTrue)

	cond := meta.FindStatusCondition(updated.Status.Conditions, "Evicted")
	if cond.Reason != evictionrequest.TargetTerminalReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.TargetTerminalReason, cond.Reason)
	}

	if meta.IsStatusConditionTrue(updated.Status.Conditions, "Canceled") {
		t.Error("should not have Canceled condition when pod is terminal")
	}
}

// TestInterceptorStatusPreservedOnAdvancement verifies that when the controller advances
// to the next interceptor, the completed interceptor's status fields are preserved.
func TestInterceptorStatusPreservedOnAdvancement(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-status-preserved", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	pod := newTestPod("test-pod")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: "first.example.com"},
	}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for first interceptor to become active
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.ActiveInterceptors) == 1 &&
				er.Status.ActiveInterceptors[0] == "first.example.com"
		},
		"waiting for first interceptor to become active",
	)

	// Simulate interceptor completion
	now := metav1.Now()
	updated, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get EvictionRequest: %v", err)
	}
	for i := range updated.Status.Interceptors {
		if updated.Status.Interceptors[i].Name == "first.example.com" {
			updated.Status.Interceptors[i].HeartbeatTime = &now
			updated.Status.Interceptors[i].CompletionTime = &now
			updated.Status.Interceptors[i].Message = "eviction completed successfully"
			break
		}
	}
	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).UpdateStatus(tCtx, updated, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update EvictionRequest status: %v", err)
	}

	// Wait for controller to advance to imperative interceptor
	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.ActiveInterceptors) == 1 &&
				er.Status.ActiveInterceptors[0] == evictionrequest.ImperativeEvictionInterceptor
		},
		"waiting for advancement to imperative interceptor",
	)

	// Verify the completed interceptor's status fields are preserved
	var firstStatus *coordinationv1alpha1.InterceptorStatus
	for i := range final.Status.Interceptors {
		if final.Status.Interceptors[i].Name == "first.example.com" {
			firstStatus = &final.Status.Interceptors[i]
			break
		}
	}
	if firstStatus == nil {
		t.Fatal("first interceptor status not found after advancement")
	}
	if firstStatus.CompletionTime == nil {
		t.Error("expected CompletionTime to be preserved")
	}
	if firstStatus.HeartbeatTime == nil {
		t.Error("expected HeartbeatTime to be preserved")
	}
	if firstStatus.Message != "eviction completed successfully" {
		t.Errorf("expected Message to be preserved, got %q", firstStatus.Message)
	}

	// Verify it was moved to processed
	found := false
	for _, name := range final.Status.ProcessedInterceptors {
		if name == "first.example.com" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected first interceptor to be in processedInterceptors")
	}
}

// TestAllInterceptorsProcessed verifies that after all interceptors complete,
// activeInterceptors is cleared and processedInterceptors contains all entries.
func TestAllInterceptorsProcessed(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-all-processed", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	pod := newTestPod("test-pod")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: "first.example.com"},
	}
	pod = createPodAndWait(tCtx, t, cs, ns.Name, pod)

	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// completeActiveInterceptor simulates the active interceptor completing.
	completeActiveInterceptor := func(interceptorName string) {
		t.Helper()
		waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
			func(er *coordinationv1alpha1.EvictionRequest) bool {
				return len(er.Status.ActiveInterceptors) == 1 &&
					er.Status.ActiveInterceptors[0] == interceptorName
			},
			fmt.Sprintf("waiting for %s to become active", interceptorName),
		)

		now := metav1.Now()
		current, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get EvictionRequest: %v", err)
		}
		for i := range current.Status.Interceptors {
			if current.Status.Interceptors[i].Name == interceptorName {
				current.Status.Interceptors[i].HeartbeatTime = &now
				current.Status.Interceptors[i].CompletionTime = &now
				current.Status.Interceptors[i].Message = "done"
				break
			}
		}
		_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).UpdateStatus(tCtx, current, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update EvictionRequest status: %v", err)
		}
	}

	completeActiveInterceptor("first.example.com")
	completeActiveInterceptor(evictionrequest.ImperativeEvictionInterceptor)

	// Wait for all interceptors to be processed
	final := waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.ProcessedInterceptors) == 2 &&
				len(er.Status.ActiveInterceptors) == 0
		},
		"waiting for all interceptors to be processed",
	)

	expectedProcessed := map[string]bool{
		"first.example.com":                          false,
		evictionrequest.ImperativeEvictionInterceptor: false,
	}
	for _, name := range final.Status.ProcessedInterceptors {
		expectedProcessed[name] = true
	}
	for name, found := range expectedProcessed {
		if !found {
			t.Errorf("expected %s in processedInterceptors", name)
		}
	}
}

// TestRequestersRemovedDuringProcessing verifies that removing all requesters
// from the spec during active processing causes the EvictionRequest to be Canceled.
func TestRequestersRemovedDuringProcessing(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, c, inf, cs := setup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(cs, "evreq-requesters-removed", t)
	defer framework.DeleteNamespaceOrDie(cs, ns, t)
	defer tCtx.Cancel("test has completed")

	inf.Start(tCtx.Done())
	go c.Run(tCtx, 1)

	pod := createPodAndWait(tCtx, t, cs, ns.Name, newTestPod("test-pod"))

	er := newTestEvictionRequest(pod)
	_, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Create(tCtx, er, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create EvictionRequest: %v", err)
	}

	// Wait for controller to process (active interceptor set)
	waitForEvictionRequestStatus(tCtx, t, cs, ns.Name, string(pod.UID),
		func(er *coordinationv1alpha1.EvictionRequest) bool {
			return len(er.Status.ActiveInterceptors) == 1
		},
		"waiting for active interceptor",
	)

	// Remove all requesters (cancellation signal)
	current, err := cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Get(tCtx, string(pod.UID), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get EvictionRequest: %v", err)
	}
	current.Spec.Requesters = nil
	_, err = cs.CoordinationV1alpha1().EvictionRequests(ns.Name).Update(tCtx, current, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update EvictionRequest: %v", err)
	}

	// Wait for Canceled condition
	updated := waitForEvictionRequestCondition(tCtx, t, cs, ns.Name, string(pod.UID),
		"Canceled", metav1.ConditionTrue)

	cond := meta.FindStatusCondition(updated.Status.Conditions, "Canceled")
	if cond.Reason != evictionrequest.NoRequestersReason {
		t.Errorf("expected reason %s, got %s", evictionrequest.NoRequestersReason, cond.Reason)
	}

	if meta.IsStatusConditionTrue(updated.Status.Conditions, "Evicted") {
		t.Error("should not have Evicted condition when requesters are removed")
	}
}
