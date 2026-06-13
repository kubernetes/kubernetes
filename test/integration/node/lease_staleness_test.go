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

package node

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils"
)

func TestLeaseCircuitBreaker(t *testing.T) {
	blockLeaseWatch := make(chan struct{})
	defer close(blockLeaseWatch)

	nodeName := "target-node"
	var directGetHits int32

	// Standard Integration API server initialization
	testCtx := testutils.InitTestAPIServer(t, "lease-test", nil)

	// Configure the external clientset used by the Informer Factory to block target lease watch events
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return utils.NewInterceptingTransport(rt, []utils.InterceptionRule{
			{
				Method:    "GET",
				Group:     "coordination.k8s.io",
				Resource:  "leases",
				Namespace: v1.NamespaceNodeLease,
				Name:      nodeName,
				IsWatch:   true,
				Hook: func(req *http.Request, eventBytes []byte) {
					// Suspend watch processing by blocking here
					select {
					case <-testCtx.Ctx.Done():
					case <-blockLeaseWatch:
					}
				},
			},
		})
	})
	externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)

	// Configure the controller's internal clientset to count direct GET fallback calls
	controllerClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	controllerClientConfig.QPS = -1
	controllerClientConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return utils.NewInterceptingTransport(rt, []utils.InterceptionRule{
			{
				Method:    "GET",
				Group:     "coordination.k8s.io",
				Resource:  "leases",
				Namespace: v1.NamespaceNodeLease,
				Name:      nodeName,
				IsWatch:   false,
				Hook: func(req *http.Request, eventBytes []byte) {
					atomic.AddInt32(&directGetHits, 1)
				},
			},
		})
	})
	controllerClientset := clientset.NewForConfigOrDie(controllerClientConfig)

	// Use strict timing: 2s stale duration with rapid checks
	nodeGrace := 2 * time.Second
	monitorPeriod := 100 * time.Millisecond

	nc, err := nodelifecycle.NewNodeLifecycleController(
		testCtx.Ctx,
		externalInformers.Coordination().V1().Leases(),
		externalInformers.Core().V1().Pods(),
		externalInformers.Core().V1().Nodes(),
		externalInformers.Apps().V1().DaemonSets(),
		controllerClientset,
		nodeGrace,
		nodeGrace, // same startup grace period
		monitorPeriod,
		100, 100, 50, 0.55,
	)
	if err != nil {
		t.Fatalf("Failed to create node controller: %v", err)
	}

	// Start informers and sync caches normally (Initial Lists proceed successfully)
	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

	// Run controller loop asynchronously
	go nc.Run(testCtx.Ctx)

	// Create the targeted Node object
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeName,
			Labels: map[string]string{"node.kubernetes.io/exclude-disruption": "true"},
		},
		Spec: v1.NodeSpec{},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}
	_, err = testCtx.ClientSet.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to initialize node: %v", err)
	}

	// Ensure lease namespace exists (integration API env is minimal)
	nsObj := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: v1.NamespaceNodeLease}}
	_, err = testCtx.ClientSet.CoreV1().Namespaces().Create(testCtx.Ctx, nsObj, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("Failed to ensure namespace %q: %v", v1.NamespaceNodeLease, err)
	}

	// Pre-create the initial Lease object via Direct API Write!
	// This will trigger a WATCH event that gets BLOCKED at the HTTP transport layer,
	// so it stays in the API server but NEVER reaches the Informer cache!
	duration := int32(40)
	testLease := &coordv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodeName,
			Namespace: v1.NamespaceNodeLease,
		},
		Spec: coordv1.LeaseSpec{
			HolderIdentity:       &nodeName,
			LeaseDurationSeconds: &duration,
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}
	_, err = testCtx.ClientSet.CoordinationV1().Leases(v1.NamespaceNodeLease).Create(testCtx.Ctx, testLease, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to initialize lease object directly: %v", err)
	}

	// The controller initialized at creation T=0. Wait for 2s grace window to elapse.
	// Once expired, and with our client-level blocking of WATCH updates, the node controller
	// SHOULD execute the direct GET fallback block.
	t.Log("Waiting for NodeGrace period to elapse and trigger the first direct GET fallback.")

	err = wait.PollUntilContextTimeout(testCtx.Ctx, 500*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		count := atomic.LoadInt32(&directGetHits)
		if count > 0 {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		t.Fatalf("FAIL: Node lifecycle controller did not trigger direct GET fallback code path within timeout.")
	}

	// Once the first direct GET fallback has triggered, wait for an additional 2 seconds
	// (approx 20 monitor cycles) to verify that subsequent reconciliation cycles do NOT
	// trigger more direct GET lookups, thanks to the ConsistencyStore caching protection.
	t.Log("First fallback triggered. Waiting for 2 seconds to allow subsequent reconciliation cycles to run.")
	time.Sleep(2 * time.Second)

	finalHits := atomic.LoadInt32(&directGetHits)
	t.Logf("Validation complete. Total direct GET hits: %d", finalHits)

	if finalHits != 1 {
		t.Errorf("FAIL: Expected exactly 1 direct GET hit (protected by ConsistencyStore), but got %d.", finalHits)
	} else {
		t.Log("SUCCESS: Dependency Injection wrapper successfully validated that the ConsistencyStore protects the apiserver from redundant direct GET lookups.")
	}
}
