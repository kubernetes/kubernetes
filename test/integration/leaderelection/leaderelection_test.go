/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// testRoundTripper can be used to fail requests on demand
type testRoundTripper struct {
	shouldFail atomic.Bool
	rt         http.RoundTripper
}

func (t *testRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	if t.shouldFail.Load() {
		return nil, fmt.Errorf("failure injection")
	} else {
		return t.rt.RoundTrip(r)
	}
}

func TestLeaderElection(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	// make transport cancellable
	config := rest.CopyConfig(server.ClientConfig)
	testRT := &testRoundTripper{}
	config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		testRT.rt = rt
		return testRT
	})

	testClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	for _, testcase := range []string{"cancel-context", "error-injection"} {
		t.Run("test-"+testcase, func(t *testing.T) {
			namespaceName := "ns-" + testcase

			ns := framework.CreateNamespaceOrDie(client, namespaceName, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// we use the Lease lock type since edits to Leases are less common
			// and fewer objects in the cluster watch "all Leases".
			lock := &resourcelock.LeaseLock{
				LeaseMeta: metav1.ObjectMeta{
					Name:      "test-lease",
					Namespace: namespaceName,
				},
				Client: testClient.CoordinationV1(),
				LockConfig: resourcelock.ResourceLockConfig{
					Identity: "fake-id",
				},
			}

			startLeadingCh := make(chan struct{})
			exitLeadingCh := make(chan struct{})
			stopLeadingCh := make(chan struct{})

			run := func(ctx context.Context) {
				close(startLeadingCh)
				<-ctx.Done()
				close(exitLeadingCh)
			}

			tCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// start the leader election code loop
			go func() {
				leaderelection.RunOrDie(tCtx, leaderelection.LeaderElectionConfig{
					Lock:            lock,
					ReleaseOnCancel: false,
					LeaseDuration:   5 * time.Second,
					RenewDeadline:   2 * time.Second,
					RetryPeriod:     1 * time.Second,
					Callbacks: leaderelection.LeaderCallbacks{
						OnStartedLeading: func(ctx context.Context) {
							t.Log("leader acquired")
							run(ctx)
						},
						OnStoppedLeading: func() {
							// we can do cleanup here
							t.Log("leader lost")
							close(stopLeadingCh)
						},
					},
				})
			}()

			select {
			case <-startLeadingCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("not able to acquire the leader")
			}

			switch testcase {
			case "cancel-context":
				// cancel the context to stop leading
				cancel()
			case "error-injection":
				// fail to renew the lease
				testRT.shouldFail.Store(true)
			default:
				t.Fatalf("unknown testcase %s", testcase)
			}

			select {
			case <-exitLeadingCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("not closing the leading running function on stop leading")
			}

			select {
			case <-stopLeadingCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("not executing stop leading callback")
			}
		})
	}
}
