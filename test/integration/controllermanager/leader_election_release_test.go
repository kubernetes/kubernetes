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

package controllermanager

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanagertesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/kubeconfig"
)

// TestLeaderElectionReleaseOnCancel verifies that when the
// ControllerManagerReleaseLeaderElectionLockOnExit feature gate is enabled,
// tearing down the KCM (cancelling its context) causes the leader election
// lease to be released (holder identity cleared). This allows a new leader
// to acquire the lock without waiting for the full lease duration to expire.
//
// Note: We only test the feature-enabled case here. The feature-disabled case
// retains the legacy behavior where OnStoppedLeading calls klog.FlushAndExit
// (os.Exit), which cannot be tested in-process.
func TestLeaderElectionReleaseOnCancel(t *testing.T) {
	// Start a real API server.
	apiServer := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(apiServer.TearDownFn)

	client, err := kubernetes.NewForConfig(apiServer.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Write a kubeconfig file for the KCM to use.
	clientConfig := kubeconfig.CreateKubeConfig(apiServer.ClientConfig)
	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig.yaml")
	if err := clientcmd.WriteToFile(*clientConfig, kubeConfigFile); err != nil {
		t.Fatalf("failed to write kubeconfig: %v", err)
	}

	// Start a full KCM with leader election and the feature gate enabled.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	kcm, err := kubecontrollermanagertesting.StartTestServer(t, ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--leader-elect=true",
		"--feature-gates=ControllerManagerReleaseLeaderElectionLockOnExit=true",
	})
	if err != nil {
		t.Fatalf("failed to start KCM: %v", err)
	}
	t.Cleanup(kcm.TearDownFn)

	// Wait for the lease to exist and have a holder. There can be a brief gap
	// between the KCM's /healthz becoming ready and the lease being acquired.
	var lease *coordinationv1.Lease
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, err = client.CoordinationV1().Leases("kube-system").Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if err != nil {
			return false, nil // retry on not-found or transient errors
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != "", nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for lease to have a holder: %v", err)
	}
	t.Logf("lease holder before shutdown: %q", *lease.Spec.HolderIdentity)

	// Tear down the KCM (cancels its context), which should release the lease.
	kcm.TearDownFn()

	// Poll until the lease holder is cleared.
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, err = client.CoordinationV1().Leases("kube-system").Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity == "", nil
	})
	if err != nil {
		holder := "<nil>"
		if lease.Spec.HolderIdentity != nil {
			holder = *lease.Spec.HolderIdentity
		}
		t.Fatalf("expected lease holder to be cleared after shutdown, but got %q: %v", holder, err)
	}
	t.Log("lease holder cleared after shutdown")
}
