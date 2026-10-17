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
	"errors"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/util/keyutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanagertesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/kubeconfig"
)

// TestMain runs the goroutine leak check for this package, like every other
// integration package that embeds a kube-apiserver. The pre-existing test in
// this package was never covered by it.
func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
}

// TestLeaderElectionRecoveryWriteGate verifies that a gated client's write is
// rejected with ErrWriteGateClosed while the elector is inactive, and succeeds
// after the gate reopens.
func TestLeaderElectionRecoveryWriteGate(t *testing.T) {
	apiServer := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(apiServer.TearDownFn)

	var failLeaseRequests atomic.Bool
	lockConfig := restclient.CopyConfig(apiServer.ClientConfig)
	lockConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if failLeaseRequests.Load() && strings.Contains(req.URL.Path, "/leases") {
				return &http.Response{
					StatusCode: http.StatusServiceUnavailable,
					Header:     make(http.Header),
					Body:       http.NoBody,
					Request:    req,
				}, nil
			}
			return rt.RoundTrip(req)
		})
	})

	rl, err := resourcelock.NewFromKubeconfig(
		resourcelock.LeasesResourceLock,
		metav1.NamespaceSystem,
		"recovery-writegate-test",
		resourcelock.ResourceLockConfig{
			Identity: "holder-1",
		},
		lockConfig,
		500*time.Millisecond,
	)
	if err != nil {
		t.Fatalf("failed to create resource lock: %v", err)
	}

	startedCh := make(chan struct{})
	stoppedCh := make(chan struct{}, 4)

	le, err := leaderelection.NewLeaderElector(leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: 2 * time.Second,
		RenewDeadline: 600 * time.Millisecond,
		RetryPeriod:   100 * time.Millisecond,
		Name:          "recovery-writegate-test",
		Recovery:      &leaderelection.RecoveryConfig{},
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				close(startedCh)
			},
			OnStoppedLeading: func() {
				select {
				case stoppedCh <- struct{}{}:
				default:
				}
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create LeaderElector: %v", err)
	}

	gatedConfig := restclient.CopyConfig(apiServer.ClientConfig)
	gatedConfig.Wrap(le.WriteGate())

	gatedClient, err := kubernetes.NewForConfig(gatedConfig)
	if err != nil {
		t.Fatalf("failed to create gated client: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Before starting election, the write gate is closed: reads pass, writes fail with ErrWriteGateClosed.
	if _, err := gatedClient.CoreV1().Namespaces().Get(ctx, metav1.NamespaceSystem, metav1.GetOptions{}); err != nil {
		t.Fatalf("expected GET through closed write gate to succeed, got: %v", err)
	}
	_, err = gatedClient.CoreV1().ConfigMaps(metav1.NamespaceDefault).Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "rejected-before-leading"},
	}, metav1.CreateOptions{})
	if !errors.Is(err, leaderelection.ErrWriteGateClosed) {
		t.Fatalf("expected ErrWriteGateClosed before leading, got: %v", err)
	}

	go le.Run(ctx)

	select {
	case <-startedCh:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for OnStartedLeading")
	}

	// Simulate lease renewal outage so the renew deadline expires and the write gate closes.
	// Drop any loss recorded by a spurious renew failure before this point so
	// that the next signal is the one caused by the failing lease requests.
	for len(stoppedCh) > 0 {
		<-stoppedCh
	}
	failLeaseRequests.Store(true)
	select {
	case <-stoppedCh:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for OnStoppedLeading after lease renewal failure")
	}

	// Direct write on the gated client must be rejected with ErrWriteGateClosed while inactive.
	_, err = gatedClient.CoreV1().ConfigMaps(metav1.NamespaceDefault).Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "rejected-while-inactive"},
	}, metav1.CreateOptions{})
	if !errors.Is(err, leaderelection.ErrWriteGateClosed) {
		t.Fatalf("expected ErrWriteGateClosed while inactive, got: %v", err)
	}

	// Restore lease renewals so the elector recovers the lease and reopens the write gate.
	failLeaseRequests.Store(false)

	// Gated client write succeeds once the gate reopens.
	err = wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		_, createErr := gatedClient.CoreV1().ConfigMaps(metav1.NamespaceDefault).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "succeeded-after-recovery"},
		}, metav1.CreateOptions{})
		if errors.Is(createErr, leaderelection.ErrWriteGateClosed) {
			return false, nil
		}
		return createErr == nil || apierrors.IsAlreadyExists(createErr), createErr
	})
	if err != nil {
		t.Fatalf("expected gated client write to succeed after gate reopened: %v", err)
	}
}

// TestKubeControllerManagerLeaderElectionRecovery verifies that kube-controller-manager
// continues to run across a lease renewal interruption, blocks writes while its
// write gate is closed, and recovers leadership without restarting once the
// interruption ends.
func TestKubeControllerManagerLeaderElectionRecovery(t *testing.T) {
	apiServer := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(apiServer.TearDownFn)

	client, err := kubernetes.NewForConfig(apiServer.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	targetURL, err := url.Parse(apiServer.ClientConfig.Host)
	if err != nil {
		t.Fatalf("failed to parse apiserver URL: %v", err)
	}
	rt, err := restclient.TransportFor(apiServer.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create apiserver transport: %v", err)
	}

	var failLeaseRenewals atomic.Bool
	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	proxy.Transport = rt
	proxy.FlushInterval = -1

	proxyServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if failLeaseRenewals.Load() && strings.Contains(r.URL.Path, "/apis/coordination.k8s.io/v1/namespaces/kube-system/leases/kube-controller-manager") {
			http.Error(w, "simulated apiserver lease unavailability", http.StatusServiceUnavailable)
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(proxyServer.Close)

	kcmClientConfig := restclient.CopyConfig(apiServer.ClientConfig)
	kcmClientConfig.Host = proxyServer.URL
	kcmClientConfig.TLSClientConfig.Insecure = true
	kcmClientConfig.TLSClientConfig.CAFile = ""
	kcmClientConfig.TLSClientConfig.CAData = nil

	kubeConfig := kubeconfig.CreateKubeConfig(kcmClientConfig)
	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig.yaml")
	if err := clientcmd.WriteToFile(*kubeConfig, kubeConfigFile); err != nil {
		t.Fatalf("failed to write kubeconfig: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// A signing key makes the KCM run the serviceaccount-token controller as
	// well, as every real controller manager does.
	keyFile := filepath.Join(t.TempDir(), "sa.key")
	keyPEM, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("failed to generate a signing key: %v", err)
	}
	if err := keyutil.WriteKey(keyFile, keyPEM); err != nil {
		t.Fatalf("failed to write the signing key: %v", err)
	}

	kcm, err := kubecontrollermanagertesting.StartTestServer(t, ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--service-account-private-key-file=" + keyFile,
		"--leader-elect=true",
		"--leader-elect-lease-duration=4s",
		"--leader-elect-renew-deadline=1500ms",
		"--leader-elect-retry-period=200ms",
		"--leader-elect-recovery-deadline=30s",
		// ControllerManagerReleaseLeaderElectionLockOnExit lets KCM return from
		// leader election on context cancellation instead of calling
		// klog.FlushAndExit, which would take the test binary down with it.
		"--feature-gates=LeaderElectionRecovery=true,ControllerManagerReleaseLeaderElectionLockOnExit=true",
	})
	if err != nil {
		t.Fatalf("failed to start KCM: %v", err)
	}
	t.Cleanup(kcm.TearDownFn)

	// Wait for KCM to acquire the lease and reconcile an initial namespace.
	var initialLease *coordinationv1.Lease
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, getErr := client.CoordinationV1().Leases(metav1.NamespaceSystem).Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if getErr != nil {
			return false, nil
		}
		if lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != "" {
			initialLease = lease
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for KCM lease acquisition: %v", err)
	}

	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "kcm-recovery-ns-1"},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create kcm-recovery-ns-1: %v", err)
	}
	if err := waitForDefaultServiceAccount(ctx, client, "kcm-recovery-ns-1", 15*time.Second); err != nil {
		t.Fatalf("timed out waiting for default ServiceAccount in kcm-recovery-ns-1: %v", err)
	}

	// Block lease renewals so KCM enters recovery mode. The gate closes once
	// the renew poll, which starts at most one retry period after the last
	// successful renewal, has failed for a full renew deadline. Renewals are
	// blocked, so a renew time that old proves the gate has closed.
	const renewDeadline, retryPeriod = 1500 * time.Millisecond, 200 * time.Millisecond
	failLeaseRenewals.Store(true)
	var staleLease *coordinationv1.Lease
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, getErr := client.CoordinationV1().Leases(metav1.NamespaceSystem).Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if getErr != nil || lease.Spec.RenewTime == nil {
			return false, nil
		}
		staleLease = lease
		return time.Since(lease.Spec.RenewTime.Time) > renewDeadline+2*retryPeriod, nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for KCM lease renewals to stop: %v", err)
	}

	// Create a second namespace while KCM's write gate is closed.
	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "kcm-recovery-ns-2"},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create kcm-recovery-ns-2: %v", err)
	}
	// Negative check: it can only pass falsely, never flake.
	time.Sleep(400 * time.Millisecond)
	if _, err := client.CoreV1().ServiceAccounts("kcm-recovery-ns-2").Get(ctx, "default", metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected default ServiceAccount in kcm-recovery-ns-2 not to exist while KCM write gate is closed, got err=%v", err)
	}

	// Restore lease renewals and verify KCM recovers the same lease and reconciles kcm-recovery-ns-2 without restarting.
	// A renew time newer than the stale lease's proves a renewal after recovery.
	failLeaseRenewals.Store(false)

	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, getErr := client.CoordinationV1().Leases(metav1.NamespaceSystem).Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if getErr != nil {
			return false, nil
		}
		if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != *initialLease.Spec.HolderIdentity {
			t.Fatalf("expected lease holder identity %q to be preserved across recovery, got %v", *initialLease.Spec.HolderIdentity, lease.Spec.HolderIdentity)
		}
		return lease.Spec.RenewTime != nil && lease.Spec.RenewTime.After(staleLease.Spec.RenewTime.Time), nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for KCM to renew lease after recovery: %v", err)
	}

	if err := waitForDefaultServiceAccount(ctx, client, "kcm-recovery-ns-2", 15*time.Second); err != nil {
		t.Fatalf("timed out waiting for KCM to reconcile kcm-recovery-ns-2 after recovery: %v", err)
	}
}

// TestLeaderElectionRecoveryKCMStarts starts a kube-controller-manager with
// the LeaderElectionRecovery gate and a recovery deadline, and checks that it
// acquires its lease and that a controller writes through its gated client.
func TestLeaderElectionRecoveryKCMStarts(t *testing.T) {
	apiServer := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(apiServer.TearDownFn)

	client, err := kubernetes.NewForConfig(apiServer.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	clientConfig := kubeconfig.CreateKubeConfig(apiServer.ClientConfig)
	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig.yaml")
	if err := clientcmd.WriteToFile(*clientConfig, kubeConfigFile); err != nil {
		t.Fatalf("failed to write kubeconfig: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	kcm, err := kubecontrollermanagertesting.StartTestServer(t, ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--leader-elect=true",
		// ControllerManagerReleaseLeaderElectionLockOnExit lets the KCM return
		// on teardown instead of exiting the test process.
		"--feature-gates=LeaderElectionRecovery=true,ControllerManagerReleaseLeaderElectionLockOnExit=true",
		"--leader-elect-recovery-deadline=5s",
	})
	if err != nil {
		t.Fatalf("failed to start KCM: %v", err)
	}
	t.Cleanup(kcm.TearDownFn)

	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		lease, err := client.CoordinationV1().Leases("kube-system").Get(ctx, "kube-controller-manager", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != "", nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for the KCM to acquire its lease: %v", err)
	}

	// The service account controller creates the default service account in
	// a new namespace through a client that carries the write gate.
	ns, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: "recovery-"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	if err := waitForDefaultServiceAccount(ctx, client, ns.Name, 30*time.Second); err != nil {
		t.Fatalf("controller write through the gated client did not happen: %v", err)
	}
}

func waitForDefaultServiceAccount(ctx context.Context, client kubernetes.Interface, namespace string, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, timeout, true, func(ctx context.Context) (bool, error) {
		_, err := client.CoreV1().ServiceAccounts(namespace).Get(ctx, "default", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return err == nil, err
	})
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
