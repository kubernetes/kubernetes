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

package apiserver

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestSingleLeaseCandidate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	ctx, cancel := context.WithCancel(context.Background())
	cletest := setupCLE(config, ctx, cancel, t)
	defer cletest.cleanup()
	go cletest.createAndRunFakeController("foo1", "default", "foo", "1.20.0", "1.20.0")
	cletest.pollForLease("foo", "default", "foo1")
}

func TestMultipleLeaseCandidate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	ctx, cancel := context.WithCancel(context.Background())
	cletest := setupCLE(config, ctx, cancel, t)
	defer cletest.cleanup()
	go cletest.createAndRunFakeController("foo1", "default", "foo", "1.20.0", "1.20.0")
	go cletest.createAndRunFakeController("foo2", "default", "foo", "1.20.0", "1.19.0")
	go cletest.createAndRunFakeController("foo3", "default", "foo", "1.19.0", "1.19.0")
	go cletest.createAndRunFakeController("foo4", "default", "foo", "1.2.0", "1.19.0")
	go cletest.createAndRunFakeController("foo5", "default", "foo", "1.20.0", "1.19.0")
	cletest.pollForLease("foo", "default", "foo3")
}

func TestLeaseSwapIfBetterAvailable(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	ctx, cancel := context.WithCancel(context.Background())
	cletest := setupCLE(config, ctx, cancel, t)
	defer cletest.cleanup()

	go cletest.createAndRunFakeController("bar1", "default", "bar", "1.20.0", "1.20.0")
	cletest.pollForLease("bar", "default", "bar1")
	go cletest.createAndRunFakeController("bar2", "default", "bar", "1.19.0", "1.19.0")
	cletest.pollForLease("bar", "default", "bar2")
}

// TestUpgradeSkew tests that a legacy client and a CLE aware client operating on the same lease do not cause errors
func TestUpgradeSkew(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	ctx, cancel := context.WithCancel(context.Background())
	cletest := setupCLE(config, ctx, cancel, t)
	defer cletest.cleanup()

	go cletest.createAndRunFakeLegacyController("foo1-130", "default", "foo")
	cletest.pollForLease("foo", "default", "foo1-130")
	go cletest.createAndRunFakeController("foo1-131", "default", "foo", "1.31.0", "1.31.0")
	// running a new controller should not kick off old leader
	cletest.pollForLease("foo", "default", "foo1-130")
	cletest.cancelController("foo1-130", "default")
	cletest.pollForLease("foo", "default", "foo1-131")
}

func TestLeaseCandidateCleanup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	expiredLC := &v1alpha1.LeaseCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "expired",
			Namespace: "default",
		},
		Spec: v1alpha1.LeaseCandidateSpec{
			LeaseName:           "foobar",
			BinaryVersion:       "0.1.0",
			EmulationVersion:    "0.1.0",
			PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
			RenewTime:           &metav1.MicroTime{Time: time.Now().Add(-2 * time.Hour)},
			PingTime:            &metav1.MicroTime{Time: time.Now().Add(-1 * time.Hour)},
		},
	}
	ctx := context.Background()
	_, err = clientset.CoordinationV1alpha1().LeaseCandidates("default").Create(ctx, expiredLC, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1000*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (done bool, err error) {
		_, err = clientset.CoordinationV1alpha1().LeaseCandidates("default").Get(ctx, "expired", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timieout waiting for lease gc")
	}

}

type ctxCancelPair struct {
	ctx    context.Context
	cancel func()
}
type cleTest struct {
	config    *rest.Config
	clientset *kubernetes.Clientset
	t         *testing.T
	ctxList   map[string]ctxCancelPair
}

func (t cleTest) createAndRunFakeLegacyController(name string, namespace string, targetLease string) {
	ctx, cancel := context.WithCancel(context.Background())
	t.ctxList[name+"/"+namespace] = ctxCancelPair{ctx, cancel}

	electionChecker := leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
	go leaderElectAndRunUncoordinated(ctx, t.config, name, electionChecker,
		namespace,
		"leases",
		targetLease,
		leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				klog.Info("Elected leader, starting..")
			},
			OnStoppedLeading: func() {
				klog.Errorf("%s Lost leadership, stopping", name)
			},
		})

}
func (t cleTest) createAndRunFakeController(name string, namespace string, targetLease string, binaryVersion string, compatibilityVersion string) {
	identityLease, _, err := leaderelection.NewCandidate(
		t.clientset,
		namespace,
		name,
		targetLease,
		binaryVersion,
		compatibilityVersion,
		[]v1.CoordinatedLeaseStrategy{"OldestEmulationVersion"},
	)
	if err != nil {
		t.t.Error(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	t.ctxList[name+"/"+namespace] = ctxCancelPair{ctx, cancel}
	go identityLease.Run(ctx)

	electionChecker := leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
	go leaderElectAndRunCoordinated(ctx, t.config, name, electionChecker,
		namespace,
		"leases",
		targetLease,
		leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				klog.Info("Elected leader, starting..")
			},
			OnStoppedLeading: func() {
				klog.Errorf("%s Lost leadership, stopping", name)
				// klog.FlushAndExit(klog.ExitFlushTimeout, 1)
			},
		})
}

func leaderElectAndRunUncoordinated(ctx context.Context, kubeconfig *rest.Config, lockIdentity string, electionChecker *leaderelection.HealthzAdaptor, resourceNamespace, resourceLock, leaseName string, callbacks leaderelection.LeaderCallbacks) {
	leaderElectAndRun(ctx, kubeconfig, lockIdentity, electionChecker, resourceNamespace, resourceLock, leaseName, callbacks, false)
}

func leaderElectAndRunCoordinated(ctx context.Context, kubeconfig *rest.Config, lockIdentity string, electionChecker *leaderelection.HealthzAdaptor, resourceNamespace, resourceLock, leaseName string, callbacks leaderelection.LeaderCallbacks) {
	leaderElectAndRun(ctx, kubeconfig, lockIdentity, electionChecker, resourceNamespace, resourceLock, leaseName, callbacks, true)
}

func leaderElectAndRun(ctx context.Context, kubeconfig *rest.Config, lockIdentity string, electionChecker *leaderelection.HealthzAdaptor, resourceNamespace, resourceLock, leaseName string, callbacks leaderelection.LeaderCallbacks, coordinated bool) {
	logger := klog.FromContext(ctx)
	rl, err := resourcelock.NewFromKubeconfig(resourceLock,
		resourceNamespace,
		leaseName,
		resourcelock.ResourceLockConfig{
			Identity: lockIdentity,
		},
		kubeconfig,
		5*time.Second)
	if err != nil {
		logger.Error(err, "Error creating lock")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	leaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: 5 * time.Second,
		RenewDeadline: 3 * time.Second,
		RetryPeriod:   2 * time.Second,
		Callbacks:     callbacks,
		WatchDog:      electionChecker,
		Name:          leaseName,
		Coordinated:   coordinated,
	})
}

func (t cleTest) pollForLease(name, namespace, holder string) {
	err := wait.PollUntilContextTimeout(t.ctxList["main"].ctx, 1000*time.Millisecond, 25*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := t.clientset.CoordinationV1().Leases(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity == holder, nil
	})
	if err != nil {
		t.t.Fatalf("timeout awiting for Lease %s %s err: %v", name, namespace, err)
	}
}

func (t cleTest) cancelController(name, namespace string) {
	t.ctxList[name+"/"+namespace].cancel()
	delete(t.ctxList, name+"/"+namespace)
}

func (t cleTest) cleanup() {
	err := t.clientset.CoordinationV1().Leases("kube-system").Delete(context.TODO(), "leader-election-controller", metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		t.t.Error(err)
	}
	for _, c := range t.ctxList {
		c.cancel()
	}
}

func setupCLE(config *rest.Config, ctx context.Context, cancel func(), t *testing.T) cleTest {
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	a := ctxCancelPair{ctx, cancel}
	return cleTest{
		config:    config,
		clientset: clientset,
		ctxList:   map[string]ctxCancelPair{"main": a},
		t:         t,
	}
}
