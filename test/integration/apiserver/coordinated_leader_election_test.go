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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubeinformers "k8s.io/client-go/informers"
	identityleaseinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/clock"
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
	go cletest.createAndRunFakeController("foo1", "default", "foo", "1.20", "1.20")
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
	go cletest.createAndRunFakeController("foo1", "default", "foo", "1.20", "1.20")
	go cletest.createAndRunFakeController("foo2", "default", "foo", "1.20", "1.19")
	go cletest.createAndRunFakeController("foo3", "default", "foo", "1.19", "1.19")
	go cletest.createAndRunFakeController("foo4", "default", "foo", "1.20", "1.19")
	go cletest.createAndRunFakeController("foo5", "default", "foo", "1.20", "1.19")
	cletest.pollForLease("foo", "default", "foo3")
}

func TestControllerDisappear(t *testing.T) {
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
	go cletest.createAndRunFakeController("foo1", "default", "foo", "1.20", "1.20")
	go cletest.createAndRunFakeController("foo2", "default", "foo", "1.20", "1.19")
	cletest.pollForLease("foo", "default", "foo2")
	cletest.cancelController("foo2", "default")
	cletest.pollForLease("foo", "default", "foo1")
}

type ctxCancelPair struct {
	ctx    context.Context
	cancel func()
}
type cleTest struct {
	config    *rest.Config
	clientset *kubernetes.Clientset
	informer  identityleaseinformers.LeaseCandidateInformer
	t         *testing.T
	ctxList   map[string]ctxCancelPair
}

func (t cleTest) createAndRunFakeController(name string, namespace string, targetLease string, binaryVersion string, compatibilityVersion string) {
	identityLease := &leaderelection.LeaseCandidate{
		LeaseClient:            t.clientset.CoordinationV1alpha1().LeaseCandidates(namespace),
		LeaseCandidateInformer: t.informer,
		LeaseName:              name,
		LeaseNamespace:         namespace,
		LeaseDurationSeconds:   10,
		Clock:                  clock.RealClock{},
		TargetLease:            targetLease,
		RenewInterval:          5,
		BinaryVersion:          binaryVersion,
		CompatibilityVersion:   compatibilityVersion,
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.ctxList[name+"/"+namespace] = ctxCancelPair{ctx, cancel}
	identityLease.Run(ctx)

	electionChecker := leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
	go leaderElectAndRun(ctx, t.config, name, electionChecker,
		namespace,
		"coordinatedLeases",
		targetLease,
		leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				klog.Info("Elected leader, starting..")
			},
			OnStoppedLeading: func() {
				klog.Error("Lost leadership, stopping")
				// klog.FlushAndExit(klog.ExitFlushTimeout, 1)
			},
		})
}

func leaderElectAndRun(ctx context.Context, kubeconfig *rest.Config, lockIdentity string, electionChecker *leaderelection.HealthzAdaptor, resourceNamespace, resourceLock, leaseName string, callbacks leaderelection.LeaderCallbacks) {
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
		Lock:                      rl,
		LeaseDuration:             20 * time.Second,
		RenewDeadline:             10 * time.Second,
		RetryPeriod:               5 * time.Second,
		Callbacks:                 callbacks,
		WatchDog:                  electionChecker,
		Name:                      leaseName,
		CoordinatedLeaderElection: true,
	})
}

func (t cleTest) pollForLease(name, namespace, holder string) {
	err := wait.PollUntilContextTimeout(t.ctxList["main"].ctx, 1000*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := t.clientset.CoordinationV1().Leases(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity == holder, nil
	})
	if err != nil {
		t.t.Fatal(err)
	}
}

func (t cleTest) cancelController(name, namespace string) {
	t.ctxList[name+"/"+namespace].cancel()
}

func (t cleTest) cleanup() {
	for _, c := range t.ctxList {
		c.cancel()
	}
}

func setupCLE(config *rest.Config, ctx context.Context, cancel func(), t *testing.T) cleTest {
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	k8sI := kubeinformers.NewSharedInformerFactory(clientset, 0)

	a := ctxCancelPair{ctx, cancel}
	return cleTest{
		config:    config,
		clientset: clientset,
		informer:  k8sI.Coordination().V1alpha1().LeaseCandidates(),
		ctxList:   map[string]ctxCancelPair{"main": a},
		t:         t,
	}
}
