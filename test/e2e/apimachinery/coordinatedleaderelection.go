/*
Copyright 2025 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"
	"sync"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("CoordinatedLeaderElection", feature.CoordinatedLeaderElection, framework.WithFeatureGate(kubefeatures.CoordinatedLeaderElection), func() {
	f := framework.NewDefaultFramework("cle")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var clientset clientset.Interface
	var config *rest.Config
	var ns string

	ginkgo.BeforeEach(func() {
		clientset = f.ClientSet
		config = f.ClientConfig()
		ns = f.Namespace.Name
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		_ = clientset.CoordinationV1().Leases(ns).Delete(ctx, "foo", metav1.DeleteOptions{})
		_ = clientset.CoordinationV1().Leases(ns).Delete(ctx, "baz", metav1.DeleteOptions{})
		_ = clientset.CoordinationV1().Leases(ns).Delete(ctx, "bar", metav1.DeleteOptions{})
		_ = clientset.CoordinationV1().Leases(ns).Delete(ctx, "foobar", metav1.DeleteOptions{})
		_ = clientset.CoordinationV1alpha2().LeaseCandidates(ns).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{})
	})

	/*
		Release : v1.33
		Testname: single LeaseCandidate
		Description: Create a lease candidate. A lease must be created and renewed.
	*/
	ginkgo.It("single LeaseCandidate", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)
		go cletest.createAndRunFakeController("foo1", ns, "foo", "1.20.0", "1.20.0", coordinationv1.OldestEmulationVersion)
		cletest.pollForLease(ctx, "foo", ns, ptr.To("foo1"))
	})

	/*
		Release : v1.33
		Testname: multiple LeaseCandidate
		Description: Create multiple lease candidates. The best candidate must be selected.
	*/
	ginkgo.It("multiple LeaseCandidate", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)
		go cletest.createAndRunFakeController("baz1", ns, "baz", "1.20.0", "1.20.0", coordinationv1.OldestEmulationVersion)
		go cletest.createAndRunFakeController("baz2", ns, "baz", "1.20.0", "1.19.0", coordinationv1.OldestEmulationVersion)
		go cletest.createAndRunFakeController("baz3", ns, "baz", "1.19.0", "1.19.0", coordinationv1.OldestEmulationVersion)
		go cletest.createAndRunFakeController("baz4", ns, "baz", "1.20.0", "1.19.0", coordinationv1.OldestEmulationVersion)
		cletest.pollForLease(ctx, "baz", ns, ptr.To("baz3"))
	})

	/*
		Release : v1.33
		Testname: multiple LeaseCandidate third party strategy
		Description: Create multiple lease candidates. The leader lease MUST be created but with the holder identity unset.
	*/
	ginkgo.It("multiple LeaseCandidates third party strategy", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)
		go cletest.createAndRunFakeController("baz1", ns, "baz", "1.20.0", "1.20.0", coordinationv1.CoordinatedLeaseStrategy("foo.com/bar"))
		go cletest.createAndRunFakeController("baz2", ns, "baz", "1.20.0", "1.19.0", coordinationv1.CoordinatedLeaseStrategy("foo.com/bar"))
		cletest.pollForLease(ctx, "baz", ns, nil)
	})

	/*
		Release : v1.33
		Testname: CLE Preemption
		Description: Create a lease candidate. When another more suitable
		candidate is created, the leader lease MUST transition to the new
		candidate.
	*/
	ginkgo.It("CLE Preemption", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)
		go cletest.createAndRunFakeController("bar1", ns, "bar", "1.20.0", "1.20.0", coordinationv1.OldestEmulationVersion)
		cletest.pollForLease(ctx, "bar", ns, ptr.To("bar1"))
		go cletest.createAndRunFakeController("bar2", ns, "bar", "1.19.0", "1.19.0", coordinationv1.OldestEmulationVersion)
		cletest.pollForLease(ctx, "bar", ns, ptr.To("bar2"))
	})

	/*
		Release : v1.33
		Testname: CLE upgrade to enabled
		Description: Create a lease candidate. When another candidate is added
		with coordinated leader election supported, the lease should not
		transition. When the old controller is shutdown, leader election should
		transition to the new controller.
	*/
	ginkgo.It("CLE upgrade to enabled", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)

		go cletest.createAndRunFakeLegacyController("foo1-130", "default", "foobar")
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-130"))
		go cletest.createAndRunFakeController("foo1-131", "default", "foobar", "1.31.0", "1.31.0", coordinationv1.OldestEmulationVersion)
		// running a new controller should not kick off old leader
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-130"))
		// If the 130 (non CLE) controller is stopped, leader should transition to 131 (CLE)
		cletest.cancelController("foo1-130", "default")
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-131"))
	})

	/*
		Release : v1.33
		Testname: CLE downgrade to disabled
		Description: Create a lease candidate with coordinated leader election
		enabled. When another candidate is added without CLE, the lease should
		not transition. When the old controller is shutdown, leader election
		should transition to the new controller.
	*/
	ginkgo.It("CLE downgrade to disabled", func(ctx context.Context) {
		ctxWithCancel, cancel := context.WithCancel(ctx)
		defer cancel()
		cletest := setupCLE(config, clientset, ctxWithCancel)

		go cletest.createAndRunFakeController("foo1-131", "default", "foobar", "1.31.0", "1.31.0", coordinationv1.OldestEmulationVersion)
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-131"))
		go cletest.createAndRunFakeLegacyController("foo1-130", "default", "foobar")
		// running a new controller should not kick off old leader
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-131"))
		// If the 131 (CLE) controller is stopped, leader should transition to 130 (non-CLE)
		cletest.cancelController("foo1-131", "default")
		cletest.pollForLease(ctx, "foobar", "default", ptr.To("foo1-130"))
	})

})

func setupCLE(config *rest.Config, clientset clientset.Interface, ctx context.Context) *cleTest {
	return &cleTest{
		config:    config,
		clientset: clientset,
		ctx:       ctx,
		ctxList:   map[string]ctxCancelPair{},
	}
}

type ctxCancelPair struct {
	ctx    context.Context
	cancel func()
}
type cleTest struct {
	config    *rest.Config
	clientset clientset.Interface
	ctx       context.Context
	mu        sync.Mutex
	ctxList   map[string]ctxCancelPair
}

func (t *cleTest) createAndRunFakeLegacyController(name string, namespace string, targetLease string) {
	ctx, cancel := context.WithCancel(t.ctx)
	t.mu.Lock()
	defer t.mu.Unlock()
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
func (t *cleTest) createAndRunFakeController(name string, namespace string, targetLease string, binaryVersion string, compatibilityVersion string, preferredStrategy coordinationv1.CoordinatedLeaseStrategy) {
	identityLease, _, err := leaderelection.NewCandidate(
		t.clientset,
		namespace,
		name,
		targetLease,
		binaryVersion,
		compatibilityVersion,
		preferredStrategy,
	)
	framework.ExpectNoError(err)
	ctx, cancel := context.WithCancel(t.ctx)
	t.mu.Lock()
	t.ctxList[name+"/"+namespace] = ctxCancelPair{ctx, cancel}
	t.mu.Unlock()

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

func (t *cleTest) pollForLease(ctx context.Context, name, namespace string, holder *string) {
	err := wait.PollUntilContextTimeout(ctx, 1000*time.Millisecond, 25*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := t.clientset.CoordinationV1().Leases(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		if holder == nil {
			return lease.Spec.HolderIdentity == nil, nil
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity == *holder, nil
	})
	framework.ExpectNoError(err)

}

func (t *cleTest) cancelController(name, namespace string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ctxList[name+"/"+namespace].cancel()
	delete(t.ctxList, name+"/"+namespace)
}
