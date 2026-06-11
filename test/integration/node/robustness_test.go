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
	"fmt"
	"testing"
	"time"

	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/robustness"
)

const (
	nodeMonitorGracePeriod = 5 * time.Second
	// heartbeatPeriod is the simulated kubelet's lease renewal interval.
	heartbeatPeriod = 200 * time.Millisecond
	// leaseWatchStall must exceed the grace period so the cached lease expires
	// while the live one stays fresh.
	leaseWatchStall = 8 * time.Second

	liveLeaseGetRule = "LiveLeaseVerify"
)

// TestLeaseReconcileRobustnessMatrix runs the node lifecycle controller against
// the chaos matrix with a simulated kubelet heartbeat as the faulted write. The
// write faults hold the lease still for most of the grace period; the node must
// stay Ready. Only StaleRootWatch reaches the controller's own lease logic.
//
// StaleRootWatch stalls the controller's lease watch past the grace period,
// exercising NodeControllerLeaseCircuitBreaker.
func TestLeaseReconcileRobustnessMatrix(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeControllerLeaseCircuitBreaker, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.AtomicFIFO, true) // store RV tracking

	tCtx := ktesting.Init(t)
	nodeName := "target-node"
	var scenarioStartTime time.Time

	suite := robustness.NewTestSuite(tCtx, robustness.ControllerProfile{
		Name: "lease-renewal",
		Root: robustness.ResourceRef{
			Group:     "coordination.k8s.io",
			Resource:  "leases",
			Namespace: v1.NamespaceNodeLease,
			Name:      nodeName,
		},
		WritesRoot: true,
		WatchStall: leaseWatchStall,
		// 3s of rejected renewals: the controller sees a lease that stops
		// advancing for most of the 5s grace period and must not act on it.
		WriteFaultBurst: int((3 * time.Second) / heartbeatPeriod),
	})

	suite.SetControllerSetup(func(tCtx ktesting.TContext, fixture *robustness.RobustnessTestFixture) {
		informerFactory := informers.NewSharedInformerFactory(fixture.ClientSet(), 0)
		nodeInformer := fixture.WrapNodeInformer(informerFactory.Core().V1().Nodes())
		podInformer := fixture.WrapPodInformer(informerFactory.Core().V1().Pods())
		leaseInformer := informerFactory.Coordination().V1().Leases()
		dsInformer := fixture.WrapDaemonSetInformer(informerFactory.Apps().V1().DaemonSets())

		nc, err := nodelifecycle.NewNodeLifecycleController(
			fixture.Context(),
			leaseInformer,
			podInformer,
			nodeInformer,
			dsInformer,
			fixture.ClientSet(),
			100*time.Millisecond,   // node monitor period
			time.Minute,            // node startup grace period
			nodeMonitorGracePeriod, // node monitor grace period
			100,                    // Eviction limiter QPS
			100,                    // Secondary eviction limiter QPS
			50,                     // Large cluster threshold
			0.55,                   // Unhealthy zone threshold
		)
		if err != nil {
			tCtx.Fatalf("Failed to create NodeLifecycle controller: %v", err)
		}
		fixture.StartInformers(informerFactory)
		fixture.Go(func(ctx context.Context) { nc.Run(ctx) })
	})

	suite.SetScenarioAction(func(ctx context.Context, fixture *robustness.RobustnessTestFixture) error {
		adminClient := fixture.AdminClientSet()
		wrappedClient := fixture.ClientSet()

		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: nodeName},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						LastHeartbeatTime:  metav1.Now(),
						LastTransitionTime: metav1.Now(),
					},
				},
			},
		}
		_, err := adminClient.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}

		nsObj := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: v1.NamespaceNodeLease}}
		_, err = adminClient.CoreV1().Namespaces().Create(ctx, nsObj, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}

		duration := int32(40)
		testLease := &coordv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      nodeName,
				Namespace: v1.NamespaceNodeLease,
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "Node",
						Name:       nodeName,
					},
				},
			},
			Spec: coordv1.LeaseSpec{
				HolderIdentity:       &nodeName,
				LeaseDurationSeconds: &duration,
				RenewTime:            &metav1.MicroTime{Time: time.Now()},
			},
		}

		createdNode, err := adminClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		testLease.OwnerReferences[0].UID = createdNode.UID

		createdLease, err := adminClient.CoordinationV1().Leases(v1.NamespaceNodeLease).Create(ctx, testLease, metav1.CreateOptions{})
		if err != nil {
			return err
		}
		if createdLease.Spec.RenewTime != nil {
			scenarioStartTime = createdLease.Spec.RenewTime.Time
		} else {
			scenarioStartTime = time.Now()
		}

		// Simulated kubelet heartbeat. Writes go through the wrapped client;
		// reads use the admin client so a named lease GET on the wrapped
		// transport is unambiguously the controller's (see liveLeaseGetRule).
		fixture.Go(func(ctx context.Context) {
			ticker := time.NewTicker(heartbeatPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					latest, err := adminClient.CoordinationV1().Leases(v1.NamespaceNodeLease).Get(ctx, nodeName, metav1.GetOptions{})
					if err != nil {
						continue
					}
					latest.Spec.RenewTime = &metav1.MicroTime{Time: time.Now()}
					_, _ = wrappedClient.CoordinationV1().Leases(v1.NamespaceNodeLease).Update(ctx, latest, metav1.UpdateOptions{})
				}
			}
		})

		return nil
	})

	suite.AddSafetyInvariant("SingleLeaseObject", robustness.CountAtMost(1, "Lease",
		func(ctx context.Context, c clientset.Interface) (int, error) {
			leases, err := c.CoordinationV1().Leases(v1.NamespaceNodeLease).List(ctx, metav1.ListOptions{})
			if err != nil {
				return 0, err
			}
			return len(leases.Items), nil
		}))

	suite.AddSafetyInvariant("NodeAlwaysReady", func(ctx context.Context, c clientset.Interface) error {
		node, err := c.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}
		for _, cond := range node.Status.Conditions {
			if cond.Type == v1.NodeReady {
				if cond.Status != v1.ConditionTrue {
					return fmt.Errorf("node %q is not Ready: status=%v, reason=%v, message=%v", nodeName, cond.Status, cond.Reason, cond.Message)
				}
				return nil
			}
		}
		return fmt.Errorf("node %q does not have NodeReady condition", nodeName)
	})

	// StaleRootWatch stalls the lease watch past the grace period. The circuit
	// breaker must double-check the live lease exactly once and then skip until
	// the cache catches up; only the breaker GETs a lease by name.
	suite.AddScenarioFaults("StaleRootWatch", func(p robustness.ControllerProfile, _ *robustness.FaultRegistry) []robustness.FaultRule {
		m := p.RootWriteMatch()
		m.Verbs = []string{"GET"}
		return []robustness.FaultRule{{Name: liveLeaseGetRule, Match: m, Condition: robustness.TriggerAlways(), ExpectTriggered: true}}
	})
	suite.AddScenarioVerify("StaleRootWatch", func(_ robustness.ControllerProfile, fixture *robustness.RobustnessTestFixture) {
		if hits := fixture.Registry().GetHitCount(liveLeaseGetRule); hits != 1 {
			fixture.TCtx().Errorf("controller performed %d live lease GETs; want exactly 1 (the consistency store should suppress the rest until the cache catches up)", hits)
		}
	})

	suite.AddLivenessInvariant("LeaseEventuallyRenewed", robustness.ObjectSatisfies(
		func(ctx context.Context, c clientset.Interface) (*coordv1.Lease, error) {
			return c.CoordinationV1().Leases(v1.NamespaceNodeLease).Get(ctx, nodeName, metav1.GetOptions{})
		},
		func(latest *coordv1.Lease) error {
			if latest.Spec.RenewTime == nil {
				return fmt.Errorf("lease Spec.RenewTime is nil")
			}
			if !latest.Spec.RenewTime.After(scenarioStartTime) {
				return fmt.Errorf("lease has not been renewed yet: renewTime=%v, scenarioStartTime=%v", latest.Spec.RenewTime.Time, scenarioStartTime)
			}
			return nil
		}))

	suite.Run()
}
