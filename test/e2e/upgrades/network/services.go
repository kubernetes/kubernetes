/*
Copyright 2017 The Kubernetes Authors.

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

package network

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
)

// ServiceUpgradeTest tests that a service is available before and
// after a cluster upgrade. During a master-only upgrade, it will test
// that a service remains available during the upgrade.
type ServiceUpgradeTest struct {
	jig          *e2eservice.TestJig
	tcpService   *v1.Service
	tcpIngressIP string
	svcPort      int
}

// Name returns the tracking name of the test.
func (ServiceUpgradeTest) Name() string { return "service-upgrade" }

// Setup creates a service with a load balancer and makes sure it's reachable.
func (t *ServiceUpgradeTest) Setup(ctx context.Context, f *framework.Framework) {
	serviceName := "service-test"
	jig := e2eservice.NewTestJig(f.ClientSet, f.Namespace.Name, serviceName)

	ns := f.Namespace
	cs := f.ClientSet

	ginkgo.By("creating a TCP service " + serviceName + " with type=LoadBalancer in namespace " + ns.Name)
	_, err := jig.CreateTCPService(ctx, func(s *v1.Service) {
		s.Spec.Type = v1.ServiceTypeLoadBalancer
	})
	framework.ExpectNoError(err)
	tcpService, err := jig.WaitForLoadBalancer(ctx, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs))
	framework.ExpectNoError(err)

	// Get info to hit it with
	tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
	svcPort := int(tcpService.Spec.Ports[0].Port)

	ginkgo.By("creating pod to be part of service " + serviceName)
	deployment, err := jig.Run(ctx, jig.AddDeploymentAntiAffinity)
	framework.ExpectNoError(err)

	ginkgo.By("creating a PodDisruptionBudget to cover the Deployment")
	_, err = jig.CreatePDB(ctx, deployment)
	framework.ExpectNoError(err)

	// Hit it once before considering ourselves ready
	ginkgo.By("hitting the pod through the service's LoadBalancer")
	timeout := e2eservice.LoadBalancerLagTimeoutDefault
	if framework.ProviderIs("aws") {
		timeout = e2eservice.LoadBalancerLagTimeoutAWS
	}
	e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, timeout)

	t.jig = jig
	t.tcpService = tcpService
	t.tcpIngressIP = tcpIngressIP
	t.svcPort = svcPort
}

// Test runs a connectivity check to the service.
func (t *ServiceUpgradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	switch upgrade {
	case upgrades.MasterUpgrade, upgrades.ClusterUpgrade:
		t.test(ctx, f, done, true, true)
	case upgrades.NodeUpgrade:
		t.test(ctx, f, done, true, false)
	default:
		t.test(ctx, f, done, false, false)
	}
}

// Teardown cleans up any remaining resources.
func (t *ServiceUpgradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func (t *ServiceUpgradeTest) test(ctx context.Context, f *framework.Framework, done <-chan struct{}, testDuringDisruption, testFinalizer bool) {
	if testDuringDisruption {
		// Continuous validation
		ginkgo.By("continuously hitting the pod through the service's LoadBalancer")
		// TODO (pohly): add context support
		wait.Until(func() {
			e2eservice.TestReachableHTTP(ctx, t.tcpIngressIP, t.svcPort, e2eservice.LoadBalancerLagTimeoutDefault)
		}, framework.Poll, done)
	} else {
		// Block until upgrade is done
		ginkgo.By("waiting for upgrade to finish without checking if service remains up")
		<-done
	}

	// Hit it once more
	ginkgo.By("hitting the pod through the service's LoadBalancer")
	e2eservice.TestReachableHTTP(ctx, t.tcpIngressIP, t.svcPort, e2eservice.LoadBalancerLagTimeoutDefault)
	if testFinalizer {
		defer func() {
			ginkgo.By("Check that service can be deleted with finalizer")
			e2eservice.WaitForServiceDeletedWithFinalizer(ctx, t.jig.Client, t.tcpService.Namespace, t.tcpService.Name)
		}()
		ginkgo.By("Check that finalizer is present on loadBalancer type service")
		e2eservice.WaitForServiceUpdatedWithFinalizer(ctx, t.jig.Client, t.tcpService.Namespace, t.tcpService.Name, true)
	}
}
