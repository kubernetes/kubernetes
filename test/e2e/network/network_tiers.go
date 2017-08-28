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
	"fmt"
	"time"

	computealpha "google.golang.org/api/compute/v0.alpha"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Services [Feature:GCEAlphaFeature][Slow]", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface
	var internalClientset internalclientset.Interface
	serviceLBNames := []string{}

	BeforeEach(func() {
		// This test suite requires the GCE environment.
		framework.SkipUnlessProviderIs("gce")
		cs = f.ClientSet
		internalClientset = f.InternalClientset
	})

	AfterEach(func() {
		if CurrentGinkgoTestDescription().Failed {
			framework.DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning gce resource for %s", lb)
			framework.CleanupServiceGCEResources(cs, lb, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})
	It("should be able to create and tear down a standard-tier load balancer [Slow]", func() {
		lagTimeout := framework.LoadBalancerLagTimeoutDefault
		createTimeout := framework.GetServiceLoadBalancerCreationTimeout(cs)

		svcName := "net-tiers-svc"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, svcName)

		By("creating a pod to be part of the service " + svcName)
		jig.RunOrFail(ns, nil)

		By("creating a Service of type LoadBalancer using the standard network tier")
		svc := jig.CreateTCPServiceOrFail(ns, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			setNetworkTier(svc, gcecloud.NetworkTierAnnotationStandard)
		})
		// Record the LB name for test cleanup.
		serviceLBNames = append(serviceLBNames, cloudprovider.GetLoadBalancerName(svc))

		svc = jig.WaitForLoadBalancerOrFail(ns, svcName, createTimeout)
		lbIngress := &svc.Status.LoadBalancer.Ingress[0]
		svcPort := int(svc.Spec.Ports[0].Port)
		ingressIP := framework.GetIngressPoint(lbIngress)

		By("running sanity and reachability checks")
		jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
		jig.TestReachableHTTP(ingressIP, svcPort, lagTimeout)
		// Check the network tier of the forwarding rule.
		netTier, err := getLBNetworkTierByIP(ingressIP)
		Expect(err).NotTo(HaveOccurred(), "failed to get the network tier of the load balancer")
		Expect(netTier).To(Equal(gcecloud.NetworkTierStandard))

		By("updating the Service to use the premium (default) tier")
		existingIP := ingressIP
		svc = jig.UpdateServiceOrFail(ns, svcName, func(svc *v1.Service) {
			clearNetworkTier(svc)
		})
		// Wait until the ingress IP changes. Each tier has its own pool of
		// IPs, so changing tiers implies changing IPs.
		svc = jig.WaitForNewIngressIPOrFail(ns, svcName, existingIP, createTimeout)
		lbIngress = &svc.Status.LoadBalancer.Ingress[0]
		ingressIP = framework.GetIngressPoint(lbIngress)

		By("running sanity and reachability checks")
		jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
		jig.TestReachableHTTP(ingressIP, svcPort, lagTimeout)
		// Check the network tier of the forwarding rule.
		netTier, err = getLBNetworkTierByIP(ingressIP)
		Expect(err).NotTo(HaveOccurred(), "failed to get the network tier of the load balancer")
		Expect(netTier).To(Equal(gcecloud.NetworkTierPremium))

		// TODO: Add tests for user-requested IPs.
	})
})

func getLBNetworkTierByIP(ip string) (gcecloud.NetworkTier, error) {
	var rule *computealpha.ForwardingRule
	// Retry a few times to tolerate flakes.
	err := wait.PollImmediate(5*time.Second, 15*time.Second, func() (bool, error) {
		obj, err := getGCEForwardingRuleByIP(ip)
		if err != nil {
			return false, err
		}
		rule = obj
		return true, nil
	})
	if err != nil {
		return "", err
	}
	return gcecloud.NetworkTierGCEValueToType(rule.NetworkTier), nil
}

func getGCEForwardingRuleByIP(ip string) (*computealpha.ForwardingRule, error) {
	cloud, err := framework.GetGCECloud()
	if err != nil {
		return nil, err
	}
	ruleList, err := cloud.ListAlphaRegionForwardingRules(cloud.Region())
	if err != nil {
		return nil, err
	}
	for _, rule := range ruleList.Items {
		if rule.IPAddress == ip {
			return rule, nil
		}
	}
	return nil, fmt.Errorf("forwarding rule with ip %q not found", ip)
}

func setNetworkTier(svc *v1.Service, tier string) {
	key := gcecloud.NetworkTierAnnotationKey
	if svc.ObjectMeta.Annotations == nil {
		svc.ObjectMeta.Annotations = map[string]string{}
	}
	svc.ObjectMeta.Annotations[key] = tier
}

func clearNetworkTier(svc *v1.Service) {
	key := gcecloud.NetworkTierAnnotationKey
	if svc.ObjectMeta.Annotations == nil {
		return
	}
	delete(svc.ObjectMeta.Annotations, key)
}
