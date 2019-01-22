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
	"net/http"
	"time"

	computealpha "google.golang.org/api/compute/v0.alpha"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Services [Feature:GCEAlphaFeature][Slow]", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface
	serviceLBNames := []string{}

	BeforeEach(func() {
		// This test suite requires the GCE environment.
		framework.SkipUnlessProviderIs("gce")
		cs = f.ClientSet
	})

	AfterEach(func() {
		if CurrentGinkgoTestDescription().Failed {
			framework.DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning gce resource for %s", lb)
			framework.TestContext.CloudConfig.Provider.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
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

		// Test 1: create a standard tiered LB for the Service.
		By("creating a Service of type LoadBalancer using the standard network tier")
		svc := jig.CreateTCPServiceOrFail(ns, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			setNetworkTier(svc, string(gcecloud.NetworkTierAnnotationStandard))
		})
		// Verify that service has been updated properly.
		svcTier, err := gcecloud.GetServiceNetworkTier(svc)
		Expect(err).NotTo(HaveOccurred())
		Expect(svcTier).To(Equal(cloud.NetworkTierStandard))
		// Record the LB name for test cleanup.
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))

		// Wait and verify the LB.
		ingressIP := waitAndVerifyLBWithTier(jig, ns, svcName, "", createTimeout, lagTimeout)

		// Test 2: re-create a LB of a different tier for the updated Service.
		By("updating the Service to use the premium (default) tier")
		svc = jig.UpdateServiceOrFail(ns, svcName, func(svc *v1.Service) {
			clearNetworkTier(svc)
		})
		// Verify that service has been updated properly.
		svcTier, err = gcecloud.GetServiceNetworkTier(svc)
		Expect(err).NotTo(HaveOccurred())
		Expect(svcTier).To(Equal(cloud.NetworkTierDefault))

		// Wait until the ingress IP changes. Each tier has its own pool of
		// IPs, so changing tiers implies changing IPs.
		ingressIP = waitAndVerifyLBWithTier(jig, ns, svcName, ingressIP, createTimeout, lagTimeout)

		// Test 3: create a standard-tierd LB with a user-requested IP.
		By("reserving a static IP for the load balancer")
		requestedAddrName := fmt.Sprintf("e2e-ext-lb-net-tier-%s", framework.RunId)
		gceCloud, err := gce.GetGCECloud()
		Expect(err).NotTo(HaveOccurred())
		requestedIP, err := reserveAlphaRegionalAddress(gceCloud, requestedAddrName, cloud.NetworkTierStandard)
		Expect(err).NotTo(HaveOccurred(), "failed to reserve a STANDARD tiered address")
		defer func() {
			if requestedAddrName != "" {
				// Release GCE static address - this is not kube-managed and will not be automatically released.
				if err := gceCloud.DeleteRegionAddress(requestedAddrName, gceCloud.Region()); err != nil {
					framework.Logf("failed to release static IP address %q: %v", requestedAddrName, err)
				}
			}
		}()
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Allocated static IP to be used by the load balancer: %q", requestedIP)

		By("updating the Service to use the standard tier with a requested IP")
		svc = jig.UpdateServiceOrFail(ns, svc.Name, func(svc *v1.Service) {
			svc.Spec.LoadBalancerIP = requestedIP
			setNetworkTier(svc, string(gcecloud.NetworkTierAnnotationStandard))
		})
		// Verify that service has been updated properly.
		Expect(svc.Spec.LoadBalancerIP).To(Equal(requestedIP))
		svcTier, err = gcecloud.GetServiceNetworkTier(svc)
		Expect(err).NotTo(HaveOccurred())
		Expect(svcTier).To(Equal(cloud.NetworkTierStandard))

		// Wait until the ingress IP changes and verifies the LB.
		ingressIP = waitAndVerifyLBWithTier(jig, ns, svcName, ingressIP, createTimeout, lagTimeout)
	})
})

func waitAndVerifyLBWithTier(jig *framework.ServiceTestJig, ns, svcName, existingIP string, waitTimeout, checkTimeout time.Duration) string {
	var svc *v1.Service
	if existingIP == "" {
		// Creating the LB for the first time; wait for any ingress IP to show
		// up.
		svc = jig.WaitForNewIngressIPOrFail(ns, svcName, existingIP, waitTimeout)
	} else {
		// Re-creating the LB; wait for the ingress IP to change.
		svc = jig.WaitForNewIngressIPOrFail(ns, svcName, existingIP, waitTimeout)
	}

	svcPort := int(svc.Spec.Ports[0].Port)
	lbIngress := &svc.Status.LoadBalancer.Ingress[0]
	ingressIP := framework.GetIngressPoint(lbIngress)

	By("running sanity and reachability checks")
	if svc.Spec.LoadBalancerIP != "" {
		// Verify that the new ingress IP is the requested IP if it's set.
		Expect(ingressIP).To(Equal(svc.Spec.LoadBalancerIP))
	}
	jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
	// If the IP has been used by previous test, sometimes we get the lingering
	// 404 errors even after the LB is long gone. Tolerate and retry until the
	// the new LB is fully established since this feature is still Alpha in GCP.
	jig.TestReachableHTTPWithRetriableErrorCodes(ingressIP, svcPort, []int{http.StatusNotFound}, checkTimeout)

	// Verify the network tier matches the desired.
	svcNetTier, err := gcecloud.GetServiceNetworkTier(svc)
	Expect(err).NotTo(HaveOccurred())
	netTier, err := getLBNetworkTierByIP(ingressIP)
	Expect(err).NotTo(HaveOccurred(), "failed to get the network tier of the load balancer")
	Expect(netTier).To(Equal(svcNetTier))

	return ingressIP
}

func getLBNetworkTierByIP(ip string) (cloud.NetworkTier, error) {
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
	return cloud.NetworkTierGCEValueToType(rule.NetworkTier), nil
}

func getGCEForwardingRuleByIP(ip string) (*computealpha.ForwardingRule, error) {
	cloud, err := gce.GetGCECloud()
	if err != nil {
		return nil, err
	}
	ruleList, err := cloud.ListAlphaRegionForwardingRules(cloud.Region())
	if err != nil {
		return nil, err
	}
	for _, rule := range ruleList {
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

// TODO: add retries if this turns out to be flaky.
// TODO(#51665): remove this helper function once Network Tiers becomes beta.
func reserveAlphaRegionalAddress(cloud *gcecloud.Cloud, name string, netTier cloud.NetworkTier) (string, error) {
	alphaAddr := &computealpha.Address{
		Name:        name,
		NetworkTier: netTier.ToGCEValue(),
	}

	if err := cloud.ReserveAlphaRegionAddress(alphaAddr, cloud.Region()); err != nil {
		return "", err
	}

	addr, err := cloud.GetRegionAddress(name, cloud.Region())
	if err != nil {
		return "", err
	}

	return addr.Address, nil
}
