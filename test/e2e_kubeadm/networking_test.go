/*
Copyright 2019 The Kubernetes Authors.

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

package kubeadm

import (
	"context"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"

	"github.com/onsi/ginkgo/v2"
)

var (
	dualStack                    bool
	podSubnetInKubeadmConfig     bool
	serviceSubnetInKubeadmConfig bool
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the networking as expected.
// in case you want to skip this test use SKIP=setup-networking
var _ = Describe("networking [setup-networking]", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("networking")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	// Tests are either single-stack (either IPv4 or IPv6 address)	or dual-stack
	// We can determine this from the kubeadm config by looking at the number of
	// IPs specified in networking.podSubnet
	ginkgo.BeforeEach(func() {
		// gets the ClusterConfiguration from the kubeadm kubeadm-config ConfigMap as a untyped map
		cc := getClusterConfiguration(f.ClientSet)
		// Extract the networking.podSubnet
		// Please note that this test assumes that the user does not alter network configs
		// using the extraArgs option.
		if _, ok := cc["networking"]; ok {
			netCC := cc["networking"].(map[interface{}]interface{})
			if ps, ok := netCC["podSubnet"]; ok {
				// If podSubnet is not specified, podSubnet cases will be skipped.
				// Note that kubeadm does not currently apply defaults for PodSubnet, so we skip.
				podSubnetInKubeadmConfig = true
				cidrs := strings.Split(ps.(string), ",")
				if len(cidrs) > 1 {
					dualStack = true
				}
			}
			if _, ok := netCC["serviceSubnet"]; ok {
				// If serviceSubnet is not specified, serviceSubnet cases will be skipped.
				serviceSubnetInKubeadmConfig = true
			}
		}
	})

	ginkgo.Context("single-stack", func() {
		ginkgo.Context("podSubnet", func() {
			ginkgo.It("should be properly configured if specified in kubeadm-config", func() {
				if dualStack {
					e2eskipper.Skipf("Skipping because cluster is dual-stack")
				}
				if !podSubnetInKubeadmConfig {
					e2eskipper.Skipf("Skipping because podSubnet was not specified in kubeadm-config")
				}
				cc := getClusterConfiguration(f.ClientSet)
				if _, ok := cc["networking"]; ok {
					netCC := cc["networking"].(map[interface{}]interface{})
					if ps, ok := netCC["podSubnet"]; ok {
						// Check that the pod CIDR allocated to the node(s) is within the kubeadm-config podCIDR.
						nodes, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
						framework.ExpectNoError(err, "error listing nodes")
						for _, node := range nodes.Items {
							if !subnetWithinSubnet(ps.(string), node.Spec.PodCIDR) {
								framework.Failf("failed due to node(%v) IP %v not inside configured pod subnet: %s", node.Name, node.Spec.PodCIDR, ps)
							}
						}
					}
				}
			})
		})
		ginkgo.Context("serviceSubnet", func() {
			ginkgo.It("should be properly configured if specified in kubeadm-config", func() {
				if dualStack {
					e2eskipper.Skipf("Skipping because cluster is dual-stack")
				}
				if !serviceSubnetInKubeadmConfig {
					e2eskipper.Skipf("Skipping because serviceSubnet was not specified in kubeadm-config")
				}
				cc := getClusterConfiguration(f.ClientSet)
				if _, ok := cc["networking"]; ok {
					netCC := cc["networking"].(map[interface{}]interface{})
					if ss, ok := netCC["serviceSubnet"]; ok {
						// Get the kubernetes service in the default namespace.
						// Check that service CIDR allocated is within the serviceSubnet range.
						svc, err := f.ClientSet.CoreV1().Services("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
						framework.ExpectNoError(err, "error getting Service %q from namespace %q", "kubernetes", "default")
						if !ipWithinSubnet(ss.(string), svc.Spec.ClusterIP) {
							framework.Failf("failed due to service(%v) cluster-IP %v not inside configured service subnet: %s", svc.Name, svc.Spec.ClusterIP, ss)
						}
					}
				}
			})
		})
	})
	ginkgo.Context("dual-stack", func() {
		ginkgo.Context("podSubnet", func() {
			ginkgo.It("should be properly configured if specified in kubeadm-config", func() {
				if !dualStack {
					e2eskipper.Skipf("Skipping because cluster is not dual-stack")
				}
				if !podSubnetInKubeadmConfig {
					e2eskipper.Skipf("Skipping because podSubnet was not specified in kubeadm-config")
				}
				cc := getClusterConfiguration(f.ClientSet)
				if _, ok := cc["networking"]; ok {
					netCC := cc["networking"].(map[interface{}]interface{})
					if ps, ok := netCC["podSubnet"]; ok {
						nodes, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
						framework.ExpectNoError(err, "error listing nodes")
						// Check that the pod CIDRs allocated to the node(s) are within the kubeadm-config podCIDR.
						var found bool
						configCIDRs := strings.Split(ps.(string), ",")
						for _, node := range nodes.Items {
							for _, nCIDR := range node.Spec.PodCIDRs {
								found = false
								for _, cCIDR := range configCIDRs {
									if subnetWithinSubnet(cCIDR, nCIDR) {
										found = true
										break
									}
								}
								if !found {
									framework.Failf("failed due to the PodCIDRs (%v) of Node %q not being inside the configuration podSubnet CIDR %q", node.Spec.PodCIDRs, node.Name, configCIDRs)
								}
							}
						}
					}
				}
			})
		})
	})
})

// ipWithinSubnet returns true if an IP (targetIP) falls within the reference subnet (refIPNet)
func ipWithinSubnet(refIPNet, targetIP string) bool {
	_, rNet, _ := netutils.ParseCIDRSloppy(refIPNet)
	tIP := netutils.ParseIPSloppy(targetIP)
	return rNet.Contains(tIP)
}

// subnetWithinSubnet returns true if a subnet (targetNet) falls within the reference subnet (refIPNet)
func subnetWithinSubnet(refIPNet, targetNet string) bool {
	_, rNet, _ := netutils.ParseCIDRSloppy(refIPNet)
	tNet, _, _ := netutils.ParseCIDRSloppy(targetNet)
	return rNet.Contains(tNet)
}
