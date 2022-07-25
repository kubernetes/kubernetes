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
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	dnsService = "kube-dns"

	coreDNSServiceAccountName = "coredns"
	coreDNSConfigMap          = "coredns"
	coreDNSConfigMapKey       = "Corefile"
	coreDNSRoleName           = "system:coredns"
	coreDNSRoleBindingName    = coreDNSRoleName
	coreDNSDeploymentName     = "coredns"

	kubeDNSServiceAccountName = "kube-dns"
	kubeDNSDeploymentName     = "kube-dns"
)

var (
	dnsType = ""
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the dns as expected
var _ = Describe("DNS addon", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("DNS")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	// kubeadm supports two type of DNS addon, and so
	// it is necessary to get it from the kubeadm-config ConfigMap before testing
	ginkgo.BeforeEach(func() {
		// if the dnsType name is already known exit
		if dnsType != "" {
			return
		}

		// gets the ClusterConfiguration from the kubeadm kubeadm-config ConfigMap as a untyped map
		m := getClusterConfiguration(f.ClientSet)

		// Extract the dnsType
		dnsType = "CoreDNS"
		if _, ok := m["dns"]; ok {
			d := m["dns"].(map[interface{}]interface{})
			if t, ok := d["type"]; ok {
				dnsType = t.(string)
			}
		}
	})

	ginkgo.Context("kube-dns", func() {
		ginkgo.Context("kube-dns ServiceAccount", func() {
			ginkgo.It("should exist", func() {
				if dnsType != "kube-dns" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				ExpectServiceAccount(f.ClientSet, kubeSystemNamespace, kubeDNSServiceAccountName)
			})
		})

		ginkgo.Context("kube-dns Deployment", func() {
			ginkgo.It("should exist and be properly configured", func() {
				if dnsType != "kube-dns" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				d := GetDeployment(f.ClientSet, kubeSystemNamespace, kubeDNSDeploymentName)

				framework.ExpectEqual(d.Spec.Template.Spec.ServiceAccountName, kubeDNSServiceAccountName)
			})
		})
	})

	ginkgo.Context("CoreDNS", func() {
		ginkgo.Context("CoreDNS ServiceAccount", func() {
			ginkgo.It("should exist", func() {
				if dnsType != "CoreDNS" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				ExpectServiceAccount(f.ClientSet, kubeSystemNamespace, coreDNSServiceAccountName)
			})

			ginkgo.It("should have related ClusterRole and ClusterRoleBinding", func() {
				if dnsType != "CoreDNS" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				ExpectClusterRole(f.ClientSet, coreDNSRoleName)
				ExpectClusterRoleBinding(f.ClientSet, coreDNSRoleBindingName)
			})
		})

		ginkgo.Context("CoreDNS ConfigMap", func() {
			ginkgo.It("should exist and be properly configured", func() {
				if dnsType != "CoreDNS" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				cm := GetConfigMap(f.ClientSet, kubeSystemNamespace, coreDNSConfigMap)

				gomega.Expect(cm.Data).To(gomega.HaveKey(coreDNSConfigMapKey))
			})
		})

		ginkgo.Context("CoreDNS Deployment", func() {
			ginkgo.It("should exist and be properly configured", func() {
				if dnsType != "CoreDNS" {
					e2eskipper.Skipf("Skipping because DNS type is %s", dnsType)
				}

				d := GetDeployment(f.ClientSet, kubeSystemNamespace, coreDNSDeploymentName)

				framework.ExpectEqual(d.Spec.Template.Spec.ServiceAccountName, coreDNSServiceAccountName)
			})
		})
	})

	ginkgo.Context("DNS Service", func() {
		ginkgo.It("should exist", func() {
			ExpectService(f.ClientSet, kubeSystemNamespace, dnsService)
		})
	})
})
