/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

const (
	FederationIngressName = "federation-ingress"
)

// Create/delete ingress api objects
var _ = framework.KubeDescribe("Federation ingresses [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-ingress")

	Describe("Ingress objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)

			// Delete registered ingresses.
			ingressList, err := f.FederationClientset_1_4.Extensions().Ingresses(f.Namespace.Name).List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, ingress := range ingressList.Items {
				err := f.FederationClientset_1_4.Extensions().Ingresses(f.Namespace.Name).Delete(ingress.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)
			ingress := createIngressOrFail(f.FederationClientset_1_4, f.Namespace.Name)
			By(fmt.Sprintf("Creation of ingress %q in namespace %q succeeded.  Deleting ingress.", ingress.Name, f.Namespace.Name))
			// Cleanup
			err := f.FederationClientset_1_4.Extensions().Ingresses(f.Namespace.Name).Delete(ingress.Name, &api.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting ingress %q in namespace %q", ingress.Name, ingress.Namespace)
			By(fmt.Sprintf("Deletion of ingress %q in namespace %q succeeded.", ingress.Name, f.Namespace.Name))
		})

	})
})

func createIngressOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1beta1.Ingress {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createIngressOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federated ingress %q in namespace %q", FederationIngressName, namespace))

	ingress := &v1beta1.Ingress{
		ObjectMeta: v1.ObjectMeta{
			Name: FederationIngressName,
		},
		Spec: v1beta1.IngressSpec{
			Backend: &v1beta1.IngressBackend{
				ServiceName: "testservice",
				ServicePort: intstr.FromInt(80),
			},
		},
	}

	_, err := clientset.Extensions().Ingresses(namespace).Create(ingress)
	framework.ExpectNoError(err, "Creating ingress %q in namespace %q", ingress.Name, namespace)
	By(fmt.Sprintf("Successfully created federated ingress %q in namespace %q", FederationIngressName, namespace))
	return ingress
}
