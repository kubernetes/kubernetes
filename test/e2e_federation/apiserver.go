/*
Copyright 2015 The Kubernetes Authors.

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

package e2e_federation

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
)

// Create/delete cluster api objects
var _ = framework.KubeDescribe("Federation apiserver [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-cluster")
	testClusterPrefix := "test"

	Describe("Cluster objects [Serial]", func() {
		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// Delete registered clusters.
			// This is if a test failed, it should not affect other tests.
			clusterList, err := f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{LabelSelector: "prefix=" + testClusterPrefix})
			Expect(err).NotTo(HaveOccurred())
			for _, cluster := range clusterList.Items {
				err := f.FederationClientset.Federation().Clusters().Delete(cluster.Name, &metav1.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			contexts := f.GetUnderlyingFederatedContexts()

			By(fmt.Sprintf("Creating %d cluster objects", len(contexts)))
			for _, context := range contexts {
				createClusterObjectOrFail(f, &context, testClusterPrefix)
			}

			By(fmt.Sprintf("Checking that %d clusters are ready", len(contexts)))
			for _, context := range contexts {
				fedframework.ClusterIsReadyOrFail(f, context.Name)
			}
			framework.Logf("%d clusters are Ready", len(contexts))

			// Verify that deletion works.
			framework.Logf("Deleting %d clusters", len(contexts))
			for _, context := range contexts {
				clusterName := testClusterPrefix + context.Name
				framework.Logf("Deleting cluster object: %s (%s, secret: %s)", clusterName, context.Cluster.Cluster.Server, context.Name)
				err := f.FederationClientset.Federation().Clusters().Delete(clusterName, &metav1.DeleteOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("unexpected error in deleting cluster %s: %+v", clusterName, err))
				framework.Logf("Successfully deleted cluster object: %s (%s, secret: %s)", clusterName, context.Cluster.Cluster.Server, context.Name)
			}

			// There should not be any remaining cluster.
			framework.Logf("Verifying that zero test clusters remain")
			clusterList, err := f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{LabelSelector: "prefix=" + testClusterPrefix})
			Expect(err).NotTo(HaveOccurred())
			if len(clusterList.Items) != 0 {
				framework.Failf("there should not have been any remaining clusters. Found: %+v", clusterList)
			}
			framework.Logf("Verified that zero clusters remain")
		})
	})

	Describe("Admission control [NoCluster]", func() {
		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
		})

		It("should not be able to create resources if namespace does not exist", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// Creating a service in a non-existing namespace should fail.
			svcNamespace := "federation-admission-test-ns"
			svcName := "myns"
			clientset := f.FederationClientset
			framework.Logf("Trying to create service %s in namespace %s, expect to get error", svcName, svcNamespace)
			if _, err := clientset.Core().Services(svcNamespace).Create(newService(svcName, svcNamespace)); err == nil {
				framework.Failf("Expected to get an error while creating a service in a non-existing namespace")
			}

			// Note: We have other tests that verify that we can create resources in existing namespaces, so we dont test it again here.
		})
	})
})

func newService(name, namespace string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{
					Port: 80,
				},
			},
		},
	}
}
