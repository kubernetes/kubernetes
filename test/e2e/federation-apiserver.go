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

package e2e

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

// Create/delete cluster api objects
var _ = framework.KubeDescribe("Federation apiserver [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-cluster")

	Describe("Cluster objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)

			// Delete registered clusters.
			// This is if a test failed, it should not affect other tests.
			clusterList, err := f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, cluster := range clusterList.Items {
				err := f.FederationClientset_1_4.Federation().Clusters().Delete(cluster.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			contexts := f.GetUnderlyingFederatedContexts()

			framework.Logf("Creating %d cluster objects", len(contexts))
			for _, context := range contexts {
				createClusterObjectOrFail(f, &context)
			}

			framework.Logf("Checking that %d clusters are Ready", len(contexts))
			for _, context := range contexts {
				clusterIsReadyOrFail(f, &context)
			}
			framework.Logf("%d clusters are Ready", len(contexts))

			// Verify that deletion works.
			framework.Logf("Deleting %d clusters", len(contexts))
			for _, context := range contexts {
				framework.Logf("Deleting cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
				err := f.FederationClientset_1_4.Federation().Clusters().Delete(context.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("unexpected error in deleting cluster %s: %+v", context.Name, err))
				framework.Logf("Successfully deleted cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
			}

			// There should not be any remaining cluster.
			framework.Logf("Verifying that zero clusters remain")
			clusterList, err := f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			if len(clusterList.Items) != 0 {
				framework.Failf("there should not have been any remaining clusters. Found: %+v", clusterList)
			}
			framework.Logf("Verified that zero clusters remain")
		})
	})
	Describe("Admission control", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)
		})

		It("should not be able to create resources if namespace does not exist", func() {
			framework.SkipUnlessFederated(f.Client)

			// Creating a service in a non-existing namespace should fail.
			svcNamespace := "federation-admission-test-ns"
			svcName := "myns"
			clientset := f.FederationClientset_1_4
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
		ObjectMeta: v1.ObjectMeta{
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

// Verify that the cluster is marked ready.
func isReady(clusterName string, clientset *federation_release_1_4.Clientset) error {
	return wait.PollImmediate(time.Second, 5*time.Minute, func() (bool, error) {
		c, err := clientset.Federation().Clusters().Get(clusterName)
		if err != nil {
			return false, err
		}
		for _, condition := range c.Status.Conditions {
			if condition.Type == federationapi.ClusterReady && condition.Status == v1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
}
