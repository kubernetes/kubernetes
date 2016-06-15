/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	federationapi "k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

// Create/delete cluster api objects
var _ = framework.KubeDescribe("Federation apiserver [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-cluster")

	AfterEach(func() {
		framework.SkipUnlessFederated(f.Client)

		// Delete registered clusters.
		// This is if a test failed, it should not affect other tests.
		clusterList, err := f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		for _, cluster := range clusterList.Items {
			err := f.FederationClientset.Federation().Clusters().Delete(cluster.Name, &api.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("should allow creation and deletion of cluster api objects", func() {
		framework.SkipUnlessFederated(f.Client)

		contexts := f.GetUnderlyingFederatedContexts()

		for _, context := range contexts {
			framework.Logf("Creating cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
			cluster := federationapi.Cluster{
				ObjectMeta: api.ObjectMeta{
					Name: context.Name,
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: context.Cluster.Cluster.Server,
						},
					},
					SecretRef: &api.LocalObjectReference{
						// Note: Name must correlate with federation build script secret name,
						//       which currently matches the cluster name.
						//       See federation/cluster/common.sh:132
						Name: context.Name,
					},
				},
			}
			_, err := f.FederationClientset.Federation().Clusters().Create(&cluster)
			framework.ExpectNoError(err, fmt.Sprintf("creating cluster: %+v", err))
		}

		for _, context := range contexts {
			c, err := f.FederationClientset.Federation().Clusters().Get(context.Name)
			framework.ExpectNoError(err, fmt.Sprintf("get cluster: %+v", err))
			if c.ObjectMeta.Name != context.Name {
				framework.Failf("cluster name does not match input context: actual=%+v, expected=%+v", c, context)
			}
			err = isReady(context.Name, f.FederationClientset)
			framework.ExpectNoError(err, fmt.Sprintf("unexpected error in verifying if cluster %s is ready: %+v", context.Name, err))
		}

		// Verify that deletion works.
		for _, context := range contexts {
			framework.Logf("Deleting cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
			err := f.FederationClientset.Federation().Clusters().Delete(context.Name, &api.DeleteOptions{})
			framework.ExpectNoError(err, fmt.Sprintf("unexpected error in deleting cluster %s: %+v", context.Name, err))
		}

		// There should not be any remaining cluster.
		clusterList, err := f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		if len(clusterList.Items) != 0 {
			framework.Failf("there should not have been any remaining clusters. Found: %+v", clusterList)
		}
	})
})

// Verify that the cluster is marked ready.
func isReady(clusterName string, clientset *federation_internalclientset.Clientset) error {
	return wait.PollImmediate(time.Second, 5*time.Minute, func() (bool, error) {
		c, err := clientset.Federation().Clusters().Get(clusterName)
		if err != nil {
			return false, err
		}
		for _, condition := range c.Status.Conditions {
			if condition.Type == federationapi.ClusterReady && condition.Status == api.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
}
