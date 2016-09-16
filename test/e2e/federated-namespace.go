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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	namespacePrefix = "e2e-namespace-test-"
)

// Create/delete ingress api objects
var _ = framework.KubeDescribe("Federation namespace [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-namespace")
	clusterClientSet := make(map[string]*release_1_3.Clientset)

	Describe("Namespace objects", func() {
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			clusters := buildClustersOrFail_14(f)
			for _, cluster := range clusters {
				if _, found := clusterClientSet[cluster.Name]; !found {
					clientset := createClientsetForCluster(*cluster, 1, "e2e-test")
					clusterClientSet[cluster.Name] = clientset
				}
			}
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)
			deleteAllTestNamespaces(
				f.FederationClientset_1_4.Core().Namespaces().List,
				f.FederationClientset_1_4.Core().Namespaces().Delete)
			for _, clientset := range clusterClientSet {
				deleteAllTestNamespaces(
					clientset.Core().Namespaces().List,
					clientset.Core().Namespaces().Delete)
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			ns := api_v1.Namespace{
				ObjectMeta: api_v1.ObjectMeta{
					Name: api.SimpleNameGenerator.GenerateName(namespacePrefix),
				},
			}
			By(fmt.Sprintf("Creating namespace %s", ns.Name))
			_, err := f.FederationClientset_1_4.Core().Namespaces().Create(&ns)
			framework.ExpectNoError(err, "Failed to create namespace %s", ns.Name)

			// Check subclusters if the namespace was created there.
			err = wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
				for _, client := range clusterClientSet {
					_, err := client.Core().Namespaces().Get(ns.Name)
					if err != nil && !errors.IsNotFound(err) {
						return false, err
					}
					if err != nil {
						return false, nil
					}
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Not all namespaces created")

			deleteAllTestNamespaces(
				f.FederationClientset_1_4.Core().Namespaces().List,
				f.FederationClientset_1_4.Core().Namespaces().Delete)
		})
	})
})

func deleteAllTestNamespaces(lister func(api.ListOptions) (*api_v1.NamespaceList, error), deleter func(string, *api.DeleteOptions) error) {
	list, err := lister(api.ListOptions{})
	if err != nil {
		framework.Failf("Failed to get all namespaes: %v", err)
		return
	}
	for _, namespace := range list.Items {
		if strings.HasPrefix(namespace.Name, namespacePrefix) {
			err := deleter(namespace.Name, &api.DeleteOptions{})
			if err != nil {
				framework.Failf("Failed to set %s for deletion: %v", namespace.Name, err)
			}
		}
	}
	waitForNoTestNamespaces(lister)
}

func waitForNoTestNamespaces(lister func(api.ListOptions) (*api_v1.NamespaceList, error)) {
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		list, err := lister(api.ListOptions{})
		if err != nil {
			return false, err
		}
		for _, namespace := range list.Items {
			if strings.HasPrefix(namespace.Name, namespacePrefix) {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		framework.Failf("Namespaces not deleted: %v", err)
	}
}
