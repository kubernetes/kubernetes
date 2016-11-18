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
	"os"
	"strings"
	"time"

	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	namespacePrefix = "e2e-namespace-test-"
	eventNamePrefix = "e2e-namespace-test-event-"
)

// Create/delete ingress api objects
var _ = framework.KubeDescribe("Federation namespace [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-namespace")

	Describe("Namespace objects", func() {
		var federationName string
		var clusters map[string]*cluster // All clusters, keyed by cluster name

		BeforeEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)

			// TODO: Federation API server should be able to answer this.
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}

			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, federationName, f)
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			deleteAllTestNamespaces(nil,
				f.FederationClientset_1_5.Core().Namespaces().List,
				f.FederationClientset_1_5.Core().Namespaces().Delete)
			for _, cluster := range clusters {
				deleteAllTestNamespaces(nil,
					cluster.Core().Namespaces().List,
					cluster.Core().Namespaces().Delete)
			}
			unregisterClusters(clusters, f)
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			nsName := createNamespace(f.FederationClientset_1_5.Core().Namespaces())

			By(fmt.Sprintf("Deleting namespace %s", nsName))
			deleteAllTestNamespaces(nil,
				f.FederationClientset_1_5.Core().Namespaces().List,
				f.FederationClientset_1_5.Core().Namespaces().Delete)
			By(fmt.Sprintf("Verified that deletion succeeded"))
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			orphanDependents := false
			verifyNsCascadingDeletion(f.FederationClientset_1_5.Core().Namespaces(), clusters, &orphanDependents)
			By(fmt.Sprintf("Verified that namespaces were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			orphanDependents := true
			verifyNsCascadingDeletion(f.FederationClientset_1_5.Core().Namespaces(), clusters, &orphanDependents)
			By(fmt.Sprintf("Verified that namespaces were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			verifyNsCascadingDeletion(f.FederationClientset_1_5.Core().Namespaces(), clusters, nil)
			By(fmt.Sprintf("Verified that namespaces were not deleted from underlying clusters"))
		})

		It("all resources in the namespace should be deleted when namespace is deleted", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			nsName := createNamespace(f.FederationClientset_1_5.Core().Namespaces())

			// Create resources in the namespace.
			event := api_v1.Event{
				ObjectMeta: api_v1.ObjectMeta{
					Name:      api.SimpleNameGenerator.GenerateName(eventNamePrefix),
					Namespace: nsName,
				},
				InvolvedObject: api_v1.ObjectReference{
					Kind:      "Pod",
					Namespace: nsName,
					Name:      "sample-pod",
				},
			}
			By(fmt.Sprintf("Creating event %s in namespace %s", event.Name, nsName))
			_, err := f.FederationClientset_1_5.Core().Events(nsName).Create(&event)
			if err != nil {
				framework.Failf("Failed to create event %v in namespace %s, err: %s", event, nsName, err)
			}

			By(fmt.Sprintf("Deleting namespace %s", nsName))
			deleteAllTestNamespaces(nil,
				f.FederationClientset_1_5.Core().Namespaces().List,
				f.FederationClientset_1_5.Core().Namespaces().Delete)

			By(fmt.Sprintf("Verify that event %s was deleted as well", event.Name))
			latestEvent, err := f.FederationClientset_1_5.Core().Events(nsName).Get(event.Name)
			if !errors.IsNotFound(err) {
				framework.Failf("Event %s should have been deleted. Found: %v", event.Name, latestEvent)
			}
			By(fmt.Sprintf("Verified that deletion succeeded"))
		})
	})
})

// verifyNsCascadingDeletion verifies that namespaces are deleted from
// underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyNsCascadingDeletion(nsClient clientset.NamespaceInterface, clusters map[string]*cluster, orphanDependents *bool) {
	nsName := createNamespace(nsClient)
	// Check subclusters if the namespace was created there.
	By(fmt.Sprintf("Waiting for namespace %s to be created in all underlying clusters", nsName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Core().Namespaces().Get(nsName)
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

	By(fmt.Sprintf("Deleting namespace %s", nsName))
	deleteAllTestNamespaces(orphanDependents, nsClient.List, nsClient.Delete)

	By(fmt.Sprintf("Verifying namespaces %s in underlying clusters", nsName))
	errMessages := []string{}
	// namespace should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Core().Namespaces().Get(nsName)
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for namespace %s in cluster %s, expected namespace to exist", nsName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for namespace %s in cluster %s, got error: %v", nsName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func createNamespace(nsClient clientset.NamespaceInterface) string {
	ns := api_v1.Namespace{
		ObjectMeta: api_v1.ObjectMeta{
			Name: api.SimpleNameGenerator.GenerateName(namespacePrefix),
		},
	}
	By(fmt.Sprintf("Creating namespace %s", ns.Name))
	_, err := nsClient.Create(&ns)
	framework.ExpectNoError(err, "Failed to create namespace %s", ns.Name)
	By(fmt.Sprintf("Created namespace %s", ns.Name))
	return ns.Name
}

func deleteAllTestNamespaces(orphanDependents *bool, lister func(api_v1.ListOptions) (*api_v1.NamespaceList, error), deleter func(string, *api_v1.DeleteOptions) error) {
	list, err := lister(api_v1.ListOptions{})
	if err != nil {
		framework.Failf("Failed to get all namespaes: %v", err)
		return
	}
	for _, namespace := range list.Items {
		if strings.HasPrefix(namespace.Name, namespacePrefix) {
			By(fmt.Sprintf("Deleting ns: %s, found by listing", namespace.Name))
			err := deleter(namespace.Name, &api_v1.DeleteOptions{OrphanDependents: orphanDependents})
			if err != nil {
				framework.Failf("Failed to set %s for deletion: %v", namespace.Name, err)
			}
		}
	}
	waitForNoTestNamespaces(lister)
}

func waitForNoTestNamespaces(lister func(api_v1.ListOptions) (*api_v1.NamespaceList, error)) {
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		list, err := lister(api_v1.ListOptions{})
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
