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

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1"
	fedframework "k8s.io/kubernetes/federation/test/e2e/framework"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	eventNamePrefix      = "e2e-namespace-test-event-"
	namespacePrefix      = "e2e-namespace-test-"
	replicaSetNamePrefix = "e2e-namespace-test-rs-"
)

// Create/delete ingress api objects
var _ = framework.KubeDescribe("Federation namespace [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-namespace")

	Describe("Namespace objects", func() {
		var clusters fedframework.ClusterSlice

		var nsName string

		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters = f.GetRegisteredClusters()
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			deleteNamespace(nil, nsName,
				f.FederationClientset.Core().Namespaces().Get,
				f.FederationClientset.Core().Namespaces().Delete)
			for _, cluster := range clusters {
				deleteNamespace(nil, nsName,
					cluster.CoreV1().Namespaces().Get,
					cluster.CoreV1().Namespaces().Delete)
			}
		})

		// See https://github.com/kubernetes/kubernetes/issues/38225
		It("deletes replicasets in the namespace when the namespace is deleted", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName = createNamespace(f.FederationClientset.Core().Namespaces())
			rsName := k8s_api_v1.SimpleNameGenerator.GenerateName(replicaSetNamePrefix)
			replicaCount := int32(2)
			rs := &v1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      rsName,
					Namespace: nsName,
				},
				Spec: v1beta1.ReplicaSetSpec{
					Replicas: &replicaCount,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"name": "myrs"},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"name": "myrs"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			}

			By(fmt.Sprintf("Creating replicaset %s in namespace %s", rsName, nsName))
			_, err := f.FederationClientset.Extensions().ReplicaSets(nsName).Create(rs)
			if err != nil {
				framework.Failf("Failed to create replicaset %v in namespace %s, err: %s", rs, nsName, err)
			}

			By(fmt.Sprintf("Deleting namespace %s", nsName))
			deleteNamespace(nil, nsName,
				f.FederationClientset.Core().Namespaces().Get,
				f.FederationClientset.Core().Namespaces().Delete)

			By(fmt.Sprintf("Verify that replicaset %s was deleted as well", rsName))

			waitForReplicaSetToBeDeletedOrFail(f.FederationClientset, nsName, rsName)
		})

		It("all resources in the namespace should be deleted when namespace is deleted", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName = createNamespace(f.FederationClientset.Core().Namespaces())

			// Create resources in the namespace.
			event := v1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      k8s_api_v1.SimpleNameGenerator.GenerateName(eventNamePrefix),
					Namespace: nsName,
				},
				InvolvedObject: v1.ObjectReference{
					Kind:      "Pod",
					Namespace: nsName,
					Name:      "sample-pod",
				},
			}
			By(fmt.Sprintf("Creating event %s in namespace %s", event.Name, nsName))
			_, err := f.FederationClientset.Core().Events(nsName).Create(&event)
			if err != nil {
				framework.Failf("Failed to create event %v in namespace %s, err: %s", event, nsName, err)
			}

			By(fmt.Sprintf("Deleting namespace %s", nsName))
			deleteNamespace(nil, nsName,
				f.FederationClientset.Core().Namespaces().Get,
				f.FederationClientset.Core().Namespaces().Delete)

			By(fmt.Sprintf("Verify that event %s was deleted as well", event.Name))
			latestEvent, err := f.FederationClientset.Core().Events(nsName).Get(event.Name, metav1.GetOptions{})
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
func verifyNsCascadingDeletion(nsClient clientset.NamespaceInterface, clusters fedframework.ClusterSlice, orphanDependents *bool) string {
	nsName := createNamespace(nsClient)
	// Check subclusters if the namespace was created there.
	By(fmt.Sprintf("Waiting for namespace %s to be created in all underlying clusters", nsName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.CoreV1().Namespaces().Get(nsName, metav1.GetOptions{})
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
	deleteNamespace(orphanDependents, nsName, nsClient.Get, nsClient.Delete)

	By(fmt.Sprintf("Verifying namespaces %s in underlying clusters", nsName))
	errMessages := []string{}
	// namespace should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for _, cluster := range clusters {
		clusterName := cluster.Name
		_, err := cluster.CoreV1().Namespaces().Get(nsName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for namespace %s in cluster %s, expected namespace to exist", nsName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for namespace %s in cluster %s, got error: %v", nsName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
	return nsName
}

func createNamespace(nsClient clientset.NamespaceInterface) string {
	ns := v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: k8s_api_v1.SimpleNameGenerator.GenerateName(namespacePrefix),
		},
	}
	By(fmt.Sprintf("Creating namespace %s", ns.Name))
	_, err := nsClient.Create(&ns)
	framework.ExpectNoError(err, "Failed to create namespace %s", ns.Name)
	By(fmt.Sprintf("Created namespace %s", ns.Name))
	return ns.Name
}

func deleteNamespace(orphanDependents *bool, namespace string, getter func(name string, options metav1.GetOptions) (*v1.Namespace, error), deleter func(string, *metav1.DeleteOptions) error) {
	By(fmt.Sprintf("Deleting namespace: %s", namespace))
	err := deleter(namespace, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if errors.IsNotFound(err) {
		return
	} else if err != nil {
		framework.Failf("Failed to set %s for deletion: %v", namespace, err)
	}
	waitForNamespaceDeletion(namespace, getter)
}

func waitForNamespaceDeletion(namespace string, getter func(name string, options metav1.GetOptions) (*v1.Namespace, error)) {
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		_, err := getter(namespace, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		} else if err != nil {
			return false, err
		}
		return false, nil
	})
	if err != nil {
		framework.Failf("Namespaces not deleted: %v", err)
	}
}
