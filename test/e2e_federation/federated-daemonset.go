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

package e2e_federation

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
)

const (
	FederatedDaemonSetName       = "federated-daemonset"
	FederatedDaemonSetTimeout    = 60 * time.Second
	FederatedDaemonSetMaxRetries = 3
)

// Create/delete daemonset api objects
var _ = framework.KubeDescribe("Federation daemonsets [Feature:Federation]", func() {
	var clusters map[string]*cluster // All clusters, keyed by cluster name

	f := fedframework.NewDefaultFederatedFramework("federated-daemonset")

	Describe("DaemonSet objects", func() {

		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters, _ = getRegisteredClusters(UserAgentName, f)
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			// Delete all daemonsets.
			nsName := f.FederationNamespace.Name
			deleteAllDaemonSetsOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			daemonset := createDaemonSetOrFail(f.FederationClientset, nsName)
			defer func() { // Cleanup
				By(fmt.Sprintf("Deleting daemonset %q in namespace %q", daemonset.Name, nsName))
				err := f.FederationClientset.Extensions().DaemonSets(nsName).Delete(daemonset.Name, &metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting daemonset %q in namespace %q", daemonset.Name, nsName)
			}()
			// wait for daemonset shards being created
			waitForDaemonSetShardsOrFail(nsName, daemonset, clusters)
			daemonset = updateDaemonSetOrFail(f.FederationClientset, nsName)
			waitForDaemonSetShardsUpdatedOrFail(nsName, daemonset, clusters)
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForDS(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that daemonsets were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForDS(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that daemonsets were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForDS(f.FederationClientset, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that daemonsets were not deleted from underlying clusters"))
		})
	})
})

// deleteAllDaemonSetsOrFail deletes all DaemonSets in the given namespace name.
func deleteAllDaemonSetsOrFail(clientset *fedclientset.Clientset, nsName string) {
	DaemonSetList, err := clientset.Extensions().DaemonSets(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, daemonSet := range DaemonSetList.Items {
		deleteDaemonSetOrFail(clientset, nsName, daemonSet.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForDS verifies that daemonsets are deleted from
// underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForDS(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	daemonset := createDaemonSetOrFail(clientset, nsName)
	daemonsetName := daemonset.Name
	// Check subclusters if the daemonset was created there.
	By(fmt.Sprintf("Waiting for daemonset %s to be created in all underlying clusters", daemonsetName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Extensions().DaemonSets(nsName).Get(daemonsetName, metav1.GetOptions{})
			if err != nil && errors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all daemonsets created")

	By(fmt.Sprintf("Deleting daemonset %s", daemonsetName))
	deleteDaemonSetOrFail(clientset, nsName, daemonsetName, orphanDependents)

	By(fmt.Sprintf("Verifying daemonsets %s in underlying clusters", daemonsetName))
	errMessages := []string{}
	// daemon set should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Extensions().DaemonSets(nsName).Get(daemonsetName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for daemonset %s in cluster %s, expected daemonset to exist", daemonsetName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for daemonset %s in cluster %s, got error: %v", daemonsetName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func createDaemonSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.DaemonSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createDaemonSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	daemonset := &v1beta1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      FederatedDaemonSetName,
			Namespace: namespace,
		},
		Spec: v1beta1.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"aaa": "bbb"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container1",
							Image: "gcr.io/google_containers/serve_hostname:v1.4",
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating daemonset %q in namespace %q", daemonset.Name, namespace))
	_, err := clientset.Extensions().DaemonSets(namespace).Create(daemonset)
	framework.ExpectNoError(err, "Failed to create daemonset %s", daemonset.Name)
	By(fmt.Sprintf("Successfully created federated daemonset %q in namespace %q", FederatedDaemonSetName, namespace))
	return daemonset
}

func deleteDaemonSetOrFail(clientset *fedclientset.Clientset, nsName string, daemonsetName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting daemonset %q in namespace %q", daemonsetName, nsName))
	err := clientset.Extensions().DaemonSets(nsName).Delete(daemonsetName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting daemonset %q in namespace %q", daemonsetName, nsName)
	}

	// Wait for the daemonset to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Extensions().DaemonSets(nsName).Get(daemonsetName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting daemonset %s: %v", daemonsetName, err)
	}
}

func updateDaemonSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.DaemonSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateDaemonSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	var newDaemonSet *v1beta1.DaemonSet
	for retryCount := 0; retryCount < FederatedDaemonSetMaxRetries; retryCount++ {
		daemonset, err := clientset.Extensions().DaemonSets(namespace).Get(FederatedDaemonSetName, metav1.GetOptions{})
		if err != nil {
			framework.Failf("failed to get daemonset %q: %v", FederatedDaemonSetName, err)
		}

		// Update one of the data in the daemonset.
		daemonset.Annotations = map[string]string{"ccc": "ddd"}
		newDaemonSet, err = clientset.Extensions().DaemonSets(namespace).Update(daemonset)
		if err == nil {
			return newDaemonSet
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update daemonset %q: %v", FederatedDaemonSetName, err)
		}
	}
	framework.Failf("too many retries updating daemonset %q", FederatedDaemonSetName)
	return newDaemonSet
}

func waitForDaemonSetShardsOrFail(namespace string, daemonset *v1beta1.DaemonSet, clusters map[string]*cluster) {
	framework.Logf("Waiting for daemonset %q in %d clusters", daemonset.Name, len(clusters))
	for _, c := range clusters {
		waitForDaemonSetOrFail(c.Clientset, namespace, daemonset, true, FederatedDaemonSetTimeout)
	}
}

func waitForDaemonSetOrFail(clientset *kubeclientset.Clientset, namespace string, daemonset *v1beta1.DaemonSet, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated daemonset shard of daemonset %q in namespace %q from cluster", daemonset.Name, namespace))
	var clusterDaemonSet *v1beta1.DaemonSet
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterDaemonSet, err := clientset.Extensions().DaemonSets(namespace).Get(daemonset.Name, metav1.GetOptions{})
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated daemonset %q in namespace %q in cluster is absent", daemonset.Name, namespace))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated daemonset %q in namespace %q in cluster is present", daemonset.Name, namespace))
			return true, nil // Success
		}
		By(fmt.Sprintf("DaemonSet %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", daemonset.Name, namespace, clusterDaemonSet != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify daemonset %q in namespace %q in cluster: Present=%v", daemonset.Name, namespace, present)

	if present && clusterDaemonSet != nil {
		Expect(util.ObjectMetaAndSpecEquivalent(clusterDaemonSet, daemonset))
	}
}

func waitForDaemonSetShardsUpdatedOrFail(namespace string, daemonset *v1beta1.DaemonSet, clusters map[string]*cluster) {
	framework.Logf("Waiting for daemonset %q in %d clusters", daemonset.Name, len(clusters))
	for _, c := range clusters {
		waitForDaemonSetUpdateOrFail(c.Clientset, namespace, daemonset, FederatedDaemonSetTimeout)
	}
}

func waitForDaemonSetUpdateOrFail(clientset *kubeclientset.Clientset, namespace string, daemonset *v1beta1.DaemonSet, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated daemonset shard of daemonset %q in namespace %q from cluster", daemonset.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterDaemonSet, err := clientset.Extensions().DaemonSets(namespace).Get(daemonset.Name, metav1.GetOptions{})
		if err == nil { // We want it present, and the Get succeeded, so we're all good.
			if util.ObjectMetaAndSpecEquivalent(clusterDaemonSet, daemonset) {
				By(fmt.Sprintf("Success: shard of federated daemonset %q in namespace %q in cluster is updated", daemonset.Name, namespace))
				return true, nil
			} else {
				By(fmt.Sprintf("Expected equal daemonsets. expected: %+v\nactual: %+v", *daemonset, *clusterDaemonSet))
			}
			By(fmt.Sprintf("DaemonSet %q in namespace %q in cluster, waiting for daemonset being updated, trying again in %s (err=%v)", daemonset.Name, namespace, framework.Poll, err))
			return false, nil
		}
		By(fmt.Sprintf("DaemonSet %q in namespace %q in cluster, waiting for being updated, trying again in %s (err=%v)", daemonset.Name, namespace, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify daemonset %q in namespace %q in cluster", daemonset.Name, namespace)
}
