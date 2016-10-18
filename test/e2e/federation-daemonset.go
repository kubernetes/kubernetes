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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	FederatedDaemonSetName       = "federated-daemonset"
	FederatedDaemonSetTimeout    = 60 * time.Second
	FederatedDaemonSetMaxRetries = 3
)

// Create/delete daemonset api objects
var _ = framework.KubeDescribe("Federation daemonsets [Feature:Federation12]", func() {
	var clusters map[string]*cluster // All clusters, keyed by cluster name

	f := framework.NewDefaultFederatedFramework("federated-daemonset")

	Describe("DaemonSet objects", func() {

		BeforeEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, "", f)
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			unregisterClusters(clusters, f)
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			daemonset := createDaemonSetOrFail(f.FederationClientset_1_5, nsName)
			defer func() { // Cleanup
				By(fmt.Sprintf("Deleting daemonset %q in namespace %q", daemonset.Name, nsName))
				err := f.FederationClientset_1_5.Extensions().DaemonSets(nsName).Delete(daemonset.Name, &v1.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting daemonset %q in namespace %q", daemonset.Name, nsName)
			}()
			// wait for daemonset shards being created
			waitForDaemonSetShardsOrFail(nsName, daemonset, clusters)
			daemonset = updateDaemonSetOrFail(f.FederationClientset_1_5, nsName)
			waitForDaemonSetShardsUpdatedOrFail(nsName, daemonset, clusters)
		})
	})
})

func createDaemonSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.DaemonSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createDaemonSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	daemonset := &v1beta1.DaemonSet{
		ObjectMeta: v1.ObjectMeta{
			Name:      FederatedDaemonSetName,
			Namespace: namespace,
		},
		Spec: v1beta1.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
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

func updateDaemonSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.DaemonSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateDaemonSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	var newDaemonSet *v1beta1.DaemonSet
	for retryCount := 0; retryCount < FederatedDaemonSetMaxRetries; retryCount++ {
		daemonset, err := clientset.Extensions().DaemonSets(namespace).Get(FederatedDaemonSetName)
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
		clusterDaemonSet, err := clientset.Extensions().DaemonSets(namespace).Get(daemonset.Name)
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
		clusterDaemonSet, err := clientset.Extensions().DaemonSets(namespace).Get(daemonset.Name)
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
