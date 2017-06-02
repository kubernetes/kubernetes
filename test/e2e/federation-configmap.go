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
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	configmapNamePrefix       = "e2e-configmap-test-"
	FederatedConfigMapTimeout = 60 * time.Second
)

// Create/delete configmap api objects
var _ = framework.KubeDescribe("Federation configmaps [Feature:Federation]", func() {
	var clusters map[string]*cluster // All clusters, keyed by cluster name

	f := framework.NewDefaultFederatedFramework("federated-configmap")

	Describe("ConfigMap objects", func() {

		BeforeEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, "", f)
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			// Delete all configmaps.
			nsName := f.FederationNamespace.Name
			deleteAllConfigMapsOrFail(f.FederationClientset_1_5, nsName)
			unregisterClusters(clusters, f)

		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			configmap := createConfigMapOrFail(f.FederationClientset_1_5, nsName)
			// wait for configmap shards being created
			waitForConfigMapShardsOrFail(nsName, configmap, clusters)
			configmap = updateConfigMapOrFail(f.FederationClientset_1_5, nsName, configmap.Name)
			waitForConfigMapShardsUpdatedOrFail(nsName, configmap, clusters)
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForConfigMap(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that configmaps were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForConfigMap(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that configmaps were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForConfigMap(f.FederationClientset_1_5, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that configmaps were not deleted from underlying clusters"))
		})
	})
})

// deleteAllConfigMapsOrFail deletes all configmaps in the given namespace name.
func deleteAllConfigMapsOrFail(clientset *fedclientset.Clientset, nsName string) {
	ConfigMapList, err := clientset.Core().ConfigMaps(nsName).List(v1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, ConfigMap := range ConfigMapList.Items {
		deleteConfigMapOrFail(clientset, nsName, ConfigMap.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForConfigMap verifies that configmaps are deleted from
// underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForConfigMap(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	configmap := createConfigMapOrFail(clientset, nsName)
	configmapName := configmap.Name
	// Check subclusters if the configmap was created there.
	By(fmt.Sprintf("Waiting for configmap %s to be created in all underlying clusters", configmapName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Core().ConfigMaps(nsName).Get(configmapName)
			if err != nil {
				if !errors.IsNotFound(err) {
					return false, err
				}
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all configmaps created")

	By(fmt.Sprintf("Deleting configmap %s", configmapName))
	deleteConfigMapOrFail(clientset, nsName, configmapName, orphanDependents)

	By(fmt.Sprintf("Verifying configmaps %s in underlying clusters", configmapName))
	errMessages := []string{}
	// configmap should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Core().ConfigMaps(nsName).Get(configmapName)
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for configmap %s in cluster %s, expected configmap to exist", configmapName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for configmap %s in cluster %s, got error: %v", configmapName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func createConfigMapOrFail(clientset *fedclientset.Clientset, nsName string) *v1.ConfigMap {
	if len(nsName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createConfigMapOrFail: namespace: %v", nsName))
	}

	configmap := &v1.ConfigMap{
		ObjectMeta: v1.ObjectMeta{
			Name:      v1.SimpleNameGenerator.GenerateName(configmapNamePrefix),
			Namespace: nsName,
		},
	}
	By(fmt.Sprintf("Creating configmap %q in namespace %q", configmap.Name, nsName))
	_, err := clientset.Core().ConfigMaps(nsName).Create(configmap)
	framework.ExpectNoError(err, "Failed to create configmap %s", configmap.Name)
	By(fmt.Sprintf("Successfully created federated configmap %q in namespace %q", configmap.Name, nsName))
	return configmap
}

func deleteConfigMapOrFail(clientset *fedclientset.Clientset, nsName string, configmapName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting configmap %q in namespace %q", configmapName, nsName))
	err := clientset.Core().ConfigMaps(nsName).Delete(configmapName, &v1.DeleteOptions{OrphanDependents: orphanDependents})
	framework.ExpectNoError(err, "Error deleting configmap %q in namespace %q", configmapName, nsName)

	// Wait for the configmap to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Core().ConfigMaps(nsName).Get(configmapName)
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting configmap %s: %v", configmapName, err)
	}
}

func updateConfigMapOrFail(clientset *fedclientset.Clientset, nsName string, configmapName string) *v1.ConfigMap {
	if clientset == nil || len(nsName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateConfigMapOrFail: clientset: %v, namespace: %v", clientset, nsName))
	}

	var newConfigMap *v1.ConfigMap
	for retryCount := 0; retryCount < MaxRetries; retryCount++ {
		configmap, err := clientset.Core().ConfigMaps(nsName).Get(configmapName)
		if err != nil {
			framework.Failf("failed to get configmap %q: %v", configmapName, err)
		}

		// Update one of the data in the configmap.
		configmap.Data = map[string]string{
			"key": "value",
		}
		newConfigMap, err = clientset.Core().ConfigMaps(nsName).Update(configmap)
		if err == nil {
			return newConfigMap
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update configmap %q: %v", configmapName, err)
		}
	}
	framework.Failf("too many retries updating configmap %q", configmapName)
	return newConfigMap
}

func waitForConfigMapShardsOrFail(nsName string, configmap *v1.ConfigMap, clusters map[string]*cluster) {
	framework.Logf("Waiting for configmap %q in %d clusters", configmap.Name, len(clusters))
	for _, c := range clusters {
		waitForConfigMapOrFail(c.Clientset, nsName, configmap, true, FederatedConfigMapTimeout)
	}
}

func waitForConfigMapOrFail(clientset *kubeclientset.Clientset, nsName string, configmap *v1.ConfigMap, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated configmap shard of configmap %q in namespace %q from cluster", configmap.Name, nsName))
	var clusterConfigMap *v1.ConfigMap
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterConfigMap, err := clientset.Core().ConfigMaps(nsName).Get(configmap.Name)
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated configmap %q in namespace %q in cluster is absent", configmap.Name, nsName))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated configmap %q in namespace %q in cluster is present", configmap.Name, nsName))
			return true, nil // Success
		}
		By(fmt.Sprintf("ConfigMap %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", configmap.Name, nsName, clusterConfigMap != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify configmap %q in namespace %q in cluster: Present=%v", configmap.Name, nsName, present)

	if present && clusterConfigMap != nil {
		Expect(util.ConfigMapEquivalent(clusterConfigMap, configmap))
	}
}

func waitForConfigMapShardsUpdatedOrFail(nsName string, configmap *v1.ConfigMap, clusters map[string]*cluster) {
	framework.Logf("Waiting for configmap %q in %d clusters", configmap.Name, len(clusters))
	for _, c := range clusters {
		waitForConfigMapUpdateOrFail(c.Clientset, nsName, configmap, FederatedConfigMapTimeout)
	}
}

func waitForConfigMapUpdateOrFail(clientset *kubeclientset.Clientset, nsName string, configmap *v1.ConfigMap, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated configmap shard of configmap %q in namespace %q from cluster", configmap.Name, nsName))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterConfigMap, err := clientset.Core().ConfigMaps(nsName).Get(configmap.Name)
		if err == nil { // We want it present, and the Get succeeded, so we're all good.
			if util.ConfigMapEquivalent(clusterConfigMap, configmap) {
				By(fmt.Sprintf("Success: shard of federated configmap %q in namespace %q in cluster is updated", configmap.Name, nsName))
				return true, nil
			} else {
				By(fmt.Sprintf("Expected equal configmaps. expected: %+v\nactual: %+v", *configmap, *clusterConfigMap))
			}
			By(fmt.Sprintf("ConfigMap %q in namespace %q in cluster, waiting for configmap being updated, trying again in %s (err=%v)", configmap.Name, nsName, framework.Poll, err))
			return false, nil
		}
		By(fmt.Sprintf("ConfigMap %q in namespace %q in cluster, waiting for being updated, trying again in %s (err=%v)", configmap.Name, nsName, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify configmap %q in namespace %q in cluster", configmap.Name, nsName)
}
