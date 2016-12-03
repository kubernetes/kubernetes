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

	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	"reflect"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/errors"
)

const (
	FederationReplicaSetName   = "federation-replicaset"
	FederatedReplicaSetTimeout = 120 * time.Second
)

// Create/delete replicaset api objects
var _ = framework.KubeDescribe("Federation replicasets [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-replicaset")

	Describe("ReplicaSet objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)

			// Delete all replicasets.
			nsName := f.FederationNamespace.Name
			deleteAllReplicaSetsOrFail(f.FederationClientset_1_5, nsName)
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			replicaset := createReplicaSetOrFail(f.FederationClientset_1_5, nsName)
			By(fmt.Sprintf("Creation of replicaset %q in namespace %q succeeded.  Deleting replicaset.", replicaset.Name, nsName))
			// Cleanup
			err := f.FederationClientset_1_5.Extensions().ReplicaSets(nsName).Delete(replicaset.Name, &v1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting replicaset %q in namespace %q", replicaset.Name, replicaset.Namespace)
			By(fmt.Sprintf("Deletion of replicaset %q in namespace %q succeeded.", replicaset.Name, nsName))
		})

	})

	// e2e cases for federated replicaset controller
	Describe("Federated ReplicaSet", func() {
		var (
			clusters       map[string]*cluster
			federationName string
		)
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}
			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, federationName, f)
		})

		AfterEach(func() {
			// Delete all replicasets.
			nsName := f.FederationNamespace.Name
			deleteAllReplicaSetsOrFail(f.FederationClientset_1_5, nsName)
			unregisterClusters(clusters, f)
		})

		It("should create and update matching replicasets in underling clusters", func() {
			nsName := f.FederationNamespace.Name
			rs := createReplicaSetOrFail(f.FederationClientset_1_5, nsName)
			defer func() {
				// cleanup. deletion of replicasets is not supported for underlying clusters
				By(fmt.Sprintf("Preparing replicaset %q/%q for deletion by setting replicas to zero", nsName, rs.Name))
				replicas := int32(0)
				rs.Spec.Replicas = &replicas
				f.FederationClientset_1_5.ReplicaSets(nsName).Update(rs)
				waitForReplicaSetOrFail(f.FederationClientset_1_5, nsName, rs.Name, clusters)
				f.FederationClientset_1_5.ReplicaSets(nsName).Delete(rs.Name, &v1.DeleteOptions{})
			}()

			waitForReplicaSetOrFail(f.FederationClientset_1_5, nsName, rs.Name, clusters)
			By(fmt.Sprintf("Successfuly created and synced replicaset %q/%q to clusters", nsName, rs.Name))
			updateReplicaSetOrFail(f.FederationClientset_1_5, nsName)
			waitForReplicaSetOrFail(f.FederationClientset_1_5, nsName, rs.Name, clusters)
			By(fmt.Sprintf("Successfuly updated and synced replicaset %q/%q to clusters", nsName, rs.Name))
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForReplicaSet(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that replica sets were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForReplicaSet(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that replica sets were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForReplicaSet(f.FederationClientset_1_5, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that replica sets were not deleted from underlying clusters"))
		})
	})
})

// deleteAllReplicaSetsOrFail deletes all replicasets in the given namespace name.
func deleteAllReplicaSetsOrFail(clientset *fedclientset.Clientset, nsName string) {
	replicasetList, err := clientset.Extensions().ReplicaSets(nsName).List(v1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, replicaset := range replicasetList.Items {
		deleteReplicaSetOrFail(clientset, nsName, replicaset.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForReplicaSet verifies that replicaSets are deleted
// from underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForReplicaSet(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	replicaSet := createReplicaSetOrFail(clientset, nsName)
	replicaSetName := replicaSet.Name
	// Check subclusters if the replicaSet was created there.
	By(fmt.Sprintf("Waiting for replica sets %s to be created in all underlying clusters", replicaSetName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Extensions().ReplicaSets(nsName).Get(replicaSetName)
			if err != nil && errors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all replica sets created")

	By(fmt.Sprintf("Deleting replica set %s", replicaSetName))
	deleteReplicaSetOrFail(clientset, nsName, replicaSetName, orphanDependents)

	By(fmt.Sprintf("Verifying replica sets %s in underlying clusters", replicaSetName))
	errMessages := []string{}
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Extensions().ReplicaSets(nsName).Get(replicaSetName)
		if (orphanDependents == nil || *orphanDependents == true) && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for replica set %s in cluster %s, expected replica set to exist", replicaSetName, clusterName))
		} else if (orphanDependents != nil && *orphanDependents == false) && (err == nil || !errors.IsNotFound(err)) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for replica set %s in cluster %s, got error: %v", replicaSetName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func waitForReplicaSetOrFail(c *fedclientset.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster) {
	err := waitForReplicaSet(c, namespace, replicaSetName, clusters)
	framework.ExpectNoError(err, "Failed to verify replica set %q/%q, err: %v", namespace, replicaSetName, err)
}

func waitForReplicaSet(c *fedclientset.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster) error {
	err := wait.Poll(10*time.Second, FederatedReplicaSetTimeout, func() (bool, error) {
		frs, err := c.ReplicaSets(namespace).Get(replicaSetName)
		if err != nil {
			return false, err
		}
		specReplicas, statusReplicas := int32(0), int32(0)
		for _, cluster := range clusters {
			rs, err := cluster.ReplicaSets(namespace).Get(replicaSetName)
			if err != nil && !errors.IsNotFound(err) {
				By(fmt.Sprintf("Failed getting replicaset: %q/%q/%q, err: %v", cluster.name, namespace, replicaSetName, err))
				return false, err
			}
			if err == nil {
				if !equivalentReplicaSet(frs, rs) {
					By(fmt.Sprintf("Replicaset meta or spec not match for cluster %q:\n    federation: %v\n    cluster: %v", cluster.name, frs, rs))
					return false, nil
				}
				specReplicas += *rs.Spec.Replicas
				statusReplicas += rs.Status.Replicas
			}
		}
		if statusReplicas == frs.Status.Replicas && specReplicas >= *frs.Spec.Replicas {
			return true, nil
		}
		By(fmt.Sprintf("Replicas not match, federation replicas: %v/%v, clusters replicas: %v/%v\n", *frs.Spec.Replicas, frs.Status.Replicas, specReplicas, statusReplicas))
		return false, nil
	})

	return err
}

func equivalentReplicaSet(fedReplicaSet, localReplicaSet *v1beta1.ReplicaSet) bool {
	localReplicaSetSpec := localReplicaSet.Spec
	localReplicaSetSpec.Replicas = fedReplicaSet.Spec.Replicas
	return fedutil.ObjectMetaEquivalent(fedReplicaSet.ObjectMeta, localReplicaSet.ObjectMeta) &&
		reflect.DeepEqual(fedReplicaSet.Spec, localReplicaSetSpec)
}

func createReplicaSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.ReplicaSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createReplicaSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))

	replicaset := newReplicaSet(namespace, FederationReplicaSetName, 5)

	_, err := clientset.Extensions().ReplicaSets(namespace).Create(replicaset)
	framework.ExpectNoError(err, "Creating replicaset %q in namespace %q", replicaset.Name, namespace)
	By(fmt.Sprintf("Successfully created federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))
	return replicaset
}

func deleteReplicaSetOrFail(clientset *fedclientset.Clientset, nsName string, replicaSetName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting replica set %q in namespace %q", replicaSetName, nsName))
	err := clientset.Extensions().ReplicaSets(nsName).Delete(replicaSetName, &v1.DeleteOptions{OrphanDependents: orphanDependents})
	framework.ExpectNoError(err, "Error deleting replica set %q in namespace %q", replicaSetName, nsName)

	// Wait for the replicaSet to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Extensions().ReplicaSets(nsName).Get(replicaSetName)
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting replica set %s: %v", replicaSetName, err)
	}
}

func updateReplicaSetOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.ReplicaSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateReplicaSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Updating federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))

	replicaset := newReplicaSet(namespace, FederationReplicaSetName, 15)

	newRs, err := clientset.ReplicaSets(namespace).Update(replicaset)
	framework.ExpectNoError(err, "Updating replicaset %q in namespace %q", replicaset.Name, namespace)
	By(fmt.Sprintf("Successfully updated federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))

	return newRs
}

func newReplicaSet(namespace string, name string, replicas int32) *v1beta1.ReplicaSet {
	return &v1beta1.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1beta1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"name": "myrs"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
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
}
