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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/federation/apis/federation"
	fedreplicsetcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/replicaset"
)

const (
	FederationReplicaSetPrefix = "federation-replicaset-"
)

// Create/delete replicaset api objects
var _ = framework.KubeDescribe("Federated ReplicaSet [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-replicaset")

	Describe("ReplicaSet objects [NoCluster]", func() {
		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// Delete all replicasets.
			nsName := f.FederationNamespace.Name
			deleteAllReplicaSetsOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			rs := createReplicaSetOrFail(f.FederationClientset, newReplicaSet(nsName, FederationReplicaSetPrefix, 5, nil))
			By(fmt.Sprintf("Creation of replicaset %q in namespace %q succeeded.  Deleting replicaset.", rs.Name, nsName))
			// Cleanup
			err := f.FederationClientset.Extensions().ReplicaSets(nsName).Delete(rs.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting replicaset %q in namespace %q", rs.Name, rs.Namespace)
			By(fmt.Sprintf("Deletion of replicaset %q in namespace %q succeeded.", rs.Name, nsName))
		})

	})

	// e2e cases for federated replicaset controller
	Describe("Features", func() {
		var (
			clusters map[string]*cluster
		)

		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters, _ = getRegisteredClusters(UserAgentName, f)
		})

		// e2e cases for federated replicaset controller
		Describe("CRUD", func() {
			var (
				rs *v1beta1.ReplicaSet
			)

			BeforeEach(func() {
				nsName := f.FederationNamespace.Name

				By(fmt.Sprintf("Creating a new replicaset in namespace %q", nsName))
				rs = createAndWaitForReplicasetOrFail(f.FederationClientset, nsName, clusters)
			})

			AfterEach(func() {
				// Delete all replicasets.
				nsName := f.FederationNamespace.Name

				By(fmt.Sprintf("Deleting replicaset \"%s/%s\"", nsName, rs.Name))
				orphanDependents := false
				deleteReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, &orphanDependents)
			})

			It("should create and update matching replicasets in underlying clusters", func() {
				nsName := f.FederationNamespace.Name

				// As part of the update, we scale the replicaset here.
				rs = newReplicaSetWithName(nsName, rs.Name, 15, nil)
				updateReplicaSetOrFail(f.FederationClientset, rs)
				waitForReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, clusters, nil)
				By(fmt.Sprintf("Successfully updated and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))
			})

			It("should be deleted from underlying clusters when OrphanDependents is false", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				nsName := f.FederationNamespace.Name
				orphanDependents := false
				verifyCascadingDeletionForReplicaSet(f.FederationClientset, clusters, &orphanDependents, nsName, rs.Name)
				By(fmt.Sprintf("Verified that replicasets were deleted from underlying clusters"))
			})

			It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				nsName := f.FederationNamespace.Name
				orphanDependents := true
				verifyCascadingDeletionForReplicaSet(f.FederationClientset, clusters, &orphanDependents, nsName, rs.Name)
				By(fmt.Sprintf("Verified that replicasets were not deleted from underlying clusters"))
			})

			It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				nsName := f.FederationNamespace.Name
				verifyCascadingDeletionForReplicaSet(f.FederationClientset, clusters, nil, nsName, rs.Name)
				By(fmt.Sprintf("Verified that replicasets were not deleted from underlying clusters"))
			})
		})

		// e2e cases for federated replicaset controller
		Describe("Preferences", func() {
			var (
				rs *v1beta1.ReplicaSet
			)

			AfterEach(func() {
				// Delete all replicasets.
				nsName := f.FederationNamespace.Name
				if rs != nil {
					orphanDependents := false
					By(fmt.Sprintf("Deleting replicaset \"%s/%s\"", nsName, rs.Name))
					deleteReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, &orphanDependents)
					rs = nil
				}
			})

			It("should create replicasets with weight preference", func() {
				pref, replicas, expect := generateFedRSPrefsWithWeight(clusters)
				rs = createAndUpdateFedRSWithPref(f.FederationClientset, f.FederationNamespace.Name, clusters, pref, replicas, expect)
			})

			It("should create replicasets with min replicas preference", func() {
				pref, replicas, expect := generateFedRSPrefsWithMin(clusters)
				rs = createAndUpdateFedRSWithPref(f.FederationClientset, f.FederationNamespace.Name, clusters, pref, replicas, expect)
			})

			It("should create replicasets with max replicas preference", func() {
				pref, replicas, expect := generateFedRSPrefsWithMax(clusters)
				rs = createAndUpdateFedRSWithPref(f.FederationClientset, f.FederationNamespace.Name, clusters, pref, replicas, expect)
			})

			// test for rebalancing
			PIt("should create replicasets and rebalance them", func() {
				nsName := f.FederationNamespace.Name
				pref1, pref2, replicas, expect1, expect2 := generateFedRSPrefsForRebalancing(clusters)

				By("Testing replicaset rebalancing")
				framework.Logf("Replicas: %d", replicas)
				framework.Logf("Preference 1: %#v", pref1)
				framework.Logf("Preference 2: %#v", pref2)

				rs = newReplicaSet(nsName, FederationReplicaSetPrefix, replicas, pref1)
				rs = createReplicaSetOrFail(f.FederationClientset, rs)
				waitForReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, clusters, expect1)
				By(fmt.Sprintf("Successfully created and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))

				rs = newReplicaSetWithName(nsName, rs.Name, replicas, pref2)
				updateReplicaSetOrFail(f.FederationClientset, rs)
				waitForReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, clusters, expect1)
				By(fmt.Sprintf("Successfully updated and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))

				pref2 = updateFedRSPrefsRebalance(pref2, true)
				rs = newReplicaSetWithName(nsName, rs.Name, replicas, pref2)
				updateReplicaSetOrFail(f.FederationClientset, rs)
				waitForReplicaSetOrFail(f.FederationClientset, nsName, rs.Name, clusters, expect2)
				By(fmt.Sprintf("Successfully updated and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))
			})
		})
	})
})

func createAndWaitForReplicasetOrFail(clientset *fedclientset.Clientset, nsName string, clusters map[string]*cluster) *v1beta1.ReplicaSet {
	rs := createReplicaSetOrFail(clientset, newReplicaSet(nsName, FederationReplicaSetPrefix, 5, nil))
	// Check subclusters if the replicaSet was created there.
	By(fmt.Sprintf("Waiting for replica sets %s to be created in all underlying clusters", rs.Name))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Extensions().ReplicaSets(nsName).Get(rs.Name, metav1.GetOptions{})
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
	return rs
}

func createAndUpdateFedRSWithPref(clientset *fedclientset.Clientset, nsName string, clusters map[string]*cluster, pref *federation.FederatedReplicaSetPreferences, replicas int32, expect map[string]int32) *v1beta1.ReplicaSet {
	framework.Logf("Replicas: %d, Preference: %#v", replicas, pref)
	rs := newReplicaSet(nsName, FederationReplicaSetPrefix, replicas, pref)
	rs = createReplicaSetOrFail(clientset, rs)

	waitForReplicaSetOrFail(clientset, nsName, rs.Name, clusters, expect)
	By(fmt.Sprintf("Successfully created and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))

	rs = newReplicaSetWithName(nsName, rs.Name, 0, pref)
	updateReplicaSetOrFail(clientset, rs)
	waitForReplicaSetOrFail(clientset, nsName, rs.Name, clusters, nil)
	By(fmt.Sprintf("Successfully updated and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))

	rs = newReplicaSetWithName(nsName, rs.Name, replicas, pref)
	updateReplicaSetOrFail(clientset, rs)
	waitForReplicaSetOrFail(clientset, nsName, rs.Name, clusters, expect)
	By(fmt.Sprintf("Successfully updated and synced replicaset \"%s/%s\" (%v/%v) to clusters", nsName, rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))

	return rs
}

// deleteAllReplicaSetsOrFail deletes all replicasets in the given namespace name.
func deleteAllReplicaSetsOrFail(clientset *fedclientset.Clientset, nsName string) {
	replicasetList, err := clientset.Extensions().ReplicaSets(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, replicaset := range replicasetList.Items {
		deleteReplicaSetOrFail(clientset, nsName, replicaset.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForReplicaSet verifies that replicaSets are deleted
// from underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForReplicaSet(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName, rsName string) {
	By(fmt.Sprintf("Deleting replica set %s", rsName))
	deleteReplicaSetOrFail(clientset, nsName, rsName, orphanDependents)

	By(fmt.Sprintf("Verifying replica sets %s in underlying clusters", rsName))
	errMessages := []string{}
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Extensions().ReplicaSets(nsName).Get(rsName, metav1.GetOptions{})
		if (orphanDependents == nil || *orphanDependents == true) && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for replica set %s in cluster %s, expected replica set to exist", rsName, clusterName))
		} else if (orphanDependents != nil && *orphanDependents == false) && (err == nil || !errors.IsNotFound(err)) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for replica set %s in cluster %s, got error: %v", rsName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func generateFedRSPrefsWithWeight(clusters map[string]*cluster) (pref *federation.FederatedReplicaSetPreferences, replicas int32, expect map[string]int32) {
	By("Generating replicaset preferences with weights")
	clusterNames := extraceClusterNames(clusters)
	pref = &federation.FederatedReplicaSetPreferences{
		Clusters: map[string]federation.ClusterReplicaSetPreferences{},
	}
	replicas = 0
	expect = map[string]int32{}

	for i, clusterName := range clusterNames {
		if i != 0 { // do not set weight for cluster[0] thus it should have no replicas scheduled
			pref.Clusters[clusterName] = federation.ClusterReplicaSetPreferences{
				Weight: int64(i),
			}
			replicas += int32(i)
			expect[clusterName] = int32(i)
		}
	}
	return
}

func generateFedRSPrefsWithMin(clusters map[string]*cluster) (pref *federation.FederatedReplicaSetPreferences, replicas int32, expect map[string]int32) {
	By("Generating replicaset preferences with min replicas")
	clusterNames := extraceClusterNames(clusters)
	pref = &federation.FederatedReplicaSetPreferences{
		Clusters: map[string]federation.ClusterReplicaSetPreferences{
			clusterNames[0]: {Weight: 100},
		},
	}
	replicas = 0
	expect = map[string]int32{}

	for i, clusterName := range clusterNames {
		if i != 0 { // do not set weight and minReplicas for cluster[0] thus it should have no replicas scheduled
			pref.Clusters[clusterName] = federation.ClusterReplicaSetPreferences{
				Weight:      int64(1),
				MinReplicas: int64(i + 2),
			}
			replicas += int32(i + 2)
			expect[clusterName] = int32(i + 2)
		}
	}
	// the extra replica goes to cluster[0] which has the highest weight
	replicas += 1
	expect[clusterNames[0]] = 1
	return
}

func generateFedRSPrefsWithMax(clusters map[string]*cluster) (pref *federation.FederatedReplicaSetPreferences, replicas int32, expect map[string]int32) {
	By("Generating replicaset preferences with max replicas")
	clusterNames := extraceClusterNames(clusters)
	pref = &federation.FederatedReplicaSetPreferences{
		Clusters: map[string]federation.ClusterReplicaSetPreferences{
			clusterNames[0]: {Weight: 1},
		},
	}
	replicas = 0
	expect = map[string]int32{}

	for i, clusterName := range clusterNames {
		if i != 0 { // do not set maxReplicas for cluster[0] thus replicas exceeds the total maxReplicas turned to cluster[0]
			maxReplicas := int64(i)
			pref.Clusters[clusterName] = federation.ClusterReplicaSetPreferences{
				Weight:      int64(100),
				MaxReplicas: &maxReplicas,
			}
			replicas += int32(i)
			expect[clusterName] = int32(i)
		}
	}
	// extra replicas go to cluster[0] although it has the lowest weight as others hit the MaxReplicas
	replicas += 5
	expect[clusterNames[0]] = 5
	return
}

func updateFedRSPrefsRebalance(pref *federation.FederatedReplicaSetPreferences, rebalance bool) *federation.FederatedReplicaSetPreferences {
	pref.Rebalance = rebalance
	return pref
}

func generateFedRSPrefsForRebalancing(clusters map[string]*cluster) (pref1, pref2 *federation.FederatedReplicaSetPreferences, replicas int32, expect1, expect2 map[string]int32) {
	By("Generating replicaset for rebalancing")
	clusterNames := extraceClusterNames(clusters)
	replicas = 3

	pref1 = &federation.FederatedReplicaSetPreferences{
		Clusters: map[string]federation.ClusterReplicaSetPreferences{
			clusterNames[0]: {Weight: 1},
			clusterNames[1]: {Weight: 2},
		},
	}
	expect1 = map[string]int32{
		clusterNames[0]: 1,
		clusterNames[1]: 2,
	}
	pref2 = &federation.FederatedReplicaSetPreferences{
		Clusters: map[string]federation.ClusterReplicaSetPreferences{
			clusterNames[0]: {Weight: 2},
			clusterNames[1]: {Weight: 1},
		},
	}
	expect2 = map[string]int32{
		clusterNames[0]: 2,
		clusterNames[1]: 1,
	}
	return
}

func waitForReplicaSetOrFail(c *fedclientset.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster, expect map[string]int32) {
	err := waitForReplicaSet(c, namespace, replicaSetName, clusters, expect)
	framework.ExpectNoError(err, "Failed to verify replica set \"%s/%s\", err: %v", namespace, replicaSetName, err)
}

func waitForReplicaSet(c *fedclientset.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster, expect map[string]int32) error {
	framework.Logf("waitForReplicaSet: %s/%s; clusters: %v; expect: %v", namespace, replicaSetName, clusters, expect)
	err := wait.Poll(10*time.Second, federatedDefaultTestTimeout, func() (bool, error) {
		frs, err := c.ReplicaSets(namespace).Get(replicaSetName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		specReplicas, statusReplicas := int32(0), int32(0)
		for _, cluster := range clusters {
			rs, err := cluster.ReplicaSets(namespace).Get(replicaSetName, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				framework.Logf("Failed getting replicaset: \"%s/%s/%s\", err: %v", cluster.name, namespace, replicaSetName, err)
				return false, err
			}
			if errors.IsNotFound(err) {
				if expect != nil && expect[cluster.name] > 0 {
					framework.Logf("Replicaset \"%s/%s/%s\" with replica count %d does not exist", cluster.name, namespace, replicaSetName, expect[cluster.name])
					return false, nil
				}
			} else {
				if !equivalentReplicaSet(frs, rs) {
					framework.Logf("Replicaset meta or spec does not match for cluster %q:\n    federation: %v\n    cluster: %v", cluster.name, frs, rs)
					return false, nil
				}
				if expect != nil && *rs.Spec.Replicas < expect[cluster.name] {
					framework.Logf("Replicas do not match for \"%s/%s/%s\": expected: >= %v, actual: %v", cluster.name, namespace, replicaSetName, expect[cluster.name], *rs.Spec.Replicas)
					return false, nil
				}
				specReplicas += *rs.Spec.Replicas
				statusReplicas += rs.Status.Replicas
			}
		}
		if *frs.Spec.Replicas == 0 && frs.Status.Replicas != 0 {
			framework.Logf("ReplicaSet \"%s/%s\" with zero replicas should match the status as no overflow happens: expected: 0, actual: %v", namespace, replicaSetName, frs.Status.Replicas)
			return false, nil
		}
		if statusReplicas == frs.Status.Replicas && specReplicas >= *frs.Spec.Replicas {
			return true, nil
		}
		framework.Logf("Replicas do not match, federation replicas: %v/%v, cluster replicas: %v/%v", frs.Status.Replicas, *frs.Spec.Replicas, statusReplicas, specReplicas)
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

func createReplicaSetOrFail(clientset *fedclientset.Clientset, replicaset *v1beta1.ReplicaSet) *v1beta1.ReplicaSet {
	namespace := replicaset.Namespace
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createReplicaSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation replicaset %q in namespace %q", replicaset.Name, namespace))

	newRS, err := clientset.Extensions().ReplicaSets(namespace).Create(replicaset)
	framework.ExpectNoError(err, "Creating replicaset %q in namespace %q", replicaset.Name, namespace)
	By(fmt.Sprintf("Successfully created federation replicaset %q in namespace %q", newRS.Name, namespace))
	return newRS
}

func deleteReplicaSetOrFail(clientset *fedclientset.Clientset, nsName string, replicaSetName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting replica set %q in namespace %q", replicaSetName, nsName))
	err := clientset.Extensions().ReplicaSets(nsName).Delete(replicaSetName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting replica set %q in namespace %q", replicaSetName, nsName)
	}

	waitForReplicaSetToBeDeletedOrFail(clientset, nsName, replicaSetName)
}

func updateReplicaSetOrFail(clientset *fedclientset.Clientset, replicaset *v1beta1.ReplicaSet) *v1beta1.ReplicaSet {
	namespace := replicaset.Namespace
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateReplicaSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Updating federation replicaset %q in namespace %q", replicaset.Name, namespace))

	newRS, err := clientset.ReplicaSets(namespace).Update(replicaset)
	framework.ExpectNoError(err, "Updating replicaset %q in namespace %q", replicaset.Name, namespace)
	By(fmt.Sprintf("Successfully updated federation replicaset %q in namespace %q", replicaset.Name, namespace))

	return newRS
}

func newReplicaSetObj(namespace string, replicas int32, pref *federation.FederatedReplicaSetPreferences) *v1beta1.ReplicaSet {
	// When the tests are run in parallel, replicasets from different tests can
	// collide with each other. Prevent that by creating a unique label and
	// label selector for each created replica set.
	uuidString := string(uuid.NewUUID())
	rsLabel := fmt.Sprintf("myrs-%s", uuidString)

	rs := &v1beta1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec: v1beta1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"name": rsLabel},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": rsLabel},
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
	if pref != nil {
		prefBytes, _ := json.Marshal(pref)
		prefString := string(prefBytes)
		rs.Annotations[fedreplicsetcontroller.FedReplicaSetPreferencesAnnotation] = prefString
	}
	return rs

}

func newReplicaSet(namespace string, prefix string, replicas int32, pref *federation.FederatedReplicaSetPreferences) *v1beta1.ReplicaSet {
	rs := newReplicaSetObj(namespace, replicas, pref)
	rs.GenerateName = prefix
	return rs
}

func newReplicaSetWithName(namespace string, name string, replicas int32, pref *federation.FederatedReplicaSetPreferences) *v1beta1.ReplicaSet {
	rs := newReplicaSetObj(namespace, replicas, pref)
	rs.Name = name
	return rs
}

func extraceClusterNames(clusters map[string]*cluster) []string {
	clusterNames := make([]string, 0, len(clusters))
	for clusterName := range clusters {
		clusterNames = append(clusterNames, clusterName)
	}
	return clusterNames
}
