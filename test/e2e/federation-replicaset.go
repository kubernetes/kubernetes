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
	"time"

	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/errors"
	"reflect"
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
			framework.SkipUnlessFederated(f.Client)

			// Delete registered replicasets.
			nsName := f.FederationNamespace.Name
			replicasetList, err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, replicaset := range replicasetList.Items {
				err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).Delete(replicaset.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			nsName := f.FederationNamespace.Name
			replicaset := createReplicaSetOrFail(f.FederationClientset_1_4, nsName)
			By(fmt.Sprintf("Creation of replicaset %q in namespace %q succeeded.  Deleting replicaset.", replicaset.Name, nsName))
			// Cleanup
			err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).Delete(replicaset.Name, &api.DeleteOptions{})
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
			framework.SkipUnlessFederated(f.Client)
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}
			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, federationName, f)
		})

		AfterEach(func() {
			unregisterClusters(clusters, f)
		})

		It("should create and update matching replicasets in underling clusters", func() {
			nsName := f.FederationNamespace.Name
			rs := createReplicaSetOrFail(f.FederationClientset_1_4, nsName)
			defer func() {
				// cleanup. deletion of replicasets is not supported for underlying clusters
				By(fmt.Sprintf("Preparing replicaset %q/%q for deletion by setting replicas to zero", nsName, rs.Name))
				replicas := int32(0)
				rs.Spec.Replicas = &replicas
				f.FederationClientset_1_4.ReplicaSets(nsName).Update(rs)
				waitForReplicaSetOrFail(f.FederationClientset_1_4, nsName, rs.Name, clusters)
				f.FederationClientset_1_4.ReplicaSets(nsName).Delete(rs.Name, &api.DeleteOptions{})
			}()

			waitForReplicaSetOrFail(f.FederationClientset_1_4, nsName, rs.Name, clusters)
			By(fmt.Sprintf("Successfuly created and synced replicaset %q/%q to clusters", nsName, rs.Name))
			updateReplicaSetOrFail(f.FederationClientset_1_4, nsName)
			waitForReplicaSetOrFail(f.FederationClientset_1_4, nsName, rs.Name, clusters)
			By(fmt.Sprintf("Successfuly updated and synced replicaset %q/%q to clusters", nsName, rs.Name))
		})
	})
})

func waitForReplicaSetOrFail(c *federation_release_1_4.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster) {
	err := waitForReplicaSet(c, namespace, replicaSetName, clusters)
	framework.ExpectNoError(err, "Failed to verify replicaset %q/%q, err: %v", namespace, replicaSetName, err)
}

func waitForReplicaSet(c *federation_release_1_4.Clientset, namespace string, replicaSetName string, clusters map[string]*cluster) error {
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

func createReplicaSetOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1beta1.ReplicaSet {
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

func updateReplicaSetOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1beta1.ReplicaSet {
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
			Selector: &v1beta1.LabelSelector{
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
