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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/errors"
)

const (
	FederationDeploymentName   = "federation-deployment"
	FederatedDeploymentTimeout = 120 * time.Second
)

// Create/delete deployment api objects
var _ = framework.KubeDescribe("Federation deployments [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-deployment")

	Describe("Deployment objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)

			// Delete all deployments.
			nsName := f.FederationNamespace.Name
			deleteAllDeploymentsOrFail(f.FederationClientset_1_5, nsName)
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			deployment := createDeploymentOrFail(f.FederationClientset_1_5, nsName)
			By(fmt.Sprintf("Creation of deployment %q in namespace %q succeeded.  Deleting deployment.", deployment.Name, nsName))
			// Cleanup
			err := f.FederationClientset_1_5.Extensions().Deployments(nsName).Delete(deployment.Name, &v1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting deployment %q in namespace %q", deployment.Name, deployment.Namespace)
			By(fmt.Sprintf("Deletion of deployment %q in namespace %q succeeded.", deployment.Name, nsName))
		})

	})

	// e2e cases for federated deployment controller
	Describe("Federated Deployment", func() {
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
			nsName := f.FederationNamespace.Name
			deleteAllDeploymentsOrFail(f.FederationClientset_1_5, nsName)
			unregisterClusters(clusters, f)
		})

		It("should create and update matching deployments in underling clusters", func() {
			nsName := f.FederationNamespace.Name
			dep := createDeploymentOrFail(f.FederationClientset_1_5, nsName)
			defer func() {
				// cleanup. deletion of deployments is not supported for underlying clusters
				By(fmt.Sprintf("Preparing deployment %q/%q for deletion by setting replicas to zero", nsName, dep.Name))
				replicas := int32(0)
				dep.Spec.Replicas = &replicas
				f.FederationClientset_1_5.Deployments(nsName).Update(dep)
				waitForDeploymentOrFail(f.FederationClientset_1_5, nsName, dep.Name, clusters)
				f.FederationClientset_1_5.Deployments(nsName).Delete(dep.Name, &v1.DeleteOptions{})
			}()

			waitForDeploymentOrFail(f.FederationClientset_1_5, nsName, dep.Name, clusters)
			By(fmt.Sprintf("Successfuly created and synced deployment %q/%q to clusters", nsName, dep.Name))
			updateDeploymentOrFail(f.FederationClientset_1_5, nsName)
			waitForDeploymentOrFail(f.FederationClientset_1_5, nsName, dep.Name, clusters)
			By(fmt.Sprintf("Successfuly updated and synced deployment %q/%q to clusters", nsName, dep.Name))
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForDeployment(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that deployments were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForDeployment(f.FederationClientset_1_5, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that deployments were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			framework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForDeployment(f.FederationClientset_1_5, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that deployments were not deleted from underlying clusters"))
		})

	})
})

// deleteAllDeploymentsOrFail deletes all deployments in the given namespace name.
func deleteAllDeploymentsOrFail(clientset *fedclientset.Clientset, nsName string) {
	deploymentList, err := clientset.Extensions().Deployments(nsName).List(v1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, deployment := range deploymentList.Items {
		deleteDeploymentOrFail(clientset, nsName, deployment.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForDeployment verifies that deployments are deleted
// from underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForDeployment(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	deployment := createDeploymentOrFail(clientset, nsName)
	deploymentName := deployment.Name
	// Check subclusters if the deployment was created there.
	By(fmt.Sprintf("Waiting for deployment %s to be created in all underlying clusters", deploymentName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Extensions().Deployments(nsName).Get(deploymentName)
			if err != nil && errors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all deployments created")

	By(fmt.Sprintf("Deleting deployment %s", deploymentName))
	deleteDeploymentOrFail(clientset, nsName, deploymentName, orphanDependents)

	By(fmt.Sprintf("Verifying deployments %s in underlying clusters", deploymentName))
	errMessages := []string{}
	// deployment should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Extensions().Deployments(nsName).Get(deploymentName)
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for deployment %s in cluster %s, expected deployment to exist", deploymentName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for deployment %s in cluster %s, got error: %v", deploymentName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func waitForDeploymentOrFail(c *fedclientset.Clientset, namespace string, deploymentName string, clusters map[string]*cluster) {
	err := waitForDeployment(c, namespace, deploymentName, clusters)
	framework.ExpectNoError(err, "Failed to verify deployment %q/%q, err: %v", namespace, deploymentName, err)
}

func waitForDeployment(c *fedclientset.Clientset, namespace string, deploymentName string, clusters map[string]*cluster) error {
	err := wait.Poll(10*time.Second, FederatedDeploymentTimeout, func() (bool, error) {
		fdep, err := c.Deployments(namespace).Get(deploymentName)
		if err != nil {
			return false, err
		}
		specReplicas, statusReplicas := int32(0), int32(0)
		for _, cluster := range clusters {
			dep, err := cluster.Deployments(namespace).Get(deploymentName)
			if err != nil && !errors.IsNotFound(err) {
				By(fmt.Sprintf("Failed getting deployment: %q/%q/%q, err: %v", cluster.name, namespace, deploymentName, err))
				return false, err
			}
			if err == nil {
				if !verifyDeployment(fdep, dep) {
					By(fmt.Sprintf("Deployment meta or spec not match for cluster %q:\n    federation: %v\n    cluster: %v", cluster.name, fdep, dep))
					return false, nil
				}
				specReplicas += *dep.Spec.Replicas
				statusReplicas += dep.Status.Replicas
			}
		}
		if statusReplicas == fdep.Status.Replicas && specReplicas >= *fdep.Spec.Replicas {
			return true, nil
		}
		By(fmt.Sprintf("Replicas not match, federation replicas: %v/%v, clusters replicas: %v/%v\n", *fdep.Spec.Replicas, fdep.Status.Replicas, specReplicas, statusReplicas))
		return false, nil
	})

	return err
}

func verifyDeployment(fedDeployment, localDeployment *v1beta1.Deployment) bool {
	localDeployment = fedutil.DeepCopyDeployment(localDeployment)
	localDeployment.Spec.Replicas = fedDeployment.Spec.Replicas
	return fedutil.DeploymentEquivalent(fedDeployment, localDeployment)
}

func createDeploymentOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.Deployment {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createDeploymentOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation deployment %q in namespace %q", FederationDeploymentName, namespace))

	deployment := newDeploymentForFed(namespace, FederationDeploymentName, 5)

	_, err := clientset.Extensions().Deployments(namespace).Create(deployment)
	framework.ExpectNoError(err, "Creating deployment %q in namespace %q", deployment.Name, namespace)
	By(fmt.Sprintf("Successfully created federation deployment %q in namespace %q", FederationDeploymentName, namespace))
	return deployment
}

func updateDeploymentOrFail(clientset *fedclientset.Clientset, namespace string) *v1beta1.Deployment {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateDeploymentOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Updating federation deployment %q in namespace %q", FederationDeploymentName, namespace))

	deployment := newDeploymentForFed(namespace, FederationDeploymentName, 15)

	newRs, err := clientset.Deployments(namespace).Update(deployment)
	framework.ExpectNoError(err, "Updating deployment %q in namespace %q", deployment.Name, namespace)
	By(fmt.Sprintf("Successfully updated federation deployment %q in namespace %q", FederationDeploymentName, namespace))

	return newRs
}

func deleteDeploymentOrFail(clientset *fedclientset.Clientset, nsName string, deploymentName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting deployment %q in namespace %q", deploymentName, nsName))
	err := clientset.Extensions().Deployments(nsName).Delete(deploymentName, &v1.DeleteOptions{OrphanDependents: orphanDependents})
	framework.ExpectNoError(err, "Error deleting deployment %q in namespace %q", deploymentName, nsName)

	// Wait for the deployment to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Extensions().Deployments(nsName).Get(deploymentName)
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting deployment %s: %v", deploymentName, err)
	}
}

func newDeploymentForFed(namespace string, name string, replicas int32) *v1beta1.Deployment {
	return &v1beta1.Deployment{
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1beta1.DeploymentSpec{
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
