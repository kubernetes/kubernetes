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
	"reflect"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	hpautil "k8s.io/kubernetes/federation/pkg/federation-controller/util/hpa"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	testHpaPrefix        = "fed-hpa"
	testHpaTargetObjKind = "Deployment"
	testDeploymentPrefix = "fed-dep"
	testHpaScalableImage = "gcr.io/google_containers/hpa-example"
	testHpaPollTimeout   = 5 * time.Second
)

var _ = framework.KubeDescribe("Federated Hpa [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-hpa")

	Describe("Federated HPA [NoCluster]", func() {
		var (
			nsName string
		)

		BeforeEach(func() {
			nsName = f.FederationNamespace.Name
			fedframework.SkipUnlessFederated(f.ClientSet)
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// Delete all hpa objects.
			nsName := f.FederationNamespace.Name
			deleteAllHpasOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			hpa := createHpaOrFail(f.FederationClientset, newHpa(nsName, testHpaPrefix, testDeploymentPrefix, NewInt32(1), NewInt32(50), 4))
			By(fmt.Sprintf("Creation of hpa %q in namespace %q succeeded.  Deleting hpa.", hpa.Name, nsName))
			// Cleanup
			err := f.FederationClientset.Autoscaling().HorizontalPodAutoscalers(nsName).Delete(hpa.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting hpa %q in namespace %q", hpa.Name, hpa.Namespace)
			By(fmt.Sprintf("Deletion of hpa %q in namespace %q succeeded.", hpa.Name, nsName))
		})

		Describe("with valid target obj in spec", func() {
			AfterEach(func() {
				// Delete all hpa target objects.
				deleteAllTargetObjsOrFail(f.FederationClientset, nsName)
			})

			It("should be able to set correct selected cluster list on target object", func() {
				// Hpa replicas are distributed randomly; the default distribution can thus
				// land into any of the clusters. This test verifies against the length of cluster
				// list. Hpa with min, max = 1, 1 is bound to set the hpa and the target object
				// only for 1 cluster
				clusterListLen := 1
				dep := newHpaTargetObj(nsName, testDeploymentPrefix, NewInt32(0), "")
				createTargetObjOrFail(f.FederationClientset, dep)
				hpa := newHpa(nsName, testHpaPrefix, dep.Name, NewInt32(1), NewInt32(50), 1)
				createHpaOrFail(f.FederationClientset, hpa)
				waitForClusterListOnTargetObjOrFail(f.FederationClientset, nsName, dep.Name, clusterListLen)
			})
		})
	})

	Describe("Federated HPA: ", func() {
		var (
			clusters       fedframework.ClusterSlice
			nsName         string
			loadGenPodName string
		)

		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters = f.GetRegisteredClusters()
			nsName = f.FederationNamespace.Name
			loadGenPodName = ""
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			// Delete all hpa objects.
			deleteAllHpasOrFail(f.FederationClientset, nsName)
		})

		It("given that sufficient replicas are there, hpa replicas get distributed to all clusters", func() {
			expected, max := getExpectedForHpaFromMaxReplicas(clusters, 3)
			hpa := createHpaOrFail(f.FederationClientset, newHpa(nsName, testHpaPrefix, testDeploymentPrefix, NewInt32(1), NewInt32(50), max))
			By(fmt.Sprintf("Creation of hpa %q in namespace %q succeeded.", hpa.Name, nsName))
			waitForHpaOrFail(f.FederationClientset, f.FederationNamespace.Name, hpa.Name, clusters, expected)
		})

		Describe("With valid and existing hpa target object: ", func() {
			AfterEach(func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				// Delete all target objects.
				deleteAllTargetObjsOrFail(f.FederationClientset, nsName)
			})

			It("fed hpa controls target object", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

				// create the deployment object with only 1 replica to ensure
				// that its created in only 1 cluster, if hpa exists in more clusters, then
				// it ensures that target object also gets created in more clusters (with 0
				// replicas where the target object controller could not give replicas).
				dep := createTargetObjOrFail(f.FederationClientset, newHpaTargetObj(nsName, testDeploymentPrefix, NewInt32(1), ""))
				expected, max := getExpectedForHpaFromMaxReplicas(clusters, 3)
				hpa := createHpaOrFail(f.FederationClientset, newHpa(nsName, testHpaPrefix, dep.Name, NewInt32(1), NewInt32(50), max))
				By(fmt.Sprintf("Creation of hpa %q in namespace %q succeeded.", hpa.Name, nsName))

				waitForHpaOrFail(f.FederationClientset, nsName, hpa.Name, clusters, expected)
				waitForTargetObjInClustersOrFail(nsName, dep.Name, clusters, expected, false)
			})

			// The below test does following stuff:
			// It assumes that there are minimum of 2 clusters in this federation.
			// Creates resource consumer deployment object with 1 replica into first 2 clusters.
			// The resource consumer is created directly into the clusters to reutilize it and the
			// utils around it (as written for local cluster e2e tests) unchanged.
			//
			// Create an hpa with min|max : 1|5, to cap the limits during test.
			// This test will then drive utilisation up or down in clusters to test:
			// 1. Originally the hpa gets partitioned with equal replicas in 2 clusters.
			// 2. When utilization is driven up in first cluster, replicas move there.
			// 3. When utilization is driven up in second cluster, replicas move there.
			It("app replicas move where they are needed most", func() {
				// resource consumer enables consuming desired amount of resources
				const rcName = "fed-hpa-dep-rc"
				const timeToWait = 10 * time.Minute

				rc := common.NewDynamicResourceConsumer(rcName, nsName, common.KindDeployment, 1, 100, 0, 0, 100, 200, clusters[0].Clientset, clusters[0].InternalClientSet)
				// TODO: bug this cleanup sometimes hangs and for some reason pods are not removed
				// however removing the namespace, deletes all resources.
				defer rc.CleanUp()
				//rc.Pause()

				// get a copy of the load generating rc deployment from one cluster and create in federation
				// to ensure its controlled by federated hpa.
				dep := fedutil.DeepCopyDeployment(getRcObjectFromClusterOrFail(clusters[0], nsName, rcName))
				createTargetObjOrFail(f.FederationClientset, dep)

				// create the hpa
				totalMaxHpaReplicas := 5
				hpa := createHpaOrFail(f.FederationClientset, newHpa(nsName, testHpaPrefix, rcName, NewInt32(1), NewInt32(20), int32(totalMaxHpaReplicas)))
				expected, _ := getExpectedForHpaFromMaxReplicas(clusters, totalMaxHpaReplicas)
				waitForHpaOrFail(f.FederationClientset, nsName, hpa.Name, clusters, expected)

				//rc.Resume()
				// consume cpu
				rc.ConsumeCPU(200)
				// all replicas should move to loaded cluster leaving 1 each in remaining.
				maxExpectedInLoadedCluster := totalMaxHpaReplicas - (len(clusters) - 1)
				rc.WaitForReplicas(maxExpectedInLoadedCluster, timeToWait)

			})
		})
	})
})

func getRcObjectFromClusterOrFail(cluster *fedframework.Cluster, namespace, targetObjName string) *v1beta1.Deployment {
	dep := &v1beta1.Deployment{}
	err := wait.Poll(10*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		var err error = nil
		dep, err = cluster.Extensions().Deployments(namespace).Get(targetObjName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Could not get the target object from federated cluster")
	return dep
}

func waitForClusterListOnTargetObjOrFail(clientset *fedclientset.Clientset, nsName, targetObjName string, clusterListLen int) bool {
	By(fmt.Sprintf("Waiting for hpa set cluster list to be updated on target object %s", targetObjName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		rs, err := clientset.Extensions().Deployments(nsName).Get(targetObjName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		clusterNames, err := hpautil.GetHpaTargetClusterList(rs)
		if err != nil {
			return true, err
		}
		if clusterNames != nil && len(clusterNames.Names) == clusterListLen {
			return true, nil
		}
		By(fmt.Sprintf("Selected ClusterNames which appear on hpa targetted obj %q, are %s", targetObjName, clusterNames.String()))
		return false, nil
	})
	framework.ExpectNoError(err, "The correct cluster list did not appear on target object")
	return true
}

func waitForHpaOrFail(c *fedclientset.Clientset, namespace string, hpaName string, clusters fedframework.ClusterSlice, expect map[string]int32) {
	err := waitForHpaInClusters(c, namespace, hpaName, clusters, expect)
	framework.ExpectNoError(err, "Failed to verify hpa \"%s/%s\", err: %v", namespace, hpaName, err)
}

func waitForHpaInClusters(c *fedclientset.Clientset, namespace string, hpaName string, clusters fedframework.ClusterSlice, expected map[string]int32) error {
	framework.Logf("waitForHpaInClusters: %s/%s; clusters: %v", namespace, hpaName, clusters)
	err := wait.Poll(10*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		fedHpa, err := c.AutoscalingV1().HorizontalPodAutoscalers(namespace).Get(hpaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		minReplicas, maxReplicas := int32(0), int32(0)
		for _, cluster := range clusters {
			localHpa, err := cluster.AutoscalingV1().HorizontalPodAutoscalers(namespace).Get(hpaName, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				framework.Logf("Failed getting hpa: \"%s/%s/%s\", err: %v", cluster.Name, namespace, hpaName, err)
				return false, err
			}
			if errors.IsNotFound(err) {
				if expected != nil && expected[cluster.Name] > 0 {
					framework.Logf("Hpa \"%s/%s/%s\" does not exist", cluster.Name, namespace, hpaName)
					return false, nil
				}
			} else {
				if !hpaEquivalentIgnoringReplicas(fedHpa, localHpa) {
					framework.Logf("Hpa meta or spec does not match for cluster %q:\n    federation: %v\n    cluster: %v", cluster.Name, fedHpa, localHpa)
					return false, nil
				}
				if localHpa.Spec.MinReplicas != nil {
					minReplicas += *localHpa.Spec.MinReplicas
				}
				maxReplicas += localHpa.Spec.MaxReplicas
			}
		}

		// total cluster min should not be less then fed min
		// total cluster max should be same as fed max
		if minReplicas >= *fedHpa.Spec.MinReplicas && maxReplicas == fedHpa.Spec.MaxReplicas {
			return true, nil
		}
		framework.Logf("Replicas min/max do not match, federation replicas: %v/%v, cluster replicas: %v/%v", *fedHpa.Spec.MinReplicas, fedHpa.Spec.MaxReplicas, minReplicas, maxReplicas)
		return false, nil
	})

	return err
}

func waitForTargetObjInClustersOrFail(namespace string, targetObjName string, clusters fedframework.ClusterSlice, expected map[string]int32, checkReplicas bool) {
	err := waitForTargetObjInClusters(namespace, targetObjName, clusters, expected, checkReplicas)
	framework.ExpectNoError(err, "Failed to verify target obj \"%s/%s\", err: %v", namespace, targetObjName, err)
}

func waitForTargetObjInClusters(namespace string, targetObjName string, clusters fedframework.ClusterSlice, expected map[string]int32, checkReplicas bool) error {
	framework.Logf("waitForTargetObjClusters: %s/%s; clusters: %v", namespace, targetObjName, clusters)
	err := wait.Poll(10*time.Second, 4*fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		var localObj *v1beta1.Deployment = nil
		var err error = nil
		okClusters := 0
		for _, cluster := range clusters {
			localObj, err = cluster.Extensions().Deployments(namespace).Get(targetObjName, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				framework.Logf("Failed getting targetObj: \"%s/%s/%s\", err: %v", cluster.Name, namespace, targetObjName, err)
				return false, err
			}
			if errors.IsNotFound(err) {
				if expected != nil && expected[cluster.Name] > 0 {
					framework.Logf("TargetObj \"%s/%s/%s\" does not exist yet", cluster.Name, namespace, targetObjName)
					return false, nil
				}
			}

			if checkReplicas && localObj.Spec.Replicas != nil &&
				expected[cluster.Name] == *localObj.Spec.Replicas {
				okClusters++
			}
		}
		if checkReplicas {
			if okClusters == len(clusters) {
				return true, nil
			}
			return false, nil
		}
		return true, nil
	})
	return err
}

func deleteLoadGenPodOrFail(cluster *fedframework.Cluster, loadGenPodName, nsName string) {
	if loadGenPodName == "" {
		// this test did not create this pod
		return
	}
	gracePeriod := int64(0)
	By(fmt.Sprintf("Deleting load gen pod %q in namespace %q", loadGenPodName, nsName))
	err := cluster.CoreV1().Pods(nsName).Delete(loadGenPodName, &metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
	if err != nil {
		if errors.IsNotFound(err) {
			return
		}
		framework.ExpectNoError(err, "Error deleting load gen pod %q in namespace %q", loadGenPodName, nsName)
	}
	waitForPodToBeDeletedFromClusterOrFail(testHpaPollTimeout, cluster, nsName, loadGenPodName)
}

func deleteAllHpasOrFail(clientset *fedclientset.Clientset, nsName string) {
	hpaList, err := clientset.AutoscalingV1().HorizontalPodAutoscalers(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, hpa := range hpaList.Items {
		deleteHpaOrFail(clientset, nsName, hpa.Name, &orphanDependents)
	}
}

func deleteAllTargetObjsOrFail(clientset *fedclientset.Clientset, nsName string) {
	// we carry out our e2e tests only with deployment objects as hpa target
	depList, err := clientset.Extensions().Deployments(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, dep := range depList.Items {
		deleteTargetObjOrFail(clientset, nsName, dep.Name, &orphanDependents)
	}
}

func createHpaOrFail(clientset *fedclientset.Clientset, hpa *autoscalingv1.HorizontalPodAutoscaler) *autoscalingv1.HorizontalPodAutoscaler {
	namespace := hpa.Namespace
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createHpaOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation hpa %q in namespace %q", hpa.Name, namespace))

	newHpa, err := clientset.AutoscalingV1().HorizontalPodAutoscalers(namespace).Create(hpa)
	framework.ExpectNoError(err, "Creating hpa %q in namespace %q", newHpa.Name, namespace)
	By(fmt.Sprintf("Successfully created federation hpa %q in namespace %q", newHpa.Name, namespace))
	return newHpa
}
func createTargetObjOrFail(clientset *fedclientset.Clientset, dep *v1beta1.Deployment) *v1beta1.Deployment {
	namespace := dep.Namespace
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createTargetObjOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating hpa target obj %q in namespace %q", dep.Name, namespace))

	newDep, err := clientset.Extensions().Deployments(namespace).Create(dep)
	framework.ExpectNoError(err, "Creating hpa target obj %q in namespace %q", dep.Name, namespace)
	By(fmt.Sprintf("Successfully created hpa target obj %q in namespace %q", newDep.Name, namespace))
	return newDep
}

func deleteHpaOrFail(clientset *fedclientset.Clientset, nsName string, hpaName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting hpa %q in namespace %q", hpaName, nsName))
	err := clientset.AutoscalingV1().HorizontalPodAutoscalers(nsName).Delete(hpaName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting hpa %q in namespace %q", hpaName, nsName)
	}

	waitForHpaToBeDeletedOrFail(testHpaPollTimeout, clientset, nsName, hpaName)
}

func deleteTargetObjOrFail(clientset *fedclientset.Clientset, nsName string, depName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting deployment %q in namespace %q", depName, nsName))
	err := clientset.Extensions().Deployments(nsName).Delete(depName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting deployment %q in namespace %q", depName, nsName)
	}

	waitForDeploymentToBeDeletedOrFail(testHpaPollTimeout, clientset, nsName, depName)
}

func newHpaWithName(namespace, hpaName, targetObjName string, min, targetUtilisation *int32, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	hpa := newHpaObj(namespace, targetObjName, min, targetUtilisation, max)
	hpa.Name = hpaName
	return hpa
}

func newHpa(namespace, hpaNamePrefix, targetObjName string, min, targetUtilisation *int32, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	hpa := newHpaObj(namespace, targetObjName, min, targetUtilisation, max)
	uuidString := string(uuid.NewUUID())
	hpa.Name = fmt.Sprintf("%s-%s", hpaNamePrefix, uuidString)
	return hpa
}

func newHpaObj(namespace, targetObjName string, min, targetUtilisation *int32, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: testHpaTargetObjKind,
				Name: targetObjName,
			},
			MinReplicas:                    min,
			MaxReplicas:                    max,
			TargetCPUUtilizationPercentage: targetUtilisation,
		},
	}
}

func newHpaTargetObj(namespace, targetObjPrefix string, replicas *int32, matchLabel string) *v1beta1.Deployment {
	uuidString := string(uuid.NewUUID())
	targetObjName := fmt.Sprintf("%s-%s", targetObjPrefix, uuidString)
	dep := &v1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        targetObjName,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"name": matchLabel},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": matchLabel},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "hpa-example",
							Image: testHpaScalableImage,
							Resources: v1.ResourceRequirements{
								Requests: map[v1.ResourceName]resource.Quantity{
									v1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
				},
			},
		},
	}
	return dep
}

// hpaObjectMetaEquivalent checks if cluster-independent, user provided data in two given
// ObjectMeta are equal. The annotations are skipped here, as local hpa controller updates
// events as annotations on the local hpa object.
func hpaObjectMetaEquivalent(a, b metav1.ObjectMeta) bool {
	if a.Name != b.Name {
		return false
	}
	if a.Namespace != b.Namespace {
		return false
	}
	if !reflect.DeepEqual(a.Labels, b.Labels) && (len(a.Labels) != 0 || len(b.Labels) != 0) {
		return false
	}
	return true
}

func hpaEquivalentIgnoringReplicas(fedHpa, localHpa *autoscalingv1.HorizontalPodAutoscaler) bool {
	localHpaSpec := localHpa.Spec
	if fedHpa.Spec.MinReplicas == nil {
		localHpaSpec.MinReplicas = nil
	} else if localHpaSpec.MinReplicas == nil {
		var r int32 = *fedHpa.Spec.MinReplicas
		localHpaSpec.MinReplicas = &r
	} else {
		*localHpaSpec.MinReplicas = *fedHpa.Spec.MinReplicas
	}
	localHpaSpec.MaxReplicas = fedHpa.Spec.MaxReplicas
	return hpaObjectMetaEquivalent(fedHpa.ObjectMeta, localHpa.ObjectMeta) &&
		reflect.DeepEqual(fedHpa.Spec, localHpaSpec)
}

func getExpectedForHpaFromMaxReplicas(clusters fedframework.ClusterSlice, max int) (map[string]int32, int32) {
	// distribute total replicas of max equally in clusters.
	expected := make(map[string]int32)
	for i := 0; i < max; i++ {
		for _, cluster := range clusters {
			if _, found := expected[cluster.Name]; !found {
				expected[cluster.Name] = int32(0)
			}
			expected[cluster.Name] += int32(1)
		}
	}

	return expected, int32(max)
}
