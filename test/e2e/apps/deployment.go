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

package apps

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/tools/cache"

	"encoding/json"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unstructuredv1 "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	appsclient "k8s.io/client-go/kubernetes/typed/apps/v1"
	watchtools "k8s.io/client-go/tools/watch"
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutil "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"
)

const (
	poll            = 2 * time.Second
	pollLongTimeout = 5 * time.Minute
	dRetryPeriod    = 2 * time.Second
	dRetryTimeout   = 5 * time.Minute
)

var (
	nilRs *appsv1.ReplicaSet
)

var _ = SIGDescribe("Deployment", func() {
	var ns string
	var c clientset.Interface
	var dc dynamic.Interface

	ginkgo.AfterEach(func() {
		failureTrap(c, ns)
	})

	f := framework.NewDefaultFramework("deployment")

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		dc = f.DynamicClient
	})

	ginkgo.It("deployment reaping should cascade to its replica sets and pods", func() {
		testDeleteDeployment(f)
	})
	/*
	  Release: v1.12
	  Testname: Deployment RollingUpdate
	  Description: A conformant Kubernetes distribution MUST support the Deployment with RollingUpdate strategy.
	*/
	framework.ConformanceIt("RollingUpdateDeployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	/*
	  Release: v1.12
	  Testname: Deployment Recreate
	  Description: A conformant Kubernetes distribution MUST support the Deployment with Recreate strategy.
	*/
	framework.ConformanceIt("RecreateDeployment should delete old pods and create new ones", func() {
		testRecreateDeployment(f)
	})
	/*
	  Release: v1.12
	  Testname: Deployment RevisionHistoryLimit
	  Description: A conformant Kubernetes distribution MUST clean up Deployment's ReplicaSets based on
	  the Deployment's `.spec.revisionHistoryLimit`.
	*/
	framework.ConformanceIt("deployment should delete old replica sets", func() {
		testDeploymentCleanUpPolicy(f)
	})
	/*
	  Release: v1.12
	  Testname: Deployment Rollover
	  Description: A conformant Kubernetes distribution MUST support Deployment rollover,
	    i.e. allow arbitrary number of changes to desired state during rolling update
	    before the rollout finishes.
	*/
	framework.ConformanceIt("deployment should support rollover", func() {
		testRolloverDeployment(f)
	})
	ginkgo.It("iterative rollouts should eventually progress", func() {
		testIterativeDeployments(f)
	})
	ginkgo.It("test Deployment ReplicaSet orphaning and adoption regarding controllerRef", func() {
		testDeploymentsControllerRef(f)
	})
	/*
	  Release: v1.12
	  Testname: Deployment Proportional Scaling
	  Description: A conformant Kubernetes distribution MUST support Deployment
	    proportional scaling, i.e. proportionally scale a Deployment's ReplicaSets
	    when a Deployment is scaled.
	*/
	framework.ConformanceIt("deployment should support proportional scaling", func() {
		testProportionalScalingDeployment(f)
	})
	ginkgo.It("should not disrupt a cloud load-balancer's connectivity during rollout", func() {
		e2eskipper.SkipUnlessProviderIs("aws", "azure", "gce", "gke")
		nodes, err := e2enode.GetReadySchedulableNodes(c)
		framework.ExpectNoError(err)
		e2eskipper.SkipUnlessAtLeast(len(nodes.Items), 3, "load-balancer test requires at least 3 schedulable nodes")
		testRollingUpdateDeploymentWithLocalTrafficLoadBalancer(f)
	})
	// TODO: add tests that cover deployment.Spec.MinReadySeconds once we solved clock-skew issues
	// See https://github.com/kubernetes/kubernetes/issues/29229

	ginkgo.It("should run the lifecycle of a Deployment", func() {
		zero := int64(0)
		deploymentResource := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
		testNamespaceName := f.Namespace.Name
		testDeploymentName := "test-deployment"
		testDeploymentInitialImage := imageutils.GetE2EImage(imageutils.Agnhost)
		testDeploymentPatchImage := imageutils.GetE2EImage(imageutils.Pause)
		testDeploymentUpdateImage := imageutils.GetE2EImage(imageutils.Httpd)
		testDeploymentDefaultReplicas := int32(2)
		testDeploymentMinimumReplicas := int32(1)
		testDeploymentNoReplicas := int32(0)
		testDeploymentLabels := map[string]string{"test-deployment-static": "true"}
		testDeploymentLabelsFlat := "test-deployment-static=true"
		testDeploymentLabelSelectors := metav1.LabelSelector{
			MatchLabels: testDeploymentLabels,
		}
		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = testDeploymentLabelsFlat
				return f.ClientSet.AppsV1().Deployments(testNamespaceName).Watch(context.TODO(), options)
			},
		}
		deploymentsList, err := f.ClientSet.AppsV1().Deployments("").List(context.TODO(), metav1.ListOptions{LabelSelector: testDeploymentLabelsFlat})
		framework.ExpectNoError(err, "failed to list Deployments")

		ginkgo.By("creating a Deployment")
		testDeployment := appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:   testDeploymentName,
				Labels: map[string]string{"test-deployment-static": "true"},
			},
			Spec: appsv1.DeploymentSpec{
				Replicas: &testDeploymentDefaultReplicas,
				Selector: &testDeploymentLabelSelectors,
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: testDeploymentLabelSelectors.MatchLabels,
					},
					Spec: v1.PodSpec{
						TerminationGracePeriodSeconds: &zero,
						Containers: []v1.Container{{
							Name:  testDeploymentName,
							Image: testDeploymentInitialImage,
						}},
					},
				},
			},
		}
		_, err = f.ClientSet.AppsV1().Deployments(testNamespaceName).Create(context.TODO(), &testDeployment, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Deployment %v in namespace %v", testDeploymentName, testNamespaceName)

		ginkgo.By("waiting for Deployment to be created")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Added:
				if deployment, ok := event.Object.(*appsv1.Deployment); ok {
					found := deployment.ObjectMeta.Name == testDeployment.Name &&
						deployment.ObjectMeta.Labels["test-deployment-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Added)

		ginkgo.By("waiting for all Replicas to be Ready")
		ctx, cancel = context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if deployment, ok := event.Object.(*appsv1.Deployment); ok {
				found := deployment.ObjectMeta.Name == testDeployment.Name &&
					deployment.ObjectMeta.Labels["test-deployment-static"] == "true" &&
					deployment.Status.ReadyReplicas == testDeploymentDefaultReplicas
				if !found {
					framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v and labels %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas, deployment.ObjectMeta.Labels)
				}
				framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v and labels %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas, deployment.ObjectMeta.Labels)
				return found, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see replicas of %v in namespace %v scale to requested amount of %v", testDeployment.Name, testNamespaceName, testDeploymentDefaultReplicas)

		ginkgo.By("patching the Deployment")
		deploymentPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{"test-deployment": "patched"},
			},
			"spec": map[string]interface{}{
				"replicas": testDeploymentMinimumReplicas,
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"TerminationGracePeriodSeconds": &zero,
						"containers": [1]map[string]interface{}{{
							"name":    testDeploymentName,
							"image":   testDeploymentPatchImage,
							"command": []string{"/bin/sleep", "100000"},
						}},
					},
				},
			},
		})
		framework.ExpectNoError(err, "failed to Marshal Deployment JSON patch")
		_, err = f.ClientSet.AppsV1().Deployments(testNamespaceName).Patch(context.TODO(), testDeploymentName, types.StrategicMergePatchType, []byte(deploymentPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Deployment")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if deployment, ok := event.Object.(*appsv1.Deployment); ok {
					found := deployment.ObjectMeta.Name == testDeployment.Name &&
						deployment.ObjectMeta.Labels["test-deployment-static"] == "true"
					if !found {
						framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
					}
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("waiting for Replicas to scale")
		ctx, cancel = context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if deployment, ok := event.Object.(*appsv1.Deployment); ok {
				found := deployment.ObjectMeta.Name == testDeployment.Name &&
					deployment.ObjectMeta.Labels["test-deployment-static"] == "true" &&
					deployment.Status.ReadyReplicas == testDeploymentMinimumReplicas &&
					deployment.Spec.Template.Spec.Containers[0].Image == testDeploymentPatchImage
				if !found {
					framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
				}
				framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
				return found, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see replicas of %v in namespace %v scale to requested amount of %v", testDeployment.Name, testNamespaceName, testDeploymentMinimumReplicas)

		ginkgo.By("listing Deployments")
		deploymentsList, err = f.ClientSet.AppsV1().Deployments("").List(context.TODO(), metav1.ListOptions{LabelSelector: testDeploymentLabelsFlat})
		framework.ExpectNoError(err, "failed to list Deployments")
		foundDeployment := false
		for _, deploymentItem := range deploymentsList.Items {
			if deploymentItem.ObjectMeta.Name == testDeploymentName &&
				deploymentItem.ObjectMeta.Namespace == testNamespaceName &&
				deploymentItem.ObjectMeta.Labels["test-deployment-static"] == "true" {
				foundDeployment = true
				framework.Logf("Found %v with labels: %v", deploymentItem.ObjectMeta.Name, deploymentItem.ObjectMeta.Labels)
				break
			}
		}
		framework.ExpectEqual(foundDeployment, true, "unable to find the Deployment in list", deploymentsList)

		ginkgo.By("updating the Deployment")
		testDeploymentUpdate := testDeployment
		testDeploymentUpdate.ObjectMeta.Labels["test-deployment"] = "updated"
		testDeploymentUpdate.Spec.Template.Spec.Containers[0].Image = testDeploymentUpdateImage
		testDeploymentDefaultReplicasPointer := &testDeploymentDefaultReplicas
		testDeploymentUpdate.Spec.Replicas = testDeploymentDefaultReplicasPointer
		testDeploymentUpdateUnstructuredMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&testDeploymentUpdate)
		framework.ExpectNoError(err, "failed to convert to unstructured")
		testDeploymentUpdateUnstructured := unstructuredv1.Unstructured{
			Object: testDeploymentUpdateUnstructuredMap,
		}
		// currently this hasn't been able to hit the endpoint replaceAppsV1NamespacedDeploymentStatus
		_, err = dc.Resource(deploymentResource).Namespace(testNamespaceName).Update(context.TODO(), &testDeploymentUpdateUnstructured, metav1.UpdateOptions{}) //, "status")
		framework.ExpectNoError(err, "failed to update the DeploymentStatus")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if deployment, ok := event.Object.(*appsv1.Deployment); ok {
					found := deployment.ObjectMeta.Name == testDeployment.Name &&
						deployment.ObjectMeta.Labels["test-deployment-static"] == "true"
					if !found {
						framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
					}
					framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the DeploymentStatus")
		deploymentGetUnstructured, err := dc.Resource(deploymentResource).Namespace(testNamespaceName).Get(context.TODO(), testDeploymentName, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "failed to fetch the Deployment")
		deploymentGet := appsv1.Deployment{}
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(deploymentGetUnstructured.Object, &deploymentGet)
		framework.ExpectNoError(err, "failed to convert the unstructured response to a Deployment")
		framework.ExpectEqual(deploymentGet.Spec.Template.Spec.Containers[0].Image, testDeploymentUpdateImage, "failed to update image")
		framework.ExpectEqual(deploymentGet.ObjectMeta.Labels["test-deployment"], "updated", "failed to update labels")

		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if deployment, ok := event.Object.(*appsv1.Deployment); ok {
				found := deployment.ObjectMeta.Name == testDeployment.Name &&
					deployment.ObjectMeta.Labels["test-deployment-static"] == "true" &&
					deployment.Status.ReadyReplicas == testDeploymentDefaultReplicas
				if !found {
					framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v and labels %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas, deployment.ObjectMeta.Labels)
				}
				return found, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see replicas of %v in namespace %v scale to requested amount of %v", testDeployment.Name, testNamespaceName, testDeploymentDefaultReplicas)

		ginkgo.By("patching the DeploymentStatus")
		deploymentStatusPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{"test-deployment": "patched-status"},
			},
			"status": map[string]interface{}{
				"readyReplicas": testDeploymentNoReplicas,
			},
		})
		framework.ExpectNoError(err, "failed to Marshal Deployment JSON patch")
		dc.Resource(deploymentResource).Namespace(testNamespaceName).Patch(context.TODO(), testDeploymentName, types.StrategicMergePatchType, []byte(deploymentStatusPatch), metav1.PatchOptions{}, "status")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if deployment, ok := event.Object.(*appsv1.Deployment); ok {
					found := deployment.ObjectMeta.Name == testDeployment.Name &&
						deployment.ObjectMeta.Labels["test-deployment-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the DeploymentStatus")
		deploymentGetUnstructured, err = dc.Resource(deploymentResource).Namespace(testNamespaceName).Get(context.TODO(), testDeploymentName, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "failed to fetch the DeploymentStatus")
		deploymentGet = appsv1.Deployment{}
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(deploymentGetUnstructured.Object, &deploymentGet)
		framework.ExpectNoError(err, "failed to convert the unstructured response to a Deployment")
		framework.ExpectEqual(deploymentGet.Spec.Template.Spec.Containers[0].Image, testDeploymentUpdateImage, "failed to update image")
		framework.ExpectEqual(deploymentGet.ObjectMeta.Labels["test-deployment"], "updated", "failed to update labels")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if deployment, ok := event.Object.(*appsv1.Deployment); ok {
				found := deployment.ObjectMeta.Name == testDeployment.Name &&
					deployment.ObjectMeta.Labels["test-deployment-static"] == "true" &&
					deployment.Status.ReadyReplicas == testDeploymentDefaultReplicas &&
					deployment.Spec.Template.Spec.Containers[0].Image == testDeploymentUpdateImage
				if !found {
					framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
				}
				return found, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see replicas of %v in namespace %v scale to requested amount of %v", testDeployment.Name, testNamespaceName, testDeploymentDefaultReplicas)

		ginkgo.By("deleting the Deployment")
		err = f.ClientSet.AppsV1().Deployments(testNamespaceName).DeleteCollection(context.TODO(), metav1.DeleteOptions{GracePeriodSeconds: &zero}, metav1.ListOptions{LabelSelector: testDeploymentLabelsFlat})
		framework.ExpectNoError(err, "failed to delete Deployment via collection")

		ctx, cancel = context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()
		_, err = watchtools.Until(ctx, deploymentsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				if deployment, ok := event.Object.(*appsv1.Deployment); ok {
					found := deployment.ObjectMeta.Name == testDeployment.Name &&
						deployment.ObjectMeta.Labels["test-deployment-static"] == "true"
					if !found {
						framework.Logf("observed Deployment %v in namespace %v with ReadyReplicas %v", deployment.ObjectMeta.Name, deployment.ObjectMeta.Namespace, deployment.Status.ReadyReplicas)
					}
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Deleted)
	})
})

func failureTrap(c clientset.Interface, ns string) {
	deployments, err := c.AppsV1().Deployments(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		framework.Logf("Could not list Deployments in namespace %q: %v", ns, err)
		return
	}
	for i := range deployments.Items {
		d := deployments.Items[i]

		framework.Logf(spew.Sprintf("Deployment %q:\n%+v\n", d.Name, d))
		_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(&d, c.AppsV1())
		if err != nil {
			framework.Logf("Could not list ReplicaSets for Deployment %q: %v", d.Name, err)
			return
		}
		testutil.LogReplicaSetsOfDeployment(&d, allOldRSs, newRS, framework.Logf)
		rsList := allOldRSs
		if newRS != nil {
			rsList = append(rsList, newRS)
		}
		testutil.LogPodsOfDeployment(c, &d, rsList, framework.Logf)
	}
	// We need print all the ReplicaSets if there are no Deployment object created
	if len(deployments.Items) != 0 {
		return
	}
	framework.Logf("Log out all the ReplicaSets if there is no deployment created")
	rss, err := c.AppsV1().ReplicaSets(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		framework.Logf("Could not list ReplicaSets in namespace %q: %v", ns, err)
		return
	}
	for _, rs := range rss.Items {
		framework.Logf(spew.Sprintf("ReplicaSet %q:\n%+v\n", rs.Name, rs))
		selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			framework.Logf("failed to get selector of ReplicaSet %s: %v", rs.Name, err)
		}
		options := metav1.ListOptions{LabelSelector: selector.String()}
		podList, err := c.CoreV1().Pods(rs.Namespace).List(context.TODO(), options)
		if err != nil {
			framework.Logf("Failed to list Pods in namespace %s: %v", rs.Namespace, err)
			continue
		}
		for _, pod := range podList.Items {
			framework.Logf(spew.Sprintf("pod: %q:\n%+v\n", pod.Name, pod))
		}
	}
}

func intOrStrP(num int) *intstr.IntOrString {
	intstr := intstr.FromInt(num)
	return &intstr
}

func stopDeployment(c clientset.Interface, ns, deploymentName string) {
	deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	framework.Logf("Deleting deployment %s", deploymentName)
	err = e2eresource.DeleteResourceAndWaitForGC(c, appsinternal.Kind("Deployment"), ns, deployment.Name)
	framework.ExpectNoError(err)

	framework.Logf("Ensuring deployment %s was deleted", deploymentName)
	_, err = c.AppsV1().Deployments(ns).Get(context.TODO(), deployment.Name, metav1.GetOptions{})
	framework.ExpectError(err)
	framework.ExpectEqual(apierrors.IsNotFound(err), true)
	framework.Logf("Ensuring deployment %s's RSes were deleted", deploymentName)
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	framework.ExpectNoError(err)
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rss, err := c.AppsV1().ReplicaSets(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	gomega.Expect(rss.Items).Should(gomega.HaveLen(0))
	framework.Logf("Ensuring deployment %s's Pods were deleted", deploymentName)
	var pods *v1.PodList
	if err := wait.PollImmediate(time.Second, timeout, func() (bool, error) {
		pods, err = c.CoreV1().Pods(ns).List(context.TODO(), options)
		if err != nil {
			return false, err
		}
		// Pods may be created by overlapping deployments right after this deployment is deleted, ignore them
		if len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		framework.Failf("Err : %s\n. Failed to remove deployment %s pods : %+v", err, deploymentName, pods)
	}
}

func testDeleteDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-new-deployment"
	podLabels := map[string]string{"name": WebserverImageName}
	replicas := int32(1)
	framework.Logf("Creating simple deployment %s", deploymentName)
	d := e2edeployment.NewDeployment(deploymentName, replicas, podLabels, WebserverImageName, WebserverImage, appsv1.RollingUpdateDeploymentStrategyType)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", v1.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	deploy, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Wait for it to be updated to revision 1
	err = e2edeployment.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", WebserverImage)
	framework.ExpectNoError(err)

	err = e2edeployment.WaitForDeploymentComplete(c, deploy)
	framework.ExpectNoError(err)

	deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	framework.ExpectNoError(err)
	framework.ExpectNotEqual(newRS, nilRs)
	stopDeployment(c, ns, deploymentName)
}

func testRollingUpdateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create webserver pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  WebserverImageName,
	}

	rsName := "test-rolling-update-controller"
	replicas := int32(1)
	rsRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rsRevision
	rs := newRS(rsName, replicas, rsPodLabels, WebserverImageName, WebserverImage, nil)
	rs.Annotations = annotations
	framework.Logf("Creating replica set %q (going to be adopted)", rs.Name)
	_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), rs, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	// Verify that the required pods have come up.
	err = e2epod.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %s", err)

	// Create a deployment to delete webserver pods and instead bring up agnhost pods.
	deploymentName := "test-rolling-update-deployment"
	framework.Logf("Creating deployment %q", deploymentName)
	d := e2edeployment.NewDeployment(deploymentName, replicas, deploymentPodLabels, AgnhostImageName, AgnhostImage, appsv1.RollingUpdateDeploymentStrategyType)
	deploy, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Wait for it to be updated to revision 3546343826724305833.
	framework.Logf("Ensuring deployment %q gets the next revision from the one the adopted replica set %q has", deploy.Name, rs.Name)
	err = e2edeployment.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3546343826724305833", AgnhostImage)
	framework.ExpectNoError(err)

	framework.Logf("Ensuring status for deployment %q is the expected", deploy.Name)
	err = e2edeployment.WaitForDeploymentComplete(c, deploy)
	framework.ExpectNoError(err)

	// There should be 1 old RS (webserver-controller, which is adopted)
	framework.Logf("Ensuring deployment %q has one old replica set (the one it adopted)", deploy.Name)
	deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c.AppsV1())
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(allOldRSs), 1)
}

func testRecreateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	// Create a deployment that brings up agnhost pods.
	deploymentName := "test-recreate-deployment"
	framework.Logf("Creating deployment %q", deploymentName)
	d := e2edeployment.NewDeployment(deploymentName, int32(1), map[string]string{"name": "sample-pod-3"}, AgnhostImageName, AgnhostImage, appsv1.RecreateDeploymentStrategyType)
	deployment, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Wait for it to be updated to revision 1
	framework.Logf("Waiting deployment %q to be updated to revision 1", deploymentName)
	err = e2edeployment.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", AgnhostImage)
	framework.ExpectNoError(err)

	framework.Logf("Waiting deployment %q to complete", deploymentName)
	err = e2edeployment.WaitForDeploymentComplete(c, deployment)
	framework.ExpectNoError(err)

	// Update deployment to delete agnhost pods and bring up webserver pods.
	framework.Logf("Triggering a new rollout for deployment %q", deploymentName)
	deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deploymentName, func(update *appsv1.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = WebserverImageName
		update.Spec.Template.Spec.Containers[0].Image = WebserverImage
	})
	framework.ExpectNoError(err)

	framework.Logf("Watching deployment %q to verify that new pods will not run with olds pods", deploymentName)
	err = watchRecreateDeployment(c, deployment)
	framework.ExpectNoError(err)
}

// testDeploymentCleanUpPolicy tests that deployment supports cleanup policy
func testDeploymentCleanUpPolicy(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create webserver pods.
	deploymentPodLabels := map[string]string{"name": "cleanup-pod"}
	rsPodLabels := map[string]string{
		"name": "cleanup-pod",
		"pod":  WebserverImageName,
	}
	rsName := "test-cleanup-controller"
	replicas := int32(1)
	revisionHistoryLimit := utilpointer.Int32Ptr(0)
	_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), newRS(rsName, replicas, rsPodLabels, WebserverImageName, WebserverImage, nil), metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Verify that the required pods have come up.
	err = e2epod.VerifyPodsRunning(c, ns, "cleanup-pod", false, replicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	// Create a deployment to delete webserver pods and instead bring up agnhost pods.
	deploymentName := "test-cleanup-deployment"
	framework.Logf("Creating deployment %s", deploymentName)

	pods, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String()})
	framework.ExpectNoError(err, "Failed to query for pods: %v", err)

	options := metav1.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	w, err := c.CoreV1().Pods(ns).Watch(context.TODO(), options)
	framework.ExpectNoError(err)
	go func() {
		defer ginkgo.GinkgoRecover()
		// There should be only one pod being created, which is the pod with the agnhost image.
		// The old RS shouldn't create new pod when deployment controller adding pod template hash label to its selector.
		numPodCreation := 1
		for {
			select {
			case event := <-w.ResultChan():
				if event.Type != watch.Added {
					continue
				}
				numPodCreation--
				if numPodCreation < 0 {
					framework.Failf("Expect only one pod creation, the second creation event: %#v\n", event)
				}
				pod, ok := event.Object.(*v1.Pod)
				if !ok {
					framework.Failf("Expect event Object to be a pod")
				}
				if pod.Spec.Containers[0].Name != AgnhostImageName {
					framework.Failf("Expect the created pod to have container name %s, got pod %#v\n", AgnhostImageName, pod)
				}
			case <-stopCh:
				return
			}
		}
	}()
	d := e2edeployment.NewDeployment(deploymentName, replicas, deploymentPodLabels, AgnhostImageName, AgnhostImage, appsv1.RollingUpdateDeploymentStrategyType)
	d.Spec.RevisionHistoryLimit = revisionHistoryLimit
	_, err = c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting for deployment %s history to be cleaned up", deploymentName))
	err = waitForDeploymentOldRSsNum(c, ns, deploymentName, int(*revisionHistoryLimit))
	framework.ExpectNoError(err)
}

// testRolloverDeployment tests that deployment supports rollover.
// i.e. we can change desired state and kick off rolling update, then change desired state again before it finishes.
func testRolloverDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	podName := "rollover-pod"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  WebserverImageName,
	}

	rsName := "test-rollover-controller"
	rsReplicas := int32(1)
	_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), newRS(rsName, rsReplicas, rsPodLabels, WebserverImageName, WebserverImage, nil), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	// Verify that the required pods have come up.
	err = e2epod.VerifyPodsRunning(c, ns, podName, false, rsReplicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	// Wait for replica set to become ready before adopting it.
	framework.Logf("Waiting for pods owned by replica set %q to become ready", rsName)
	err = e2ereplicaset.WaitForReadyReplicaSet(c, ns, rsName)
	framework.ExpectNoError(err)

	// Create a deployment to delete webserver pods and instead bring up redis-slave pods.
	// We use a nonexistent image here, so that we make sure it won't finish
	deploymentName, deploymentImageName := "test-rollover-deployment", "redis-slave"
	deploymentReplicas := int32(1)
	deploymentImage := "gcr.io/google_samples/gb-redisslave:nonexistent"
	deploymentStrategyType := appsv1.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %q", deploymentName)
	newDeployment := e2edeployment.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	newDeployment.Spec.Strategy.RollingUpdate = &appsv1.RollingUpdateDeployment{
		MaxUnavailable: intOrStrP(0),
		MaxSurge:       intOrStrP(1),
	}
	newDeployment.Spec.MinReadySeconds = int32(10)
	_, err = c.AppsV1().Deployments(ns).Create(context.TODO(), newDeployment, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Verify that the pods were scaled up and down as expected.
	deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	framework.Logf("Make sure deployment %q performs scaling operations", deploymentName)
	// Make sure the deployment starts to scale up and down replica sets by checking if its updated replicas >= 1
	err = waitForDeploymentUpdatedReplicasGTE(c, ns, deploymentName, deploymentReplicas, deployment.Generation)
	framework.ExpectNoError(err)
	// Check if it's updated to revision 1 correctly
	framework.Logf("Check revision of new replica set for deployment %q", deploymentName)
	err = checkDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	framework.ExpectNoError(err)

	framework.Logf("Ensure that both replica sets have 1 created replica")
	oldRS, err := c.AppsV1().ReplicaSets(ns).Get(context.TODO(), rsName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ensureReplicas(oldRS, int32(1))
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	framework.ExpectNoError(err)
	ensureReplicas(newRS, int32(1))

	// The deployment is stuck, update it to rollover the above 2 ReplicaSets and bring up agnhost pods.
	framework.Logf("Rollover old replica sets for deployment %q with new image update", deploymentName)
	updatedDeploymentImageName, updatedDeploymentImage := AgnhostImageName, AgnhostImage
	deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, newDeployment.Name, func(update *appsv1.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	framework.ExpectNoError(err)

	// Use observedGeneration to determine if the controller noticed the pod template update.
	framework.Logf("Wait deployment %q to be observed by the deployment controller", deploymentName)
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	framework.ExpectNoError(err)

	// Wait for it to be updated to revision 2
	framework.Logf("Wait for revision update of deployment %q to 2", deploymentName)
	err = e2edeployment.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	framework.ExpectNoError(err)

	framework.Logf("Make sure deployment %q is complete", deploymentName)
	err = waitForDeploymentCompleteAndCheckRolling(c, deployment)
	framework.ExpectNoError(err)

	framework.Logf("Ensure that both old replica sets have no replicas")
	oldRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), rsName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ensureReplicas(oldRS, int32(0))
	// Not really the new replica set anymore but we GET by name so that's fine.
	newRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), newRS.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ensureReplicas(newRS, int32(0))
}

func ensureReplicas(rs *appsv1.ReplicaSet, replicas int32) {
	framework.ExpectEqual(*rs.Spec.Replicas, replicas)
	framework.ExpectEqual(rs.Status.Replicas, replicas)
}

func randomScale(d *appsv1.Deployment, i int) {
	switch r := rand.Float32(); {
	case r < 0.3:
		framework.Logf("%02d: scaling up", i)
		*(d.Spec.Replicas)++
	case r < 0.6:
		if *(d.Spec.Replicas) > 1 {
			framework.Logf("%02d: scaling down", i)
			*(d.Spec.Replicas)--
		}
	}
}

func testIterativeDeployments(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": WebserverImageName}
	replicas := int32(6)
	zero := int64(0)
	two := int32(2)

	// Create a webserver deployment.
	deploymentName := "webserver"
	thirty := int32(30)
	d := e2edeployment.NewDeployment(deploymentName, replicas, podLabels, WebserverImageName, WebserverImage, appsv1.RollingUpdateDeploymentStrategyType)
	d.Spec.ProgressDeadlineSeconds = &thirty
	d.Spec.RevisionHistoryLimit = &two
	d.Spec.Template.Spec.TerminationGracePeriodSeconds = &zero
	framework.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	iterations := 20
	for i := 0; i < iterations; i++ {
		if r := rand.Float32(); r < 0.6 {
			time.Sleep(time.Duration(float32(i) * r * float32(time.Second)))
		}

		switch n := rand.Float32(); {
		case n < 0.2:
			// trigger a new deployment
			framework.Logf("%02d: triggering a new rollout for deployment %q", i, deployment.Name)
			deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
				newEnv := v1.EnvVar{Name: fmt.Sprintf("A%d", i), Value: fmt.Sprintf("%d", i)}
				update.Spec.Template.Spec.Containers[0].Env = append(update.Spec.Template.Spec.Containers[0].Env, newEnv)
				randomScale(update, i)
			})
			framework.ExpectNoError(err)

		case n < 0.4:
			// rollback to the previous version
			framework.Logf("%02d: rolling back a rollout for deployment %q", i, deployment.Name)
			deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
				if update.Annotations == nil {
					update.Annotations = make(map[string]string)
				}
				update.Annotations[appsv1.DeprecatedRollbackTo] = "0"
			})
			framework.ExpectNoError(err)

		case n < 0.6:
			// just scaling
			framework.Logf("%02d: scaling deployment %q", i, deployment.Name)
			deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
				randomScale(update, i)
			})
			framework.ExpectNoError(err)

		case n < 0.8:
			// toggling the deployment
			if deployment.Spec.Paused {
				framework.Logf("%02d: pausing deployment %q", i, deployment.Name)
				deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
					update.Spec.Paused = true
					randomScale(update, i)
				})
				framework.ExpectNoError(err)
			} else {
				framework.Logf("%02d: resuming deployment %q", i, deployment.Name)
				deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
					update.Spec.Paused = false
					randomScale(update, i)
				})
				framework.ExpectNoError(err)
			}

		default:
			// arbitrarily delete deployment pods
			framework.Logf("%02d: arbitrarily deleting one or more deployment pods for deployment %q", i, deployment.Name)
			selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
			framework.ExpectNoError(err)
			opts := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.CoreV1().Pods(ns).List(context.TODO(), opts)
			framework.ExpectNoError(err)
			if len(podList.Items) == 0 {
				framework.Logf("%02d: no deployment pods to delete", i)
				continue
			}
			for p := range podList.Items {
				if rand.Float32() < 0.5 {
					continue
				}
				name := podList.Items[p].Name
				framework.Logf("%02d: deleting deployment pod %q", i, name)
				err := c.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err)
				}
			}
		}
	}

	// unpause the deployment if we end up pausing it
	deployment, err = c.AppsV1().Deployments(ns).Get(context.TODO(), deployment.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	if deployment.Spec.Paused {
		deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
			update.Spec.Paused = false
		})
		framework.ExpectNoError(err)
	}

	framework.Logf("Waiting for deployment %q to be observed by the controller", deploymentName)
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	framework.ExpectNoError(err)

	framework.Logf("Waiting for deployment %q status", deploymentName)
	err = e2edeployment.WaitForDeploymentComplete(c, deployment)
	framework.ExpectNoError(err)

	framework.Logf("Checking deployment %q for a complete condition", deploymentName)
	err = waitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.NewRSAvailableReason, appsv1.DeploymentProgressing)
	framework.ExpectNoError(err)
}

func testDeploymentsControllerRef(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-orphan-deployment"
	framework.Logf("Creating Deployment %q", deploymentName)
	podLabels := map[string]string{"name": WebserverImageName}
	replicas := int32(1)
	d := e2edeployment.NewDeployment(deploymentName, replicas, podLabels, WebserverImageName, WebserverImage, appsv1.RollingUpdateDeploymentStrategyType)
	deploy, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2edeployment.WaitForDeploymentComplete(c, deploy)
	framework.ExpectNoError(err)

	framework.Logf("Verifying Deployment %q has only one ReplicaSet", deploymentName)
	rsList := listDeploymentReplicaSets(c, ns, podLabels)
	framework.ExpectEqual(len(rsList.Items), 1)

	framework.Logf("Obtaining the ReplicaSet's UID")
	orphanedRSUID := rsList.Items[0].UID

	framework.Logf("Checking the ReplicaSet has the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	framework.ExpectNoError(err)

	framework.Logf("Deleting Deployment %q and orphaning its ReplicaSet", deploymentName)
	err = orphanDeploymentReplicaSets(c, deploy)
	framework.ExpectNoError(err)

	ginkgo.By("Wait for the ReplicaSet to be orphaned")
	err = wait.Poll(dRetryPeriod, dRetryTimeout, waitDeploymentReplicaSetsOrphaned(c, ns, podLabels))
	framework.ExpectNoError(err, "error waiting for Deployment ReplicaSet to be orphaned")

	deploymentName = "test-adopt-deployment"
	framework.Logf("Creating Deployment %q to adopt the ReplicaSet", deploymentName)
	d = e2edeployment.NewDeployment(deploymentName, replicas, podLabels, WebserverImageName, WebserverImage, appsv1.RollingUpdateDeploymentStrategyType)
	deploy, err = c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2edeployment.WaitForDeploymentComplete(c, deploy)
	framework.ExpectNoError(err)

	framework.Logf("Waiting for the ReplicaSet to have the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	framework.ExpectNoError(err)

	framework.Logf("Verifying no extra ReplicaSet is created (Deployment %q still has only one ReplicaSet after adoption)", deploymentName)
	rsList = listDeploymentReplicaSets(c, ns, podLabels)
	framework.ExpectEqual(len(rsList.Items), 1)

	framework.Logf("Verifying the ReplicaSet has the same UID as the orphaned ReplicaSet")
	framework.ExpectEqual(rsList.Items[0].UID, orphanedRSUID)
}

// testProportionalScalingDeployment tests that when a RollingUpdate Deployment is scaled in the middle
// of a rollout (either in progress or paused), then the Deployment will balance additional replicas
// in existing active ReplicaSets (ReplicaSets with more than 0 replica) in order to mitigate risk.
func testProportionalScalingDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": WebserverImageName}
	replicas := int32(10)

	// Create a webserver deployment.
	deploymentName := "webserver-deployment"
	d := e2edeployment.NewDeployment(deploymentName, replicas, podLabels, WebserverImageName, WebserverImage, appsv1.RollingUpdateDeploymentStrategyType)
	d.Spec.Strategy.RollingUpdate = new(appsv1.RollingUpdateDeployment)
	d.Spec.Strategy.RollingUpdate.MaxSurge = intOrStrP(3)
	d.Spec.Strategy.RollingUpdate.MaxUnavailable = intOrStrP(2)

	framework.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	framework.Logf("Waiting for observed generation %d", deployment.Generation)
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	framework.ExpectNoError(err)

	// Verify that the required pods have come up.
	framework.Logf("Waiting for all required pods to come up")
	err = e2epod.VerifyPodsRunning(c, ns, WebserverImageName, false, *(deployment.Spec.Replicas))
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	framework.Logf("Waiting for deployment %q to complete", deployment.Name)
	err = e2edeployment.WaitForDeploymentComplete(c, deployment)
	framework.ExpectNoError(err)

	firstRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	framework.ExpectNoError(err)

	// Update the deployment with a non-existent image so that the new replica set
	// will be blocked to simulate a partial rollout.
	framework.Logf("Updating deployment %q with a non-existent image", deploymentName)
	deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *appsv1.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "webserver:404"
	})
	framework.ExpectNoError(err)

	framework.Logf("Waiting for observed generation %d", deployment.Generation)
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	framework.ExpectNoError(err)

	// Checking state of first rollout's replicaset.
	maxUnavailable, err := intstr.GetScaledValueFromIntOrPercent(deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, int(*(deployment.Spec.Replicas)), false)
	framework.ExpectNoError(err)

	// First rollout's replicaset should have Deployment's (replicas - maxUnavailable) = 10 - 2 = 8 available replicas.
	minAvailableReplicas := replicas - int32(maxUnavailable)
	framework.Logf("Waiting for the first rollout's replicaset to have .status.availableReplicas = %d", minAvailableReplicas)
	err = e2ereplicaset.WaitForReplicaSetTargetAvailableReplicas(c, firstRS, minAvailableReplicas)
	framework.ExpectNoError(err)

	// First rollout's replicaset should have .spec.replicas = 8 too.
	framework.Logf("Waiting for the first rollout's replicaset to have .spec.replicas = %d", minAvailableReplicas)
	err = waitForReplicaSetTargetSpecReplicas(c, firstRS, minAvailableReplicas)
	framework.ExpectNoError(err)

	// The desired replicas wait makes sure that the RS controller has created expected number of pods.
	framework.Logf("Waiting for the first rollout's replicaset of deployment %q to have desired number of replicas", deploymentName)
	firstRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), firstRS.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	err = waitForReplicaSetDesiredReplicas(c.AppsV1(), firstRS)
	framework.ExpectNoError(err)

	// Checking state of second rollout's replicaset.
	secondRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	framework.ExpectNoError(err)

	maxSurge, err := intstr.GetScaledValueFromIntOrPercent(deployment.Spec.Strategy.RollingUpdate.MaxSurge, int(*(deployment.Spec.Replicas)), false)
	framework.ExpectNoError(err)

	// Second rollout's replicaset should have 0 available replicas.
	framework.Logf("Verifying that the second rollout's replicaset has .status.availableReplicas = 0")
	framework.ExpectEqual(secondRS.Status.AvailableReplicas, int32(0))

	// Second rollout's replicaset should have Deployment's (replicas + maxSurge - first RS's replicas) = 10 + 3 - 8 = 5 for .spec.replicas.
	newReplicas := replicas + int32(maxSurge) - minAvailableReplicas
	framework.Logf("Waiting for the second rollout's replicaset to have .spec.replicas = %d", newReplicas)
	err = waitForReplicaSetTargetSpecReplicas(c, secondRS, newReplicas)
	framework.ExpectNoError(err)

	// The desired replicas wait makes sure that the RS controller has created expected number of pods.
	framework.Logf("Waiting for the second rollout's replicaset of deployment %q to have desired number of replicas", deploymentName)
	secondRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), secondRS.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	err = waitForReplicaSetDesiredReplicas(c.AppsV1(), secondRS)
	framework.ExpectNoError(err)

	// Check the deployment's minimum availability.
	framework.Logf("Verifying that deployment %q has minimum required number of available replicas", deploymentName)
	if deployment.Status.AvailableReplicas < minAvailableReplicas {
		err = fmt.Errorf("observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, minAvailableReplicas)
		framework.ExpectNoError(err)
	}

	// Scale the deployment to 30 replicas.
	newReplicas = int32(30)
	framework.Logf("Scaling up the deployment %q from %d to %d", deploymentName, replicas, newReplicas)
	_, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *appsv1.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	framework.ExpectNoError(err)

	framework.Logf("Waiting for the replicasets of deployment %q to have desired number of replicas", deploymentName)
	firstRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), firstRS.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	secondRS, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), secondRS.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	// First rollout's replicaset should have .spec.replicas = 8 + (30-10)*(8/13) = 8 + 12 = 20 replicas.
	// Note that 12 comes from rounding (30-10)*(8/13) to nearest integer.
	framework.Logf("Verifying that first rollout's replicaset has .spec.replicas = 20")
	err = waitForReplicaSetTargetSpecReplicas(c, firstRS, 20)
	framework.ExpectNoError(err)

	// Second rollout's replicaset should have .spec.replicas = 5 + (30-10)*(5/13) = 5 + 8 = 13 replicas.
	// Note that 8 comes from rounding (30-10)*(5/13) to nearest integer.
	framework.Logf("Verifying that second rollout's replicaset has .spec.replicas = 13")
	err = waitForReplicaSetTargetSpecReplicas(c, secondRS, 13)
	framework.ExpectNoError(err)
}

func checkDeploymentReplicaSetsControllerRef(c clientset.Interface, ns string, uid types.UID, label map[string]string) error {
	rsList := listDeploymentReplicaSets(c, ns, label)
	for _, rs := range rsList.Items {
		// This rs is adopted only when its controller ref is update
		if controllerRef := metav1.GetControllerOf(&rs); controllerRef == nil || controllerRef.UID != uid {
			return fmt.Errorf("ReplicaSet %s has unexpected controllerRef %v", rs.Name, controllerRef)
		}
	}
	return nil
}

func waitDeploymentReplicaSetsOrphaned(c clientset.Interface, ns string, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		rsList := listDeploymentReplicaSets(c, ns, label)
		for _, rs := range rsList.Items {
			// This rs is orphaned only when controller ref is cleared
			if controllerRef := metav1.GetControllerOf(&rs); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func listDeploymentReplicaSets(c clientset.Interface, ns string, label map[string]string) *appsv1.ReplicaSetList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rsList, err := c.AppsV1().ReplicaSets(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	gomega.Expect(len(rsList.Items)).To(gomega.BeNumerically(">", 0))
	return rsList
}

func orphanDeploymentReplicaSets(c clientset.Interface, d *appsv1.Deployment) error {
	trueVar := true
	deleteOptions := metav1.DeleteOptions{OrphanDependents: &trueVar}
	deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(d.UID))
	return c.AppsV1().Deployments(d.Namespace).Delete(context.TODO(), d.Name, deleteOptions)
}

func testRollingUpdateDeploymentWithLocalTrafficLoadBalancer(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	name := "test-rolling-update-with-lb"
	framework.Logf("Creating Deployment %q", name)
	podLabels := map[string]string{"name": name}
	replicas := int32(3)
	d := e2edeployment.NewDeployment(name, replicas, podLabels, AgnhostImageName, AgnhostImage, appsv1.RollingUpdateDeploymentStrategyType)
	// NewDeployment assigned the same value to both d.Spec.Selector and
	// d.Spec.Template.Labels, so mutating the one would mutate the other.
	// Thus we need to set d.Spec.Template.Labels to a new value if we want
	// to mutate it alone.
	d.Spec.Template.Labels = map[string]string{
		"iteration": "0",
		"name":      name,
	}
	d.Spec.Template.Spec.Containers[0].Args = []string{"netexec", "--http-port=80", "--udp-port=80"}
	// To ensure that a node that had a local endpoint prior to a rolling
	// update continues to have a local endpoint throughout the rollout, we
	// need an affinity policy that will cause pods to be scheduled on the
	// same nodes as old pods, and we need the deployment to scale up a new
	// pod before deleting an old pod.  This affinity policy will define
	// inter-pod affinity for pods of different rollouts and anti-affinity
	// for pods of the same rollout, so it will need to be updated when
	// performing a rollout.
	setAffinities(d, false)
	d.Spec.Strategy.RollingUpdate = &appsv1.RollingUpdateDeployment{
		MaxSurge:       intOrStrP(1),
		MaxUnavailable: intOrStrP(0),
	}
	deployment, err := c.AppsV1().Deployments(ns).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2edeployment.WaitForDeploymentComplete(c, deployment)
	framework.ExpectNoError(err)

	framework.Logf("Creating a service %s with type=LoadBalancer and externalTrafficPolicy=Local in namespace %s", name, ns)
	jig := e2eservice.NewTestJig(c, ns, name)
	jig.Labels = podLabels
	service, err := jig.CreateLoadBalancerService(e2eservice.GetServiceLoadBalancerCreationTimeout(c), func(svc *v1.Service) {
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
	})
	framework.ExpectNoError(err)

	lbNameOrAddress := e2eservice.GetIngressPoint(&service.Status.LoadBalancer.Ingress[0])
	svcPort := int(service.Spec.Ports[0].Port)

	framework.Logf("Hitting the replica set's pods through the service's load balancer")
	timeout := e2eservice.LoadBalancerLagTimeoutDefault
	if framework.ProviderIs("aws") {
		timeout = e2eservice.LoadBalancerLagTimeoutAWS
	}
	e2eservice.TestReachableHTTP(lbNameOrAddress, svcPort, timeout)

	framework.Logf("Starting a goroutine to watch the service's endpoints in the background")
	done := make(chan struct{})
	failed := make(chan struct{})
	defer close(done)
	go func() {
		defer ginkgo.GinkgoRecover()
		expectedNodes, err := jig.GetEndpointNodeNames()
		framework.ExpectNoError(err)
		// The affinity policy should ensure that before an old pod is
		// deleted, a new pod will have been created on the same node.
		// Thus the set of nodes with local endpoints for the service
		// should remain unchanged.
		wait.Until(func() {
			actualNodes, err := jig.GetEndpointNodeNames()
			framework.ExpectNoError(err)
			if !actualNodes.Equal(expectedNodes) {
				framework.Logf("The set of nodes with local endpoints changed; started with %v, now have %v", expectedNodes.List(), actualNodes.List())
				failed <- struct{}{}
			}
		}, framework.Poll, done)
	}()

	framework.Logf("Triggering a rolling deployment several times")
	for i := 1; i <= 3; i++ {
		framework.Logf("Updating label deployment %q pod spec (iteration #%d)", name, i)
		deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *appsv1.Deployment) {
			update.Spec.Template.Labels["iteration"] = fmt.Sprintf("%d", i)
			setAffinities(update, true)
		})
		framework.ExpectNoError(err)

		framework.Logf("Waiting for observed generation %d", deployment.Generation)
		err = waitForObservedDeployment(c, ns, name, deployment.Generation)
		framework.ExpectNoError(err)

		framework.Logf("Make sure deployment %q is complete", name)
		err = waitForDeploymentCompleteAndCheckRolling(c, deployment)
		framework.ExpectNoError(err)
	}

	select {
	case <-failed:
		framework.Failf("Connectivity to the load balancer was interrupted")
	case <-time.After(1 * time.Minute):
	}
}

// setAffinities set PodAntiAffinity across pods from the same generation
// of Deployment and if, explicitly requested, also affinity with pods
// from other generations.
// It is required to make those "Required" so that in large clusters where
// scheduler may not score all nodes if a lot of them are feasible, the
// test will also have a chance to pass.
func setAffinities(d *appsv1.Deployment, setAffinity bool) {
	affinity := &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					TopologyKey: "kubernetes.io/hostname",
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "name",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{d.Spec.Template.Labels["name"]},
							},
							{
								Key:      "iteration",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{d.Spec.Template.Labels["iteration"]},
							},
						},
					},
				},
			},
		},
	}
	if setAffinity {
		affinity.PodAffinity = &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					TopologyKey: "kubernetes.io/hostname",
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "name",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{d.Spec.Template.Labels["name"]},
							},
							{
								Key:      "iteration",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{d.Spec.Template.Labels["iteration"]},
							},
						},
					},
				},
			},
		}
	}
	d.Spec.Template.Spec.Affinity = affinity
}

// watchRecreateDeployment watches Recreate deployments and ensures no new pods will run at the same time with
// old pods.
func watchRecreateDeployment(c clientset.Interface, d *appsv1.Deployment) error {
	if d.Spec.Strategy.Type != appsv1.RecreateDeploymentStrategyType {
		return fmt.Errorf("deployment %q does not use a Recreate strategy: %s", d.Name, d.Spec.Strategy.Type)
	}

	fieldSelector := fields.OneTermEqualSelector("metadata.name", d.Name).String()
	w := &cache.ListWatch{
		WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
			options.FieldSelector = fieldSelector
			return c.AppsV1().Deployments(d.Namespace).Watch(context.TODO(), options)
		},
	}

	status := d.Status

	condition := func(event watch.Event) (bool, error) {
		d := event.Object.(*appsv1.Deployment)
		status = d.Status

		if d.Status.UpdatedReplicas > 0 && d.Status.Replicas != d.Status.UpdatedReplicas {
			_, allOldRSs, err := deploymentutil.GetOldReplicaSets(d, c.AppsV1())
			newRS, nerr := deploymentutil.GetNewReplicaSet(d, c.AppsV1())
			if err == nil && nerr == nil {
				framework.Logf("%+v", d)
				testutil.LogReplicaSetsOfDeployment(d, allOldRSs, newRS, framework.Logf)
				testutil.LogPodsOfDeployment(c, d, append(allOldRSs, newRS), framework.Logf)
			}
			return false, fmt.Errorf("deployment %q is running new pods alongside old pods: %#v", d.Name, status)
		}

		return *(d.Spec.Replicas) == d.Status.Replicas &&
			*(d.Spec.Replicas) == d.Status.UpdatedReplicas &&
			d.Generation <= d.Status.ObservedGeneration, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	_, err := watchtools.Until(ctx, d.ResourceVersion, w, condition)
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("deployment %q never completed: %#v", d.Name, status)
	}
	return err
}

// waitForDeploymentOldRSsNum waits for the deployment to clean up old rcs.
func waitForDeploymentOldRSsNum(c clientset.Interface, ns, deploymentName string, desiredRSNum int) error {
	var oldRSs []*appsv1.ReplicaSet
	var d *appsv1.Deployment

	pollErr := wait.PollImmediate(poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		d = deployment

		_, oldRSs, err = deploymentutil.GetOldReplicaSets(deployment, c.AppsV1())
		if err != nil {
			return false, err
		}
		return len(oldRSs) == desiredRSNum, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("%d old replica sets were not cleaned up for deployment %q", len(oldRSs)-desiredRSNum, deploymentName)
		testutil.LogReplicaSetsOfDeployment(d, oldRSs, nil, framework.Logf)
	}
	return pollErr
}

// waitForReplicaSetDesiredReplicas waits until the replicaset has desired number of replicas.
func waitForReplicaSetDesiredReplicas(rsClient appsclient.ReplicaSetsGetter, replicaSet *appsv1.ReplicaSet) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := rsClient.ReplicaSets(replicaSet.Namespace).Get(context.TODO(), replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.Replicas == *(replicaSet.Spec.Replicas) && rs.Status.Replicas == *(rs.Spec.Replicas), nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never had desired number of replicas", replicaSet.Name)
	}
	return err
}

// waitForReplicaSetTargetSpecReplicas waits for .spec.replicas of a RS to equal targetReplicaNum
func waitForReplicaSetTargetSpecReplicas(c clientset.Interface, replicaSet *appsv1.ReplicaSet, targetReplicaNum int32) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(replicaSet.Namespace).Get(context.TODO(), replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && *rs.Spec.Replicas == targetReplicaNum, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never had desired number of .spec.replicas", replicaSet.Name)
	}
	return err
}

// checkDeploymentRevisionAndImage checks if the input deployment's and its new replica set's revision and image are as expected.
func checkDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName, revision, image string) error {
	return testutil.CheckDeploymentRevisionAndImage(c, ns, deploymentName, revision, image)
}

// waitForObservedDeployment waits for the specified deployment generation.
func waitForObservedDeployment(c clientset.Interface, ns, deploymentName string, desiredGeneration int64) error {
	return testutil.WaitForObservedDeployment(c, ns, deploymentName, desiredGeneration)
}

// waitForDeploymentWithCondition waits for the specified deployment condition.
func waitForDeploymentWithCondition(c clientset.Interface, ns, deploymentName, reason string, condType appsv1.DeploymentConditionType) error {
	return testutil.WaitForDeploymentWithCondition(c, ns, deploymentName, reason, condType, framework.Logf, poll, pollLongTimeout)
}

// waitForDeploymentCompleteAndCheckRolling waits for the deployment to complete, and check rolling update strategy isn't broken at any times.
// Rolling update strategy should not be broken during a rolling update.
func waitForDeploymentCompleteAndCheckRolling(c clientset.Interface, d *appsv1.Deployment) error {
	return testutil.WaitForDeploymentCompleteAndCheckRolling(c, d, framework.Logf, poll, pollLongTimeout)
}

// waitForDeploymentUpdatedReplicasGTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func waitForDeploymentUpdatedReplicasGTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int32, desiredGeneration int64) error {
	return testutil.WaitForDeploymentUpdatedReplicasGTE(c, ns, deploymentName, minUpdatedReplicas, desiredGeneration, poll, pollLongTimeout)
}
