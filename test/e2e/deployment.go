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
	"math/rand"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// nginxImage defined in kubectl.go
	nginxImageName = "nginx"
	redisImage     = "gcr.io/google_containers/redis:e2e"
	redisImageName = "redis"
)

var _ = framework.KubeDescribe("Deployment", func() {
	f := framework.NewDefaultFramework("deployment")

	// TODO: Add failure traps once we have JustAfterEach
	// See https://github.com/onsi/ginkgo/issues/303

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("deployment reaping should cascade to its replica sets and pods", func() {
		testDeleteDeployment(f)
	})
	It("scaled rollout deployment should not block on annotation check", func() {
		testScaledRolloutDeployment(f)
	})
	It("overlapping deployment should not fight with each other", func() {
		testOverlappingDeployment(f)
	})
	// TODO: add tests that cover deployment.Spec.MinReadySeconds once we solved clock-skew issues
	// See https://github.com/kubernetes/kubernetes/issues/29229
})

func newDeploymentRollback(name string, annotations map[string]string, revision int64) *extensions.DeploymentRollback {
	return &extensions.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: annotations,
		RollbackTo:         extensions.RollbackConfig{Revision: revision},
	}
}

// checkDeploymentRevision checks if the input deployment's and its new replica set's revision and images are as expected.
func checkDeploymentRevision(c clientset.Interface, ns, deploymentName, revision, imageName, image string) (*extensions.Deployment, *extensions.ReplicaSet) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	// Check revision of the new replica set of this deployment
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS.Annotations).NotTo(Equal(nil))
	Expect(newRS.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	// Check revision of This deployment
	Expect(deployment.Annotations).NotTo(Equal(nil))
	Expect(deployment.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	if len(imageName) > 0 {
		// Check the image the new replica set creates
		Expect(newRS.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(newRS.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
		// Check the image the deployment creates
		Expect(deployment.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(deployment.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
	}
	return deployment, newRS
}

func stopDeploymentOverlap(c clientset.Interface, internalClient internalclientset.Interface, ns, deploymentName, overlapWith string) {
	stopDeploymentMaybeOverlap(c, internalClient, ns, deploymentName, overlapWith)
}

func stopDeployment(c clientset.Interface, internalClient internalclientset.Interface, ns, deploymentName string) {
	stopDeploymentMaybeOverlap(c, internalClient, ns, deploymentName, "")
}

func stopDeploymentMaybeOverlap(c clientset.Interface, internalClient internalclientset.Interface, ns, deploymentName, overlapWith string) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Deleting deployment %s", deploymentName)
	reaper, err := kubectl.ReaperFor(extensionsinternal.Kind("Deployment"), internalClient)
	Expect(err).NotTo(HaveOccurred())
	timeout := 1 * time.Minute

	err = reaper.Stop(ns, deployment.Name, timeout, metav1.NewDeleteOptions(0))
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Ensuring deployment %s was deleted", deploymentName)
	_, err = c.Extensions().Deployments(ns).Get(deployment.Name, metav1.GetOptions{})
	Expect(err).To(HaveOccurred())
	Expect(errors.IsNotFound(err)).To(BeTrue())
	framework.Logf("Ensuring deployment %s's RSes were deleted", deploymentName)
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rss, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	// RSes may be created by overlapping deployments right after this deployment is deleted, ignore them
	if len(overlapWith) == 0 {
		Expect(rss.Items).Should(HaveLen(0))
	} else {
		noOverlapRSes := []extensions.ReplicaSet{}
		for _, rs := range rss.Items {
			if !strings.HasPrefix(rs.Name, overlapWith) {
				noOverlapRSes = append(noOverlapRSes, rs)
			}
		}
		Expect(noOverlapRSes).Should(HaveLen(0))
	}
	framework.Logf("Ensuring deployment %s's Pods were deleted", deploymentName)
	var pods *v1.PodList
	if err := wait.PollImmediate(time.Second, timeout, func() (bool, error) {
		pods, err = c.Core().Pods(ns).List(options)
		if err != nil {
			return false, err
		}
		// Pods may be created by overlapping deployments right after this deployment is deleted, ignore them
		if len(overlapWith) == 0 && len(pods.Items) == 0 {
			return true, nil
		} else if len(overlapWith) != 0 {
			noOverlapPods := []v1.Pod{}
			for _, pod := range pods.Items {
				if !strings.HasPrefix(pod.Name, overlapWith) {
					noOverlapPods = append(noOverlapPods, pod)
				}
			}
			if len(noOverlapPods) == 0 {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		framework.Failf("Err : %s\n. Failed to remove deployment %s pods : %+v", err, deploymentName, pods)
	}
}

func testNewDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-new-deployment"
	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(1)
	framework.Logf("Creating simple deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", annotations.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	// Check new RS annotations
	Expect(newRS.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(newRS.Annotations[annotations.LastAppliedConfigAnnotation]).Should(Equal(""))
	Expect(deployment.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(deployment.Annotations[annotations.LastAppliedConfigAnnotation]).Should(Equal("should-not-copy-to-replica-set"))
}

func testDeleteDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	internalClient := f.InternalClientset

	deploymentName := "test-new-deployment"
	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(1)
	framework.Logf("Creating simple deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", annotations.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if newRS == nil {
		err = fmt.Errorf("expected a replica set, got nil")
		Expect(err).NotTo(HaveOccurred())
	}
	stopDeployment(c, internalClient, ns, deploymentName)
}

func testScaledRolloutDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(10)

	// Create a nginx deployment.
	deploymentName := "nginx"
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.Strategy.RollingUpdate = new(extensions.RollingUpdateDeployment)
	d.Spec.Strategy.RollingUpdate.MaxSurge = func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(3)
	d.Spec.Strategy.RollingUpdate.MaxUnavailable = func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(2)

	By(fmt.Sprintf("Creating deployment %q", deploymentName))
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for observed generation %d", deployment.Generation))
	Expect(framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	By("Waiting for all required pods to come up")
	err = framework.VerifyPods(f.ClientSet, ns, nginxImageName, false, *(deployment.Spec.Replicas))
	if err != nil {
		framework.Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	By(fmt.Sprintf("Waiting for deployment %q to complete", deployment.Name))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	first, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	// Update the deployment with a non-existent image so that the new replica set will be blocked.
	By(fmt.Sprintf("Updating deployment %q with a non-existent image", deploymentName))
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "nginx:404"
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for observed generation %d", deployment.Generation))
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	if deployment.Status.AvailableReplicas < deploymentutil.MinAvailable(deployment) {
		Expect(fmt.Errorf("Observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))).NotTo(HaveOccurred())
	}

	By(fmt.Sprintf("Checking that the replica sets for %q are synced", deploymentName))
	second, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	first, err = c.Extensions().ReplicaSets(first.Namespace).Get(first.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	firstCond := replicaSetHasDesiredReplicas(c.Extensions(), first)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, firstCond)
	Expect(err).NotTo(HaveOccurred())

	secondCond := replicaSetHasDesiredReplicas(c.Extensions(), second)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, secondCond)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Updating the size (up) and template at the same time for deployment %q", deploymentName))
	newReplicas := int32(20)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = nautilusImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for deployment status to sync (current available: %d, minimum available: %d)", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment)))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	oldRSs, _, rs, err := deploymentutil.GetAllReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	for _, rs := range append(oldRSs, rs) {
		By(fmt.Sprintf("Ensuring replica set %q has the correct desiredReplicas annotation", rs.Name))
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(rs)
		if !ok || desired == *(deployment.Spec.Replicas) {
			continue
		}
		err = fmt.Errorf("unexpected desiredReplicas annotation %d for replica set %q", desired, rs.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment with a non-existent image so that the new replica set will be blocked.
	By(fmt.Sprintf("Updating deployment %q with a non-existent image", deploymentName))
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "nginx:404"
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for observed generation %d", deployment.Generation))
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	if deployment.Status.AvailableReplicas < deploymentutil.MinAvailable(deployment) {
		Expect(fmt.Errorf("Observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))).NotTo(HaveOccurred())
	}

	By(fmt.Sprintf("Checking that the replica sets for %q are synced", deploymentName))
	oldRs, err := c.Extensions().ReplicaSets(rs.Namespace).Get(rs.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	newRs, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	oldCond := replicaSetHasDesiredReplicas(c.Extensions(), oldRs)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, oldCond)
	Expect(err).NotTo(HaveOccurred())

	newCond := replicaSetHasDesiredReplicas(c.Extensions(), newRs)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, newCond)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Updating the size (down) and template at the same time for deployment %q", deploymentName))
	newReplicas = int32(5)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = kittenImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for deployment status to sync (current available: %d, minimum available: %d)", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment)))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	oldRSs, _, rs, err = deploymentutil.GetAllReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	for _, rs := range append(oldRSs, rs) {
		By(fmt.Sprintf("Ensuring replica set %q has the correct desiredReplicas annotation", rs.Name))
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(rs)
		if !ok || desired == *(deployment.Spec.Replicas) {
			continue
		}
		err = fmt.Errorf("unexpected desiredReplicas annotation %d for replica set %q", desired, rs.Name)
		Expect(err).NotTo(HaveOccurred())
	}
}

func testOverlappingDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	internalClient := f.InternalClientset

	deploymentName := "first-deployment"
	podLabels := map[string]string{"name": redisImageName}
	replicas := int32(1)
	By(fmt.Sprintf("Creating deployment %q", deploymentName))
	d := framework.NewDeployment(deploymentName, replicas, podLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the first deployment")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploy.Name, "1", redisImage)
	Expect(err).NotTo(HaveOccurred(), "The first deployment failed to update to revision 1")

	Expect(err).NotTo(HaveOccurred())
	deploymentName = "second-deployment"
	By(fmt.Sprintf("Creating deployment %q with overlapping selector", deploymentName))
	podLabels["other-label"] = "random-label"
	d = framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deployOverlapping, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the second deployment")

	// Wait for overlapping annotation updated to both deployments
	By("Waiting for the overlapping deployment to have overlapping annotation")
	err = framework.WaitForOverlappingAnnotationMatch(c, ns, deployOverlapping.Name, deploy.Name)
	Expect(err).NotTo(HaveOccurred(), "Failed to update the second deployment's overlapping annotation")
	err = framework.WaitForOverlappingAnnotationMatch(c, ns, deploy.Name, "")
	Expect(err).NotTo(HaveOccurred(), "The deployment that holds the oldest selector shouldn't have the overlapping annotation")

	// Only the first deployment is synced
	By("Checking only the first overlapping deployment is synced")
	options := metav1.ListOptions{}
	rsList, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred(), "Failed listing all replica sets in namespace %s", ns)
	Expect(rsList.Items).To(HaveLen(int(replicas)))
	Expect(rsList.Items[0].Spec.Template.Spec.Containers).To(HaveLen(1))
	Expect(rsList.Items[0].Spec.Template.Spec.Containers[0].Image).To(Equal(deploy.Spec.Template.Spec.Containers[0].Image))

	By("Deleting the first deployment")
	stopDeploymentOverlap(c, internalClient, ns, deploy.Name, deployOverlapping.Name)

	// Wait for overlapping annotation cleared
	By("Waiting for the second deployment to clear overlapping annotation")
	err = framework.WaitForOverlappingAnnotationMatch(c, ns, deployOverlapping.Name, "")
	Expect(err).NotTo(HaveOccurred(), "Failed to clear the second deployment's overlapping annotation")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deployOverlapping.Name, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred(), "The second deployment failed to update to revision 1")

	// Now the second deployment is synced
	By("Checking the second overlapping deployment is synced")
	rsList, err = c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred(), "Failed listing all replica sets in namespace %s", ns)
	Expect(rsList.Items).To(HaveLen(int(replicas)))
	Expect(rsList.Items[0].Spec.Template.Spec.Containers).To(HaveLen(1))
	Expect(rsList.Items[0].Spec.Template.Spec.Containers[0].Image).To(Equal(deployOverlapping.Spec.Template.Spec.Containers[0].Image))

	deploymentName = "third-deployment"
	podLabels = map[string]string{"name": nginxImageName}
	By(fmt.Sprintf("Creating deployment %q", deploymentName))
	d = framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	thirdDeployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the third deployment")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, thirdDeployment.Name, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred(), "The third deployment failed to update to revision 1")

	// Update the second deployment's selector to make it overlap with the third deployment
	By(fmt.Sprintf("Updating deployment %q selector to make it overlap with existing one", deployOverlapping.Name))
	deployOverlapping, err = framework.UpdateDeploymentWithRetries(c, ns, deployOverlapping.Name, func(update *extensions.Deployment) {
		update.Spec.Selector = thirdDeployment.Spec.Selector
		update.Spec.Template.Labels = thirdDeployment.Spec.Template.Labels
		update.Spec.Template.Spec.Containers[0].Image = redisImage
	})
	Expect(err).NotTo(HaveOccurred())

	// Wait for overlapping annotation updated to both deployments
	By("Waiting for the second deployment to have the overlapping annotation")
	err = framework.WaitForOverlappingAnnotationMatch(c, ns, deployOverlapping.Name, thirdDeployment.Name)
	Expect(err).NotTo(HaveOccurred(), "Failed to update the second deployment's overlapping annotation")
	err = framework.WaitForOverlappingAnnotationMatch(c, ns, thirdDeployment.Name, "")
	Expect(err).NotTo(HaveOccurred(), "The deployment that holds the oldest selector shouldn't have the overlapping annotation")

	// The second deployment shouldn't be synced
	By("Checking the second deployment is not synced")
	Expect(deployOverlapping.Annotations[deploymentutil.RevisionAnnotation]).To(Equal("1"))

	// Update the second deployment's selector to make it not overlap with the third deployment
	By(fmt.Sprintf("Updating deployment %q selector to make it not overlap with existing one", deployOverlapping.Name))
	deployOverlapping, err = framework.UpdateDeploymentWithRetries(c, ns, deployOverlapping.Name, func(update *extensions.Deployment) {
		update.Spec.Selector = deploy.Spec.Selector
		update.Spec.Template.Labels = deploy.Spec.Template.Labels
	})
	Expect(err).NotTo(HaveOccurred())

	// Wait for the second deployment to be synced
	By("Checking the second deployment is now synced")
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deployOverlapping.Name, "2", redisImage)
	Expect(err).NotTo(HaveOccurred(), "The second deployment failed to update to revision 2")
}

func randomScale(d *extensions.Deployment, i int) {
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

func replicaSetHasDesiredReplicas(rsClient extensionsclient.ReplicaSetsGetter, replicaSet *extensions.ReplicaSet) wait.ConditionFunc {
	desiredGeneration := replicaSet.Generation
	return func() (bool, error) {
		rs, err := rsClient.ReplicaSets(replicaSet.Namespace).Get(replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.Replicas == *(rs.Spec.Replicas), nil
	}
}
