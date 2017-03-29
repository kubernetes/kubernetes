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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	dRetryPeriod  = 2 * time.Second
	dRetryTimeout = 5 * time.Minute

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
	It("RollingUpdateDeployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	It("RecreateDeployment should delete old pods and create new ones", func() {
		testRecreateDeployment(f)
	})
	It("deployment should delete old replica sets", func() {
		testDeploymentCleanUpPolicy(f)
	})
	It("deployment should support rollover", func() {
		testRolloverDeployment(f)
	})
	It("paused deployment should be ignored by the controller", func() {
		testPausedDeployment(f)
	})
	It("deployment should support rollback", func() {
		testRollbackDeployment(f)
	})
	It("deployment should support rollback when there's replica set with no revision", func() {
		testRollbackDeploymentRSNoRevision(f)
	})
	It("deployment should label adopted RSs and pods", func() {
		testDeploymentLabelAdopted(f)
	})
	It("paused deployment should be able to scale", func() {
		testScalePausedDeployment(f)
	})
	It("scaled rollout deployment should not block on annotation check", func() {
		testScaledRolloutDeployment(f)
	})
	It("overlapping deployment should not fight with each other", func() {
		testOverlappingDeployment(f)
	})
	It("lack of progress should be reported in the deployment status", func() {
		testFailedDeployment(f)
	})
	It("iterative rollouts should eventually progress", func() {
		testIterativeDeployments(f)
	})
	It("test Deployment ReplicaSet orphaning and adoption regarding controllerRef", func() {
		testDeploymentsControllerRef(f)
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

func stopDeployment(c clientset.Interface, internalClient internalclientset.Interface, ns, deploymentName string) {
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
	Expect(rss.Items).Should(HaveLen(0))
	framework.Logf("Ensuring deployment %s's Pods were deleted", deploymentName)
	var pods *v1.PodList
	if err := wait.PollImmediate(time.Second, timeout, func() (bool, error) {
		pods, err = c.Core().Pods(ns).List(options)
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

func testRollingUpdateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  nginxImageName,
	}

	rsName := "test-rolling-update-controller"
	replicas := int32(1)
	rsRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rsRevision
	rs := newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage)
	rs.Annotations = annotations
	By(fmt.Sprintf("Creating replica set %q (going to be adopted)", rs.Name))
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	if err != nil {
		framework.Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-rolling-update-deployment"
	By(fmt.Sprintf("Creating deployment %q", deploymentName))
	d := framework.NewDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 3546343826724305833.
	By(fmt.Sprintf("Ensuring deployment %q gets the next revision from the one the adopted replica set %q has", deploy.Name, rs.Name))
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3546343826724305833", redisImage)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Ensuring status for deployment %q is the expected", deploy.Name))
	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// There should be 1 old RS (nginx-controller, which is adopted)
	By(fmt.Sprintf("Ensuring deployment %q has one old replica set (the one it adopted)", deploy.Name))
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(allOldRSs)).Should(Equal(1))
	// The old RS should contain pod-template-hash in its selector, label, and template label
	Expect(len(allOldRSs[0].Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
}

func testRecreateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	// Create a deployment that brings up redis pods.
	deploymentName := "test-recreate-deployment"
	By(fmt.Sprintf("Creating deployment %q", deploymentName))
	d := framework.NewDeployment(deploymentName, int32(3), map[string]string{"name": "sample-pod-3"}, redisImageName, redisImage, extensions.RecreateDeploymentStrategyType)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	By(fmt.Sprintf("Waiting deployment %q to be updated to revision 1", deploymentName))
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", redisImage)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting deployment %q to complete", deploymentName))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	// Update deployment to delete redis pods and bring up nginx pods.
	By(fmt.Sprintf("Triggering a new rollout for deployment %q", deploymentName))
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deploymentName, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = nginxImageName
		update.Spec.Template.Spec.Containers[0].Image = nginxImage
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Watching deployment %q to verify that new pods will not run with olds pods", deploymentName))
	Expect(framework.WatchRecreateDeployment(c, deployment)).NotTo(HaveOccurred())
}

// testDeploymentCleanUpPolicy tests that deployment supports cleanup policy
func testDeploymentCleanUpPolicy(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "cleanup-pod"}
	rsPodLabels := map[string]string{
		"name": "cleanup-pod",
		"pod":  nginxImageName,
	}
	rsName := "test-cleanup-controller"
	replicas := int32(1)
	revisionHistoryLimit := util.Int32Ptr(0)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "cleanup-pod", false, 1)
	if err != nil {
		framework.Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-cleanup-deployment"
	By(fmt.Sprintf("Creating deployment %s", deploymentName))

	pods, err := c.Core().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		Expect(err).NotTo(HaveOccurred(), "Failed to query for pods: %v", err)
	}
	options := metav1.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	stopCh := make(chan struct{})
	w, err := c.Core().Pods(ns).Watch(options)
	Expect(err).NotTo(HaveOccurred())
	go func() {
		// There should be only one pod being created, which is the pod with the redis image.
		// The old RS shouldn't create new pod when deployment controller adding pod template hash label to its selector.
		numPodCreation := 1
		for {
			select {
			case event, _ := <-w.ResultChan():
				if event.Type != watch.Added {
					continue
				}
				numPodCreation--
				if numPodCreation < 0 {
					framework.Failf("Expect only one pod creation, the second creation event: %#v\n", event)
				}
				pod, ok := event.Object.(*v1.Pod)
				if !ok {
					Fail("Expect event Object to be a pod")
				}
				if pod.Spec.Containers[0].Name != redisImageName {
					framework.Failf("Expect the created pod to have container name %s, got pod %#v\n", redisImageName, pod)
				}
			case <-stopCh:
				return
			}
		}
	}()
	d := framework.NewDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.RevisionHistoryLimit = revisionHistoryLimit
	_, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for deployment %s history to be cleaned up", deploymentName))
	err = framework.WaitForDeploymentOldRSsNum(c, ns, deploymentName, int(*revisionHistoryLimit))
	Expect(err).NotTo(HaveOccurred())
	close(stopCh)
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
		"pod":  nginxImageName,
	}

	rsName := "test-rollover-controller"
	rsReplicas := int32(3)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, rsReplicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, podName, false, rsReplicas)
	if err != nil {
		framework.Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Wait for replica set to become ready before adopting it.
	framework.Logf("Waiting for pods owned by replica set %q to become ready", rsName)
	Expect(framework.WaitForReadyReplicaSet(c, ns, rsName)).NotTo(HaveOccurred())

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	// We use a nonexistent image here, so that we make sure it won't finish
	deploymentName, deploymentImageName := "test-rollover-deployment", "redis-slave"
	deploymentReplicas := int32(3)
	deploymentImage := "gcr.io/google_samples/gb-redisslave:nonexistent"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %q", deploymentName)
	newDeployment := framework.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
		MaxUnavailable: func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(1),
		MaxSurge:       func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(1),
	}
	newDeployment.Spec.MinReadySeconds = int32(10)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the pods were scaled up and down as expected.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Make sure deployment %q performs scaling operations", deploymentName)
	// Make sure the deployment starts to scale up and down replica sets by checking if its updated replicas >= 1
	err = framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, 1, deployment.Generation)
	// Check if it's updated to revision 1 correctly
	framework.Logf("Check revision of new replica set for deployment %q", deploymentName)
	_, newRS := checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Before the deployment finishes, update the deployment to rollover the above 2 ReplicaSets and bring up redis pods.
	Expect(*newRS.Spec.Replicas).Should(BeNumerically("<", deploymentReplicas))
	framework.Logf("Make sure deployment %q with new image", deploymentName)
	updatedDeploymentImageName, updatedDeploymentImage := redisImageName, redisImage
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, newDeployment.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	framework.Logf("Wait deployment %q to be observed by the deployment controller", deploymentName)
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	framework.Logf("Wait for revision update of deployment %q to 2", deploymentName)
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Make sure deployment %q is complete", deploymentName)
	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())
}

func testPausedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	deploymentName := "test-paused-deployment"
	podLabels := map[string]string{"name": nginxImageName}
	d := framework.NewDeployment(deploymentName, 1, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.Paused = true
	tgps := int64(1)
	d.Spec.Template.Spec.TerminationGracePeriodSeconds = &tgps
	framework.Logf("Creating paused deployment %s", deploymentName)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Verify that there is no latest state realized for the new deployment.
	rs, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if rs != nil {
		err = fmt.Errorf("unexpected new rs/%s for deployment/%s", rs.Name, deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment to run
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = false
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the resume.
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	opts := metav1.ListOptions{LabelSelector: selector.String()}
	w, err := c.Extensions().ReplicaSets(ns).Watch(opts)
	Expect(err).NotTo(HaveOccurred())

	select {
	case <-w.ResultChan():
		// this is it
	case <-time.After(time.Minute):
		err = fmt.Errorf("expected a new replica set to be created")
		Expect(err).NotTo(HaveOccurred())
	}

	// Pause the deployment and delete the replica set.
	// The paused deployment shouldn't recreate a new one.
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = true
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pause.
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Update the deployment template - the new replicaset should stay the same
	framework.Logf("Updating paused deployment %q", deploymentName)
	newTGPS := int64(0)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.TerminationGracePeriodSeconds = &newTGPS
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Looking for new replicaset for paused deployment %q (there should be none)", deploymentName)
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if newRS != nil {
		err = fmt.Errorf("No replica set should match the deployment template but there is %q", newRS.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	_, allOldRs, err := deploymentutil.GetOldReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if len(allOldRs) != 1 {
		err = fmt.Errorf("expected an old replica set")
		Expect(err).NotTo(HaveOccurred())
	}
	framework.Logf("Comparing deployment diff with old replica set %q", allOldRs[0].Name)
	if *allOldRs[0].Spec.Template.Spec.TerminationGracePeriodSeconds == newTGPS {
		err = fmt.Errorf("TerminationGracePeriodSeconds on the replica set should be %d but is %d", tgps, newTGPS)
		Expect(err).NotTo(HaveOccurred())
	}
}

// testRollbackDeployment tests that a deployment is created (revision 1) and updated (revision 2), and
// then rollback to revision 1 (should update template to revision 1, and then update revision 1 to 3),
// and then rollback to last revision.
func testRollbackDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}

	// 1. Create a deployment to create nginx pods.
	deploymentName, deploymentImageName := "test-rollback-deployment", nginxImageName
	deploymentReplicas := int32(1)
	deploymentImage := nginxImage
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	createAnnotation := map[string]string{"action": "create", "author": "node"}
	d.Annotations = createAnnotation
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "create"
	err = framework.CheckNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 2. Update the deployment to create redis pods.
	updatedDeploymentImage := redisImage
	updatedDeploymentImageName := redisImageName
	updateAnnotation := map[string]string{"action": "update", "log": "I need to update it"}
	deployment, err := framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
		update.Annotations = updateAnnotation
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "update"
	err = framework.CheckNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 3. Update the deploymentRollback to rollback to revision 1
	revision := int64(1)
	framework.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackDone in deployment status and check it here

	// Wait for it to be updated to revision 3
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "create", after the rollback
	err = framework.CheckNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 4. Update the deploymentRollback to rollback to last revision
	revision = 0
	framework.Logf("rolling back deployment %s to last revision", deploymentName)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 4
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "4", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "update", after the rollback
	err = framework.CheckNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	Expect(err).NotTo(HaveOccurred())
}

// testRollbackDeploymentRSNoRevision tests that deployment supports rollback even when there's old replica set without revision.
// An old replica set without revision is created, and then a deployment is created (v1). The deployment shouldn't add revision
// annotation to the old replica set. Then rollback the deployment to last revision, and it should fail.
// Then update the deployment to v2 and rollback it to v1 should succeed, now the deployment
// becomes v3. Then rollback the deployment to v10 (doesn't exist in history) should fail.
// Finally, rollback the deployment (v3) to v3 should be no-op.
// TODO: When we finished reporting rollback status in deployment status, check the rollback status here in each case.
func testRollbackDeploymentRSNoRevision(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  nginxImageName,
	}

	// Create an old RS without revision
	rsName := "test-rollback-no-revision-controller"
	rsReplicas := int32(0)
	rs := newRS(rsName, rsReplicas, rsPodLabels, nginxImageName, nginxImage)
	rs.Annotations = make(map[string]string)
	rs.Annotations["make"] = "difference"
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())

	// 1. Create a deployment to create nginx pods, which have different template than the replica set created above.
	deploymentName, deploymentImageName := "test-rollback-no-revision-deployment", nginxImageName
	deploymentReplicas := int32(1)
	deploymentImage := nginxImage
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// Check that the replica set we created still doesn't contain revision information
	rs, err = c.Extensions().ReplicaSets(ns).Get(rsName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	Expect(rs.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(""))

	// 2. Update the deploymentRollback to rollback to last revision
	//    Since there's only 1 revision in history, it should stay as revision 1
	revision := int64(0)
	framework.Logf("rolling back deployment %s to last revision", deploymentName)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no last revision
	// Check if the deployment is still revision 1 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// 3. Update the deployment to create redis pods.
	updatedDeploymentImage := redisImage
	updatedDeploymentImageName := redisImageName
	deployment, err := framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())

	// 4. Update the deploymentRollback to rollback to revision 1
	revision = 1
	framework.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackDone in deployment status and check it here

	// The pod template should be updated to the one in revision 1
	// Wait for it to be updated to revision 3
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatus(c, deployment)
	Expect(err).NotTo(HaveOccurred())

	// 5. Update the deploymentRollback to rollback to revision 10
	//    Since there's no revision 10 in history, it should stay as revision 3
	revision = 10
	framework.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no revision 10
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// 6. Update the deploymentRollback to rollback to revision 3
	//    Since it's already revision 3, it should be no-op
	revision = 3
	framework.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = framework.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackTemplateUnchanged in deployment status and check it here

	// The pod template shouldn't change since it's already revision 3
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)
}

func testDeploymentLabelAdopted(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create nginx pods.
	podName := "nginx"
	podLabels := map[string]string{"name": podName}

	rsName := "test-adopted-controller"
	replicas := int32(3)
	image := nginxImage
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, podLabels, podName, image))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, podName, false, 3)
	if err != nil {
		framework.Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a nginx deployment to adopt the old rs.
	deploymentName := "test-adopted-deployment"
	framework.Logf("Creating deployment %s", deploymentName)
	deploy, err := c.Extensions().Deployments(ns).Create(framework.NewDeployment(deploymentName, replicas, podLabels, podName, image, extensions.RollingUpdateDeploymentStrategyType))
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", image)
	Expect(err).NotTo(HaveOccurred())

	// The RS and pods should be relabeled before the status is updated by syncRollingUpdateDeployment
	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// There should be no old RSs (overlapping RS)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	oldRSs, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(oldRSs)).Should(Equal(0))
	Expect(len(allOldRSs)).Should(Equal(0))
	// New RS should contain pod-template-hash in its selector, label, and template label
	err = framework.CheckRSHashLabel(newRS)
	Expect(err).NotTo(HaveOccurred())
	// All pods targeted by the deployment should contain pod-template-hash in their labels, and there should be only 3 pods
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := metav1.ListOptions{LabelSelector: selector.String()}
	pods, err := c.Core().Pods(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	err = framework.CheckPodHashLabel(pods)
	Expect(err).NotTo(HaveOccurred())
	Expect(int32(len(pods.Items))).Should(Equal(replicas))
}

func testScalePausedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(3)

	// Create a nginx deployment.
	deploymentName := "nginx-deployment"
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	framework.Logf("Creating deployment %q", deploymentName)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	rs, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	// Pause the deployment and try to scale it.
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = true
	})
	Expect(err).NotTo(HaveOccurred())

	// Scale the paused deployment.
	framework.Logf("Scaling up the paused deployment %q", deploymentName)
	newReplicas := int32(5)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	rs, err = deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())

	if *(rs.Spec.Replicas) != newReplicas {
		err = fmt.Errorf("Expected %d replicas for the new replica set, got %d", newReplicas, *(rs.Spec.Replicas))
		Expect(err).NotTo(HaveOccurred())
	}
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
	err = framework.VerifyPodsRunning(f.ClientSet, ns, nginxImageName, false, *(deployment.Spec.Replicas))
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

	// Create first deployment.
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

	// Create second deployment with overlapping selector.
	deploymentName = "second-deployment"
	By(fmt.Sprintf("Creating deployment %q with overlapping selector", deploymentName))
	podLabels["other-label"] = "random-label"
	d = framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deployOverlapping, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the second deployment")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deployOverlapping.Name, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred(), "The second deployment failed to update to revision 1")

	// Both deployments should proceed independently.
	By("Checking each deployment creates its own replica set")
	options := metav1.ListOptions{}
	rsList, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred(), "Failed listing all replica sets in namespace %s", ns)
	Expect(rsList.Items).To(HaveLen(2))
}

func testFailedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(1)

	// Create a nginx deployment.
	deploymentName := "progress-check"
	nonExistentImage := "nginx:not-there"
	ten := int32(10)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nonExistentImage, extensions.RecreateDeploymentStrategyType)
	d.Spec.ProgressDeadlineSeconds = &ten

	framework.Logf("Creating deployment %q with progressDeadlineSeconds set to %ds and a non-existent image", deploymentName, ten)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q new replica set to come up", deploymentName)
	Expect(framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, 1, deployment.Generation))

	framework.Logf("Checking deployment %q for a timeout condition", deploymentName)
	Expect(framework.WaitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.TimedOutReason, extensions.DeploymentProgressing)).NotTo(HaveOccurred())

	framework.Logf("Updating deployment %q with a good image", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = nginxImage
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q new replica set to come up", deploymentName)
	Expect(framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, 1, deployment.Generation))

	framework.Logf("Waiting for deployment %q status", deploymentName)
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	framework.Logf("Checking deployment %q for a complete condition", deploymentName)
	Expect(framework.WaitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.NewRSAvailableReason, extensions.DeploymentProgressing)).NotTo(HaveOccurred())
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

func testIterativeDeployments(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(6)
	zero := int64(0)
	two := int32(2)

	// Create a nginx deployment.
	deploymentName := "nginx"
	thirty := int32(30)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.ProgressDeadlineSeconds = &thirty
	d.Spec.RevisionHistoryLimit = &two
	d.Spec.Template.Spec.TerminationGracePeriodSeconds = &zero
	framework.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	iterations := 20
	for i := 0; i < iterations; i++ {
		if r := rand.Float32(); r < 0.6 {
			time.Sleep(time.Duration(float32(i) * r * float32(time.Second)))
		}

		switch n := rand.Float32(); {
		case n < 0.2:
			// trigger a new deployment
			framework.Logf("%02d: triggering a new rollout for deployment %q", i, deployment.Name)
			deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
				newEnv := v1.EnvVar{Name: "A", Value: fmt.Sprintf("%d", i)}
				update.Spec.Template.Spec.Containers[0].Env = append(update.Spec.Template.Spec.Containers[0].Env, newEnv)
				randomScale(update, i)
			})
			Expect(err).NotTo(HaveOccurred())

		case n < 0.4:
			// rollback to the previous version
			framework.Logf("%02d: rolling back a rollout for deployment %q", i, deployment.Name)
			deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
				rollbackTo := &extensions.RollbackConfig{Revision: 0}
				update.Spec.RollbackTo = rollbackTo
			})
			Expect(err).NotTo(HaveOccurred())

		case n < 0.6:
			// just scaling
			framework.Logf("%02d: scaling deployment %q", i, deployment.Name)
			deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
				randomScale(update, i)
			})
			Expect(err).NotTo(HaveOccurred())

		case n < 0.8:
			// toggling the deployment
			if deployment.Spec.Paused {
				framework.Logf("%02d: pausing deployment %q", i, deployment.Name)
				deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
					update.Spec.Paused = true
					randomScale(update, i)
				})
				Expect(err).NotTo(HaveOccurred())
			} else {
				framework.Logf("%02d: resuming deployment %q", i, deployment.Name)
				deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
					update.Spec.Paused = false
					randomScale(update, i)
				})
				Expect(err).NotTo(HaveOccurred())
			}

		default:
			// arbitrarily delete deployment pods
			framework.Logf("%02d: arbitrarily deleting one or more deployment pods for deployment %q", i, deployment.Name)
			selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
			Expect(err).NotTo(HaveOccurred())
			opts := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(ns).List(opts)
			Expect(err).NotTo(HaveOccurred())
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
				err := c.Core().Pods(ns).Delete(name, nil)
				if err != nil && !errors.IsNotFound(err) {
					Expect(err).NotTo(HaveOccurred())
				}
			}
		}
	}

	// unpause the deployment if we end up pausing it
	deployment, err = c.Extensions().Deployments(ns).Get(deployment.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	if deployment.Spec.Paused {
		deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
			update.Spec.Paused = false
		})
	}

	framework.Logf("Waiting for deployment %q to be observed by the controller", deploymentName)
	Expect(framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q status", deploymentName)
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	framework.Logf("Checking deployment %q for a complete condition", deploymentName)
	Expect(framework.WaitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.NewRSAvailableReason, extensions.DeploymentProgressing)).NotTo(HaveOccurred())
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

func testDeploymentsControllerRef(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-orphan-deployment"
	By(fmt.Sprintf("Creating Deployment %q", deploymentName))
	podLabels := map[string]string{"name": nginxImageName}
	replicas := int32(1)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	By("Checking its ReplicaSet has the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Deleting Deployment %q and orphaning its ReplicaSets", deploymentName))
	err = orphanDeploymentReplicaSets(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	By("Wait for the ReplicaSet to be orphaned")
	err = wait.Poll(dRetryPeriod, dRetryTimeout, waitDeploymentReplicaSetsOrphaned(c, ns, podLabels))
	Expect(err).NotTo(HaveOccurred(), "error waiting for Deployment ReplicaSet to be orphaned")

	deploymentName = "test-adopt-deployment"
	By(fmt.Sprintf("Creating Deployment %q to adopt the ReplicaSet", deploymentName))
	d = framework.NewDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForDeploymentStatus(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	By("Waiting for the ReplicaSet to have the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	Expect(err).NotTo(HaveOccurred())
}

func waitDeploymentReplicaSetsControllerRef(c clientset.Interface, ns string, uid types.UID, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		err := checkDeploymentReplicaSetsControllerRef(c, ns, uid, label)
		if err != nil {
			return false, nil
		}
		return true, nil
	}
}

func checkDeploymentReplicaSetsControllerRef(c clientset.Interface, ns string, uid types.UID, label map[string]string) error {
	rsList := listDeploymentReplicaSets(c, ns, label)
	for _, rs := range rsList.Items {
		// This rs is adopted only when its controller ref is update
		if controllerRef := controller.GetControllerOf(&rs); controllerRef == nil || controllerRef.UID != uid {
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
			if controllerRef := controller.GetControllerOf(&rs); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func listDeploymentReplicaSets(c clientset.Interface, ns string, label map[string]string) *extensions.ReplicaSetList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rsList, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(rsList.Items)).To(BeNumerically(">", 0))
	return rsList
}

func orphanDeploymentReplicaSets(c clientset.Interface, d *extensions.Deployment) error {
	trueVar := true
	deleteOptions := &metav1.DeleteOptions{OrphanDependents: &trueVar}
	deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(d.UID))
	return c.Extensions().Deployments(d.Namespace).Delete(d.Name, deleteOptions)
}
