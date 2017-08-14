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
	"fmt"
	"math/rand"
	"time"

	"github.com/davecgh/go-spew/spew"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	extensionsclient "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/kubectl"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
	"k8s.io/kubernetes/test/e2e/framework"
	testutil "k8s.io/kubernetes/test/utils"
)

const (
	dRetryPeriod  = 2 * time.Second
	dRetryTimeout = 5 * time.Minute
)

var (
	nilRs *extensions.ReplicaSet
)

var _ = SIGDescribe("Deployment", func() {
	var ns string
	var c clientset.Interface

	AfterEach(func() {
		failureTrap(c, ns)
	})

	f := framework.NewDefaultFramework("deployment")

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
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
	It("deployment can avoid hash collisions", func() {
		testDeploymentHashCollisionAvoidance(f)
	})
	// TODO: add tests that cover deployment.Spec.MinReadySeconds once we solved clock-skew issues
	// See https://github.com/kubernetes/kubernetes/issues/29229
})

func failureTrap(c clientset.Interface, ns string) {
	deployments, err := c.Extensions().Deployments(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		framework.Logf("Could not list Deployments in namespace %q: %v", ns, err)
		return
	}
	for i := range deployments.Items {
		d := deployments.Items[i]

		framework.Logf(spew.Sprintf("Deployment %q:\n%+v\n", d.Name, d))
		_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(&d, c.ExtensionsV1beta1())
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
	rss, err := c.ExtensionsV1beta1().ReplicaSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
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
		podList, err := c.Core().Pods(rs.Namespace).List(options)
		for _, pod := range podList.Items {
			framework.Logf(spew.Sprintf("pod: %q:\n%+v\n", pod.Name, pod))
		}
	}
}

func intOrStrP(num int) *intstr.IntOrString {
	intstr := intstr.FromInt(num)
	return &intstr
}

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
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nilRs))
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

func testDeleteDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	internalClient := f.InternalClientset

	deploymentName := "test-new-deployment"
	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(1)
	framework.Logf("Creating simple deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", v1.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", NginxImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nilRs))
	stopDeployment(c, internalClient, ns, deploymentName)
}

func testRollingUpdateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  NginxImageName,
	}

	rsName := "test-rolling-update-controller"
	replicas := int32(1)
	rsRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rsRevision
	rs := newRS(rsName, replicas, rsPodLabels, NginxImageName, NginxImage)
	rs.Annotations = annotations
	framework.Logf("Creating replica set %q (going to be adopted)", rs.Name)
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	Expect(err).NotTo(HaveOccurred(), "error in waiting for pods to come up: %s", err)

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-rolling-update-deployment"
	framework.Logf("Creating deployment %q", deploymentName)
	d := framework.NewDeployment(deploymentName, replicas, deploymentPodLabels, RedisImageName, RedisImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 3546343826724305833.
	framework.Logf("Ensuring deployment %q gets the next revision from the one the adopted replica set %q has", deploy.Name, rs.Name)
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3546343826724305833", RedisImage)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Ensuring status for deployment %q is the expected", deploy.Name)
	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// There should be 1 old RS (nginx-controller, which is adopted)
	framework.Logf("Ensuring deployment %q has one old replica set (the one it adopted)", deploy.Name)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c.ExtensionsV1beta1())
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
	framework.Logf("Creating deployment %q", deploymentName)
	d := framework.NewDeployment(deploymentName, int32(1), map[string]string{"name": "sample-pod-3"}, RedisImageName, RedisImage, extensions.RecreateDeploymentStrategyType)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	framework.Logf("Waiting deployment %q to be updated to revision 1", deploymentName)
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", RedisImage)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting deployment %q to complete", deploymentName)
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	// Update deployment to delete redis pods and bring up nginx pods.
	framework.Logf("Triggering a new rollout for deployment %q", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deploymentName, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = NginxImageName
		update.Spec.Template.Spec.Containers[0].Image = NginxImage
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Watching deployment %q to verify that new pods will not run with olds pods", deploymentName)
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
		"pod":  NginxImageName,
	}
	rsName := "test-cleanup-controller"
	replicas := int32(1)
	revisionHistoryLimit := utilpointer.Int32Ptr(0)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, NginxImageName, NginxImage))
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "cleanup-pod", false, replicas)
	Expect(err).NotTo(HaveOccurred(), "error in waiting for pods to come up: %v", err)

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-cleanup-deployment"
	framework.Logf("Creating deployment %s", deploymentName)

	pods, err := c.Core().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	Expect(err).NotTo(HaveOccurred(), "Failed to query for pods: %v", err)

	options := metav1.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
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
					framework.Failf("Expect event Object to be a pod")
				}
				if pod.Spec.Containers[0].Name != RedisImageName {
					framework.Failf("Expect the created pod to have container name %s, got pod %#v\n", RedisImageName, pod)
				}
			case <-stopCh:
				return
			}
		}
	}()
	d := framework.NewDeployment(deploymentName, replicas, deploymentPodLabels, RedisImageName, RedisImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.RevisionHistoryLimit = revisionHistoryLimit
	_, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Waiting for deployment %s history to be cleaned up", deploymentName))
	err = framework.WaitForDeploymentOldRSsNum(c, ns, deploymentName, int(*revisionHistoryLimit))
	Expect(err).NotTo(HaveOccurred())
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
		"pod":  NginxImageName,
	}

	rsName := "test-rollover-controller"
	rsReplicas := int32(1)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, rsReplicas, rsPodLabels, NginxImageName, NginxImage))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, podName, false, rsReplicas)
	Expect(err).NotTo(HaveOccurred(), "error in waiting for pods to come up: %v", err)

	// Wait for replica set to become ready before adopting it.
	framework.Logf("Waiting for pods owned by replica set %q to become ready", rsName)
	Expect(framework.WaitForReadyReplicaSet(c, ns, rsName)).NotTo(HaveOccurred())

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	// We use a nonexistent image here, so that we make sure it won't finish
	deploymentName, deploymentImageName := "test-rollover-deployment", "redis-slave"
	deploymentReplicas := int32(1)
	deploymentImage := "gcr.io/google_samples/gb-redisslave:nonexistent"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %q", deploymentName)
	newDeployment := framework.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
		MaxUnavailable: intOrStrP(0),
		MaxSurge:       intOrStrP(1),
	}
	newDeployment.Spec.MinReadySeconds = int32(10)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the pods were scaled up and down as expected.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Make sure deployment %q performs scaling operations", deploymentName)
	// Make sure the deployment starts to scale up and down replica sets by checking if its updated replicas >= 1
	err = framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, deploymentReplicas, deployment.Generation)
	// Check if it's updated to revision 1 correctly
	framework.Logf("Check revision of new replica set for deployment %q", deploymentName)
	_, newRS := checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	framework.Logf("Ensure that both replica sets have 1 created replica")
	oldRS, err := c.Extensions().ReplicaSets(ns).Get(rsName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	ensureReplicas(oldRS, int32(1))
	newRS, err = c.Extensions().ReplicaSets(ns).Get(newRS.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	ensureReplicas(newRS, int32(1))

	// The deployment is stuck, update it to rollover the above 2 ReplicaSets and bring up redis pods.
	framework.Logf("Rollover old replica sets for deployment %q with new image update", deploymentName)
	updatedDeploymentImageName, updatedDeploymentImage := RedisImageName, RedisImage
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

	framework.Logf("Ensure that both old replica sets have no replicas")
	oldRS, err = c.Extensions().ReplicaSets(ns).Get(rsName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	ensureReplicas(oldRS, int32(0))
	// Not really the new replica set anymore but we GET by name so that's fine.
	newRS, err = c.Extensions().ReplicaSets(ns).Get(newRS.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	ensureReplicas(newRS, int32(0))
}

func ensureReplicas(rs *extensions.ReplicaSet, replicas int32) {
	Expect(*rs.Spec.Replicas).Should(Equal(replicas))
	Expect(rs.Status.Replicas).Should(Equal(replicas))
}

// TODO: Can be moved to a unit test.
func testPausedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	deploymentName := "test-paused-deployment"
	podLabels := map[string]string{"name": NginxImageName}
	d := framework.NewDeployment(deploymentName, 1, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
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
	rs, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	Expect(rs).To(Equal(nilRs))

	// Update the deployment to run
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = false
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the resume.
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())

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
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).To(Equal(nilRs))

	_, allOldRs, err := deploymentutil.GetOldReplicaSets(deployment, c.ExtensionsV1beta1())
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
	deploymentName, deploymentImageName := "test-rollback-deployment", NginxImageName
	deploymentReplicas := int32(1)
	deploymentImage := NginxImage
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

	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "create"
	err = framework.CheckNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 2. Update the deployment to create redis pods.
	updatedDeploymentImage := RedisImage
	updatedDeploymentImageName := RedisImageName
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
		"pod":  NginxImageName,
	}

	// Create an old RS without revision
	rsName := "test-rollback-no-revision-controller"
	rsReplicas := int32(0)
	rs := newRS(rsName, rsReplicas, rsPodLabels, NginxImageName, NginxImage)
	rs.Annotations = make(map[string]string)
	rs.Annotations["make"] = "difference"
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())

	// 1. Create a deployment to create nginx pods, which have different template than the replica set created above.
	deploymentName, deploymentImageName := "test-rollback-no-revision-deployment", NginxImageName
	deploymentReplicas := int32(1)
	deploymentImage := NginxImage
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	framework.Logf("Creating deployment %s", deploymentName)
	d := framework.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForDeploymentStatusValid(c, deploy)
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
	updatedDeploymentImage := RedisImage
	updatedDeploymentImageName := RedisImageName
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
	replicas := int32(1)
	image := NginxImage
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, podLabels, podName, image))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, podName, false, replicas)
	Expect(err).NotTo(HaveOccurred(), "error in waiting for pods to come up: %v", err)

	// Create a nginx deployment to adopt the old rs.
	deploymentName := "test-adopted-deployment"
	framework.Logf("Creating deployment %s", deploymentName)
	deploy, err := c.Extensions().Deployments(ns).Create(framework.NewDeployment(deploymentName, replicas, podLabels, podName, image, extensions.RollingUpdateDeploymentStrategyType))
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", image)
	Expect(err).NotTo(HaveOccurred())

	// The RS and pods should be relabeled before the status is updated by syncRollingUpdateDeployment
	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	// There should be no old RSs (overlapping RS)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	oldRSs, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(deployment, c.ExtensionsV1beta1())
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

// TODO: Can be moved to a unit test.
func testScalePausedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(0)

	// Create a nginx deployment.
	deploymentName := "nginx-deployment"
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	framework.Logf("Creating deployment %q", deploymentName)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q to have no running pods", deploymentName)
	Expect(framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, replicas, deployment.Generation))

	// Pause the deployment and try to scale it.
	framework.Logf("Pause deployment %q before scaling it up", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = true
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Scale the paused deployment.
	framework.Logf("Scaling up the paused deployment %q", deploymentName)
	newReplicas := int32(1)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	rs, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	Expect(*(rs.Spec.Replicas)).Should(Equal(newReplicas))
}

func testScaledRolloutDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(10)

	// Create a nginx deployment.
	deploymentName := "nginx"
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	d.Spec.Strategy.RollingUpdate = new(extensions.RollingUpdateDeployment)
	d.Spec.Strategy.RollingUpdate.MaxSurge = intOrStrP(3)
	d.Spec.Strategy.RollingUpdate.MaxUnavailable = intOrStrP(2)

	framework.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for observed generation %d", deployment.Generation)
	Expect(framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	framework.Logf("Waiting for all required pods to come up")
	err = framework.VerifyPodsRunning(f.ClientSet, ns, NginxImageName, false, *(deployment.Spec.Replicas))
	Expect(err).NotTo(HaveOccurred(), "error in waiting for pods to come up: %v", err)

	framework.Logf("Waiting for deployment %q to complete", deployment.Name)
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	first, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())

	// Update the deployment with a non-existent image so that the new replica set will be blocked.
	framework.Logf("Updating deployment %q with a non-existent image", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "nginx:404"
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for observed generation %d", deployment.Generation)
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	if deployment.Status.AvailableReplicas < deploymentutil.MinAvailable(deployment) {
		Expect(fmt.Errorf("Observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))).NotTo(HaveOccurred())
	}

	framework.Logf("Checking that the replica sets for %q are synced", deploymentName)
	second, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())

	first, err = c.Extensions().ReplicaSets(first.Namespace).Get(first.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	firstCond := replicaSetHasDesiredReplicas(c.Extensions(), first)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, firstCond)
	Expect(err).NotTo(HaveOccurred())

	secondCond := replicaSetHasDesiredReplicas(c.Extensions(), second)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, secondCond)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Updating the size (up) and template at the same time for deployment %q", deploymentName)
	newReplicas := int32(20)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = NautilusImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment status to sync (current available: %d, minimum available: %d)", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	oldRSs, _, rs, err := deploymentutil.GetAllReplicaSets(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())

	for _, rs := range append(oldRSs, rs) {
		framework.Logf("Ensuring replica set %q has the correct desiredReplicas annotation", rs.Name)
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(rs)
		if !ok || desired == *(deployment.Spec.Replicas) {
			continue
		}
		err = fmt.Errorf("unexpected desiredReplicas annotation %d for replica set %q", desired, rs.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment with a non-existent image so that the new replica set will be blocked.
	framework.Logf("Updating deployment %q with a non-existent image", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "nginx:404"
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for observed generation %d", deployment.Generation)
	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	if deployment.Status.AvailableReplicas < deploymentutil.MinAvailable(deployment) {
		Expect(fmt.Errorf("Observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))).NotTo(HaveOccurred())
	}

	framework.Logf("Checking that the replica sets for %q are synced", deploymentName)
	oldRs, err := c.Extensions().ReplicaSets(rs.Namespace).Get(rs.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	newRs, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())

	oldCond := replicaSetHasDesiredReplicas(c.Extensions(), oldRs)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, oldCond)
	Expect(err).NotTo(HaveOccurred())

	newCond := replicaSetHasDesiredReplicas(c.Extensions(), newRs)
	err = wait.PollImmediate(10*time.Millisecond, 1*time.Minute, newCond)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Updating the size (down) and template at the same time for deployment %q", deploymentName)
	newReplicas = int32(5)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = KittenImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment status to sync (current available: %d, minimum available: %d)", deployment.Status.AvailableReplicas, deploymentutil.MinAvailable(deployment))
	Expect(framework.WaitForDeploymentStatusValid(c, deployment)).NotTo(HaveOccurred())

	oldRSs, _, rs, err = deploymentutil.GetAllReplicaSets(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())

	for _, rs := range append(oldRSs, rs) {
		framework.Logf("Ensuring replica set %q has the correct desiredReplicas annotation", rs.Name)
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
	podLabels := map[string]string{"name": RedisImageName}
	replicas := int32(1)
	framework.Logf("Creating deployment %q", deploymentName)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, RedisImageName, RedisImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the first deployment")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploy.Name, "1", RedisImage)
	Expect(err).NotTo(HaveOccurred(), "The first deployment failed to update to revision 1")

	// Create second deployment with overlapping selector.
	deploymentName = "second-deployment"
	framework.Logf("Creating deployment %q with overlapping selector", deploymentName)
	podLabels["other-label"] = "random-label"
	d = framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deployOverlapping, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred(), "Failed creating the second deployment")

	// Wait for it to be updated to revision 1
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deployOverlapping.Name, "1", NginxImage)
	Expect(err).NotTo(HaveOccurred(), "The second deployment failed to update to revision 1")

	// Both deployments should proceed independently.
	framework.Logf("Checking each deployment creates its own replica set")
	options := metav1.ListOptions{}
	rsList, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred(), "Failed listing all replica sets in namespace %s", ns)
	Expect(rsList.Items).To(HaveLen(2))
}

func testFailedDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(1)

	// Create a nginx deployment.
	deploymentName := "progress-check"
	nonExistentImage := "nginx:not-there"
	ten := int32(10)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, nonExistentImage, extensions.RecreateDeploymentStrategyType)
	d.Spec.ProgressDeadlineSeconds = &ten

	framework.Logf("Creating deployment %q with progressDeadlineSeconds set to %ds and a non-existent image", deploymentName, ten)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q new replica set to come up", deploymentName)
	Expect(framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, replicas, deployment.Generation))

	framework.Logf("Checking deployment %q for a timeout condition", deploymentName)
	Expect(framework.WaitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.TimedOutReason, extensions.DeploymentProgressing)).NotTo(HaveOccurred())

	framework.Logf("Updating deployment %q with a good image", deploymentName)
	deployment, err = framework.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = NginxImage
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for deployment %q new replica set to come up", deploymentName)
	Expect(framework.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, replicas, deployment.Generation))

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

	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(6)
	zero := int64(0)
	two := int32(2)

	// Create a nginx deployment.
	deploymentName := "nginx"
	thirty := int32(30)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
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
	framework.Logf("Creating Deployment %q", deploymentName)
	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(1)
	d := framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Checking its ReplicaSet has the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Deleting Deployment %q and orphaning its ReplicaSets", deploymentName)
	err = orphanDeploymentReplicaSets(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	By("Wait for the ReplicaSet to be orphaned")
	err = wait.Poll(dRetryPeriod, dRetryTimeout, waitDeploymentReplicaSetsOrphaned(c, ns, podLabels))
	Expect(err).NotTo(HaveOccurred(), "error waiting for Deployment ReplicaSet to be orphaned")

	deploymentName = "test-adopt-deployment"
	framework.Logf("Creating Deployment %q to adopt the ReplicaSet", deploymentName)
	d = framework.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deploy, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForDeploymentStatusValid(c, deploy)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Waiting for the ReplicaSet to have the right controllerRef")
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

func testDeploymentHashCollisionAvoidance(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-hash-collision"
	framework.Logf("Creating Deployment %q", deploymentName)
	podLabels := map[string]string{"name": NginxImageName}
	d := framework.NewDeployment(deploymentName, int32(0), podLabels, NginxImageName, NginxImage, extensions.RollingUpdateDeploymentStrategyType)
	deployment, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", NginxImage)
	Expect(err).NotTo(HaveOccurred())

	// TODO: Switch this to do a non-cascading deletion of the Deployment, mutate the ReplicaSet
	// once it has no owner reference, then recreate the Deployment if we ever proceed with
	// https://github.com/kubernetes/kubernetes/issues/44237
	framework.Logf("Mock a hash collision")
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	Expect(err).NotTo(HaveOccurred())
	var nilRs *extensions.ReplicaSet
	Expect(newRS).NotTo(Equal(nilRs))
	_, err = framework.UpdateReplicaSetWithRetries(c, ns, newRS.Name, func(update *extensions.ReplicaSet) {
		*update.Spec.Template.Spec.TerminationGracePeriodSeconds = int64(5)
	})
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Expect deployment collision counter to increment")
	if err := wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
		d, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("cannot get deployment %q: %v", deploymentName, err)
			return false, nil
		}
		framework.Logf(spew.Sprintf("deployment status: %#v", d.Status))
		return d.Status.CollisionCount != nil && *d.Status.CollisionCount == int64(1), nil
	}); err != nil {
		framework.Failf("Failed to increment collision counter for deployment %q: %v", deploymentName, err)
	}

	framework.Logf("Expect a new ReplicaSet to be created")
	err = framework.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", NginxImage)
	Expect(err).NotTo(HaveOccurred())
}
