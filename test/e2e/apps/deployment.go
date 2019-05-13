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
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	apps "k8s.io/api/apps/v1"
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
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeploy "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/framework/replicaset"
	testutil "k8s.io/kubernetes/test/utils"
	utilpointer "k8s.io/utils/pointer"
)

const (
	dRetryPeriod  = 2 * time.Second
	dRetryTimeout = 5 * time.Minute
)

var (
	nilRs *apps.ReplicaSet
)

var _ = SIGDescribe("Deployment", func() {
	var ns string
	var c clientset.Interface

	ginkgo.AfterEach(func() {
		failureTrap(c, ns)
	})

	f := framework.NewDefaultFramework("deployment")

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.It("deployment reaping should cascade to its replica sets and pods", func() {
		testDeleteDeployment(f)
	})
	/*
	  Testname: Deployment RollingUpdate
	  Description: A conformant Kubernetes distribution MUST support the Deployment with RollingUpdate strategy.
	*/
	framework.ConformanceIt("RollingUpdateDeployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	/*
	  Testname: Deployment Recreate
	  Description: A conformant Kubernetes distribution MUST support the Deployment with Recreate strategy.
	*/
	framework.ConformanceIt("RecreateDeployment should delete old pods and create new ones", func() {
		testRecreateDeployment(f)
	})
	/*
	  Testname: Deployment RevisionHistoryLimit
	  Description: A conformant Kubernetes distribution MUST clean up Deployment's ReplicaSets based on
	  the Deployment's `.spec.revisionHistoryLimit`.
	*/
	framework.ConformanceIt("deployment should delete old replica sets", func() {
		testDeploymentCleanUpPolicy(f)
	})
	/*
	  Testname: Deployment Rollover
	  Description: A conformant Kubernetes distribution MUST support Deployment rollover,
	    i.e. allow arbitrary number of changes to desired state during rolling update
	    before the rollout finishes.
	*/
	framework.ConformanceIt("deployment should support rollover", func() {
		testRolloverDeployment(f)
	})
	ginkgo.It("deployment should support rollback", func() {
		testRollbackDeployment(f)
	})
	ginkgo.It("iterative rollouts should eventually progress", func() {
		testIterativeDeployments(f)
	})
	ginkgo.It("test Deployment ReplicaSet orphaning and adoption regarding controllerRef", func() {
		testDeploymentsControllerRef(f)
	})
	/*
	  Testname: Deployment Proportional Scaling
	  Description: A conformant Kubernetes distribution MUST support Deployment
	    proportional scaling, i.e. proportionally scale a Deployment's ReplicaSets
	    when a Deployment is scaled.
	*/
	framework.ConformanceIt("deployment should support proportional scaling", func() {
		testProportionalScalingDeployment(f)
	})
	// TODO: add tests that cover deployment.Spec.MinReadySeconds once we solved clock-skew issues
	// See https://github.com/kubernetes/kubernetes/issues/29229
})

func failureTrap(c clientset.Interface, ns string) {
	deployments, err := c.AppsV1().Deployments(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		e2elog.Logf("Could not list Deployments in namespace %q: %v", ns, err)
		return
	}
	for i := range deployments.Items {
		d := deployments.Items[i]

		e2elog.Logf(spew.Sprintf("Deployment %q:\n%+v\n", d.Name, d))
		_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(&d, c.AppsV1())
		if err != nil {
			e2elog.Logf("Could not list ReplicaSets for Deployment %q: %v", d.Name, err)
			return
		}
		testutil.LogReplicaSetsOfDeployment(&d, allOldRSs, newRS, e2elog.Logf)
		rsList := allOldRSs
		if newRS != nil {
			rsList = append(rsList, newRS)
		}
		testutil.LogPodsOfDeployment(c, &d, rsList, e2elog.Logf)
	}
	// We need print all the ReplicaSets if there are no Deployment object created
	if len(deployments.Items) != 0 {
		return
	}
	e2elog.Logf("Log out all the ReplicaSets if there is no deployment created")
	rss, err := c.AppsV1().ReplicaSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		e2elog.Logf("Could not list ReplicaSets in namespace %q: %v", ns, err)
		return
	}
	for _, rs := range rss.Items {
		e2elog.Logf(spew.Sprintf("ReplicaSet %q:\n%+v\n", rs.Name, rs))
		selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			e2elog.Logf("failed to get selector of ReplicaSet %s: %v", rs.Name, err)
		}
		options := metav1.ListOptions{LabelSelector: selector.String()}
		podList, err := c.CoreV1().Pods(rs.Namespace).List(options)
		if err != nil {
			e2elog.Logf("Failed to list Pods in namespace %s: %v", rs.Namespace, err)
			continue
		}
		for _, pod := range podList.Items {
			e2elog.Logf(spew.Sprintf("pod: %q:\n%+v\n", pod.Name, pod))
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

func stopDeployment(c clientset.Interface, ns, deploymentName string) {
	deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Deleting deployment %s", deploymentName)
	framework.ExpectNoError(framework.DeleteResourceAndWaitForGC(c, appsinternal.Kind("Deployment"), ns, deployment.Name))

	e2elog.Logf("Ensuring deployment %s was deleted", deploymentName)
	_, err = c.AppsV1().Deployments(ns).Get(deployment.Name, metav1.GetOptions{})
	gomega.Expect(err).To(gomega.HaveOccurred())
	gomega.Expect(errors.IsNotFound(err)).To(gomega.BeTrue())
	e2elog.Logf("Ensuring deployment %s's RSes were deleted", deploymentName)
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rss, err := c.AppsV1().ReplicaSets(ns).List(options)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(rss.Items).Should(gomega.HaveLen(0))
	e2elog.Logf("Ensuring deployment %s's Pods were deleted", deploymentName)
	var pods *v1.PodList
	if err := wait.PollImmediate(time.Second, timeout, func() (bool, error) {
		pods, err = c.CoreV1().Pods(ns).List(options)
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
	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(1)
	e2elog.Logf("Creating simple deployment %s", deploymentName)
	d := e2edeploy.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, apps.RollingUpdateDeploymentStrategyType)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", v1.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	deploy, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 1
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", NginxImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentComplete(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(newRS).NotTo(gomega.Equal(nilRs))
	stopDeployment(c, ns, deploymentName)
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
	e2elog.Logf("Creating replica set %q (going to be adopted)", rs.Name)
	_, err := c.AppsV1().ReplicaSets(ns).Create(rs)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %s", err)

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-rolling-update-deployment"
	e2elog.Logf("Creating deployment %q", deploymentName)
	d := e2edeploy.NewDeployment(deploymentName, replicas, deploymentPodLabels, RedisImageName, RedisImage, apps.RollingUpdateDeploymentStrategyType)
	deploy, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 3546343826724305833.
	e2elog.Logf("Ensuring deployment %q gets the next revision from the one the adopted replica set %q has", deploy.Name, rs.Name)
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3546343826724305833", RedisImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Ensuring status for deployment %q is the expected", deploy.Name)
	err = e2edeploy.WaitForDeploymentComplete(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// There should be 1 old RS (nginx-controller, which is adopted)
	e2elog.Logf("Ensuring deployment %q has one old replica set (the one it adopted)", deploy.Name)
	deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c.AppsV1())
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(len(allOldRSs)).Should(gomega.Equal(1))
}

func testRecreateDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	// Create a deployment that brings up redis pods.
	deploymentName := "test-recreate-deployment"
	e2elog.Logf("Creating deployment %q", deploymentName)
	d := e2edeploy.NewDeployment(deploymentName, int32(1), map[string]string{"name": "sample-pod-3"}, RedisImageName, RedisImage, apps.RecreateDeploymentStrategyType)
	deployment, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 1
	e2elog.Logf("Waiting deployment %q to be updated to revision 1", deploymentName)
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", RedisImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting deployment %q to complete", deploymentName)
	gomega.Expect(e2edeploy.WaitForDeploymentComplete(c, deployment)).NotTo(gomega.HaveOccurred())

	// Update deployment to delete redis pods and bring up nginx pods.
	e2elog.Logf("Triggering a new rollout for deployment %q", deploymentName)
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deploymentName, func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = NginxImageName
		update.Spec.Template.Spec.Containers[0].Image = NginxImage
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Watching deployment %q to verify that new pods will not run with olds pods", deploymentName)
	gomega.Expect(e2edeploy.WatchRecreateDeployment(c, deployment)).NotTo(gomega.HaveOccurred())
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
	_, err := c.AppsV1().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, NginxImageName, NginxImage))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, "cleanup-pod", false, replicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-cleanup-deployment"
	e2elog.Logf("Creating deployment %s", deploymentName)

	pods, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	framework.ExpectNoError(err, "Failed to query for pods: %v", err)

	options := metav1.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	w, err := c.CoreV1().Pods(ns).Watch(options)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
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
	d := e2edeploy.NewDeployment(deploymentName, replicas, deploymentPodLabels, RedisImageName, RedisImage, apps.RollingUpdateDeploymentStrategyType)
	d.Spec.RevisionHistoryLimit = revisionHistoryLimit
	_, err = c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	ginkgo.By(fmt.Sprintf("Waiting for deployment %s history to be cleaned up", deploymentName))
	err = e2edeploy.WaitForDeploymentOldRSsNum(c, ns, deploymentName, int(*revisionHistoryLimit))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
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
	_, err := c.AppsV1().ReplicaSets(ns).Create(newRS(rsName, rsReplicas, rsPodLabels, NginxImageName, NginxImage))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// Verify that the required pods have come up.
	err = framework.VerifyPodsRunning(c, ns, podName, false, rsReplicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	// Wait for replica set to become ready before adopting it.
	e2elog.Logf("Waiting for pods owned by replica set %q to become ready", rsName)
	gomega.Expect(replicaset.WaitForReadyReplicaSet(c, ns, rsName)).NotTo(gomega.HaveOccurred())

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	// We use a nonexistent image here, so that we make sure it won't finish
	deploymentName, deploymentImageName := "test-rollover-deployment", "redis-slave"
	deploymentReplicas := int32(1)
	deploymentImage := "gcr.io/google_samples/gb-redisslave:nonexistent"
	deploymentStrategyType := apps.RollingUpdateDeploymentStrategyType
	e2elog.Logf("Creating deployment %q", deploymentName)
	newDeployment := e2edeploy.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	newDeployment.Spec.Strategy.RollingUpdate = &apps.RollingUpdateDeployment{
		MaxUnavailable: intOrStrP(0),
		MaxSurge:       intOrStrP(1),
	}
	newDeployment.Spec.MinReadySeconds = int32(10)
	_, err = c.AppsV1().Deployments(ns).Create(newDeployment)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Verify that the pods were scaled up and down as expected.
	deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	e2elog.Logf("Make sure deployment %q performs scaling operations", deploymentName)
	// Make sure the deployment starts to scale up and down replica sets by checking if its updated replicas >= 1
	err = e2edeploy.WaitForDeploymentUpdatedReplicasGTE(c, ns, deploymentName, deploymentReplicas, deployment.Generation)
	// Check if it's updated to revision 1 correctly
	e2elog.Logf("Check revision of new replica set for deployment %q", deploymentName)
	err = e2edeploy.CheckDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Ensure that both replica sets have 1 created replica")
	oldRS, err := c.AppsV1().ReplicaSets(ns).Get(rsName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ensureReplicas(oldRS, int32(1))
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ensureReplicas(newRS, int32(1))

	// The deployment is stuck, update it to rollover the above 2 ReplicaSets and bring up redis pods.
	e2elog.Logf("Rollover old replica sets for deployment %q with new image update", deploymentName)
	updatedDeploymentImageName, updatedDeploymentImage := RedisImageName, RedisImage
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, newDeployment.Name, func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	e2elog.Logf("Wait deployment %q to be observed by the deployment controller", deploymentName)
	err = e2edeploy.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 2
	e2elog.Logf("Wait for revision update of deployment %q to 2", deploymentName)
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Make sure deployment %q is complete", deploymentName)
	err = e2edeploy.WaitForDeploymentCompleteAndCheckRolling(c, deployment)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Ensure that both old replica sets have no replicas")
	oldRS, err = c.AppsV1().ReplicaSets(ns).Get(rsName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ensureReplicas(oldRS, int32(0))
	// Not really the new replica set anymore but we GET by name so that's fine.
	newRS, err = c.AppsV1().ReplicaSets(ns).Get(newRS.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ensureReplicas(newRS, int32(0))
}

func ensureReplicas(rs *apps.ReplicaSet, replicas int32) {
	gomega.Expect(*rs.Spec.Replicas).Should(gomega.Equal(replicas))
	gomega.Expect(rs.Status.Replicas).Should(gomega.Equal(replicas))
}

// testRollbackDeployment tests that a deployment is created (revision 1) and updated (revision 2), and
// then rollback to revision 1 (should update template to revision 1, and then update revision 1 to 3),
// and then rollback to last revision (which is revision 4 that comes from revision 2).
// Then rollback the deployment to revision 10 (doesn't exist in history) should fail.
// Finally, rollback current deployment (revision 4) to revision 4 should be no-op.
func testRollbackDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}

	// 1. Create a deployment to create nginx pods.
	deploymentName, deploymentImageName := "test-rollback-deployment", NginxImageName
	deploymentReplicas := int32(1)
	deploymentImage := NginxImage
	deploymentStrategyType := apps.RollingUpdateDeploymentStrategyType
	e2elog.Logf("Creating deployment %s", deploymentName)
	d := e2edeploy.NewDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType)
	createAnnotation := map[string]string{"action": "create", "author": "node"}
	d.Annotations = createAnnotation
	deploy, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 1
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentComplete(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Current newRS annotation should be "create"
	err = replicaset.CheckNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// 2. Update the deployment to create redis pods.
	updatedDeploymentImage := RedisImage
	updatedDeploymentImageName := RedisImageName
	updateAnnotation := map[string]string{"action": "update", "log": "I need to update it"}
	deployment, err := e2edeploy.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
		update.Annotations = updateAnnotation
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = e2edeploy.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 2
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentCompleteAndCheckRolling(c, deployment)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Current newRS annotation should be "update"
	err = replicaset.CheckNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// 3. Update the deploymentRollback to rollback to revision 1
	revision := int64(1)
	e2elog.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.ExtensionsV1beta1().Deployments(ns).Rollback(rollback)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for the deployment to start rolling back
	err = e2edeploy.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// TODO: report RollbackDone in deployment status and check it here

	// Wait for it to be updated to revision 3
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "3", deploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentCompleteAndCheckRolling(c, deployment)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Current newRS annotation should be "create", after the rollback
	err = replicaset.CheckNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// 4. Update the deploymentRollback to rollback to last revision
	revision = 0
	e2elog.Logf("rolling back deployment %s to last revision", deploymentName)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.ExtensionsV1beta1().Deployments(ns).Rollback(rollback)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for it to be updated to revision 4
	err = e2edeploy.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, "4", updatedDeploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = e2edeploy.WaitForDeploymentCompleteAndCheckRolling(c, deployment)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Current newRS annotation should be "update", after the rollback
	err = replicaset.CheckNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// 5. Update the deploymentRollback to rollback to revision 10
	//    Since there's no revision 10 in history, it should stay as revision 4
	revision = 10
	e2elog.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.ExtensionsV1beta1().Deployments(ns).Rollback(rollback)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for the deployment to start rolling back
	err = e2edeploy.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no revision 10
	// Check if it's still revision 4 and still has the old pod template
	err = e2edeploy.CheckDeploymentRevisionAndImage(c, ns, deploymentName, "4", updatedDeploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// 6. Update the deploymentRollback to rollback to revision 4
	//    Since it's already revision 4, it should be no-op
	revision = 4
	e2elog.Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.ExtensionsV1beta1().Deployments(ns).Rollback(rollback)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Wait for the deployment to start rolling back
	err = e2edeploy.WaitForDeploymentRollbackCleared(c, ns, deploymentName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// TODO: report RollbackTemplateUnchanged in deployment status and check it here

	// The pod template shouldn't change since it's already revision 4
	// Check if it's still revision 4 and still has the old pod template
	err = e2edeploy.CheckDeploymentRevisionAndImage(c, ns, deploymentName, "4", updatedDeploymentImage)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
}

func randomScale(d *apps.Deployment, i int) {
	switch r := rand.Float32(); {
	case r < 0.3:
		e2elog.Logf("%02d: scaling up", i)
		*(d.Spec.Replicas)++
	case r < 0.6:
		if *(d.Spec.Replicas) > 1 {
			e2elog.Logf("%02d: scaling down", i)
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
	d := e2edeploy.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, apps.RollingUpdateDeploymentStrategyType)
	d.Spec.ProgressDeadlineSeconds = &thirty
	d.Spec.RevisionHistoryLimit = &two
	d.Spec.Template.Spec.TerminationGracePeriodSeconds = &zero
	e2elog.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	iterations := 20
	for i := 0; i < iterations; i++ {
		if r := rand.Float32(); r < 0.6 {
			time.Sleep(time.Duration(float32(i) * r * float32(time.Second)))
		}

		switch n := rand.Float32(); {
		case n < 0.2:
			// trigger a new deployment
			e2elog.Logf("%02d: triggering a new rollout for deployment %q", i, deployment.Name)
			deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
				newEnv := v1.EnvVar{Name: "A", Value: fmt.Sprintf("%d", i)}
				update.Spec.Template.Spec.Containers[0].Env = append(update.Spec.Template.Spec.Containers[0].Env, newEnv)
				randomScale(update, i)
			})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

		case n < 0.4:
			// rollback to the previous version
			e2elog.Logf("%02d: rolling back a rollout for deployment %q", i, deployment.Name)
			deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
				if update.Annotations == nil {
					update.Annotations = make(map[string]string)
				}
				update.Annotations[apps.DeprecatedRollbackTo] = "0"
			})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

		case n < 0.6:
			// just scaling
			e2elog.Logf("%02d: scaling deployment %q", i, deployment.Name)
			deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
				randomScale(update, i)
			})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

		case n < 0.8:
			// toggling the deployment
			if deployment.Spec.Paused {
				e2elog.Logf("%02d: pausing deployment %q", i, deployment.Name)
				deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
					update.Spec.Paused = true
					randomScale(update, i)
				})
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
			} else {
				e2elog.Logf("%02d: resuming deployment %q", i, deployment.Name)
				deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
					update.Spec.Paused = false
					randomScale(update, i)
				})
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
			}

		default:
			// arbitrarily delete deployment pods
			e2elog.Logf("%02d: arbitrarily deleting one or more deployment pods for deployment %q", i, deployment.Name)
			selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			opts := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.CoreV1().Pods(ns).List(opts)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			if len(podList.Items) == 0 {
				e2elog.Logf("%02d: no deployment pods to delete", i)
				continue
			}
			for p := range podList.Items {
				if rand.Float32() < 0.5 {
					continue
				}
				name := podList.Items[p].Name
				e2elog.Logf("%02d: deleting deployment pod %q", i, name)
				err := c.CoreV1().Pods(ns).Delete(name, nil)
				if err != nil && !errors.IsNotFound(err) {
					gomega.Expect(err).NotTo(gomega.HaveOccurred())
				}
			}
		}
	}

	// unpause the deployment if we end up pausing it
	deployment, err = c.AppsV1().Deployments(ns).Get(deployment.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	if deployment.Spec.Paused {
		deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
			update.Spec.Paused = false
		})
	}

	e2elog.Logf("Waiting for deployment %q to be observed by the controller", deploymentName)
	gomega.Expect(e2edeploy.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting for deployment %q status", deploymentName)
	gomega.Expect(e2edeploy.WaitForDeploymentComplete(c, deployment)).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Checking deployment %q for a complete condition", deploymentName)
	gomega.Expect(e2edeploy.WaitForDeploymentWithCondition(c, ns, deploymentName, deploymentutil.NewRSAvailableReason, apps.DeploymentProgressing)).NotTo(gomega.HaveOccurred())
}

func testDeploymentsControllerRef(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	deploymentName := "test-orphan-deployment"
	e2elog.Logf("Creating Deployment %q", deploymentName)
	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(1)
	d := e2edeploy.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, apps.RollingUpdateDeploymentStrategyType)
	deploy, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = e2edeploy.WaitForDeploymentComplete(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Verifying Deployment %q has only one ReplicaSet", deploymentName)
	rsList := listDeploymentReplicaSets(c, ns, podLabels)
	gomega.Expect(len(rsList.Items)).Should(gomega.Equal(1))

	e2elog.Logf("Obtaining the ReplicaSet's UID")
	orphanedRSUID := rsList.Items[0].UID

	e2elog.Logf("Checking the ReplicaSet has the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Deleting Deployment %q and orphaning its ReplicaSet", deploymentName)
	err = orphanDeploymentReplicaSets(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	ginkgo.By("Wait for the ReplicaSet to be orphaned")
	err = wait.Poll(dRetryPeriod, dRetryTimeout, waitDeploymentReplicaSetsOrphaned(c, ns, podLabels))
	framework.ExpectNoError(err, "error waiting for Deployment ReplicaSet to be orphaned")

	deploymentName = "test-adopt-deployment"
	e2elog.Logf("Creating Deployment %q to adopt the ReplicaSet", deploymentName)
	d = e2edeploy.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, apps.RollingUpdateDeploymentStrategyType)
	deploy, err = c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = e2edeploy.WaitForDeploymentComplete(c, deploy)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting for the ReplicaSet to have the right controllerRef")
	err = checkDeploymentReplicaSetsControllerRef(c, ns, deploy.UID, podLabels)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Verifying no extra ReplicaSet is created (Deployment %q still has only one ReplicaSet after adoption)", deploymentName)
	rsList = listDeploymentReplicaSets(c, ns, podLabels)
	gomega.Expect(len(rsList.Items)).Should(gomega.Equal(1))

	e2elog.Logf("Verifying the ReplicaSet has the same UID as the orphaned ReplicaSet")
	gomega.Expect(rsList.Items[0].UID).Should(gomega.Equal(orphanedRSUID))
}

// testProportionalScalingDeployment tests that when a RollingUpdate Deployment is scaled in the middle
// of a rollout (either in progress or paused), then the Deployment will balance additional replicas
// in existing active ReplicaSets (ReplicaSets with more than 0 replica) in order to mitigate risk.
func testProportionalScalingDeployment(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	podLabels := map[string]string{"name": NginxImageName}
	replicas := int32(10)

	// Create a nginx deployment.
	deploymentName := "nginx-deployment"
	d := e2edeploy.NewDeployment(deploymentName, replicas, podLabels, NginxImageName, NginxImage, apps.RollingUpdateDeploymentStrategyType)
	d.Spec.Strategy.RollingUpdate = new(apps.RollingUpdateDeployment)
	d.Spec.Strategy.RollingUpdate.MaxSurge = intOrStrP(3)
	d.Spec.Strategy.RollingUpdate.MaxUnavailable = intOrStrP(2)

	e2elog.Logf("Creating deployment %q", deploymentName)
	deployment, err := c.AppsV1().Deployments(ns).Create(d)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting for observed generation %d", deployment.Generation)
	gomega.Expect(e2edeploy.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(gomega.HaveOccurred())

	// Verify that the required pods have come up.
	e2elog.Logf("Waiting for all required pods to come up")
	err = framework.VerifyPodsRunning(c, ns, NginxImageName, false, *(deployment.Spec.Replicas))
	framework.ExpectNoError(err, "error in waiting for pods to come up: %v", err)

	e2elog.Logf("Waiting for deployment %q to complete", deployment.Name)
	gomega.Expect(e2edeploy.WaitForDeploymentComplete(c, deployment)).NotTo(gomega.HaveOccurred())

	firstRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Update the deployment with a non-existent image so that the new replica set
	// will be blocked to simulate a partial rollout.
	e2elog.Logf("Updating deployment %q with a non-existent image", deploymentName)
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, d.Name, func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = "nginx:404"
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting for observed generation %d", deployment.Generation)
	gomega.Expect(e2edeploy.WaitForObservedDeployment(c, ns, deploymentName, deployment.Generation)).NotTo(gomega.HaveOccurred())

	// Checking state of first rollout's replicaset.
	maxUnavailable, err := intstr.GetValueFromIntOrPercent(deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, int(*(deployment.Spec.Replicas)), false)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// First rollout's replicaset should have Deployment's (replicas - maxUnavailable) = 10 - 2 = 8 available replicas.
	minAvailableReplicas := replicas - int32(maxUnavailable)
	e2elog.Logf("Waiting for the first rollout's replicaset to have .status.availableReplicas = %d", minAvailableReplicas)
	gomega.Expect(replicaset.WaitForReplicaSetTargetAvailableReplicas(c, firstRS, minAvailableReplicas)).NotTo(gomega.HaveOccurred())

	// First rollout's replicaset should have .spec.replicas = 8 too.
	e2elog.Logf("Waiting for the first rollout's replicaset to have .spec.replicas = %d", minAvailableReplicas)
	gomega.Expect(replicaset.WaitForReplicaSetTargetSpecReplicas(c, firstRS, minAvailableReplicas)).NotTo(gomega.HaveOccurred())

	// The desired replicas wait makes sure that the RS controller has created expected number of pods.
	e2elog.Logf("Waiting for the first rollout's replicaset of deployment %q to have desired number of replicas", deploymentName)
	firstRS, err = c.AppsV1().ReplicaSets(ns).Get(firstRS.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = replicaset.WaitForReplicaSetDesiredReplicas(c.AppsV1(), firstRS)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Checking state of second rollout's replicaset.
	secondRS, err := deploymentutil.GetNewReplicaSet(deployment, c.AppsV1())
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	maxSurge, err := intstr.GetValueFromIntOrPercent(deployment.Spec.Strategy.RollingUpdate.MaxSurge, int(*(deployment.Spec.Replicas)), false)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Second rollout's replicaset should have 0 available replicas.
	e2elog.Logf("Verifying that the second rollout's replicaset has .status.availableReplicas = 0")
	gomega.Expect(secondRS.Status.AvailableReplicas).Should(gomega.Equal(int32(0)))

	// Second rollout's replicaset should have Deployment's (replicas + maxSurge - first RS's replicas) = 10 + 3 - 8 = 5 for .spec.replicas.
	newReplicas := replicas + int32(maxSurge) - minAvailableReplicas
	e2elog.Logf("Waiting for the second rollout's replicaset to have .spec.replicas = %d", newReplicas)
	gomega.Expect(replicaset.WaitForReplicaSetTargetSpecReplicas(c, secondRS, newReplicas)).NotTo(gomega.HaveOccurred())

	// The desired replicas wait makes sure that the RS controller has created expected number of pods.
	e2elog.Logf("Waiting for the second rollout's replicaset of deployment %q to have desired number of replicas", deploymentName)
	secondRS, err = c.AppsV1().ReplicaSets(ns).Get(secondRS.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = replicaset.WaitForReplicaSetDesiredReplicas(c.AppsV1(), secondRS)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Check the deployment's minimum availability.
	e2elog.Logf("Verifying that deployment %q has minimum required number of available replicas", deploymentName)
	if deployment.Status.AvailableReplicas < minAvailableReplicas {
		gomega.Expect(fmt.Errorf("observed %d available replicas, less than min required %d", deployment.Status.AvailableReplicas, minAvailableReplicas)).NotTo(gomega.HaveOccurred())
	}

	// Scale the deployment to 30 replicas.
	newReplicas = int32(30)
	e2elog.Logf("Scaling up the deployment %q from %d to %d", deploymentName, replicas, newReplicas)
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, ns, deployment.Name, func(update *apps.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	e2elog.Logf("Waiting for the replicasets of deployment %q to have desired number of replicas", deploymentName)
	firstRS, err = c.AppsV1().ReplicaSets(ns).Get(firstRS.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	secondRS, err = c.AppsV1().ReplicaSets(ns).Get(secondRS.Name, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// First rollout's replicaset should have .spec.replicas = 8 + (30-10)*(8/13) = 8 + 12 = 20 replicas.
	// Note that 12 comes from rounding (30-10)*(8/13) to nearest integer.
	e2elog.Logf("Verifying that first rollout's replicaset has .spec.replicas = 20")
	gomega.Expect(replicaset.WaitForReplicaSetTargetSpecReplicas(c, firstRS, 20)).NotTo(gomega.HaveOccurred())

	// Second rollout's replicaset should have .spec.replicas = 5 + (30-10)*(5/13) = 5 + 8 = 13 replicas.
	// Note that 8 comes from rounding (30-10)*(5/13) to nearest integer.
	e2elog.Logf("Verifying that second rollout's replicaset has .spec.replicas = 13")
	gomega.Expect(replicaset.WaitForReplicaSetTargetSpecReplicas(c, secondRS, 13)).NotTo(gomega.HaveOccurred())
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

func listDeploymentReplicaSets(c clientset.Interface, ns string, label map[string]string) *apps.ReplicaSetList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rsList, err := c.AppsV1().ReplicaSets(ns).List(options)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(len(rsList.Items)).To(gomega.BeNumerically(">", 0))
	return rsList
}

func orphanDeploymentReplicaSets(c clientset.Interface, d *apps.Deployment) error {
	trueVar := true
	deleteOptions := &metav1.DeleteOptions{OrphanDependents: &trueVar}
	deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(d.UID))
	return c.AppsV1().Deployments(d.Namespace).Delete(d.Name, deleteOptions)
}
