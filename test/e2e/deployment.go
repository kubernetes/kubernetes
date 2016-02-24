/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/util"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Deployment", func() {
	f := NewFramework("deployment")

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("RollingUpdateDeployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	It("RollingUpdateDeployment should scale up and down in the right order", func() {
		testRollingUpdateDeploymentEvents(f)
	})
	It("RecreateDeployment should delete old pods and create new ones", func() {
		testRecreateDeployment(f)
	})
	It("deployment should delete old replica sets", func() {
		testDeploymentCleanUpPolicy(f)
	})
	It("[Flaky] deployment should support rollover", func() {
		testRolloverDeployment(f)
	})
	It("[Flaky] paused deployment should be ignored by the controller", func() {
		testPausedDeployment(f)
	})
	It("[Flaky] deployment should support rollback", func() {
		testRollbackDeployment(f)
	})
	It("[Flaky] deployment should support rollback when there's replica set with no revision", func() {
		testRollbackDeploymentRSNoRevision(f)
	})
	It("deployment should label adopted RSs and pods", func() {
		testDeploymentLabelAdopted(f)
	})
})

func newRS(rsName string, replicas int, rsPodLabels map[string]string, imageName string, image string) *extensions.ReplicaSet {
	zero := int64(0)
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name: rsName,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: replicas,
			Selector: &unversioned.LabelSelector{MatchLabels: rsPodLabels},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: rsPodLabels,
				},
				Spec: api.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []api.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

func newDeployment(deploymentName string, replicas int, podLabels map[string]string, imageName string, image string, strategyType extensions.DeploymentStrategyType, revisionHistoryLimit *int) *extensions.Deployment {
	zero := int64(0)
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: replicas,
			Selector: &unversioned.LabelSelector{MatchLabels: podLabels},
			Strategy: extensions.DeploymentStrategy{
				Type: strategyType,
			},
			RevisionHistoryLimit: revisionHistoryLimit,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
				},
				Spec: api.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []api.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

func newDeploymentRollback(name string, annotations map[string]string, revision int64) *extensions.DeploymentRollback {
	return &extensions.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: annotations,
		RollbackTo:         extensions.RollbackConfig{Revision: revision},
	}
}

// checkDeploymentRevision checks if the input deployment's and its new replica set's revision and images are as expected.
func checkDeploymentRevision(c *clientset.Clientset, ns, deploymentName, revision, imageName, image string) (*extensions.Deployment, *extensions.ReplicaSet) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Check revision of the new replica set of this deployment
	newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
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

func stopDeployment(c *clientset.Clientset, oldC client.Interface, ns, deploymentName string) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	Logf("deleting deployment %s", deploymentName)
	reaper, err := kubectl.ReaperFor(extensions.Kind("Deployment"), oldC)
	Expect(err).NotTo(HaveOccurred())
	timeout := 1 * time.Minute
	err = reaper.Stop(ns, deployment.Name, timeout, api.NewDeleteOptions(0))
	Expect(err).NotTo(HaveOccurred())

	Logf("ensuring deployment %s was deleted", deploymentName)
	_, err = c.Extensions().Deployments(ns).Get(deployment.Name)
	Expect(err).To(HaveOccurred())
	Expect(errors.IsNotFound(err)).To(BeTrue())
	Logf("ensuring deployment %s rcs were deleted", deploymentName)
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := api.ListOptions{LabelSelector: selector}
	rss, err := c.Extensions().ReplicaSets(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(rss.Items).Should(HaveLen(0))
	Logf("ensuring deployment %s pods were deleted", deploymentName)
	if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		pods, err := c.Core().Pods(ns).List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		Failf("Failed to remove deployment %s pods!", deploymentName)
	}
}

func testNewDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(f.Client)

	deploymentName := "nginx-deployment"
	podLabels := map[string]string{"name": "nginx"}
	replicas := 1
	Logf("Creating simple deployment %s", deploymentName)
	d := newDeployment(deploymentName, replicas, podLabels, "nginx", "nginx", extensions.RollingUpdateDeploymentStrategyType, nil)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", kubectl.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "nginx", false, replicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// DeploymentStatus should be appropriately updated.
	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Status.Replicas).Should(Equal(replicas))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(replicas))

	// Check if it's updated to revision 1 correctly
	_, newRS := checkDeploymentRevision(c, ns, deploymentName, "1", "nginx", "nginx")
	// Check other annotations
	Expect(newRS.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(newRS.Annotations[kubectl.LastAppliedConfigAnnotation]).Should(Equal(""))
	Expect(deployment.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(deployment.Annotations[kubectl.LastAppliedConfigAnnotation]).Should(Equal("should-not-copy-to-replica-set"))
}

func testRollingUpdateDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  "nginx",
	}

	rsName := "nginx-controller"
	replicas := 3
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod", false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis", extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", "redis", "redis")

	// There should be 1 old RS (nginx-controller, which is adopted)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(allOldRSs)).Should(Equal(1))
	// The old RS should contain pod-template-hash in its selector, label, and template label
	Expect(len(allOldRSs[0].Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
}

func testRollingUpdateDeploymentEvents(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-2"}
	rsPodLabels := map[string]string{
		"name": "sample-pod-2",
		"pod":  "nginx",
	}
	rsName := "nginx-controller"
	replicas := 1

	rsRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rsRevision
	rs := newRS(rsName, replicas, rsPodLabels, "nginx", "nginx")
	rs.Annotations = annotations

	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod-2", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment-2"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis", extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Core().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new ReplicaSet and then to scale down
	// the old ReplicaSet.
	Expect(len(events.Items)).Should(Equal(2))
	newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled up replica set %s to 1", newRS.Name)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled down replica set %s to 0", rsName)))

	// Check if it's updated to revision 3546343826724305833 correctly
	checkDeploymentRevision(c, ns, deploymentName, "3546343826724305833", "redis", "redis")
}

func testRecreateDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-3"}
	rsPodLabels := map[string]string{
		"name": "sample-pod-3",
		"pod":  "nginx",
	}

	rsName := "nginx-controller"
	replicas := 3
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod-3", false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment-3"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis", extensions.RecreateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	waitForDeploymentStatus(c, ns, deploymentName, replicas, 0, replicas, 0)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Core().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new ReplicaSet and then to scale down the old ReplicaSet.
	Expect(len(events.Items)).Should(Equal(2))
	newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled down replica set %s to 0", rsName)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled up replica set %s to 3", newRS.Name)))

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", "redis", "redis")
}

// testDeploymentCleanUpPolicy tests that deployment supports cleanup policy
func testDeploymentCleanUpPolicy(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "cleanup-pod"}
	rsPodLabels := map[string]string{
		"name": "cleanup-pod",
		"pod":  "nginx",
	}
	rsName := "nginx-controller"
	replicas := 1
	revisionHistoryLimit := util.IntPtr(0)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "cleanup-pod", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis", extensions.RollingUpdateDeploymentStrategyType, revisionHistoryLimit))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	err = waitForDeploymentOldRSsNum(c, ns, deploymentName, *revisionHistoryLimit)
	Expect(err).NotTo(HaveOccurred())
}

// testRolloverDeployment tests that deployment supports rollover.
// i.e. we can change desired state and kick off rolling update, then change desired state again before it finishes.
func testRolloverDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	podName := "rollover-pod"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  "nginx",
	}

	rsName := "nginx-controller"
	rsReplicas := 4
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, rsReplicas, rsPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, podName, false, rsReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	deploymentMinReadySeconds := 5
	err = waitForPodsReady(c, ns, podName, deploymentMinReadySeconds)
	Expect(err).NotTo(HaveOccurred())

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	deploymentName, deploymentImageName := "redis-deployment", "redis-slave"
	deploymentReplicas := 4
	deploymentImage := "gcr.io/google_samples/gb-redisslave:v1"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	newDeployment := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	newDeployment.Spec.MinReadySeconds = deploymentMinReadySeconds
	newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
		MaxUnavailable: intstr.FromInt(1),
		MaxSurge:       intstr.FromInt(1),
	}
	_, err = c.Extensions().Deployments(ns).Create(newDeployment)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Make sure the deployment starts to scale up and down replica sets
	waitForPartialEvents(unversionedClient, ns, deployment, 2)
	// Check if it's updated to revision 1 correctly
	_, newRS := checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Before the deployment finishes, update the deployment to rollover the above 2 ReplicaSets and bring up redis pods.
	// If the deployment already finished here, the test would fail. When this happens, increase its minReadySeconds or replicas to prevent it.
	Expect(newRS.Spec.Replicas).Should(BeNumerically("<", deploymentReplicas))
	updatedDeploymentImage := "redis"
	_, err = updateDeploymentWithRetries(c, ns, newDeployment.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImage
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, deploymentMinReadySeconds)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 2 correctly
	checkDeploymentRevision(c, ns, deploymentName, "2", updatedDeploymentImage, updatedDeploymentImage)
}

func testPausedDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	deploymentName := "nginx"
	podLabels := map[string]string{"name": "nginx"}
	d := newDeployment(deploymentName, 1, podLabels, "nginx", "nginx", extensions.RollingUpdateDeploymentStrategyType, nil)
	d.Spec.Paused = true
	Logf("Creating paused deployment %s", deploymentName)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)
	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that there is no latest state realized for the new deployment.
	rs, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if rs != nil {
		err = fmt.Errorf("unexpected new rs/%s for deployment/%s", rs.Name, deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment to run
	deployment, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = false
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the resume.
	err = waitForObservedDeployment(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())

	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	opts := api.ListOptions{LabelSelector: selector}
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
	deployment, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = true
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pause.
	err = waitForObservedDeployment(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())

	newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(c.Extensions().ReplicaSets(ns).Delete(newRS.Name, nil)).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	if !deployment.Spec.Paused {
		err = fmt.Errorf("deployment %q should be paused", deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}
	shouldBeNil, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if shouldBeNil != nil {
		err = fmt.Errorf("deployment %q shouldn't have a replica set but there is %q", deployment.Name, shouldBeNil.Name)
		Expect(err).NotTo(HaveOccurred())
	}
}

// testRollbackDeployment tests that a deployment is created (revision 1) and updated (revision 2), and
// then rollback to revision 1 (should update template to revision 1, and then update revision 1 to 3),
// and then rollback to last revision.
func testRollbackDeployment(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}

	// Create a deployment to create nginx pods.
	deploymentName, deploymentImageName := "nginx-deployment", "nginx"
	deploymentReplicas := 1
	deploymentImage := "nginx"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	d := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "nginx", false, deploymentReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// DeploymentStatus should be appropriately updated.
	Expect(deployment.Status.Replicas).Should(Equal(deploymentReplicas))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(deploymentReplicas))

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Update the deployment to create redis pods.
	updatedDeploymentImage := "redis"
	updatedDeploymentImageName := "redis"
	_, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 2 correctly
	checkDeploymentRevision(c, ns, deploymentName, "2", updatedDeploymentImageName, updatedDeploymentImage)

	// Update the deploymentRollback to rollback to revision 1
	revision := int64(1)
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 3 correctly
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// Update the deploymentRollback to rollback to last revision
	revision = 0
	Logf("rolling back deployment %s to last revision", deploymentName)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 4 correctly
	checkDeploymentRevision(c, ns, deploymentName, "4", updatedDeploymentImageName, updatedDeploymentImage)
}

// testRollbackDeploymentRSNoRevision tests that deployment supports rollback even when there's old replica set without revision.
// An old replica set without revision is created, and then a deployment is created (v1). The deployment shouldn't add revision
// annotation to the old replica set. Then rollback the deployment to last revision, and it should fail.
// Then update the deployment to v2 and rollback it to v1 should succeed, now the deployment
// becomes v3. Then rollback the deployment to v10 (doesn't exist in history) should fail.
// Finally, rollback the deployment (v3) to v3 should be no-op.
// TODO: When we finished reporting rollback status in deployment status, check the rollback status here in each case.
func testRollbackDeploymentRSNoRevision(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(f.Client)
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  "nginx",
	}

	rsName := "nginx-controller"
	rsReplicas := 0
	rs := newRS(rsName, rsReplicas, rsPodLabels, "nginx", "nginx")
	rs.Annotations = make(map[string]string)
	rs.Annotations["make"] = "difference"
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())

	// Create a deployment to create nginx pods, which have different template than the replica set created above.
	deploymentName, deploymentImageName := "nginx-deployment", "nginx"
	deploymentReplicas := 1
	deploymentImage := "nginx"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	d := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	_, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "nginx", false, deploymentReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// DeploymentStatus should be appropriately updated.
	Expect(deployment.Status.Replicas).Should(Equal(deploymentReplicas))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(deploymentReplicas))

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Check that the replica set we created still doesn't contain revision information
	rs, err = c.Extensions().ReplicaSets(ns).Get(rsName)
	Expect(rs.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(""))

	// Update the deploymentRollback to rollback to last revision
	// Since there's only 1 revision in history, it should stay as revision 1
	revision := int64(0)
	Logf("rolling back deployment %s to last revision", deploymentName)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait until the rollback is done
	waitForRollbackDone(c, deployment)
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no last revision
	// Check if the deployment is still revision 1 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Update the deployment to create redis pods.
	updatedDeploymentImage := "redis"
	updatedDeploymentImageName := "redis"
	_, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 2 correctly
	checkDeploymentRevision(c, ns, deploymentName, "2", updatedDeploymentImageName, updatedDeploymentImage)

	// Update the deploymentRollback to rollback to revision 1
	revision = 1
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Wait until the rollback is done
	waitForRollbackDone(c, deployment)
	// TODO: report RollbackDone in deployment status and check it here

	// The pod template should be updated to the one in revision 1
	// Check if it's updated to revision 3 correctly
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// Update the deploymentRollback to rollback to revision 10
	// Since there's no revision 10 in history, it should stay as revision 3
	revision = 10
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait until the rollback is done
	waitForRollbackDone(c, deployment)
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no revision 10
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// Update the deploymentRollback to rollback to revision 3
	// Since it's already revision 3, it should be no-op
	revision = 3
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait until the rollback is done
	waitForRollbackDone(c, deployment)
	// TODO: report RollbackTemplateUnchanged in deployment status and check it here

	// The pod template shouldn't change since it's already revision 3
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)
}

func testDeploymentLabelAdopted(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	podName := "nginx"
	podLabels := map[string]string{"name": podName}

	rsName := "nginx-controller"
	replicas := 3
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, podLabels, podName, podName))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, podName, false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a nginx deployment to adopt the old rs.
	deploymentName := "nginx-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, podLabels, podName, podName, extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete replica sets with deployment
		newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Extensions().ReplicaSets(ns).Delete(newRS.Name, nil)).NotTo(HaveOccurred())
	}()

	// The RS and pods should be relabeled before the status is updated by syncRollingUpdateDeployment
	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", "nginx", "nginx")

	// There should be no old RSs (overlapping RS)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	oldRSs, allOldRSs, err := deploymentutil.GetOldReplicaSets(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(oldRSs)).Should(Equal(0))
	Expect(len(allOldRSs)).Should(Equal(0))
	// New RS should contain pod-template-hash in its selector, label, and template label
	newRS, err := deploymentutil.GetNewReplicaSet(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(newRS.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(newRS.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(newRS.Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	// All pods targeted by the deployment should contain pod-template-hash in their labels, and there should be only 3 pods
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := api.ListOptions{LabelSelector: selector}
	pods, err := c.Core().Pods(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	for _, pod := range pods.Items {
		Expect(len(pod.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	}
	Expect(len(pods.Items)).Should(Equal(replicas))
}
