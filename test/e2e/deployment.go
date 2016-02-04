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
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
	"k8s.io/kubernetes/pkg/labels"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/intstr"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Deployment [Feature:Deployment]", func() {
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
	It("deployment should delete old rcs", func() {
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
	It("deployment should support rollback when there's RC with no revision", func() {
		testRollbackDeploymentRCNoRevision(f)
	})
})

func newRC(rcName string, replicas int, rcPodLabels map[string]string, imageName string, image string) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: rcName,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: rcPodLabels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: api.PodSpec{
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
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: replicas,
			Selector: podLabels,
			Strategy: extensions.DeploymentStrategy{
				Type: strategyType,
			},
			RevisionHistoryLimit: revisionHistoryLimit,
			UniqueLabelKey:       extensions.DefaultDeploymentUniqueLabelKey,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
				},
				Spec: api.PodSpec{
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

// checkDeploymentRevision checks if the input deployment's and its new RC's revision and images are as expected.
func checkDeploymentRevision(c *clientset.Clientset, ns, deploymentName, revision, imageName, image string) (*extensions.Deployment, *api.ReplicationController) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Check revision of the new RC of this deployment
	newRC, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRC.Annotations).NotTo(Equal(nil))
	Expect(newRC.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	// Check revision of This deployment
	Expect(deployment.Annotations).NotTo(Equal(nil))
	Expect(deployment.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	if len(imageName) > 0 {
		// Check the image the new RC creates
		Expect(newRC.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(newRC.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
		// Check the image the deployment creates
		Expect(deployment.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(deployment.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
	}
	return deployment, newRC
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
	d.Annotations = map[string]string{"test": "annotation-should-copy-to-RC"}
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()
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
	_, newRC := checkDeploymentRevision(c, ns, deploymentName, "1", "nginx", "nginx")
	Expect(newRC.Annotations["test"]).Should(Equal("annotation-should-copy-to-RC"))
}

func testRollingUpdateDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rcPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	replicas := 3
	_, err := c.Legacy().ReplicationControllers(ns).Create(newRC(rcName, replicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.Legacy().ReplicationControllers(ns).Delete(rcName, nil)).NotTo(HaveOccurred())
	}()
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check if it's updated to revision 1 correctly
	checkDeploymentRevision(c, ns, deploymentName, "1", "redis", "redis")
}

func testRollingUpdateDeploymentEvents(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-2"}
	rcPodLabels := map[string]string{
		"name": "sample-pod-2",
		"pod":  "nginx",
	}
	rcName := "nginx-controller"
	replicas := 1

	rcRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rcRevision
	rc := newRC(rcName, replicas, rcPodLabels, "nginx", "nginx")
	rc.Annotations = annotations

	_, err := c.Legacy().ReplicationControllers(ns).Create(rc)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.Legacy().ReplicationControllers(ns).Delete(rcName, nil)).NotTo(HaveOccurred())
	}()
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Legacy().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new RC and then to scale down the old RC.
	Expect(len(events.Items)).Should(Equal(2))
	newRC, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRC).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled up rc %s to 1", newRC.Name)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled down rc %s to 0", rcName)))

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
	rcPodLabels := map[string]string{
		"name": "sample-pod-3",
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	replicas := 3
	_, err := c.Legacy().ReplicationControllers(ns).Create(newRC(rcName, replicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.Legacy().ReplicationControllers(ns).Delete(rcName, nil)).NotTo(HaveOccurred())
	}()
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, 0, replicas, 0)
	if err != nil {
		deployment, _ := c.Extensions().Deployments(ns).Get(deploymentName)
		Logf("deployment = %+v", deployment)
	}
	Expect(err).NotTo(HaveOccurred())

	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Legacy().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new RC and then to scale down the old RC.
	Expect(len(events.Items)).Should(Equal(2))
	newRC, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRC).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled down rc %s to 0", rcName)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled up rc %s to 3", newRC.Name)))

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
	rcPodLabels := map[string]string{
		"name": "cleanup-pod",
		"pod":  "nginx",
	}
	rcName := "nginx-controller"
	replicas := 1
	revisionHistoryLimit := new(int)
	*revisionHistoryLimit = 0
	_, err := c.Legacy().ReplicationControllers(ns).Create(newRC(rcName, replicas, rcPodLabels, "nginx", "nginx"))
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentOldRCsNum(c, ns, deploymentName, *revisionHistoryLimit)
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
	rcPodLabels := map[string]string{
		"name": podName,
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	rcReplicas := 4
	_, err := c.Legacy().ReplicationControllers(ns).Create(newRC(rcName, rcReplicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.Legacy().ReplicationControllers(ns).Delete(rcName, nil)).NotTo(HaveOccurred())
	}()
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, podName, false, rcReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	deploymentName, deploymentImageName := "redis-deployment", "redis-slave"
	deploymentReplicas := 4
	deploymentImage := "gcr.io/google_samples/gb-redisslave:v1"
	deploymentMinReadySeconds := 5
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
	}()
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Make sure the deployment starts to scale up and down RCs
	waitForPartialEvents(unversionedClient, ns, deployment, 2)
	// Check if it's updated to revision 1 correctly
	_, newRC := checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Before the deployment finishes, update the deployment to rollover the above 2 rcs and bring up redis pods.
	// If the deployment already finished here, the test would fail. When this happens, increase its minReadySeconds or replicas to prevent it.
	Expect(newRC.Spec.Replicas).Should(BeNumerically("<", deploymentReplicas))
	updatedDeploymentImage := "redis"
	newDeployment.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImage
	newDeployment.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	Logf("updating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Update(newDeployment)
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
	defer func() {
		_, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
	}()
	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that there is no latest state realized for the new deployment.
	rc, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if rc != nil {
		err = fmt.Errorf("unexpected new rc/%s for deployment/%s", rc.Name, deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment to run
	deployment.Spec.Paused = false
	deployment, err = c.Extensions().Deployments(ns).Update(deployment)
	Expect(err).NotTo(HaveOccurred())

	opts := api.ListOptions{LabelSelector: labels.Set(deployment.Spec.Selector).AsSelector()}
	w, err := c.Legacy().ReplicationControllers(ns).Watch(opts)
	Expect(err).NotTo(HaveOccurred())

	select {
	case <-w.ResultChan():
		// this is it
	case <-time.After(time.Minute):
		err = fmt.Errorf("expected a new rc to be created")
		Expect(err).NotTo(HaveOccurred())
	}

	// Pause the deployment and delete the replication controller.
	// The paused deployment shouldn't recreate a new one.
	deployment.Spec.Paused = true
	deployment.ResourceVersion = ""
	deployment, err = c.Extensions().Deployments(ns).Update(deployment)
	Expect(err).NotTo(HaveOccurred())

	newRC, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	if !deployment.Spec.Paused {
		err = fmt.Errorf("deployment %q should be paused", deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}
	shouldBeNil, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if shouldBeNil != nil {
		err = fmt.Errorf("deployment %q shouldn't have a rc but there is %q", deployment.Name, shouldBeNil.Name)
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
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
		oldRCs, _, err := deploymentutil.GetOldRCs(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		for _, oldRC := range oldRCs {
			Expect(c.Legacy().ReplicationControllers(ns).Delete(oldRC.Name, nil)).NotTo(HaveOccurred())
		}
	}()
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
	d.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
	d.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	Logf("updating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Update(d)
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

// testRollbackDeploymentRCNoRevision tests that deployment supports rollback even when there's old RC without revision.
// An old RC without revision is created, and then a deployment is created (v1). The deployment shouldn't add revision
// annotation to the old RC. Then rollback the deployment to last revision, and it should fail and emit related event.
// Then update the deployment to v2 and rollback it to v1 should succeed and emit related event, now the deployment
// becomes v3. Then rollback the deployment to v10 (doesn't exist in history) should fail and emit related event.
// Finally, rollback the deployment (v3) to v3 should be no-op and emit related event.
func testRollbackDeploymentRCNoRevision(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(f.Client)
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}
	rcPodLabels := map[string]string{
		"name": podName,
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	rcReplicas := 0
	rc := newRC(rcName, rcReplicas, rcPodLabels, "nginx", "nginx")
	rc.Annotations = make(map[string]string)
	rc.Annotations["make"] = "difference"
	_, err := c.Legacy().ReplicationControllers(ns).Create(rc)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.Legacy().ReplicationControllers(ns).Delete(rcName, nil)).NotTo(HaveOccurred())
	}()

	// Create a deployment to create nginx pods, which have different template than the rc created above.
	deploymentName, deploymentImageName := "nginx-deployment", "nginx"
	deploymentReplicas := 1
	deploymentImage := "nginx"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	d := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	_, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Extensions().Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.Legacy().ReplicationControllers(ns).Delete(newRC.Name, nil)).NotTo(HaveOccurred())
		oldRCs, _, err := deploymentutil.GetOldRCs(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		for _, oldRC := range oldRCs {
			Expect(c.Legacy().ReplicationControllers(ns).Delete(oldRC.Name, nil)).NotTo(HaveOccurred())
		}
	}()
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

	// Check that the rc we created still doesn't contain revision information
	rc, err = c.Legacy().ReplicationControllers(ns).Get(rcName)
	Expect(rc.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(""))

	// Update the deploymentRollback to rollback to last revision
	// Since there's only 1 revision in history, it should stay as revision 1
	revision := int64(0)
	Logf("rolling back deployment %s to last revision", deploymentName)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// There should be revision not found event since there's no last revision
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Events(ns).Search(deployment)
	Expect(err).NotTo(HaveOccurred())
	Expect(events.Items[1].Reason).Should(Equal(deploymentutil.RollbackRevisionNotFound))

	// Check if it's still revision 1
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Update the deployment to create redis pods.
	updatedDeploymentImage := "redis"
	updatedDeploymentImageName := "redis"
	d.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
	d.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	Logf("updating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Update(d)
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

	// There should be rollback event after we rollback to revision 1
	waitForEvents(unversionedClient, ns, deployment, 5)
	events, err = c.Events(ns).Search(deployment)
	Expect(err).NotTo(HaveOccurred())
	Expect(events.Items[4].Reason).Should(Equal(deploymentutil.RollbackDone))

	// Check if it's updated to revision 3 correctly
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// Update the deploymentRollback to rollback to revision 10
	// Since there's no revision 10 in history, it should stay as revision 3, and emit an event
	revision = 10
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// There should be revision not found event since there's no revision 10
	waitForEvents(unversionedClient, ns, deployment, 7)
	events, err = c.Events(ns).Search(deployment)
	Expect(err).NotTo(HaveOccurred())
	Expect(events.Items[6].Reason).Should(Equal(deploymentutil.RollbackRevisionNotFound))

	// Check if it's still revision 3
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// Update the deploymentRollback to rollback to revision 3
	// Since it's already revision 3, it should be no-op and emit an event
	revision = 3
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// There should be revision template unchanged event since it's already revision 3
	waitForEvents(unversionedClient, ns, deployment, 8)
	events, err = c.Events(ns).Search(deployment)
	Expect(err).NotTo(HaveOccurred())
	Expect(events.Items[7].Reason).Should(Equal(deploymentutil.RollbackTemplateUnchanged))

	// Check if it's still revision 3
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)
}
