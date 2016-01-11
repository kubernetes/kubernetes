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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/intstr"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Deployment", func() {
	f := NewFramework("deployment")

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("deployment should delete old pods and create new ones [Flaky]", func() {
		testRollingUpdateDeployment(f)
	})
	It("deployment should scale up and down in the right order [Flaky]", func() {
		testRollingUpdateDeploymentEvents(f)
	})
	It("deployment should support rollover [Flaky]", func() {
		testRolloverDeployment(f)
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

func newDeployment(deploymentName string, replicas int, podLabels map[string]string, imageName string, image string) *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas:       replicas,
			Selector:       podLabels,
			UniqueLabelKey: extensions.DefaultDeploymentUniqueLabelKey,
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

func testNewDeployment(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	deploymentName := "nginx-deployment"
	podLabels := map[string]string{"name": "nginx"}
	replicas := 1
	Logf("Creating simple deployment %s", deploymentName)
	_, err := c.Deployments(ns).Create(newDeployment(deploymentName, replicas, podLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.ReplicationControllers(ns).Delete(newRC.Name)).NotTo(HaveOccurred())
	}()
	// Check that deployment is created fine.
	deployment, err := c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "nginx", false, replicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// DeploymentStatus should be appropriately updated.
	deployment, err = c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Status.Replicas).Should(Equal(replicas))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(replicas))
}

func testRollingUpdateDeployment(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rcPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	replicas := 3
	_, err := c.ReplicationControllers(ns).Create(newRC(rcName, replicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.ReplicationControllers(ns).Delete(rcName)).NotTo(HaveOccurred())
	}()
	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "sample-pod", false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.ReplicationControllers(ns).Delete(newRC.Name)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())
}

func testRollingUpdateDeploymentEvents(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-2"}
	rcPodLabels := map[string]string{
		"name": "sample-pod-2",
		"pod":  "nginx",
	}
	rcName := "nginx-controller"
	replicas := 1
	_, err := c.ReplicationControllers(ns).Create(newRC(rcName, replicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.ReplicationControllers(ns).Delete(rcName)).NotTo(HaveOccurred())
	}()
	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "sample-pod-2", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment-2"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, "redis", "redis"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.ReplicationControllers(ns).Delete(newRC.Name)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(c, ns, deployment, 2)
	events, err := c.Events(ns).Search(deployment)
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
}

// testRolloverDeployment tests that deployment supports rollover.
// i.e. we can change desired state and kick off rolling update, then change desired state again before it finishes.
func testRolloverDeployment(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	podName := "rollover-pod"
	deploymentPodLabels := map[string]string{"name": podName}
	rcPodLabels := map[string]string{
		"name": podName,
		"pod":  "nginx",
	}

	rcName := "nginx-controller"
	rcReplicas := 4
	_, err := c.ReplicationControllers(ns).Create(newRC(rcName, rcReplicas, rcPodLabels, "nginx", "nginx"))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.ReplicationControllers(ns).Delete(rcName)).NotTo(HaveOccurred())
	}()
	// Verify that the required pods have come up.
	err = verifyPods(c, ns, podName, false, rcReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	deploymentName, deploymentImageName := "redis-deployment", "redis-slave"
	deploymentReplicas := 4
	deploymentImage := "gcr.io/google_samples/gb-redisslave:v1"
	deploymentMinReadySeconds := 5
	Logf("Creating deployment %s", deploymentName)
	newDeployment := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage)
	newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
		MaxUnavailable:  intstr.FromInt(1),
		MaxSurge:        intstr.FromInt(1),
		MinReadySeconds: deploymentMinReadySeconds,
	}
	_, err = c.Deployments(ns).Create(newDeployment)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		deployment, err := c.Deployments(ns).Get(deploymentName)
		Expect(err).NotTo(HaveOccurred())
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
		// TODO: remove this once we can delete rcs with deployment
		newRC, err := deploymentutil.GetNewRC(*deployment, c)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.ReplicationControllers(ns).Delete(newRC.Name)).NotTo(HaveOccurred())
	}()
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Deployments(ns).Get(deploymentName)
	// Make sure the deployment starts to scale up and down RCs
	waitForPartialEvents(c, ns, deployment, 2)
	newRC, err := deploymentutil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRC).NotTo(Equal(nil))

	// Before the deployment finishes, update the deployment to rollover the above 2 rcs and bring up redis pods.
	// If the deployment already finished here, the test would fail. When this happens, increase its minReadySeconds or replicas to prevent it.
	Expect(newRC.Spec.Replicas).Should(BeNumerically("<", deploymentReplicas))
	updatedDeploymentImage := "redis"
	newDeployment.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImage
	newDeployment.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	Logf("updating deployment %s", deploymentName)
	_, err = c.Deployments(ns).Update(newDeployment)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, deploymentMinReadySeconds)
	Expect(err).NotTo(HaveOccurred())

	// Make sure updated deployment contains "redis" image
	deployment, err = c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Spec.Template.Spec.Containers[0].Image).Should(Equal(updatedDeploymentImage))
	// Make sure new RC contains "redis" image
	newRC, err = deploymentutil.GetNewRC(*deployment, c)
	Expect(newRC.Spec.Template.Spec.Containers[0].Image).Should(Equal(updatedDeploymentImage))
}
