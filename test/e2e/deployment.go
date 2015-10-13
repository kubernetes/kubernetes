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
	deploymentUtil "k8s.io/kubernetes/pkg/util/deployment"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Deployment", func() {
	f := NewFramework("deployment")

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("deployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	It("deployment should scale up and down in the right order", func() {
		testRollingUpdateDeploymentEvents(f)
	})
})

func testNewDeployment(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	deploymentName := "nginx-deployment"
	podLabels := map[string]string{"name": "nginx"}
	Logf("Creating simple deployment %s", deploymentName)
	_, err := c.Deployments(ns).Create(&extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas:       1,
			Selector:       podLabels,
			UniqueLabelKey: "deployment.kubernetes.io/podTemplateHash",
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "nginx",
							Image: "nginx",
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
	}()
	// Check that deployment is created fine.
	deployment, err := c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "nginx", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// DeploymentStatus should be appropriately updated.
	deployment, err = c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Status.Replicas).Should(Equal(1))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(1))
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
	_, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: rcName,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 3,
			Selector: rcPodLabels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "nginx",
							Image: "nginx",
						},
					},
				},
			},
		},
	})
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
	newDeployment := extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas:       3,
			Selector:       deploymentPodLabels,
			UniqueLabelKey: "deployment.kubernetes.io/podTemplateHash",
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: deploymentPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "redis",
							Image: "redis",
						},
					},
				},
			},
		},
	}
	_, err = c.Deployments(ns).Create(&newDeployment)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, 3, 2, 4)
	Expect(err).NotTo(HaveOccurred())
}

func testRollingUpdateDeploymentEvents(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rcPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  "nginx",
	}
	rcName := "nginx-controller"
	_, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: rcName,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: rcPodLabels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "nginx",
							Image: "nginx",
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting replication controller %s", rcName)
		Expect(c.ReplicationControllers(ns).Delete(rcName)).NotTo(HaveOccurred())
	}()
	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "sample-pod", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "redis-deployment"
	Logf("Creating deployment %s", deploymentName)
	newDeployment := extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas:       1,
			Selector:       deploymentPodLabels,
			UniqueLabelKey: "deployment.kubernetes.io/podTemplateHash",
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: deploymentPodLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "redis",
							Image: "redis",
						},
					},
				},
			},
		},
	}
	_, err = c.Deployments(ns).Create(&newDeployment)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Logf("deleting deployment %s", deploymentName)
		Expect(c.Deployments(ns).Delete(deploymentName, nil)).NotTo(HaveOccurred())
	}()

	err = waitForDeploymentStatus(c, ns, deploymentName, 1, 0, 2)
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
	newRC, err := deploymentUtil.GetNewRC(*deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRC).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled up rc %s to 1", newRC.Name)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled down rc %s to 0", rcName)))
}
