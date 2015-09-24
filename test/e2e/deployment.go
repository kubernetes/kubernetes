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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Deployment", func() {
	f := NewFramework("deployment")

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("deployment should delete old pods and create new ones", func() {
		testDeploymentDeletesOldPods(f)
	})
})

func testNewDeployment(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	deploymentName := "nginx-deployment"
	podLabels := map[string]string{"name": "nginx"}
	Logf("Creating simple deployment %s", deploymentName)
	_, err := c.Deployments(ns).Create(&experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: experimental.DeploymentSpec{
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
	Logf("Deployment: %s", deployment)

	// Verify that the required pods have come up.
	err = verifyPods(c, ns, "nginx", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		return
	}
	// DeploymentStatus should be appropriately updated.
	deployment, err = c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Status.Replicas).Should(Equal(1))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(1))
}

func testDeploymentDeletesOldPods(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	// Create redis pods.
	podLabels := map[string]string{"name": "sample-pod"}
	rcName := "redis-controller"
	_, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: rcName,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: podLabels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
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
		return
	}

	// Create a deployment to delete redis pods and instead bring up nginx pods.
	deploymentName := "nginx-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Deployments(ns).Create(&experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: experimental.DeploymentSpec{
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

	// Verify that the required pods have come up.
	verifyPods(c, ns, "nginx", false, 1)
	// DeploymentStatus should be appropriately updated.
	deployment, err := c.Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	Expect(deployment.Status.Replicas).Should(Equal(1))
	Expect(deployment.Status.UpdatedReplicas).Should(Equal(1))
}
