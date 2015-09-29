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
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ReplicationController", func() {
	framework := NewFramework("replication-controller")

	It("should serve a basic image on each replica with a public image", func() {
		ServeImageOrFail(framework, "basic", "gcr.io/google_containers/serve_hostname:1.1")
	})

	It("should serve a basic image on each replica with a private image", func() {
		// requires private images
		SkipUnlessProviderIs("gce", "gke")

		ServeImageOrFail(framework, "private", "b.gcr.io/k8s_authenticated_test/serve_hostname:1.1")
	})
})

// A basic test to check the deployment of an image using
// a replication controller. The image serves its hostname
// which is checked for each replica.
func ServeImageOrFail(f *Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(util.NewUUID())
	replicas := 2

	// Create a replication controller for a service
	// that serves its hostname.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	By(fmt.Sprintf("Creating replication controller %s", name))
	controller, err := f.Client.ReplicationControllers(f.Namespace.Name).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		if err := DeleteRC(f.Client, f.Namespace.Name, controller.Name); err != nil {
			Logf("Failed to cleanup replication controller %v: %v.", controller.Name, err)
		}
	}()

	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))

	pods, err := podsCreated(f.Client, f.Namespace.Name, name, replicas)

	By("Ensuring each pod is running")

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = f.WaitForPodRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Verify that something is listening.
	By("Trying to dial each unique pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	err = wait.Poll(retryInterval, retryTimeout, podResponseChecker{f.Client, f.Namespace.Name, label, name, true, pods}.checkAllResponses)
	if err != nil {
		Failf("Did not get expected responses within the timeout period of %.2f seconds.", retryTimeout.Seconds())
	}
}

func isElementOf(podUID types.UID, pods *api.PodList) bool {
	for _, pod := range pods.Items {
		if pod.UID == podUID {
			return true
		}
	}
	return false
}
