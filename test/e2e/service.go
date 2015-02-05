/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Services", func() {
	var c *client.Client

	BeforeEach(func() {
		c = loadClientOrDie()
	})

	It("should provide DNS for the cluster", func() {
		if testContext.provider == "vagrant" {
			By("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
			return
		}

		podClient := c.Pods(api.NamespaceDefault)

		//TODO: Wait for skyDNS

		// All the names we need to be able to resolve.
		namesToResolve := []string{
			"kubernetes-ro",
			"kubernetes-ro.default",
			"kubernetes-ro.default.kubernetes.local",
			"google.com",
		}

		probeCmd := "for i in `seq 1 600`; do "
		for _, name := range namesToResolve {
			probeCmd += fmt.Sprintf("wget -O /dev/null %s && echo OK > /results/%s;", name, name)
		}
		probeCmd += "sleep 1; done"

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := &api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1beta1",
			},
			ObjectMeta: api.ObjectMeta{
				Name: "dns-test-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "results",
						Source: api.VolumeSource{
							EmptyDir: &api.EmptyDir{},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:  "webserver",
						Image: "kubernetes/test-webserver",
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "results",
								MountPath: "/results",
							},
						},
					},
					{
						Name:    "pinger",
						Image:   "busybox",
						Command: []string{"sh", "-c", probeCmd},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "results",
								MountPath: "/results",
							},
						},
					},
				},
			},
		}

		By("submitting the pod to kuberenetes")
		_, err := podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create %s pod: %v", pod.Name, err))
		}
		defer func() {
			By("deleting the pod")
			defer GinkgoRecover()
			podClient.Delete(pod.Name)
		}()

		By("waiting for the pod to start running")
		waitForPodRunning(c, pod.Name)

		By("retrieving the pod")
		pod, err = podClient.Get(pod.Name)
		if err != nil {
			Fail(fmt.Sprintf("Failed to get pod %s: %v", pod.Name, err))
		}

		// Try to find results for each expected name.
		By("looking for the results for each expected name")
		var failed []string
		for try := 1; try < 100; try++ {
			failed = []string{}
			for _, name := range namesToResolve {
				_, err := c.Get().
					Prefix("proxy").
					Resource("pods").
					Namespace("default").
					Name(pod.Name).
					Suffix("results", name).
					Do().Raw()
				if err != nil {
					failed = append(failed, name)
					fmt.Printf("Lookup using %s for %s failed: %v\n", pod.Name, name, err)
				}
			}
			if len(failed) == 0 {
				break
			}
			fmt.Printf("lookups using %s failed for: %v\n", pod.Name, failed)
			time.Sleep(10 * time.Second)
		}
		Expect(len(failed)).To(Equal(0))

		// TODO: probe from the host, too.

		fmt.Printf("DNS probes using %s succeeded\n", pod.Name)
	})

	It("should provide RW and RO services", func() {
		svc := api.ServiceList{}
		err := c.Get().
			Namespace("default").
			AbsPath("/api/v1beta1/proxy/services/kubernetes-ro/api/v1beta1/services").
			Do().
			Into(&svc)
		if err != nil {
			Fail(fmt.Sprintf("unexpected error listing services using ro service: %v", err))
		}
		var foundRW, foundRO bool
		for i := range svc.Items {
			if svc.Items[i].Name == "kubernetes" {
				foundRW = true
			}
			if svc.Items[i].Name == "kubernetes-ro" {
				foundRO = true
			}
		}
		Expect(foundRW).To(Equal(true))
		Expect(foundRO).To(Equal(true))
	})
})
