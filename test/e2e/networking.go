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
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Networking", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should function for pods", func() {
		if testContext.provider == "vagrant" {
			By("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
			return
		}

		// Test basic external connectivity.
		resp, err := http.Get("http://google.com/")
		if err != nil {
			Fail(fmt.Sprintf("unable to talk to the external internet: %v", err))
		}
		if resp.StatusCode != http.StatusOK {
			Fail(fmt.Sprintf("unexpected error code. expected 200, got: %v (%v)", resp.StatusCode, resp))
		}

		ns := api.NamespaceDefault
		// TODO(satnam6502): Replace call of randomSuffix with call to NewUUID when service
		//                   names have the same form as pod and replication controller names.
		name := "nettest-" + randomSuffix()

		svc, err := c.Services(ns).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": name,
				},
			},
			Spec: api.ServiceSpec{
				Port:          8080,
				ContainerPort: util.NewIntOrStringFromInt(8080),
				Selector: map[string]string{
					"name": name,
				},
			},
		})
		By(fmt.Sprintf("Creating service with name %s", svc.Name))
		if err != nil {
			Fail(fmt.Sprintf("unable to create test service %s: %v", svc.Name, err))
		}
		// Clean up service
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the service")
			if err = c.Services(ns).Delete(svc.Name); err != nil {
				Fail(fmt.Sprintf("unable to delete svc %v: %v", svc.Name, err))
			}
		}()

		By("Creating a replication controller")
		rc, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": name,
				},
			},
			Spec: api.ReplicationControllerSpec{
				Replicas: 8,
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
								Name:    "webserver",
								Image:   "kubernetes/nettest:latest",
								Command: []string{"-service=" + name},
								Ports:   []api.ContainerPort{{ContainerPort: 8080}},
							},
						},
					},
				},
			},
		})
		if err != nil {
			Fail(fmt.Sprintf("unable to create test rc: %v", err))
		}
		// Clean up rc
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the replication controller")
			rc.Spec.Replicas = 0
			rc, err = c.ReplicationControllers(ns).Update(rc)
			if err != nil {
				Fail(fmt.Sprintf("unable to modify replica count for rc %v: %v", rc.Name, err))
			}
			if err = c.ReplicationControllers(ns).Delete(rc.Name); err != nil {
				Fail(fmt.Sprintf("unable to delete rc %v: %v", rc.Name, err))
			}
		}()

		By("Waiting for connectivity to be verified")
		const maxAttempts = 60
		passed := false
		var body []byte
		for i := 0; i < maxAttempts && !passed; i++ {
			time.Sleep(2 * time.Second)
			body, err = c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("status").Do().Raw()
			if err != nil {
				fmt.Printf("Attempt %v/%v: service/pod still starting. (error: '%v')\n", i, maxAttempts, err)
				continue
			}
			switch string(body) {
			case "pass":
				fmt.Printf("Passed on attempt %v. Cleaning up.\n", i)
				passed = true
				break
			case "running":
				fmt.Printf("Attempt %v/%v: test still running\n", i, maxAttempts)
				break
			case "fail":
				if body, err = c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("read").Do().Raw(); err != nil {
					Fail(fmt.Sprintf("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err))
				} else {
					Fail(fmt.Sprintf("Failed on attempt %v. Cleaning up. Details:\n%v", i, string(body)))
				}
				break
			}
		}

		if !passed {
			if body, err = c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("read").Do().Raw(); err != nil {
				Fail(fmt.Sprintf("Timed out. Cleaning up. Error reading details: %v", err))
			} else {
				Fail(fmt.Sprintf("Timed out. Cleaning up. Details:\n%v", string(body)))
			}
		}
		Expect(string(body)).To(Equal("pass"))
	})

	It("should provide unchanging URLs", func() {
		tests := []struct {
			path string
		}{
			{path: "/validate"},
			{path: "/healthz"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := c.RESTClient.Get().
				AbsPath(test.path).
				Do().
				Raw()
			if err != nil {
				Fail(fmt.Sprintf("Failed: %v\nBody: %s", err, string(data)))
			}
		}
	})
})
