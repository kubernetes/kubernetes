/*
Copyright 2015 Google Inc. All rights reserved.

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

/*
 * This test checks that NFS volume is working. It uses one pod exporting a
 * directory and other pod consuming it, exporting its content through http.
 *
 * To export something via NFS, the pod must be privileged (it mounts
 * /proc/fs/nfsd). The test reports 'success' on clusters with
 * --allow_privileged=False (which is the default).
 */

package e2e

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("NFS", func() {
	clean := true // If 'false', the test won't clear its namespace (and pods and services) upon completion. Useful for debugging.

	// filled in BeforeEach
	var c *client.Client
	var namespace string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
		namespace = "e2e-ns-" + dateStamp() + "-0"
	})

	AfterEach(func() {
		if clean {
			c.Namespaces().Delete(namespace)
		}
	})

	It("should be able to mount NFS", func() {
		podClient := c.Pods(namespace)
		serviceClient := c.Services(namespace)

		By("creating a NFS server pod")
		nfs_pod := &api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1beta3",
			},
			ObjectMeta: api.ObjectMeta{
				Name: "nfs-server-" + string(util.NewUUID()),
				Labels: map[string]string{
					"role": "nfs-server",
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:       "nfs-server",
						Image:      "jsafrane/nfs-data",
						Privileged: true,
						Ports: []api.ContainerPort{
							{
								Name:          "nfs",
								ContainerPort: 2049,
								Protocol:      api.ProtocolTCP,
							},
						},
					},
				},
			},
		}
		defer func() {
			if clean {
				By("deleting the pod")
				defer GinkgoRecover()
				podClient.Delete(nfs_pod.Name)
			}
		}()
		if _, err := podClient.Create(nfs_pod); err != nil {
			// Skip the test when 'privileged' containers are not allowed
			if strings.Contains(err.Error(), "spec.containers[0].privileged: forbidden 'true'") {
				By("Skipping NFS test, which is supported only on on cluster with --allow_privileged=True")
				return
			}
			// Report real error otherwise
			Failf("Failed to create %s pod: %v", nfs_pod.Name, err)
		}
		expectNoError(waitForPodRunningInNamespace(c, nfs_pod.Name, namespace))

		By("creating a NFS server service")
		nfs_service := &api.Service{
			TypeMeta: api.TypeMeta{
				Kind:       "Service",
				APIVersion: "v1beta3",
			},
			ObjectMeta: api.ObjectMeta{
				Name: "nfs-server",
			},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{
					{
						Name:     "nfs",
						Port:     2049,
						Protocol: api.ProtocolTCP,
					},
				},
				Selector: map[string]string{
					"role": "nfs-server",
				},
			},
		}

		defer func() {
			if clean {
				By("deleting the NFS service")
				defer GinkgoRecover()
				serviceClient.Delete(nfs_service.Name)
			}
		}()
		if _, err := serviceClient.Create(nfs_service); err != nil {
			Failf("Failed to create %s service: %v", nfs_service.Name, err)
		}

		By("locating the NFS server service")
		srv, err := serviceClient.Get(nfs_service.Name)
		expectNoError(err, "Cannot read IP address of service %v", nfs_service.Name)
		serverIP := srv.Spec.PortalIP
		Logf("NFS server IP address: %v", serverIP)

		By("creating a NFS client pod")
		web_pod := &api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1beta3",
			},
			ObjectMeta: api.ObjectMeta{
				Name: "nfs-web-" + string(util.NewUUID()),
				Labels: map[string]string{
					"role": "web-server",
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:       "web",
						Image:      "dockerfile/nginx",
						Privileged: false,
						Ports: []api.ContainerPort{
							{
								Name:          "web",
								ContainerPort: 80,
								Protocol:      api.ProtocolTCP,
							},
						},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      "nfs",
								MountPath: "/var/www/html",
							},
						},
					},
				},
				Volumes: []api.Volume{
					{
						Name: "nfs",
						VolumeSource: api.VolumeSource{
							NFS: &api.NFSVolumeSource{
								Server:   serverIP,
								Path:     "/",
								ReadOnly: true,
							},
						},
					},
				},
			},
		}
		defer func() {
			if clean {
				By("deleting the pod")
				defer GinkgoRecover()
				podClient.Delete(web_pod.Name)
			}
		}()
		if _, err := podClient.Create(web_pod); err != nil {
			Failf("Failed to create %s pod: %v", web_pod.Name, err)
		}
		expectNoError(waitForPodRunningInNamespace(c, web_pod.Name, namespace))

		By("reading a page from the web server")
		body, err := c.Get().
			Namespace(namespace).
			Prefix("proxy").
			Resource("pods").
			Name(web_pod.Name).
			DoRaw()
		expectNoError(err, "Cannot read web page: %v", err)
		Logf("body: %v", string(body))

		By("checking the page content")
		Expect(body).To(ContainSubstring("Hello world!"))
	})
})
