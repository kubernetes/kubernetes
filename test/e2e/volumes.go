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
 * This test checks that various VolumeSources are working. For each volume
 * type it creates a server pod + service, exporing simple 'index.html' file.
 * Then it uses appropriate VolumeSource to import this file into a client pod
 * and tests, that the pod can see the file. It does so by importing the file
 * into web server root and loadind the index.html from it.
 *
 * These tests work only when privileged containers are allowed, exporting
 * various filesystems (NFS, GlusterFS, ...) usually needs some mounting or
 * other privileged magic in the server pod.
 */

package e2e
import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"strings"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Configuration of one tests. The test consist of:
// - server pod - runs serverImage, exports ports[
// - server service - exports ports[] from the server pod
// - client service - does not need any special configuration
type VolumeTestConfig struct {
	namespace string
	// Prefix of all pods and services. Typically the test name.
	prefix string
	// Name of container image for the server pod.
	serverImage string
	// Ports to export from the server pod via the server service. TCP only.
	serverPorts []int
}

// Starts a container specified by config.containerImage and exports all
// configured ports from it as a service. The returned service should be used to
// get the server IP address and create appropriate VolumeSource.
// The function may return nil - the test should be skipped in this case.
func startVolumeServer(client *client.Client, config VolumeTestConfig) *api.Service {
	podClient := client.Pods(config.namespace)
	serviceClient := client.Services(config.namespace)

	portCount := len(config.serverPorts)
	serverPodPorts := make([]api.ContainerPort, portCount)
	serverServicePorts := make([]api.ServicePort, portCount)

	for i := 0; i < portCount; i++ {
		portName := fmt.Sprintf("%s-%d", config.prefix, i)

		serverPodPorts[i] = api.ContainerPort{
			Name:          portName,
			ContainerPort: config.serverPorts[i],
			Protocol:      api.ProtocolTCP,
		}

		serverServicePorts[i] = api.ServicePort{
			Name:     portName,
			Port:     config.serverPorts[i],
			Protocol: api.ProtocolTCP,
		}
	}

	By(fmt.Sprint("creating ", config.prefix, " server pod"))
	serverPod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1beta3",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-server",
			Labels: map[string]string{
				"role": config.prefix + "-server",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:       config.prefix + "-server",
					Image:      config.serverImage,
					Privileged: true,
					Ports: serverPodPorts,
				},
			},
		},
	}
	if _, err := podClient.Create(serverPod); err != nil {
		// Skip the test when 'privileged' containers are not allowed
		if strings.Contains(err.Error(), "spec.containers[0].privileged: forbidden 'true'") {
			By(fmt.Sprint("Skipping ", config.prefix, " test, which is supported only on on cluster with --allow_privileged=True"))
			return nil
		}
		// Report real error otherwise
		Failf("Failed to create %s pod: %v", serverPod.Name, err)
	}

	expectNoError(waitForPodRunningInNamespace(client, serverPod.Name, config.namespace))
	By(fmt.Sprint("creating ", config.prefix, " server service"))
	serverService := &api.Service{
		TypeMeta: api.TypeMeta{
			Kind:       "Service",
			APIVersion: "v1beta3",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-server",
		},
		Spec: api.ServiceSpec{
			Ports: serverServicePorts,
			Selector: map[string]string{
				"role": config.prefix + "-server",
			},
		},
	}
	if _, err := serviceClient.Create(serverService); err != nil {
		Failf("Failed to create %s service: %v", serverService.Name, err)
	}

	By("locating the NFS server service")
	srv, err := serviceClient.Get(serverService.Name)
	expectNoError(err, "Cannot read IP address of service %v: %v", serverService.Name, err)

	By("sleeping a bit to give the server time to start")
	time.Sleep(1 * time.Second)
	return srv
}

// Clean both server pod+service and client pod.
func volumeTestCleanup(client *client.Client, config VolumeTestConfig) {
	By(fmt.Sprint("cleaning the environment after ", config.prefix))

	defer GinkgoRecover()

	podClient := client.Pods(config.namespace)
	serviceClient := client.Services(config.namespace)

 	// ignore all errors, the pods/services may not be even created
	podClient.Delete(config.prefix + "-client")
	serviceClient.Delete(config.prefix + "-server")
	podClient.Delete(config.prefix + "-server")

	if config.namespace != "default" {
		client.Namespaces().Delete(config.namespace)
	}
}

// Start a client pod using given VolumeSource (exported by startVolumeServer())
// and check that the pod sees the data from the server pod+service.
func testVolumeClient(client *client.Client, config VolumeTestConfig, volume api.VolumeSource, expectedContent string) {
	By(fmt.Sprint("starting ", config.prefix, " client"))
	podClient := client.Pods(config.namespace)

	clientPod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1beta3",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-client",
			Labels: map[string]string{
				"role": config.prefix + "-client",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:       config.prefix + "-client",
					Image:      "nginx",
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
							Name:      config.prefix + "-volume",
							MountPath: "/usr/share/nginx/html",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: config.prefix + "-volume",
					VolumeSource: volume,
				},
			},
		},
	}
	if _, err := podClient.Create(clientPod); err != nil {
		Failf("Failed to create %s pod: %v", clientPod.Name, err)
	}
	expectNoError(waitForPodRunningInNamespace(client, clientPod.Name, config.namespace))

	By("reading a web page from the client")
	body, err := client.Get().
		Namespace(config.namespace).
		Prefix("proxy").
		Resource("pods").
		Name(clientPod.Name).
		DoRaw()
	expectNoError(err, "Cannot read web page: %v", err)
	Logf("body: %v", string(body))

	By("checking the page content")
	Expect(body).To(ContainSubstring(expectedContent))
}

var _ = Describe("Volumes", func() {
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

	It("should be able to mount NFS", func() {
		config := VolumeTestConfig{
			namespace: namespace,
			prefix: "nfs",
			serverImage: "jsafrane/nfs-data",
			serverPorts: []int { 2049 },
		}

		defer func() {
			if clean {
				volumeTestCleanup(c, config)
			}
		}()

		srv := startVolumeServer(c, config)
		if srv == nil {
			// Skip the test, message has been already logged.
			return
		}

		serverIP := srv.Spec.PortalIP
		Logf("NFS server IP address: %v", serverIP)

		volume := api.VolumeSource{
			NFS: &api.NFSVolumeSource{
				Server:   serverIP,
				Path:     "/",
				ReadOnly: true,
			},
		}
		testVolumeClient(c, config, volume, "Hello world!")
	})
})
