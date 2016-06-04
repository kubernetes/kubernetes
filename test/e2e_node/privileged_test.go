/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO: This test was ported from test/e2e/privileged.go. We should
// re-evaluate the need of testing the feature in both suites.
const (
	privilegedPodName          = "privileged-pod"
	privilegedContainerName    = "privileged-container"
	privilegedHttpPort         = 8080
	privilegedUdpPort          = 8081
	notPrivilegedHttpPort      = 9090
	notPrivilegedUdpPort       = 9091
	notPrivilegedContainerName = "not-privileged-container"
	privilegedCommand          = "ip link add dummy1 type dummy"
)

type PrivilegedPodTestConfig struct {
	config        *restclient.Config
	client        *client.Client
	namespace     string
	hostExecPod   *api.Pod
	privilegedPod *api.Pod
}

var _ = Describe("PrivilegedPod", func() {
	var c *client.Client
	restClientConfig := &restclient.Config{Host: *apiServerAddress}
	BeforeEach(func() {
		// Setup the apiserver client
		c = client.NewOrDie(restClientConfig)
	})
	It("should test privileged pod", func() {
		namespace := "privileged-pods"
		config := &PrivilegedPodTestConfig{
			client:    c,
			config:    restClientConfig,
			namespace: namespace,
		}
		By("Creating a host exec pod")
		config.hostExecPod = createPodAndWaitUntilRunning(c, newHostExecPodSpec(config.namespace, "hostexec"))

		By("Creating a privileged pod")
		config.privilegedPod = createPodAndWaitUntilRunning(c, config.createPrivilegedPodSpec())

		By("Executing privileged command on privileged container")
		config.runPrivilegedCommandOnPrivilegedContainer()

		By("Executing privileged command on non-privileged container")
		config.runPrivilegedCommandOnNonPrivilegedContainer()
	})
})

func (config *PrivilegedPodTestConfig) createPrivilegedPodSpec() *api.Pod {
	isPrivileged := true
	notPrivileged := false
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      privilegedPodName,
			Namespace: config.namespace,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            privilegedContainerName,
					Image:           ImageRegistry[netExecImage],
					ImagePullPolicy: api.PullIfNotPresent,
					SecurityContext: &api.SecurityContext{Privileged: &isPrivileged},
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", privilegedHttpPort),
						fmt.Sprintf("--udp-port=%d", privilegedUdpPort),
					},
				},
				{
					Name:            notPrivilegedContainerName,
					Image:           ImageRegistry[netExecImage],
					ImagePullPolicy: api.PullIfNotPresent,
					SecurityContext: &api.SecurityContext{Privileged: &notPrivileged},
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", notPrivilegedHttpPort),
						fmt.Sprintf("--udp-port=%d", notPrivilegedUdpPort),
					},
				},
			},
		},
	}
	return pod
}

func (config *PrivilegedPodTestConfig) runPrivilegedCommandOnPrivilegedContainer() {
	outputMap := config.dialFromContainer(config.privilegedPod.Status.PodIP, privilegedHttpPort)
	Expect(len(outputMap["error"]) == 0).To(BeTrue(), fmt.Sprintf("Privileged command failed unexpectedly on privileged container, output: %v", outputMap))
}

func (config *PrivilegedPodTestConfig) runPrivilegedCommandOnNonPrivilegedContainer() {
	outputMap := config.dialFromContainer(config.privilegedPod.Status.PodIP, notPrivilegedHttpPort)
	Expect(len(outputMap["error"]) > 0).To(BeTrue(), fmt.Sprintf("Privileged command should have failed on non-privileged container, output: %v", outputMap))
}

func (config *PrivilegedPodTestConfig) dialFromContainer(containerIP string, containerHttpPort int) map[string]string {
	v := url.Values{}
	v.Set("shellCommand", "ip link add dummy1 type dummy")
	cmd := fmt.Sprintf("curl -q 'http://%s:%d/shell?%s'",
		containerIP,
		containerHttpPort,
		v.Encode())
	By(fmt.Sprintf("Exec-ing into container over http. Running command: %s", cmd))

	stdout, err := execCommandInContainer(config.config, config.client, config.hostExecPod.Namespace, config.hostExecPod.Name, config.hostExecPod.Spec.Containers[0].Name,
		[]string{"/bin/sh", "-c", cmd})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Error running command %q: %v", cmd, err))

	var output map[string]string
	err = json.Unmarshal([]byte(stdout), &output)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Could not unmarshal curl response: %s", stdout))
	return output
}

// newHostExecPodSpec returns the pod spec of hostexec pod
func newHostExecPodSpec(ns, name string) *api.Pod {
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "hostexec",
					Image:           ImageRegistry[hostExecImage],
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}
}

// TODO: This is a very basic helper function. Remove this once a common
// utility function has been ported from the the e2e package.
func waitForPodRunning(c *client.Client, ns string, name string) error {
	var pod *api.Pod
	var err error
	for t := time.Now(); time.Since(t) < time.Minute*2; time.Sleep(time.Second * 5) {
		pod, err = c.Pods(ns).Get(name)
		Expect(err).NotTo(HaveOccurred())
		if pod == nil {
			continue
		}
		if pod.Status.Phase != api.PodRunning {
			continue
		}
		return nil
	}
	return fmt.Errorf("Time out while waiting for pod %s/%s to become running; current status: %+v", ns, name, pod.Status)
}

func createPodAndWaitUntilRunning(c *client.Client, pod *api.Pod) *api.Pod {
	ref := fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)
	_, err := createPodWithSpec(c, pod)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create pod %q: %v", ref, err))
	err = waitForPodRunning(c, pod.Namespace, pod.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed waiting for pod %q to become running: %v", ref, err))
	runningPod, err := c.Pods(pod.Namespace).Get(pod.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to retrieve pod %q: %v", ref, err))
	return runningPod
}

func createPodWithSpec(c *client.Client, pod *api.Pod) (*api.Pod, error) {
	// Manually assign pod to node because we don't run the scheduler in node
	// e2e tests.
	assignPodToNode(pod)
	createdPod, err := c.Pods(pod.Namespace).Create(pod)
	return createdPod, err
}
