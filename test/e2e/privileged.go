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
	"encoding/json"
	"fmt"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"net/url"
)

const (
	privilegedPodName          = "privileged-pod"
	privilegedContainerName    = "privileged-container"
	privilegedHttpPort         = 8080
	privilegedUdpPort          = 8081
	notPrivilegedHttpPort      = 9090
	notPrivilegedUdpPort       = 9091
	notPrivilegedContainerName = "not-privileged-container"
	privilegedContainerImage   = "gcr.io/google_containers/netexec:1.1"
	privilegedCommand          = "ip link add dummy1 type dummy"
)

type PrivilegedPodTestConfig struct {
	privilegedPod *api.Pod
	f             *Framework
	nodes         []string
}

var _ = Describe("PrivilegedPod", func() {
	f := NewFramework("e2e-privilegedpod")
	config := &PrivilegedPodTestConfig{
		f: f,
	}
	It("should test privileged pod", func() {
		SkipUnlessProviderIs(providersWithSSH...)

		By("Getting ssh-able hosts")
		hosts, err := NodeSSHHosts(config.f.Client)
		Expect(err).NotTo(HaveOccurred())
		if len(hosts) == 0 {
			Failf("No ssh-able nodes")
		}
		config.nodes = hosts

		By("Creating a privileged pod")
		config.createPrivilegedPod()

		By("Executing privileged command on privileged container")
		config.runPrivilegedCommandOnPrivilegedContainer()

		By("Executing privileged command on non-privileged container")
		config.runPrivilegedCommandOnNonPrivilegedContainer()
	})
})

func (config *PrivilegedPodTestConfig) runPrivilegedCommandOnPrivilegedContainer() {
	outputMap := config.dialFromContainer(config.privilegedPod.Status.PodIP, privilegedHttpPort)
	if len(outputMap["error"]) > 0 {
		Failf("Privileged command failed unexpectedly on privileged container, output:%v", outputMap)
	}
}

func (config *PrivilegedPodTestConfig) runPrivilegedCommandOnNonPrivilegedContainer() {
	outputMap := config.dialFromContainer(config.privilegedPod.Status.PodIP, notPrivilegedHttpPort)
	if len(outputMap["error"]) == 0 {
		Failf("Privileged command should have failed on non-privileged container, output:%v", outputMap)
	}
}

func (config *PrivilegedPodTestConfig) dialFromContainer(containerIP string, containerHttpPort int) map[string]string {
	v := url.Values{}
	v.Set("shellCommand", "ip link add dummy1 type dummy")
	cmd := fmt.Sprintf("curl -q 'http://%s:%d/shell?%s'",
		containerIP,
		containerHttpPort,
		v.Encode())

	By(fmt.Sprintf("Exec-ing into container over http. Running command:%s", cmd))
	stdout := config.ssh(cmd)
	Logf("Output is %q", stdout)
	var output map[string]string
	err := json.Unmarshal([]byte(stdout), &output)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Could not unmarshal curl response: %s", stdout))
	Logf("Deserialized output is %v", stdout)
	return output
}

func (config *PrivilegedPodTestConfig) createPrivilegedPodSpec() *api.Pod {
	isPrivileged := true
	notPrivileged := false
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      privilegedPodName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            privilegedContainerName,
					Image:           privilegedContainerImage,
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
					Image:           privilegedContainerImage,
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

func (config *PrivilegedPodTestConfig) createPrivilegedPod() {
	podSpec := config.createPrivilegedPodSpec()
	config.privilegedPod = config.createPod(podSpec)
}

func (config *PrivilegedPodTestConfig) createPod(pod *api.Pod) *api.Pod {
	createdPod, err := config.getPodClient().Create(pod)
	if err != nil {
		Failf("Failed to create %q pod: %v", pod.Name, err)
	}
	expectNoError(config.f.WaitForPodRunning(pod.Name))
	createdPod, err = config.getPodClient().Get(pod.Name)
	if err != nil {
		Failf("Failed to retrieve %q pod: %v", pod.Name, err)
	}
	return createdPod
}

func (config *PrivilegedPodTestConfig) getPodClient() client.PodInterface {
	return config.f.Client.Pods(config.f.Namespace.Name)
}

func (config *PrivilegedPodTestConfig) getNamespaceClient() client.NamespaceInterface {
	return config.f.Client.Namespaces()
}

func (config *PrivilegedPodTestConfig) ssh(cmd string) string {
	stdout, _, code, err := SSH(cmd, config.nodes[0], testContext.Provider)
	Expect(err).NotTo(HaveOccurred(), "error while SSH-ing to node: %v (code %v)", err, code)
	Expect(code).Should(BeZero(), "command exited with non-zero code %v. cmd:%s", code, cmd)
	return stdout
}
