/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	. "github.com/onsi/ginkgo"
	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"strings"
)

const (
	kubeletEtcHostsImageName          = "gcr.io/google_containers/netexec:1.0"
	kubeletEtcHostsPodName            = "test-pod"
	kubeletEtcHostsHostNetworkPodName = "test-host-network-pod"
	etcHostsPartialContent            = "# Kubernetes-managed hosts file."
)

type KubeletManagedHostConfig struct {
	hostNetworkPod *api.Pod
	pod            *api.Pod
	f              *Framework
}

var _ = Describe("KubeletManagedEtcHosts", func() {
	f := NewFramework("e2e-kubelet-etc-hosts")
	config := &KubeletManagedHostConfig{
		f: f,
	}

	It("should test kubelet managed /etc/hosts file", func() {
		By("Setting up the test")
		config.setup()

		By("Running the test")
		config.verifyEtcHosts()
	})
})

func (config *KubeletManagedHostConfig) verifyEtcHosts() {
	By("Verifying /etc/hosts of container is kubelet-managed for pod with hostNetwork=false")
	stdout := config.getEtcHostsContent(kubeletEtcHostsPodName, "busybox-1")
	assertEtcHostsIsKubeletManaged(stdout)
	stdout = config.getEtcHostsContent(kubeletEtcHostsPodName, "busybox-2")
	assertEtcHostsIsKubeletManaged(stdout)

	By("Verifying /etc/hosts of container is not kubelet-managed since container specifies /etc/hosts mount")
	stdout = config.getEtcHostsContent(kubeletEtcHostsPodName, "busybox-3")
	assertEtcHostsIsNotKubeletManaged(stdout)

	By("Verifying /etc/hosts content of container is not kubelet-managed for pod with hostNetwork=true")
	stdout = config.getEtcHostsContent(kubeletEtcHostsHostNetworkPodName, "busybox-1")
	assertEtcHostsIsNotKubeletManaged(stdout)
	stdout = config.getEtcHostsContent(kubeletEtcHostsHostNetworkPodName, "busybox-2")
	assertEtcHostsIsNotKubeletManaged(stdout)
}

func (config *KubeletManagedHostConfig) setup() {
	By("Creating hostNetwork=false pod")
	config.createPodWithoutHostNetwork()

	By("Creating hostNetwork=true pod")
	config.createPodWithHostNetwork()
}

func (config *KubeletManagedHostConfig) createPodWithoutHostNetwork() {
	podSpec := config.createPodSpec(kubeletEtcHostsPodName)
	config.pod = config.createPod(podSpec)
}

func (config *KubeletManagedHostConfig) createPodWithHostNetwork() {
	podSpec := config.createPodSpecWithHostNetwork(kubeletEtcHostsHostNetworkPodName)
	config.hostNetworkPod = config.createPod(podSpec)
}

func (config *KubeletManagedHostConfig) createPod(podSpec *api.Pod) *api.Pod {
	createdPod, err := config.getPodClient().Create(podSpec)
	if err != nil {
		Failf("Failed to create %s pod: %v", podSpec.Name, err)
	}
	expectNoError(config.f.WaitForPodRunning(podSpec.Name))
	createdPod, err = config.getPodClient().Get(podSpec.Name)
	if err != nil {
		Failf("Failed to retrieve %s pod: %v", podSpec.Name, err)
	}
	return createdPod
}

func (config *KubeletManagedHostConfig) getPodClient() client.PodInterface {
	return config.f.Client.Pods(config.f.Namespace.Name)
}

func assertEtcHostsIsKubeletManaged(etcHostsContent string) {
	isKubeletManaged := strings.Contains(etcHostsContent, etcHostsPartialContent)
	if !isKubeletManaged {
		Failf("/etc/hosts file should be kubelet managed, but is not: %q", etcHostsContent)
	}
}

func assertEtcHostsIsNotKubeletManaged(etcHostsContent string) {
	isKubeletManaged := strings.Contains(etcHostsContent, etcHostsPartialContent)
	if isKubeletManaged {
		Failf("/etc/hosts file should not be kubelet managed, but is: %q", etcHostsContent)
	}
}

func (config *KubeletManagedHostConfig) getEtcHostsContent(podName, containerName string) string {
	cmd := kubectlCmd("exec", fmt.Sprintf("--namespace=%v", config.f.Namespace.Name), podName, "-c", containerName, "cat", "/etc/hosts")
	stdout, stderr, err := startCmdAndStreamOutput(cmd)
	if err != nil {
		Failf("Failed to retrieve /etc/hosts, err: %q", err)
	}
	defer stdout.Close()
	defer stderr.Close()

	buf := make([]byte, 1000)
	var n int
	Logf("reading from `kubectl exec` command's stdout")
	if n, err = stdout.Read(buf); err != nil {
		Failf("Failed to read from kubectl exec stdout: %v", err)
	}
	return string(buf[:n])
}

func (config *KubeletManagedHostConfig) createPodSpec(podName string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      podName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "busybox-1",
					Image:           kubeletEtcHostsImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
				},
				{
					Name:            "busybox-2",
					Image:           kubeletEtcHostsImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
				},
				{
					Name:            "busybox-3",
					Image:           kubeletEtcHostsImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: "/etc/hosts",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "host-etc-hosts",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/etc/hosts",
						},
					},
				},
			},
		},
	}
	return pod
}

func (config *KubeletManagedHostConfig) createPodSpecWithHostNetwork(podName string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      podName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
			Containers: []api.Container{
				{
					Name:            "busybox-1",
					Image:           kubeletEtcHostsImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
				},
				{
					Name:            "busybox-2",
					Image:           kubeletEtcHostsImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
				},
			},
		},
	}
	return pod
}
