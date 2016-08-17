/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"strings"

	. "github.com/onsi/ginkgo"
	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	kubeletEtcHostsImageName          = "gcr.io/google_containers/netexec:1.4"
	kubeletEtcHostsPodName            = "test-pod"
	kubeletEtcHostsHostNetworkPodName = "test-host-network-pod"
	etcHostsPartialContent            = "# Kubernetes-managed hosts file."
)

type KubeletManagedHostConfig struct {
	hostNetworkPod *api.Pod
	pod            *api.Pod
	f              *framework.Framework
}

var _ = framework.KubeDescribe("KubeletManagedEtcHosts", func() {
	f := framework.NewDefaultFramework("e2e-kubelet-etc-hosts")
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
	config.pod = config.f.PodClient().CreateSync(podSpec)
}

func (config *KubeletManagedHostConfig) createPodWithHostNetwork() {
	podSpec := config.createPodSpecWithHostNetwork(kubeletEtcHostsHostNetworkPodName)
	config.hostNetworkPod = config.f.PodClient().CreateSync(podSpec)
}

func assertEtcHostsIsKubeletManaged(etcHostsContent string) {
	isKubeletManaged := strings.Contains(etcHostsContent, etcHostsPartialContent)
	if !isKubeletManaged {
		framework.Failf("/etc/hosts file should be kubelet managed, but is not: %q", etcHostsContent)
	}
}

func assertEtcHostsIsNotKubeletManaged(etcHostsContent string) {
	isKubeletManaged := strings.Contains(etcHostsContent, etcHostsPartialContent)
	if isKubeletManaged {
		framework.Failf("/etc/hosts file should not be kubelet managed, but is: %q", etcHostsContent)
	}
}

func (config *KubeletManagedHostConfig) getEtcHostsContent(podName, containerName string) string {
	return config.f.ExecCommandInContainer(podName, containerName, "cat", "/etc/hosts")
}

func (config *KubeletManagedHostConfig) createPodSpec(podName string) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
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
		ObjectMeta: api.ObjectMeta{
			Name: podName,
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
