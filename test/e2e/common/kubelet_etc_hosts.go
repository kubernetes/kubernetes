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
	"time"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	etcHostsPodName            = "test-pod"
	etcHostsHostNetworkPodName = "test-host-network-pod"
	etcHostsPartialContent     = "# Kubernetes-managed hosts file."
	etcHostsPath               = "/etc/hosts"
	etcHostsOriginalPath       = "/etc/hosts-original"
)

var etcHostsImageName = imageutils.GetE2EImage(imageutils.Netexec)

type KubeletManagedHostConfig struct {
	hostNetworkPod *v1.Pod
	pod            *v1.Pod
	f              *framework.Framework
}

var _ = framework.KubeDescribe("KubeletManagedEtcHosts", func() {
	f := framework.NewDefaultFramework("e2e-kubelet-etc-hosts")
	config := &KubeletManagedHostConfig{
		f: f,
	}

	/*
		Release : v1.9
		Testname: Kubelet, managed etc hosts
		Description: Create a Pod with containers with hostNetwork set to false, one of the containers mounts the /etc/hosts file form the host. Create a second Pod with hostNetwork set to true.
			1. The Pod with hostNetwork=false MUST have /etc/hosts of containers managed by the Kubelet.
			2. The Pod with hostNetwork=false but the container mounts /etc/hosts file from the host. The /etc/hosts file MUST not be managed by the Kubelet.
			3. The Pod with hostNetwork=true , /etc/hosts file MUST not be managed by the Kubelet.
	*/
	framework.ConformanceIt("should test kubelet managed /etc/hosts file [NodeConformance]", func() {
		By("Setting up the test")
		config.setup()

		By("Running the test")
		config.verifyEtcHosts()
	})
})

func (config *KubeletManagedHostConfig) verifyEtcHosts() {
	By("Verifying /etc/hosts of container is kubelet-managed for pod with hostNetwork=false")
	assertManagedStatus(config, etcHostsPodName, true, "busybox-1")
	assertManagedStatus(config, etcHostsPodName, true, "busybox-2")

	By("Verifying /etc/hosts of container is not kubelet-managed since container specifies /etc/hosts mount")
	assertManagedStatus(config, etcHostsPodName, false, "busybox-3")

	By("Verifying /etc/hosts content of container is not kubelet-managed for pod with hostNetwork=true")
	assertManagedStatus(config, etcHostsHostNetworkPodName, false, "busybox-1")
	assertManagedStatus(config, etcHostsHostNetworkPodName, false, "busybox-2")
}

func (config *KubeletManagedHostConfig) setup() {
	By("Creating hostNetwork=false pod")
	config.createPodWithoutHostNetwork()

	By("Creating hostNetwork=true pod")
	config.createPodWithHostNetwork()
}

func (config *KubeletManagedHostConfig) createPodWithoutHostNetwork() {
	podSpec := config.createPodSpec(etcHostsPodName)
	config.pod = config.f.PodClient().CreateSync(podSpec)
}

func (config *KubeletManagedHostConfig) createPodWithHostNetwork() {
	podSpec := config.createPodSpecWithHostNetwork(etcHostsHostNetworkPodName)
	config.hostNetworkPod = config.f.PodClient().CreateSync(podSpec)
}

func assertManagedStatus(
	config *KubeletManagedHostConfig, podName string, expectedIsManaged bool, name string) {
	// TODO: workaround for https://github.com/kubernetes/kubernetes/issues/34256
	//
	// Retry until timeout for the contents of /etc/hosts to show
	// up. Note: if /etc/hosts is properly mounted, then this will
	// succeed immediately.
	const retryTimeout = 30 * time.Second

	retryCount := 0
	etcHostsContent := ""

	for startTime := time.Now(); time.Since(startTime) < retryTimeout; {
		etcHostsContent = config.getFileContents(podName, name, etcHostsPath)
		etcHostsOriginalContent := config.getFileContents(podName, name, etcHostsOriginalPath)

		// Make sure there is some content in both files
		if len(etcHostsContent) > 0 && len(etcHostsOriginalContent) > 0 {
			// if the files match, kubernetes did not touch the file at all
			// if the file has the header, kubernetes is not using host network
			// and is constructing the file based on Pod IP
			isManaged := strings.HasPrefix(etcHostsContent, etcHostsPartialContent) &&
				etcHostsContent != etcHostsOriginalContent
			if expectedIsManaged == isManaged {
				return
			}
		}

		glog.Warningf(
			"For pod: %s, name: %s, expected %t, (/etc/hosts was %q), (/etc/hosts-original was %q), retryCount: %d",
			podName, name, expectedIsManaged, etcHostsContent, etcHostsOriginalContent, retryCount)

		retryCount++
		time.Sleep(100 * time.Millisecond)
	}

	if expectedIsManaged {
		framework.Failf(
			"/etc/hosts file should be kubelet managed (name: %s, retries: %d). /etc/hosts contains %q",
			name, retryCount, etcHostsContent)
	} else {
		framework.Failf(
			"/etc/hosts file should no be kubelet managed (name: %s, retries: %d). /etc/hosts contains %q",
			name, retryCount, etcHostsContent)
	}
}

func (config *KubeletManagedHostConfig) getFileContents(podName, containerName, path string) string {
	return config.f.ExecCommandInContainer(podName, containerName, "cat", path)
}

func (config *KubeletManagedHostConfig) createPodSpec(podName string) *v1.Pod {
	hostPathType := new(v1.HostPathType)
	*hostPathType = v1.HostPathType(string(v1.HostPathFileOrCreate))
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "busybox-1",
					Image:           etcHostsImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsOriginalPath,
						},
					},
				},
				{
					Name:            "busybox-2",
					Image:           etcHostsImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsOriginalPath,
						},
					},
				},
				{
					Name:            "busybox-3",
					Image:           etcHostsImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsPath,
						},
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsOriginalPath,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "host-etc-hosts",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: etcHostsPath,
							Type: hostPathType,
						},
					},
				},
			},
		},
	}
	return pod
}

func (config *KubeletManagedHostConfig) createPodSpecWithHostNetwork(podName string) *v1.Pod {
	hostPathType := new(v1.HostPathType)
	*hostPathType = v1.HostPathType(string(v1.HostPathFileOrCreate))
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			HostNetwork:     true,
			SecurityContext: &v1.PodSecurityContext{},
			Containers: []v1.Container{
				{
					Name:            "busybox-1",
					Image:           etcHostsImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsOriginalPath,
						},
					},
				},
				{
					Name:            "busybox-2",
					Image:           etcHostsImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"900",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host-etc-hosts",
							MountPath: etcHostsOriginalPath,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "host-etc-hosts",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: etcHostsPath,
							Type: hostPathType,
						},
					},
				},
			},
		},
	}
	return pod
}
