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
	"encoding/json"
	"fmt"
	"net/url"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	privilegedPodName          = "privileged-pod"
	privilegedContainerName    = "privileged-container"
	privilegedHttpPort         = 8080
	privilegedUdpPort          = 8081
	notPrivilegedHttpPort      = 9090
	notPrivilegedUdpPort       = 9091
	notPrivilegedContainerName = "not-privileged-container"
	privilegedContainerImage   = "gcr.io/google_containers/netexec:1.4"
	privilegedCommand          = "ip link add dummy1 type dummy"
)

type PrivilegedPodTestConfig struct {
	privilegedPod *api.Pod
	f             *framework.Framework
	hostExecPod   *api.Pod
}

var _ = framework.KubeDescribe("PrivilegedPod", func() {
	f := framework.NewDefaultFramework("e2e-privilegedpod")
	config := &PrivilegedPodTestConfig{
		f: f,
	}
	It("should test privileged pod", func() {
		By("Creating a hostexec pod")
		config.createHostExecPod()

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
		framework.Failf("Privileged command failed unexpectedly on privileged container, output:%v", outputMap)
	}
}

func (config *PrivilegedPodTestConfig) runPrivilegedCommandOnNonPrivilegedContainer() {
	outputMap := config.dialFromContainer(config.privilegedPod.Status.PodIP, notPrivilegedHttpPort)
	if len(outputMap["error"]) == 0 {
		framework.Failf("Privileged command should have failed on non-privileged container, output:%v", outputMap)
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
	stdout := config.f.ExecShellInPod(config.hostExecPod.Name, cmd)
	var output map[string]string
	err := json.Unmarshal([]byte(stdout), &output)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Could not unmarshal curl response: %s", stdout))
	framework.Logf("Deserialized output is %v", stdout)
	return output
}

func (config *PrivilegedPodTestConfig) createPrivilegedPodSpec() *api.Pod {
	isPrivileged := true
	notPrivileged := false
	pod := &api.Pod{
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

func (config *PrivilegedPodTestConfig) createHostExecPod() {
	podSpec := framework.NewHostExecPodSpec(config.f.Namespace.Name, "hostexec")
	config.hostExecPod = config.f.PodClient().CreateSync(podSpec)
}

func (config *PrivilegedPodTestConfig) createPrivilegedPod() {
	podSpec := config.createPrivilegedPodSpec()
	config.privilegedPod = config.f.PodClient().CreateSync(podSpec)
}
