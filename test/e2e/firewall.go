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
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"net/http"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type testConfig struct {
	podNamespace string
	podName      string
	forwardingIP string
	publicIP     string
	testPort     int
	f            *Framework
}

var _ = Describe("GCEFirewall", func() {
	if !providerIs("gce") {
		return
	}

	f := NewFramework("gce-firewall")
	config := &testConfig{
		podNamespace: f.Namespace.Name,
		podName:      "firewall-test-" + string(util.NewUUID()),
		f:            f,
	}

	defer config.teardown()
	It("should provide GCE Firewall for the cluster", config.createFirewallPod)
	It("verify setup", config.verifySetup)
	It("read forwarding and public ips", config.readIPs)
	It("should verify public ip is not reachable", config.verifyPublicIpIsNotReachable)
	It("should verify forwarding ip is not reachable", config.verifyForwardingIpIsNotReachable)
	It("should enable forwarding ip via firewall", config.openFirewall)
	It("should verify public ip is not reachable", config.verifyPublicIpIsNotReachable)
	It("should verify forwarding ip is reachable", config.verifyForwardingIpIsReachable)
})

func (config *testConfig) getFirewallPodSpec() *api.Pod {
	pod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      config.podName,
			Namespace: config.podNamespace,
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "etcssl",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/etc/ssl"},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "firewalltest",
					Image: "artfulcoder/gce-firewall:1.1",
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							ContainerPort: config.testPort,
							HostPort:      config.testPort,
						},
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "etcssl",
							MountPath: "/etc/ssl",
						},
					},
				},
			},
		},
	}

	return pod
}

func (config *testConfig) createFirewallPod() {
	By("creating a pod to  Firewall Pod")
	podClient := config.f.Client.Pods(config.podNamespace)
	pod := config.getFirewallPodSpec()
	if _, err := podClient.Create(pod); err != nil {
		Failf("Failed to create %s pod: %v", pod.Name, err)
	}
	expectNoError(config.f.WaitForPodRunning(config.podName))
}

func (config *testConfig) teardown() {
	defer GinkgoRecover()
	By("calling the teardown endpoint on the test-server")
	config.httpPodGet("teardown")

	By("deleting the pod")
	defer GinkgoRecover()
	podClient := config.f.Client.Pods(config.podNamespace)
	podClient.Delete(config.podName, nil)
}

func (config *testConfig) httpPodGet(path string) string {
	var getError error
	var getBody []byte
	expectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
		getBody, getError = config.f.Client.Get().
			Prefix("proxy").
			Resource("pods").
			Namespace(config.podNamespace).
			Name(fmt.Sprintf("%s:%s", config.podName, config.testPort)).
			Suffix(path).
			Do().Raw()
		if getError != nil {
			return false, nil
		} else {
			return true, nil
		}
	}))
	Expect(getError != nil)
	return string(getBody)
}

func (config *testConfig) verifySetup() {
	By("invoking verify-setup on test server")
	body := config.httpPodGet("verify-setup")
	Expect(strings.Contains(body, "SUCCESS:"))
}

func (config *testConfig) readIPs() {
	By("reading forwarding ip")
	body := config.httpPodGet("forwarding-ip")
	Expect(len(body) > 0)
	config.forwardingIP = body

	By("reading public ip")
	body = config.httpPodGet("public-ip")
	Expect(len(body) > 0)
	config.publicIP = body
}

func (config *testConfig) openFirewall() {
	By("calling the open-firewall endpoint")
	body := config.httpPodGet("open-firewall")
	Expect(strings.Contains(body, "SUCCESS:"))
}

func verifyReachability(endpoint string, port int, isReachable bool) {
	_, err := http.Get(fmt.Sprintf("http://%s:%s/echo", endpoint, port))
	Expect((err != nil && !isReachable) || (err == nil && isReachable))
}

func (config *testConfig) verifyPublicIpIsNotReachable() {
	By(fmt.Sprintf("verifying that the public ip:%s is not reachable", config.publicIP))
	verifyReachability(config.publicIP, config.testPort, false)
}

func (config *testConfig) verifyForwardingIpIsNotReachable() {
	By(fmt.Sprintf("verifying that the forwarding ip:%s is not reachable", config.forwardingIP))
	verifyReachability(config.forwardingIP, config.testPort, false)
}

func (config *testConfig) verifyForwardingIpIsReachable() {
	By(fmt.Sprintf("verifying that the forwarding ip:%s is reachable", config.publicIP))
	verifyReachability(config.publicIP, config.testPort, true)
}
