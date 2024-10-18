//go:build linux
// +build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
)

var _ = SIGDescribe("SystemdWatchdog", feature.CriProxy, framework.WithSlow(), framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("systemd-watchdog")

	ginkgo.Context("Inject a list pod sandbox error exception into the CriProxy", func() {
		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
		})

		ginkgo.AfterEach(func() {
			err := resetCRIProxyInjector()
			framework.ExpectNoError(err)
		})

		ginkgo.It("should restart kubelet when watchdog notification timeout.", func(ctx context.Context) {
			ginkgo.By("Get the initial PID of the kubelet")
			initialPid := getKubeletMainPID()

			ginkgo.By("Injecting an error into CRI will cause PLEG to become unhealthy.")
			expectedErr := fmt.Errorf("ListPodSandbox failed")
			err := addCRIProxyInjector(func(apiName string) error {
				if apiName == criproxy.ListPodSandbox {
					return expectedErr
				}
				return nil
			})
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func() bool {
				currentPid := getKubeletMainPID()
				return initialPid != currentPid
			}, 10*time.Minute, 10*time.Second).Should(gomega.BeTrueBecause("Kubelet should restarted due to watchdog timeout."))

			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalseBecause("Kubelet should fail to start."))

			ginkgo.By("CRI returns to normal.")
			err = resetCRIProxyInjector()
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("Kubelet should be started after the CRI returns to normal."))
		})
	})
})

func getKubeletMainPID() string {
	kubeletServiceName := findKubeletServiceName(true)
	stdout, err := exec.Command("sudo", "systemctl", "show", kubeletServiceName, "-p", "MainPID").CombinedOutput()
	framework.ExpectNoError(err)
	return string(stdout)
}
