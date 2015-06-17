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
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Skipped", func() {
	Describe("Cluster upgrade", func() {

		framework := NewFramework("cluster-upgrade")

		Describe("kube-push", func() {
			It("of master should maintain responsive services", func() {
				testClusterUpgrade(framework, func() {
					runUpgradeScript("hack/e2e-internal/e2e-push.sh", "-m")
				})
			})
		})

		Describe("gce-upgrade", func() {
			It("of master should maintain responsive services", func() {
				if !providerIs("gce") {
					By(fmt.Sprintf("Skippingt test, which is not implemented for %s", testContext.Provider))
					return
				}
				testClusterUpgrade(framework, func() {
					runUpgradeScript("hack/e2e-internal/e2e-upgrade.sh", "-M", "-l")
				})
			})
		})
	})
})

func testClusterUpgrade(framework *Framework, upgrade func()) {
	By("Setting up test fixtures")
	ip, svcName, rcName, podName := setupUpgradeFixtures(framework)
	validateClusterUpgrade(framework, svcName, rcName, podName)

	Logf("Starting async validation")
	httpClient := http.Client{Timeout: 2 * time.Second}
	done := make(chan struct{}, 1)
	// Let's make sure we've finished the heartbeat before shutting things down.
	var wg sync.WaitGroup
	go util.Until(func() {
		defer GinkgoRecover()
		wg.Add(1)
		defer wg.Done()

		expectNoError(wait.Poll(poll, singleCallTimeout, func() (bool, error) {
			r, err := httpClient.Get("http://" + ip)
			if err != nil {
				return false, nil
			}
			if r.StatusCode < http.StatusOK || r.StatusCode >= http.StatusNotFound {
				return false, nil
			}
			return true, nil
		}))
	}, 200*time.Millisecond, done)

	By("Starting upgrade")
	upgrade()
	done <- struct{}{}
	Logf("Stopping async validation")
	wg.Wait()
	Logf("Upgrade complete.")

	By("Validating post upgrade state")
	validateClusterUpgrade(framework, svcName, rcName, podName)
}

func runUpgradeScript(scriptPath string, args ...string) {
	cmd := exec.Command(path.Join(testContext.RepoRoot, scriptPath), args...)
	upgradeLogPath := path.Join(testContext.OutputDir, "upgrade-"+string(util.NewUUID())+".log")
	Logf("Writing upgrade logs to %s", upgradeLogPath)
	upgradeLog, err := os.Create(upgradeLogPath)
	expectNoError(err)

	cmd.Stdout = io.MultiWriter(os.Stdout, upgradeLog)
	cmd.Stderr = io.MultiWriter(os.Stderr, upgradeLog)
	if err := cmd.Run(); err != nil {
		Failf("Upgrade failed: %v", err)
	}
}

func validateClusterUpgrade(framework *Framework, svcName, rcName, podName string) {
	rc, err := framework.Client.ReplicationControllers(framework.Namespace.Name).Get(rcName)
	Expect(err).NotTo(HaveOccurred())

	pods, err := framework.Client.Pods(framework.Namespace.Name).List(labels.Set(rc.Spec.Selector).AsSelector(), fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	Expect(len(pods.Items) == 1).Should(BeTrue())
	if pods.Items[0].Name != podName {
		Failf("pod name should not have changed: changed to '%s' from '%s'", pods.Items[0].Name, podName)
	}

	isRunning, err := podRunningReady(&pods.Items[0])
	Expect(err).NotTo(HaveOccurred())
	Expect(isRunning).To(BeTrue())

	svc, err := framework.Client.Services(framework.Namespace.Name).Get(svcName)
	Expect(err).NotTo(HaveOccurred())
	if svcName != svc.Name {
		Failf("Service name should not have changed")
	}
}

func setupUpgradeFixtures(framework *Framework) (ip, svcName, rcName, podName string) {
	svcName = "baz"
	webserver := NewWebserverTest(framework.Client, framework.Namespace.Name, svcName)

	svc := webserver.BuildServiceSpec()
	svcName = svc.Name
	svc.Spec.Type = api.ServiceTypeLoadBalancer
	webserver.CreateService(svc)

	rc := webserver.BuildReplicationController()
	rc, err := framework.Client.ReplicationControllers(framework.Namespace.Name).Create(rc)
	rcName = rc.Name
	Expect(err).NotTo(HaveOccurred())

	result, err := waitForLoadBalancerIngress(framework.Client, svcName, framework.Namespace.Name)
	Expect(err).NotTo(HaveOccurred())
	ingresses := result.Status.LoadBalancer.Ingress

	if len(ingresses) != 1 {
		Failf("Was expecting only 1 ingress IP but got %d (%v): %v", len(ingresses), ingresses, result)
	}
	ingress := ingresses[0]

	testLoadBalancerReachable(ingress, 80)

	pods, err := framework.Client.Pods(framework.Namespace.Name).List(labels.Set(rc.Spec.Selector).AsSelector(), fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	Expect(len(pods.Items) == 1).Should(BeTrue())
	podName = pods.Items[0].Name

	ip = ingress.IP
	if ip == "" {
		ip = ingress.Hostname
	}
	return
}
