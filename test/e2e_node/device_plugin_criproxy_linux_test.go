//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
)

// Regression test for #118559: simulate a kubelet restart where the initial CRI
// snapshot (ListPodSandbox + ListContainers) returns an error.
// buildContainerMapFromRuntime silently discards these errors. Without the
// checkpoint-trust bypass, the scenario-2 path in devicesToAllocate falls through
// to the healthyDevices check and rejects the still-running pod with
// UnexpectedAdmissionError.
//
// This file carries //go:build linux because the CRI proxy (e2eCriProxy,
// addCRIProxyInjector, resetCRIProxyInjector) is defined in linux-tagged files.
// The setup/teardown it shares with device_plugin_test.go is in platform-agnostic
// helpers there.
var _ = SIGDescribe("Device Plugin CRI Snapshot Failure", framework.WithSerial(), feature.DevicePlugin, feature.CriProxy, func() {
	f := framework.NewDefaultFramework("device-plugin-cri-snapshot")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("DevicePlugin", f.WithSerial(), f.WithDisruptive(), func() {
		var devicePluginPod *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			if e2eCriProxy == nil {
				ginkgo.Skip("test requires CRI proxy (run with --cri-proxy-enabled=true)")
			}
			ginkgo.DeferCleanup(func() error { return resetCRIProxyInjector(e2eCriProxy) })
			devicePluginPod = setupSampleDevicePluginWithRegistrationControl(ctx, f)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if devicePluginPod == nil {
				return
			}
			teardownSampleDevicePluginWithRegistrationControl(ctx, f, devicePluginPod)
		})

		framework.It("Keeps device plugin assignments across kubelet restart when CRI snapshot returns an error", func(ctx context.Context) {

			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			// Remove only the device plugin sandbox so the plugin will not re-register quickly.
			// The app container stays running in containerd (this is the production scenario
			// for #118559): kubelet restarts, containerd stays up, workload containers
			// keep running, device plugin needs to re-register.
			ginkgo.By("removing only the device plugin sandbox via CRI")
			rs, _, err := getCRIClient(ctx)
			framework.ExpectNoError(err)
			sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
			framework.ExpectNoError(err)
			removed := false
			for _, sandbox := range sandboxes {
				gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
				if sandbox.Metadata.Name != devicePluginPod.Name {
					continue
				}
				ginkgo.By(fmt.Sprintf("deleting device plugin sandbox via CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))
				framework.ExpectNoError(rs.RemovePodSandbox(ctx, sandbox.Id))
				removed = true
			}
			gomega.Expect(removed).To(gomega.BeTrueBecause("expected to find and remove the device plugin sandbox"))

			// Arm the CRI proxy to fail ListPodSandbox and ListContainers during early kubelet
			// startup. The containerManager snapshot happens within the first ~300ms, but GC
			// and stats-init also call these APIs before and after. 2s is a generous envelope —
			// wide enough to cover the snapshot on slow CI nodes, short enough that PLEG
			// (running at ~1s intervals) recovers before the 5-minute node-ready timeout.
			ginkgo.By("arming CRI proxy to fail ListPodSandbox/ListContainers during the startup snapshot window")
			var listPodSandboxCalls, listContainersCalls atomic.Int32
			armedAt := time.Now()
			const injectionWindow = 2 * time.Second
			err = addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if time.Since(armedAt) > injectionWindow {
					return nil
				}
				switch apiName {
				case criproxy.ListPodSandbox:
					listPodSandboxCalls.Add(1)
					return fmt.Errorf("injected failure: simulating transient CRI error during containerManager snapshot")
				case criproxy.ListContainers:
					listContainersCalls.Add(1)
					return fmt.Errorf("injected failure: simulating transient CRI error during containerManager snapshot")
				}
				return nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("restarting the kubelet")
			restartKubelet(ctx)

			ginkgo.By("waiting for node to be ready again")
			framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute))

			framework.Logf("CRI injector fired %d ListPodSandbox and %d ListContainers failures during the %v injection window", listPodSandboxCalls.Load(), listContainersCalls.Load(), injectionWindow)

			// Sanity check: the injector actually fired during the startup window.
			gomega.Expect(listPodSandboxCalls.Load()).To(gomega.BeNumerically(">=", int32(1)), "CRI proxy ListPodSandbox injector never fired")
			gomega.Expect(listContainersCalls.Load()).To(gomega.BeNumerically(">=", int32(1)), "CRI proxy ListContainers injector never fired")

			// The app container was running throughout. The checkpoint has its allocation.
			// An incomplete CRI snapshot must not cause re-admission to reject it.
			ginkgo.By("verifying the pod is still running with its original device assignment")
			gomega.Consistently(ctx, getPodByName).
				WithArguments(f, pod1.Name).
				WithTimeout(30*time.Second).WithPolling(2*time.Second).
				Should(BeTheSamePodStillRunning(pod1),
					"pod was evicted after kubelet restart despite being a running container with a valid checkpoint allocation — CRI snapshot failure must not cause re-admission to reject running pods (issue #118559)")

			ginkgo.By("verifying the device assignment is preserved via the podresources API")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				v1PodResources, err := getV1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node devices: %w", err)
				}
				matchErr, _ := checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, e2enode.SampleDeviceResourceName, []string{devID1})
				return matchErr
			}, time.Minute, framework.Poll).Should(gomega.Succeed())
		})
	})
})
