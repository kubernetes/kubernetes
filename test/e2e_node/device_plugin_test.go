/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubeletdevicepluginv1beta1 "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
)

var (
	appsScheme = runtime.NewScheme()
	appsCodecs = serializer.NewCodecFactory(appsScheme)
)

// Serial because the test restarts Kubelet
var _ = SIGDescribe("Device Plugin [Feature:DevicePluginProbe][NodeFeature:DevicePluginProbe][Serial]", func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	testDevicePlugin(f, kubeletdevicepluginv1beta1.DevicePluginPath)
	testDevicePluginNodeReboot(f, kubeletdevicepluginv1beta1.DevicePluginPath)
})

// readDaemonSetV1OrDie reads daemonset object from bytes. Panics on error.
func readDaemonSetV1OrDie(objBytes []byte) *appsv1.DaemonSet {
	appsv1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(appsv1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*appsv1.DaemonSet)
}

const (
	// TODO(vikasc): Instead of hard-coding number of devices, provide number of devices in the sample-device-plugin using configmap
	// and then use the same here
	expectedSampleDevsAmount int64 = 2

	// This is the sleep interval specified in the command executed in the pod to ensure container is running "forever" in the test timescale
	sleepIntervalForever string = "24h"

	// This is the sleep interval specified in the command executed in the pod so that container is restarted within the expected test run time
	sleepIntervalWithRestart string = "60s"
)

func testDevicePlugin(f *framework.Framework, pluginSockDir string) {
	pluginSockDir = filepath.Join(pluginSockDir) + "/"
	ginkgo.Context("DevicePlugin [Serial] [Disruptive]", func() {
		var devicePluginPod, dptemplate *v1.Pod
		var v1alphaPodResources *kubeletpodresourcesv1alpha1.ListPodResourcesResponse
		var v1PodResources *kubeletpodresourcesv1.ListPodResourcesResponse
		var err error

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrue())

			// Before we run the device plugin test, we need to ensure
			// that the cluster is in a clean state and there are no
			// pods running on this node.
			// This is done in a gomega.Eventually with retries since a prior test in a different test suite could've run and the deletion of it's resources may still be in progress.
			// xref: https://issue.k8s.io/115381
			gomega.Eventually(ctx, func(ctx context.Context) error {
				v1alphaPodResources, err = getV1alpha1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1alpha) podresources API endpoint: %v", err)
				}

				v1PodResources, err = getV1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1) podresources API endpoint: %v", err)
				}

				if len(v1alphaPodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1alpha pod resources to be empty, but got non-empty resources: %+v", v1alphaPodResources.PodResources)
				}

				if len(v1PodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1 pod resources to be empty, but got non-empty resources: %+v", v1PodResources.PodResources)
				}
				return nil
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.Succeed())

			ginkgo.By("Scheduling a sample device plugin pod")
			dp := getSampleDevicePluginPod(pluginSockDir)
			dptemplate = dp.DeepCopy()
			devicePluginPod = e2epod.NewPodClient(f).CreateSync(ctx, dp)

			ginkgo.By("Waiting for devices to become available on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && CountSampleDeviceCapacity(node) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			framework.Logf("Successfully created device plugin pod")

			ginkgo.By(fmt.Sprintf("Waiting for the resource exported by the sample device plugin to become available on the local node (instances: %d)", expectedSampleDevsAmount))
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					CountSampleDeviceCapacity(node) == expectedSampleDevsAmount &&
					CountSampleDeviceAllocatable(node) == expectedSampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Deleting the device plugin pod")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, time.Minute)

			ginkgo.By("Deleting any Pods created by the test")
			l, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				framework.Logf("Deleting pod: %s", p.Name)
				e2epod.NewPodClient(f).DeleteSync(ctx, p.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}

			restartKubelet(true)

			ginkgo.By("Waiting for devices to become unavailable on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && CountSampleDeviceCapacity(node) <= 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("devices now unavailable on the local node")
		})

		ginkgo.It("Can schedule a pod that requires a device", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalWithRestart)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			v1alphaPodResources, err = getV1alpha1NodeDevices(ctx)
			framework.ExpectNoError(err)

			v1PodResources, err = getV1NodeDevices(ctx)
			framework.ExpectNoError(err)

			framework.Logf("v1alphaPodResources.PodResources:%+v\n", v1alphaPodResources.PodResources)
			framework.Logf("v1PodResources.PodResources:%+v\n", v1PodResources.PodResources)
			framework.Logf("len(v1alphaPodResources.PodResources):%+v", len(v1alphaPodResources.PodResources))
			framework.Logf("len(v1PodResources.PodResources):%+v", len(v1PodResources.PodResources))

			framework.ExpectEqual(len(v1alphaPodResources.PodResources), 2)
			framework.ExpectEqual(len(v1PodResources.PodResources), 2)

			var v1alphaResourcesForOurPod *kubeletpodresourcesv1alpha1.PodResources
			for _, res := range v1alphaPodResources.GetPodResources() {
				if res.Name == pod1.Name {
					v1alphaResourcesForOurPod = res
				}
			}

			var v1ResourcesForOurPod *kubeletpodresourcesv1.PodResources
			for _, res := range v1PodResources.GetPodResources() {
				if res.Name == pod1.Name {
					v1ResourcesForOurPod = res
				}
			}

			gomega.Expect(v1alphaResourcesForOurPod).NotTo(gomega.BeNil())
			gomega.Expect(v1ResourcesForOurPod).NotTo(gomega.BeNil())

			framework.ExpectEqual(v1alphaResourcesForOurPod.Name, pod1.Name)
			framework.ExpectEqual(v1ResourcesForOurPod.Name, pod1.Name)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Namespace, pod1.Namespace)
			framework.ExpectEqual(v1ResourcesForOurPod.Namespace, pod1.Namespace)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers), 1)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Containers[0].Name, pod1.Spec.Containers[0].Name)
			framework.ExpectEqual(v1ResourcesForOurPod.Containers[0].Name, pod1.Spec.Containers[0].Name)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers[0].Devices), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers[0].Devices), 1)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Containers[0].Devices[0].ResourceName, SampleDeviceResourceName)
			framework.ExpectEqual(v1ResourcesForOurPod.Containers[0].Devices[0].ResourceName, SampleDeviceResourceName)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers[0].Devices[0].DeviceIds), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers[0].Devices[0].DeviceIds), 1)
		})

		// simulate container restart, while all other involved components (kubelet, device plugin) stay stable. To do so, in the container
		// entry point we sleep for a limited and short period of time. The device assignment should be kept and be stable across the container
		// restarts. For the sake of brevity we however check just the fist restart.
		ginkgo.It("Keeps device plugin assignments across pod restarts (no kubelet restart, device plugin re-registration)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalWithRestart)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for container to restart")
			ensurePodContainerRestart(ctx, f, pod1.Name, pod1.Name)

			// check from the device assignment is preserved and stable from perspective of the container
			ginkgo.By("Confirming that after a container restart, fake-device assignment is kept")
			devIDRestart1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			framework.ExpectEqual(devIDRestart1, devID1)

			// crosscheck from the device assignment is preserved and stable from perspective of the kubelet.
			// needs to match the container perspective.
			ginkgo.By("Verifying the device assignment after container restart using podresources API")
			v1PodResources, err = getV1NodeDevices(ctx)
			if err != nil {
				framework.ExpectNoError(err, "getting pod resources assignment after pod restart")
			}
			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after pod restart")

			ginkgo.By("Creating another pod")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod2.Name, f.Namespace.Name, 1*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By("Checking that pod got a fake device")
			devID2, err := parseLog(ctx, f, pod2.Name, pod2.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod2.Name)

			gomega.Expect(devID2).To(gomega.Not(gomega.Equal("")), "pod2 requested a device but started successfully without")

			ginkgo.By("Verifying the device assignment after extra container start using podresources API")
			v1PodResources, err = getV1NodeDevices(ctx)
			if err != nil {
				framework.ExpectNoError(err, "getting pod resources assignment after pod restart")
			}
			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after extra container restart - pod1")
			err = checkPodResourcesAssignment(v1PodResources, pod2.Namespace, pod2.Name, pod2.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID2})
			framework.ExpectNoError(err, "inconsistent device assignment after extra container restart - pod2")
		})

		// simulate kubelet restart, *but not* device plugin re-registration, while the pod and the container stays running.
		// The device assignment should be kept and be stable across the kubelet restart, because it's the kubelet which performs the device allocation,
		// and both the device plugin and the actual consumer (container) are stable.
		ginkgo.It("Keeps device plugin assignments across kubelet restarts (no pod restart, no device plugin re-registration)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Waiting for resource to become available on the local node after restart")
			gomega.Eventually(ctx, func() bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					CountSampleDeviceCapacity(node) == expectedSampleDevsAmount &&
					CountSampleDeviceAllocatable(node) == expectedSampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Waiting for the pod to fail with admission error as device plugin hasn't re-registered yet")
			gomega.Eventually(ctx, getPod).
				WithArguments(f, pod1.Name).
				WithTimeout(time.Minute).
				Should(HaveFailedWithAdmissionError(),
					"the pod succeeded to start, when it should fail with the admission error")

			// crosscheck from the device assignment is preserved and stable from perspective of the kubelet.
			// note we don't check again the logs of the container: the check is done at startup, the container
			// never restarted (runs "forever" from this test timescale perspective) hence re-doing this check
			// is useless.
			ginkgo.By("Verifying the device assignment after kubelet restart using podresources API")
			gomega.Eventually(ctx, func() error {
				v1PodResources, err = getV1NodeDevices(ctx)
				return err
			}, 30*time.Second, framework.Poll).ShouldNot(gomega.HaveOccurred(), "cannot fetch the compute resource assignment after kubelet restart")

			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after pod restart")
		})

		// simulate kubelet and container restart, *but not* device plugin re-registration.
		// The device assignment should be kept and be stable across the kubelet and container restart, because it's the kubelet which
		// performs the device allocation, and both the device plugin is stable.
		ginkgo.It("Keeps device plugin assignments across pod and kubelet restarts (no device plugin re-registration)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalWithRestart)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)

			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Waiting for container to restart")
			ensurePodContainerRestart(ctx, f, pod1.Name, pod1.Name)

			ginkgo.By("Confirming that after a container restart, fake-device assignment is kept")
			devIDRestart1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Waiting for the pod to fail with admission error as device plugin hasn't re-registered yet")
			gomega.Eventually(ctx, getPod).
				WithArguments(f, pod1.Name).
				WithTimeout(time.Minute).
				Should(HaveFailedWithAdmissionError(),
					"the pod succeeded to start, when it should fail with the admission error")

			// crosscheck from the device assignment is preserved and stable from perspective of the kubelet.
			// note we don't check again the logs of the container: the check is done at startup, the container
			// never restarted (runs "forever" from this test timescale perspective) hence re-doing this check
			// is useless.
			ginkgo.By("Verifying the device assignment after kubelet restart using podresources API")
			gomega.Eventually(ctx, func() error {
				v1PodResources, err = getV1NodeDevices(ctx)
				return err
			}, 30*time.Second, framework.Poll).ShouldNot(gomega.HaveOccurred(), "cannot fetch the compute resource assignment after kubelet restart")

			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after pod restart")
		})

		// simulate device plugin re-registration, *but not* container and kubelet restart.
		// After the device plugin has re-registered, the list healthy devices is repopulated based on the devices discovered.
		// Once Pod2 is running we determine the device that was allocated it. As long as the device allocation succeeds the
		// test should pass.

		ginkgo.It("Keeps device plugin assignments after the device plugin has been re-registered (no kubelet, pod restart)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Re-Register resources and delete the plugin pod")
			gp := int64(0)
			deleteOptions := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, deleteOptions, time.Minute)
			waitForContainerRemoval(ctx, devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Recreating the plugin pod")
			devicePluginPod = e2epod.NewPodClient(f).CreateSync(ctx, dptemplate)
			err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, devicePluginPod.Name, devicePluginPod.Namespace, 1*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for resource to become available on the local node after re-registration")
			gomega.Eventually(ctx, func() bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					CountSampleDeviceCapacity(node) == expectedSampleDevsAmount &&
					CountSampleDeviceAllocatable(node) == expectedSampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			// crosscheck that after device plugin re-registration the device assignment is preserved and
			// stable from the kubelet's perspective.
			// note we don't check again the logs of the container: the check is done at startup, the container
			// never restarted (runs "forever" from this test timescale perspective) hence re-doing this check
			// is useless.
			ginkgo.By("Verifying the device assignment after device plugin re-registration using podresources API")
			gomega.Eventually(ctx, func() error {
				v1PodResources, err = getV1NodeDevices(ctx)
				return err
			}, 30*time.Second, framework.Poll).ShouldNot(gomega.HaveOccurred(), "cannot fetch the compute resource assignment after kubelet restart")

			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after pod restart")

			ginkgo.By("Creating another pod")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod2.Name, f.Namespace.Name, 1*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By("Checking that pod got a fake device")
			devID2, err := parseLog(ctx, f, pod2.Name, pod2.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod2.Name)

			gomega.Expect(devID2).To(gomega.Not(gomega.Equal("")), "pod2 requested a device but started successfully without")
		})

		// simulate kubelet restart *and* device plugin re-registration, while the pod and the container stays running.
		// The device assignment should be kept and be stable across the kubelet/device plugin restart, as both the aforementioned components
		// orchestrate the device allocation: the actual consumer (container) is stable.
		ginkgo.It("Keeps device plugin assignments after kubelet restart and device plugin has been re-registered (no pod restart)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever) // the pod has to run "forever" in the timescale of this test
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)

			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")), "pod1 requested a device but started successfully without")

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Waiting for the pod to fail with admission error as device plugin hasn't re-registered yet")
			gomega.Eventually(ctx, getPod).
				WithArguments(f, pod1.Name).
				WithTimeout(time.Minute).
				Should(HaveFailedWithAdmissionError(),
					"the pod succeeded to start, when it should fail with the admission error")

			// crosscheck from the device assignment is preserved and stable from perspective of the kubelet.
			// note we don't check again the logs of the container: the check is done at startup, the container
			// never restarted (runs "forever" from this test timescale perspective) hence re-doing this check
			// is useless.
			ginkgo.By("Verifying the device assignment after kubelet restart using podresources API")
			gomega.Eventually(ctx, func() error {
				v1PodResources, err = getV1NodeDevices(ctx)
				return err
			}, 30*time.Second, framework.Poll).ShouldNot(gomega.HaveOccurred(), "cannot fetch the compute resource assignment after kubelet restart")

			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after pod restart")

			ginkgo.By("Re-Register resources by deleting the plugin pod")
			gp := int64(0)
			deleteOptions := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, deleteOptions, time.Minute)
			waitForContainerRemoval(ctx, devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Recreating the plugin pod")
			devicePluginPod = e2epod.NewPodClient(f).CreateSync(ctx, dptemplate)

			ginkgo.By("Waiting for resource to become available on the local node after re-registration")
			gomega.Eventually(ctx, func() bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					CountSampleDeviceCapacity(node) == expectedSampleDevsAmount &&
					CountSampleDeviceAllocatable(node) == expectedSampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Creating another pod")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))

			ginkgo.By("Checking that pod got a fake device")
			devID2, err := parseLog(ctx, f, pod2.Name, pod2.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod2.Name)

			ginkgo.By("Verifying the device assignment after kubelet restart and device plugin re-registration using podresources API")
			// note we don't use eventually: the kubelet is supposed to be running and stable by now, so the call should just succeed
			v1PodResources, err = getV1NodeDevices(ctx)
			if err != nil {
				framework.ExpectNoError(err, "getting pod resources assignment after pod restart")
			}

			err = checkPodResourcesAssignment(v1PodResources, pod2.Namespace, pod2.Name, pod2.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID2})
			framework.ExpectNoError(err, "inconsistent device assignment after extra container restart - pod2")
		})
	})
}

func testDevicePluginNodeReboot(f *framework.Framework, pluginSockDir string) {
	ginkgo.Context("DevicePlugin [Serial] [Disruptive]", func() {
		var devicePluginPod *v1.Pod
		var v1alphaPodResources *kubeletpodresourcesv1alpha1.ListPodResourcesResponse
		var v1PodResources *kubeletpodresourcesv1.ListPodResourcesResponse
		var triggerPathFile, triggerPathDir string
		var err error
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrue())

			// Before we run the device plugin test, we need to ensure
			// that the cluster is in a clean state and there are no
			// pods running on this node.
			// This is done in a gomega.Eventually with retries since a prior test in a different test suite could've run and the deletion of it's resources may still be in progress.
			// xref: https://issue.k8s.io/115381
			gomega.Eventually(ctx, func(ctx context.Context) error {
				v1alphaPodResources, err = getV1alpha1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1alpha) podresources API endpoint: %v", err)
				}

				v1PodResources, err = getV1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1) podresources API endpoint: %v", err)
				}

				if len(v1alphaPodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1alpha pod resources to be empty, but got non-empty resources: %+v", v1alphaPodResources.PodResources)
				}

				if len(v1PodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1 pod resources to be empty, but got non-empty resources: %+v", v1PodResources.PodResources)
				}
				return nil
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.Succeed())

			ginkgo.By("Setting up the directory and file for controlling registration")
			triggerPathDir = filepath.Join(devicePluginDir, "sample")
			if _, err := os.Stat(triggerPathDir); errors.Is(err, os.ErrNotExist) {
				err := os.Mkdir(triggerPathDir, os.ModePerm)
				if err != nil {
					klog.Errorf("Directory creation %s failed: %v ", triggerPathDir, err)
					panic(err)
				}
				klog.InfoS("Directory created successfully")

				triggerPathFile = filepath.Join(triggerPathDir, "registration")
				if _, err := os.Stat(triggerPathFile); errors.Is(err, os.ErrNotExist) {
					_, err = os.Create(triggerPathFile)
					if err != nil {
						klog.Errorf("File creation %s failed: %v ", triggerPathFile, err)
						panic(err)
					}
				}
			}

			ginkgo.By("Scheduling a sample device plugin pod")
			data, err := e2etestfiles.Read(SampleDevicePluginControlRegistrationDSYAML)
			if err != nil {
				framework.Fail(err.Error())
			}
			ds := readDaemonSetV1OrDie(data)

			dp := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: SampleDevicePluginName,
				},
				Spec: ds.Spec.Template.Spec,
			}

			devicePluginPod = e2epod.NewPodClient(f).CreateSync(ctx, dp)

			go func() {
				// Since autoregistration is disabled for the device plugin (as REGISTER_CONTROL_FILE
				// environment variable is specified), device plugin registration needs to be triggerred
				// manually.
				// This is done by deleting the control file at the following path:
				// `/var/lib/kubelet/device-plugins/sample/registration`.

				defer ginkgo.GinkgoRecover()
				framework.Logf("Deleting the control file: %q to trigger registration", triggerPathFile)
				err := os.Remove(triggerPathFile)
				framework.ExpectNoError(err)
			}()

			ginkgo.By("Waiting for devices to become available on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && CountSampleDeviceCapacity(node) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			framework.Logf("Successfully created device plugin pod")

			ginkgo.By(fmt.Sprintf("Waiting for the resource exported by the sample device plugin to become available on the local node (instances: %d)", expectedSampleDevsAmount))
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					CountSampleDeviceCapacity(node) == expectedSampleDevsAmount &&
					CountSampleDeviceAllocatable(node) == expectedSampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Deleting the device plugin pod")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, time.Minute)

			ginkgo.By("Deleting any Pods created by the test")
			l, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				framework.Logf("Deleting pod: %s", p.Name)
				e2epod.NewPodClient(f).DeleteSync(ctx, p.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}

			err = os.Remove(triggerPathDir)
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for devices to become unavailable on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && CountSampleDeviceCapacity(node) <= 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("devices now unavailable on the local node")
		})

		// simulate node reboot scenario by removing pods using CRI before kubelet is started. In addition to that,
		// intentionally a scenario is created where after node reboot, application pods requesting devices appear before the device plugin pod
		// exposing those devices as resource has re-registers itself to Kubelet. The expected behavior is that the application pod fails at
		// admission time.
		ginkgo.It("Keeps device plugin assignments across node reboots (no pod restart, no device plugin re-registration)", func(ctx context.Context) {
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(SampleDeviceResourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1, err := parseLog(ctx, f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectNoError(err, "getting logs for pod %q", pod1.Name)

			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("stopping the kubelet")
			startKubelet := stopKubelet()

			ginkgo.By("stopping all the local containers - using CRI")
			rs, _, err := getCRIClient()
			framework.ExpectNoError(err)
			sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
			framework.ExpectNoError(err)
			for _, sandbox := range sandboxes {
				gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
				ginkgo.By(fmt.Sprintf("deleting pod using CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))

				err := rs.RemovePodSandbox(ctx, sandbox.Id)
				framework.ExpectNoError(err)
			}

			ginkgo.By("restarting the kubelet")
			startKubelet()

			ginkgo.By("Wait for node to be ready again")
			e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)

			ginkgo.By("Waiting for the pod to fail with admission error as device plugin hasn't re-registered yet")
			gomega.Eventually(ctx, getPod).
				WithArguments(f, pod1.Name).
				WithTimeout(time.Minute).
				Should(HaveFailedWithAdmissionError(),
					"the pod succeeded to start, when it should fail with the admission error")

			// crosscheck from the device assignment is preserved and stable from perspective of the kubelet.
			// note we don't check again the logs of the container: the check is done at startup, the container
			// never restarted (runs "forever" from this test timescale perspective) hence re-doing this check
			// is useless.
			ginkgo.By("Verifying the device assignment after kubelet restart using podresources API")
			gomega.Eventually(ctx, func() error {
				v1PodResources, err = getV1NodeDevices(ctx)
				return err
			}, 30*time.Second, framework.Poll).ShouldNot(gomega.HaveOccurred(), "cannot fetch the compute resource assignment after kubelet restart")

			err = checkPodResourcesAssignment(v1PodResources, pod1.Namespace, pod1.Name, pod1.Spec.Containers[0].Name, SampleDeviceResourceName, []string{devID1})
			framework.ExpectNoError(err, "inconsistent device assignment after node reboot")

		})
	})
}

// makeBusyboxPod returns a simple Pod spec with a busybox container
// that requests SampleDeviceResourceName and runs the specified command.
func makeBusyboxPod(SampleDeviceResourceName, cmd string) *v1.Pod {
	podName := "device-plugin-test-" + string(uuid.NewUUID())
	rl := v1.ResourceList{v1.ResourceName(SampleDeviceResourceName): *resource.NewQuantity(1, resource.DecimalSI)}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{{
				Image: busyboxImage,
				Name:  podName,
				// Runs the specified command in the test pod.
				Command: []string{"sh", "-c", cmd},
				Resources: v1.ResourceRequirements{
					Limits:   rl,
					Requests: rl,
				},
			}},
		},
	}
}

// ensurePodContainerRestart confirms that pod container has restarted at least once
func ensurePodContainerRestart(ctx context.Context, f *framework.Framework, podName string, contName string) {
	var initialCount int32
	var currentCount int32
	p, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	if err != nil || len(p.Status.ContainerStatuses) < 1 {
		framework.Failf("ensurePodContainerRestart failed for pod %q: %v", podName, err)
	}
	initialCount = p.Status.ContainerStatuses[0].RestartCount
	gomega.Eventually(ctx, func() bool {
		p, err = e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
		if err != nil || len(p.Status.ContainerStatuses) < 1 {
			return false
		}
		currentCount = p.Status.ContainerStatuses[0].RestartCount
		framework.Logf("initial %v, current %v", initialCount, currentCount)
		return currentCount > initialCount
	}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
}

// parseLog returns the matching string for the specified regular expression parsed from the container logs.
func parseLog(ctx context.Context, f *framework.Framework, podName string, contName string, re string) (string, error) {
	logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, contName)
	if err != nil {
		return "", err
	}

	framework.Logf("got pod logs: %v", logs)
	regex := regexp.MustCompile(re)
	matches := regex.FindStringSubmatch(logs)
	if len(matches) < 2 {
		return "", fmt.Errorf("unexpected match in logs: %q", logs)
	}

	return matches[1], nil
}

func checkPodResourcesAssignment(v1PodRes *kubeletpodresourcesv1.ListPodResourcesResponse, podNamespace, podName, containerName, resourceName string, devs []string) error {
	for _, podRes := range v1PodRes.PodResources {
		if podRes.Namespace != podNamespace || podRes.Name != podName {
			continue
		}
		for _, contRes := range podRes.Containers {
			if contRes.Name != containerName {
				continue
			}
			return matchContainerDevices(podNamespace+"/"+podName+"/"+containerName, contRes.Devices, resourceName, devs)
		}
	}
	return fmt.Errorf("no resources found for %s/%s/%s", podNamespace, podName, containerName)
}

func matchContainerDevices(ident string, contDevs []*kubeletpodresourcesv1.ContainerDevices, resourceName string, devs []string) error {
	expected := sets.New[string](devs...)
	assigned := sets.New[string]()
	for _, contDev := range contDevs {
		if contDev.ResourceName != resourceName {
			continue
		}
		assigned = assigned.Insert(contDev.DeviceIds...)
	}
	expectedStr := strings.Join(expected.UnsortedList(), ",")
	assignedStr := strings.Join(assigned.UnsortedList(), ",")
	framework.Logf("%s: devices expected %q assigned %q", ident, expectedStr, assignedStr)
	if !assigned.Equal(expected) {
		return fmt.Errorf("device allocation mismatch for %s expected %s assigned %s", ident, expectedStr, assignedStr)
	}
	return nil
}

// getSampleDevicePluginPod returns the Sample Device Plugin pod to be used e2e tests.
func getSampleDevicePluginPod(pluginSockDir string) *v1.Pod {
	data, err := e2etestfiles.Read(SampleDevicePluginDSYAML)
	if err != nil {
		framework.Fail(err.Error())
	}

	ds := readDaemonSetV1OrDie(data)
	dp := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: SampleDevicePluginName,
		},
		Spec: ds.Spec.Template.Spec,
	}
	for i := range dp.Spec.Containers[0].Env {
		if dp.Spec.Containers[0].Env[i].Name == SampleDeviceEnvVarNamePluginSockDir {
			dp.Spec.Containers[0].Env[i].Value = pluginSockDir
		}
	}

	return dp
}
