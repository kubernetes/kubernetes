/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
	"github.com/onsi/gomega/types"
)

const (
	devicePluginDir = "/var/lib/kubelet/device-plugins"
)

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Device Manager", framework.WithSerial(), nodefeature.DeviceManager, func() {
	f := framework.NewDefaultFramework("devicemanager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		This end to end test is to simulate a scenario where after kubelet restart/node
		reboot application pods requesting devices appear before the device plugin
		pod exposing those devices as resources.

		The happy path is where after node reboot/ kubelet restart, the device plugin pod
		appears before the application pod. This PR and this e2e test
		aims to tackle the scenario where device plugin either does not appear first
		or doesn't get the chance to re-register itself.

		Since there is no way of controlling the order in which the pods appear after
		kubelet restart/node reboot, we can't guarantee that the application pod
		recovers before device plugin pod (the scenario we want to exercise here).
		If the device plugin pod is recovered before the test pod, we still can
		meaningfully reproduce the scenario by NOT sending the registration command.
		To do so sample device plugin is enhanced. For implementation details, refer to:
		`test/images/sample-device-plugin/sampledeviceplugin.go`. This enhancement
		allows auto-registration of the plugin to be controlled with the help of an environment
		variable: REGISTER_CONTROL_FILE. By default this environment variable is not present
		and the device plugin autoregisters to kubelet. For this e2e test, we use sample device
		plugin spec with REGISTER_CONTROL_FILE=/var/lib/kubelet/device-plugins/sample/registration
		to allow manual registeration of the plugin to allow an application pod (requesting devices)
		to successfully run on the node followed by kubelet restart where device plugin doesn't
		register and the application pod fails with admission error.

		   Breakdown of the steps implemented as part of this e2e test is as follows:
		   1. Create a file `registration` at path `/var/lib/kubelet/device-plugins/sample/`
		   2. Create sample device plugin with an environment variable with
		      `REGISTER_CONTROL_FILE=/var/lib/kubelet/device-plugins/sample/registration` that
			  waits for a client to delete the control file.
		   3. Trigger plugin registeration by deleting the abovementioned directory.
		   4. Create a test pod requesting devices exposed by the device plugin.
		   5. Stop kubelet.
		   6. Remove pods using CRI to ensure new pods are created after kubelet restart.
		   7. Restart kubelet.
		   8. Wait for the sample device plugin pod to be running. In this case,
		      the registration is not triggered.
		   9. Ensure that resource capacity/allocatable exported by the device plugin is zero.
		   10. The test pod should fail with `UnexpectedAdmissionError`
		   11. Delete the test pod.
		   12. Delete the sample device plugin pod.
		   13. Remove `/var/lib/kubelet/device-plugins/sample/` and its content, the directory created to control registration
	*/
	f.Context("With sample device plugin", f.WithSerial(), f.WithDisruptive(), func() {
		var deviceCount int = 2
		var devicePluginPod *v1.Pod
		var triggerPathFile, triggerPathDir string

		// this test wants to reproduce what happened in https://github.com/kubernetes/kubernetes/issues/109595
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(ctx, e2enode.TotalReady).
				WithArguments(f.ClientSet).
				WithTimeout(time.Minute).
				Should(gomega.BeEquivalentTo(1))

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

			gomega.Eventually(ctx, isNodeReadyWithSampleResources).
				WithArguments(f).
				WithTimeout(5 * time.Minute).
				Should(BeReady())

			framework.Logf("Successfully created device plugin pod")

			devsLen := int64(deviceCount) // shortcut
			ginkgo.By("Waiting for the resource exported by the sample device plugin to become available on the local node")

			gomega.Eventually(ctx, isNodeReadyWithAllocatableSampleResources).
				WithArguments(f, devsLen).
				WithTimeout(5 * time.Minute).
				Should(HaveAllocatableDevices())
		})

		framework.It("should deploy pod consuming devices first but fail with admission error after kubelet restart in case device plugin hasn't re-registered", framework.WithFlaky(), func(ctx context.Context) {
			var err error
			podCMD := "while true; do sleep 1000; done;"

			ginkgo.By(fmt.Sprintf("creating a pods requiring %d %q", deviceCount, SampleDeviceResourceName))

			pod := makeBusyboxDeviceRequiringPod(SampleDeviceResourceName, podCMD)
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			ginkgo.By("making sure all the pods are ready")

			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, testPod.Namespace, testPod.Name, "Ready", 120*time.Second, testutils.PodRunningReady)
			framework.ExpectNoError(err, "pod %s/%s did not go running", testPod.Namespace, testPod.Name)
			framework.Logf("pod %s/%s running", testPod.Namespace, testPod.Name)

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

			ginkgo.By("waiting for the kubelet to be ready again")
			// Wait for the Kubelet to be ready.

			gomega.Eventually(ctx, e2enode.TotalReady).
				WithArguments(f.ClientSet).
				WithTimeout(2 * time.Minute).
				Should(gomega.BeEquivalentTo(1))

			ginkgo.By("making sure all the pods are ready after the recovery")

			var devicePluginPodAfterRestart *v1.Pod

			devicePluginPodAfterRestart, err = e2epod.NewPodClient(f).Get(ctx, devicePluginPod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, devicePluginPodAfterRestart.Namespace, devicePluginPodAfterRestart.Name, "Ready", 120*time.Second, testutils.PodRunningReady)
			framework.ExpectNoError(err, "pod %s/%s did not go running", devicePluginPodAfterRestart.Namespace, devicePluginPodAfterRestart.Name)
			framework.Logf("pod %s/%s running", devicePluginPodAfterRestart.Namespace, devicePluginPodAfterRestart.Name)

			ginkgo.By("Waiting for the resource capacity/allocatable exported by the sample device plugin to become zero")

			// The device plugin pod has restarted but has not re-registered to kubelet (as AUTO_REGISTER= false)
			// and registration wasn't triggered manually (by writing to the unix socket exposed at
			// `/var/lib/kubelet/device-plugins/registered`). Because of this, the capacity and allocatable corresponding
			// to the resource exposed by the device plugin should be zero.

			gomega.Eventually(ctx, isNodeReadyWithAllocatableSampleResources).
				WithArguments(f, int64(0)).
				WithTimeout(5 * time.Minute).
				Should(HaveAllocatableDevices())

			ginkgo.By("Checking that pod requesting devices failed to start because of admission error")

			// NOTE: The device plugin won't re-register again and this is intentional.
			// Because of this, the testpod (requesting a device) should fail with an admission error.

			gomega.Eventually(ctx, getPod).
				WithArguments(f, testPod.Name).
				WithTimeout(time.Minute).
				Should(HaveFailedWithAdmissionError(),
					"the pod succeeded to start, when it should fail with the admission error")

			ginkgo.By("removing application pods")
			e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, 2*time.Minute)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Deleting the device plugin pod")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, time.Minute)

			ginkgo.By("Deleting the directory and file setup for controlling registration")
			err := os.RemoveAll(triggerPathDir)
			framework.ExpectNoError(err)

			ginkgo.By("Deleting any Pods created by the test")
			l, err := e2epod.NewPodClient(f).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				framework.Logf("Deleting pod: %s", p.Name)
				e2epod.NewPodClient(f).DeleteSync(ctx, p.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}

			ginkgo.By("Waiting for devices to become unavailable on the local node")
			gomega.Eventually(ctx, isNodeReadyWithoutSampleResources).
				WithArguments(f).
				WithTimeout(5 * time.Minute).
				Should(BeReady())
		})

	})

})

func makeBusyboxDeviceRequiringPod(resourceName, cmd string) *v1.Pod {
	podName := "device-manager-test-" + string(uuid.NewUUID())
	rl := v1.ResourceList{
		v1.ResourceName(resourceName): *resource.NewQuantity(2, resource.DecimalSI),
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
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

// BeReady verifies that a node is ready and devices have registered.
func BeReady() types.GomegaMatcher {
	return gomega.And(
		// This additional matcher checks for the final error condition.
		gcustom.MakeMatcher(func(ready bool) (bool, error) {
			if !ready {
				return false, fmt.Errorf("expected node to be ready=%t", ready)
			}
			return true, nil
		}),
		BeInReadyPhase(true),
	)
}

// BeInReadyPhase matches if node is ready i.e. ready is true.
func BeInReadyPhase(isReady bool) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(ready bool) (bool, error) {
		return ready == isReady, nil
	}).WithTemplate("expected Node Ready {{.To}} be in {{format .Data}}\nGot instead:\n{{.FormattedActual}}").WithTemplateData(isReady)
}

func isNodeReadyWithSampleResources(ctx context.Context, f *framework.Framework) (bool, error) {
	node, ready := getLocalTestNode(ctx, f)
	if !ready {
		return false, fmt.Errorf("expected node to be ready=%t", ready)
	}

	if CountSampleDeviceCapacity(node) <= 0 {
		return false, fmt.Errorf("expected devices to be advertised")
	}
	return true, nil
}

// HaveAllocatableDevices verifies that a node has allocatable devices.
func HaveAllocatableDevices() types.GomegaMatcher {
	return gomega.And(
		// This additional matcher checks for the final error condition.
		gcustom.MakeMatcher(func(hasAllocatable bool) (bool, error) {
			if !hasAllocatable {
				return false, fmt.Errorf("expected node to be have allocatable devices=%t", hasAllocatable)
			}
			return true, nil
		}),
		hasAllocatable(true),
	)
}

// hasAllocatable matches if node is ready i.e. ready is true.
func hasAllocatable(hasAllocatable bool) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(hasAllocatableDevices bool) (bool, error) {
		return hasAllocatableDevices == hasAllocatable, nil
	}).WithTemplate("expected Node with allocatable {{.To}} be in {{format .Data}}\nGot instead:\n{{.FormattedActual}}").WithTemplateData(hasAllocatable)
}

func isNodeReadyWithAllocatableSampleResources(ctx context.Context, f *framework.Framework, devCount int64) (bool, error) {
	node, ready := getLocalTestNode(ctx, f)
	if !ready {
		return false, fmt.Errorf("expected node to be ready=%t", ready)
	}

	if CountSampleDeviceCapacity(node) != devCount {
		return false, fmt.Errorf("expected devices capacity to be: %d", devCount)
	}

	if CountSampleDeviceAllocatable(node) != devCount {
		return false, fmt.Errorf("expected devices allocatable to be: %d", devCount)
	}
	return true, nil
}

func isNodeReadyWithoutSampleResources(ctx context.Context, f *framework.Framework) (bool, error) {
	node, ready := getLocalTestNode(ctx, f)
	if !ready {
		return false, fmt.Errorf("expected node to be ready=%t", ready)
	}

	if CountSampleDeviceCapacity(node) > 0 {
		return false, fmt.Errorf("expected devices to be not present")
	}
	return true, nil
}

// HaveFailedWithAdmissionError verifies that a pod fails at admission.
func HaveFailedWithAdmissionError() types.GomegaMatcher {
	return gomega.And(
		gcustom.MakeMatcher(func(hasFailed bool) (bool, error) {
			if !hasFailed {
				return false, fmt.Errorf("expected pod to have failed=%t", hasFailed)
			}
			return true, nil
		}),
		hasFailed(true),
	)
}

// hasFailed matches if pod has failed.
func hasFailed(hasFailed bool) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(hasPodFailed bool) (bool, error) {
		return hasPodFailed == hasFailed, nil
	}).WithTemplate("expected Pod failed {{.To}} be in {{format .Data}}\nGot instead:\n{{.FormattedActual}}").WithTemplateData(hasFailed)
}

func getPodByName(ctx context.Context, f *framework.Framework, podName string) (*v1.Pod, error) {
	return e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
}

func getPod(ctx context.Context, f *framework.Framework, podName string) (bool, error) {
	pod, err := getPodByName(ctx, f, podName)
	if err != nil {
		return false, err
	}

	expectedStatusReason := "UnexpectedAdmissionError"
	expectedStatusMessage := "Allocate failed due to no healthy devices present; cannot allocate unhealthy devices"

	// This additional matcher checks for the final error condition.
	if pod.Status.Phase != v1.PodFailed {
		return false, fmt.Errorf("expected pod to reach phase %q, got final phase %q instead.", v1.PodFailed, pod.Status.Phase)
	}
	if pod.Status.Reason != expectedStatusReason {
		return false, fmt.Errorf("expected pod status reason to be %q, got %q instead.", expectedStatusReason, pod.Status.Reason)
	}
	if !strings.Contains(pod.Status.Message, expectedStatusMessage) {
		return false, fmt.Errorf("expected pod status reason to contain %q, got %q instead.", expectedStatusMessage, pod.Status.Message)
	}
	return true, nil
}
