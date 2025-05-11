/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourceapi "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	applyv1 "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	// podStartTimeout is how long to wait for the pod to be started.
	podStartTimeout = 5 * time.Minute
)

// TODO: remove feature.DynamicResourceAllocation here and add it only for those tests
// which really need node-level support for DRA.
//
// The "DRA" label is used to select tests related to DRA in a Ginkgo label filter.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DynamicResourceAllocation), func() {
	f := framework.NewDefaultFramework("dra")

	// The driver containers have to run with sufficient privileges to
	// modify /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("kubelet", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources(10, false))
		b := newBuilder(f, driver)

		ginkgo.It("registers plugin", func() {
			ginkgo.By("the driver is running")
		})

		ginkgo.It("must retry NodePrepareResources", func(ctx context.Context) {
			// We have exactly one host.
			m := MethodInstance{driver.Nodenames()[0], NodePrepareResourcesMethod}

			driver.Fail(m, true)

			ginkgo.By("waiting for container startup to fail")
			pod, template := b.podInline()

			b.create(ctx, pod, template)

			ginkgo.By("wait for NodePrepareResources call")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				if driver.CallCount(m) == 0 {
					return errors.New("NodePrepareResources not called yet")
				}
				return nil
			}).WithTimeout(podStartTimeout).Should(gomega.Succeed())

			ginkgo.By("allowing container startup to succeed")
			callCount := driver.CallCount(m)
			driver.Fail(m, false)
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
			framework.ExpectNoError(err, "start pod with inline resource claim")
			if driver.CallCount(m) == callCount {
				framework.Fail("NodePrepareResources should have been called again")
			}
		})

		ginkgo.It("must not run a pod if a claim is not ready", func(ctx context.Context) {
			claim := b.externalClaim()
			b.create(ctx, claim)
			pod := b.podExternal()

			// This bypasses scheduling and therefore the pod gets
			// to run on the node although the claim is not ready.
			// Because the parameters are missing, the claim
			// also cannot be allocated later.
			pod.Spec.NodeName = nodes.NodeNames[0]
			b.create(ctx, pod)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				testPod, err := b.f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the test pod %s to exist: %w", pod.Name, err)
				}
				if testPod.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s: unexpected status %s, expected status: %s", pod.Name, testPod.Status.Phase, v1.PodPending)
				}
				return nil
			}, 20*time.Second, 200*time.Millisecond).Should(gomega.BeNil())
		})

		ginkgo.It("must unprepare resources for force-deleted pod", func(ctx context.Context) {
			claim := b.externalClaim()
			pod := b.podExternal()
			zero := int64(0)
			pod.Spec.TerminationGracePeriodSeconds = &zero

			b.create(ctx, claim, pod)

			b.testPod(ctx, f, pod)

			ginkgo.By(fmt.Sprintf("force delete test pod %s", pod.Name))
			err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &zero})
			if !apierrors.IsNotFound(err) {
				framework.ExpectNoError(err, "force delete test pod")
			}

			for host, plugin := range b.driver.Nodes {
				ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
				gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
			}
		})

		ginkgo.It("must call NodePrepareResources even if not used by any container", func(ctx context.Context) {
			pod, template := b.podInline()
			for i := range pod.Spec.Containers {
				pod.Spec.Containers[i].Resources.Claims = nil
			}
			b.create(ctx, pod, template)
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "start pod")
			for host, plugin := range b.driver.Nodes {
				gomega.Expect(plugin.GetPreparedResources()).ShouldNot(gomega.BeEmpty(), "claims should be prepared on host %s while pod is running", host)
			}
		})

		ginkgo.It("must map configs and devices to the right containers", func(ctx context.Context) {
			// Several claims, each with three requests and three configs.
			// One config applies to all requests, the other two only to one request each.
			claimForAllContainers := b.externalClaim()
			claimForAllContainers.Name = "all"
			claimForAllContainers.Spec.Devices.Requests = append(claimForAllContainers.Spec.Devices.Requests,
				*claimForAllContainers.Spec.Devices.Requests[0].DeepCopy(),
				*claimForAllContainers.Spec.Devices.Requests[0].DeepCopy(),
			)
			claimForAllContainers.Spec.Devices.Requests[0].Name = "req0"
			claimForAllContainers.Spec.Devices.Requests[1].Name = "req1"
			claimForAllContainers.Spec.Devices.Requests[2].Name = "req2"
			claimForAllContainers.Spec.Devices.Config = append(claimForAllContainers.Spec.Devices.Config,
				*claimForAllContainers.Spec.Devices.Config[0].DeepCopy(),
				*claimForAllContainers.Spec.Devices.Config[0].DeepCopy(),
			)
			claimForAllContainers.Spec.Devices.Config[0].Requests = nil
			claimForAllContainers.Spec.Devices.Config[1].Requests = []string{"req1"}
			claimForAllContainers.Spec.Devices.Config[2].Requests = []string{"req2"}
			claimForAllContainers.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`{"all_config0":"true"}`)
			claimForAllContainers.Spec.Devices.Config[1].Opaque.Parameters.Raw = []byte(`{"all_config1":"true"}`)
			claimForAllContainers.Spec.Devices.Config[2].Opaque.Parameters.Raw = []byte(`{"all_config2":"true"}`)

			claimForContainer0 := claimForAllContainers.DeepCopy()
			claimForContainer0.Name = "container0"
			claimForContainer0.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`{"container0_config0":"true"}`)
			claimForContainer0.Spec.Devices.Config[1].Opaque.Parameters.Raw = []byte(`{"container0_config1":"true"}`)
			claimForContainer0.Spec.Devices.Config[2].Opaque.Parameters.Raw = []byte(`{"container0_config2":"true"}`)
			claimForContainer1 := claimForAllContainers.DeepCopy()
			claimForContainer1.Name = "container1"
			claimForContainer1.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`{"container1_config0":"true"}`)
			claimForContainer1.Spec.Devices.Config[1].Opaque.Parameters.Raw = []byte(`{"container1_config1":"true"}`)
			claimForContainer1.Spec.Devices.Config[2].Opaque.Parameters.Raw = []byte(`{"container1_config2":"true"}`)

			pod := b.podExternal()
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              "all",
					ResourceClaimName: &claimForAllContainers.Name,
				},
				{
					Name:              "container0",
					ResourceClaimName: &claimForContainer0.Name,
				},
				{
					Name:              "container1",
					ResourceClaimName: &claimForContainer1.Name,
				},
			}

			// Add a second container.
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers[0].Name = "container0"
			pod.Spec.Containers[1].Name = "container1"

			// All claims use unique env variables which can be used to verify that they
			// have been mapped into the right containers. In addition, the test driver
			// also sets "claim_<claim name>_<request name>=true" with non-alphanumeric
			// replaced by underscore.

			// Both requests (claim_*_req*) and all user configs (user_*_config*).
			allContainersEnv := []string{
				"user_all_config0", "true",
				"user_all_config1", "true",
				"user_all_config2", "true",
				"claim_all_req0", "true",
				"claim_all_req1", "true",
				"claim_all_req2", "true",
			}

			// Everything from the "all" claim and everything from the "container0" claim.
			pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: "all"}, {Name: "container0"}}
			container0Env := []string{
				"user_container0_config0", "true",
				"user_container0_config1", "true",
				"user_container0_config2", "true",
				"claim_container0_req0", "true",
				"claim_container0_req1", "true",
				"claim_container0_req2", "true",
			}
			container0Env = append(container0Env, allContainersEnv...)

			// Everything from the "all" claim, but only the second request from the "container1" claim.
			// The first two configs apply.
			pod.Spec.Containers[1].Resources.Claims = []v1.ResourceClaim{{Name: "all"}, {Name: "container1", Request: "req1"}}
			container1Env := []string{
				"user_container1_config0", "true",
				"user_container1_config1", "true",
				// Does not apply: user_container1_config2
				"claim_container1_req1", "true",
			}
			container1Env = append(container1Env, allContainersEnv...)

			b.create(ctx, claimForAllContainers, claimForContainer0, claimForContainer1, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")

			testContainerEnv(ctx, f, pod, pod.Spec.Containers[0].Name, true, container0Env...)
			testContainerEnv(ctx, f, pod, pod.Spec.Containers[1].Name, true, container1Env...)
		})
	})

	// claimTests tries out several different combinations of pods with
	// claims, both inline and external.
	claimTests := func(b *builder, driver *Driver) {
		ginkgo.It("supports simple pod referencing inline resource claim", func(ctx context.Context) {
			pod, template := b.podInline()
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod)
		})

		ginkgo.It("supports inline claim referenced by multiple containers", func(ctx context.Context) {
			pod, template := b.podInlineMultiple()
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod)
		})

		ginkgo.It("supports simple pod referencing external resource claim", func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod)
			b.testPod(ctx, f, pod)
		})

		ginkgo.It("supports external claim referenced by multiple pods", func(ctx context.Context) {
			pod1 := b.podExternal()
			pod2 := b.podExternal()
			pod3 := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f, pod)
			}
		})

		ginkgo.It("supports external claim referenced by multiple containers of multiple pods", func(ctx context.Context) {
			pod1 := b.podExternalMultiple()
			pod2 := b.podExternalMultiple()
			pod3 := b.podExternalMultiple()
			claim := b.externalClaim()
			b.create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f, pod)
			}
		})

		ginkgo.It("supports init containers", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
			pod.Spec.InitContainers[0].Name += "-init"
			// This must succeed for the pod to start.
			pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
			b.create(ctx, pod, template)

			b.testPod(ctx, f, pod)
		})

		ginkgo.It("removes reservation from claim when pod is done", func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			pod.Spec.Containers[0].Command = []string{"true"}
			b.create(ctx, claim, pod)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be unreserved")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return f.ClientSet.ResourceV1beta2().ResourceClaims(pod.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.ReservedFor", gomega.BeEmpty()), "reservation should have been removed")
		})

		ginkgo.It("deletes generated claims when pod is done", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Spec.Containers[0].Command = []string{"true"}
			b.create(ctx, template, pod)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be deleted")
			gomega.Eventually(ctx, func(ctx context.Context) ([]resourceapi.ResourceClaim, error) {
				claims, err := f.ClientSet.ResourceV1beta2().ResourceClaims(pod.Namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return nil, err
				}
				return claims.Items, nil
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.BeEmpty(), "claim should have been deleted")
		})

		ginkgo.It("does not delete generated claims when pod is restarting", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Spec.Containers[0].Command = []string{"sh", "-c", "sleep 1; exit 1"}
			pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			b.create(ctx, template, pod)

			ginkgo.By("waiting for pod to restart twice")
			gomega.Eventually(ctx, func(ctx context.Context) (*v1.Pod, error) {
				return f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodStartSlow).Should(gomega.HaveField("Status.ContainerStatuses", gomega.ContainElements(gomega.HaveField("RestartCount", gomega.BeNumerically(">=", 2)))))
		})

		ginkgo.It("must deallocate after use", func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod)

			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))

			b.testPod(ctx, f, pod)

			ginkgo.By(fmt.Sprintf("deleting pod %s", klog.KObj(pod)))
			framework.ExpectNoError(b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))

			ginkgo.By("waiting for claim to get deallocated")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
		})

		f.It("must be possible for the driver to update the ResourceClaim.Status.Devices once allocated", f.WithFeatureGate(features.DRAResourceClaimDeviceStatus), func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod)

			// Waits for the ResourceClaim to be allocated and the pod to be scheduled.
			b.testPod(ctx, f, pod)

			allocatedResourceClaim, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(allocatedResourceClaim).ToNot(gomega.BeNil())
			gomega.Expect(allocatedResourceClaim.Status.Allocation).ToNot(gomega.BeNil())
			gomega.Expect(allocatedResourceClaim.Status.Allocation.Devices.Results).To(gomega.HaveLen(1))

			scheduledPod, err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(scheduledPod).ToNot(gomega.BeNil())

			ginkgo.By("Setting the device status a first time")
			allocatedResourceClaim.Status.Devices = append(allocatedResourceClaim.Status.Devices,
				resourceapi.AllocatedDeviceStatus{
					Driver:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Driver,
					Pool:       allocatedResourceClaim.Status.Allocation.Devices.Results[0].Pool,
					Device:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Device,
					Conditions: []metav1.Condition{{Type: "a", Status: "True", Message: "c", Reason: "d", LastTransitionTime: metav1.NewTime(time.Now().Truncate(time.Second))}},
					Data:       &runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)},
					NetworkData: &resourceapi.NetworkDeviceData{
						InterfaceName:   "inf1",
						IPs:             []string{"10.9.8.0/24", "2001:db8::/64"},
						HardwareAddress: "bc:1c:b6:3e:b8:25",
					},
				})

			// Updates the ResourceClaim from the driver on the same node as the pod.
			plugin, ok := driver.Nodes[scheduledPod.Spec.NodeName]
			if !ok {
				framework.Failf("pod got scheduled to node %s without a plugin", scheduledPod.Spec.NodeName)
			}
			updatedResourceClaim, err := plugin.UpdateStatus(ctx, allocatedResourceClaim)
			framework.ExpectNoError(err)
			gomega.Expect(updatedResourceClaim).ToNot(gomega.BeNil())
			gomega.Expect(updatedResourceClaim.Status.Devices).To(gomega.Equal(allocatedResourceClaim.Status.Devices))

			ginkgo.By("Updating the device status")
			updatedResourceClaim.Status.Devices[0] = resourceapi.AllocatedDeviceStatus{
				Driver:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Driver,
				Pool:       allocatedResourceClaim.Status.Allocation.Devices.Results[0].Pool,
				Device:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Device,
				Conditions: []metav1.Condition{{Type: "e", Status: "True", Message: "g", Reason: "h", LastTransitionTime: metav1.NewTime(time.Now().Truncate(time.Second))}},
				Data:       &runtime.RawExtension{Raw: []byte(`{"bar":"foo"}`)},
				NetworkData: &resourceapi.NetworkDeviceData{
					InterfaceName:   "inf2",
					IPs:             []string{"10.9.8.1/24", "2001:db8::1/64"},
					HardwareAddress: "bc:1c:b6:3e:b8:26",
				},
			}

			updatedResourceClaim2, err := plugin.UpdateStatus(ctx, updatedResourceClaim)
			framework.ExpectNoError(err)
			gomega.Expect(updatedResourceClaim2).ToNot(gomega.BeNil())
			gomega.Expect(updatedResourceClaim2.Status.Devices).To(gomega.Equal(updatedResourceClaim.Status.Devices))

			getResourceClaim, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(getResourceClaim).ToNot(gomega.BeNil())
			gomega.Expect(getResourceClaim.Status.Devices).To(gomega.Equal(updatedResourceClaim.Status.Devices))
		})
	}

	singleNodeTests := func() {
		nodes := NewNodes(f, 1, 1)
		maxAllocations := 1
		numPods := 10
		driver := NewDriver(f, nodes, driverResources(maxAllocations)) // All tests get their own driver instance.
		b := newBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.classParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.parametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports claim and class parameters", func(ctx context.Context) {
			pod, template := b.podInline()
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("supports reusing resources", func(ctx context.Context) {
			var objects []klog.KMetadata
			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pod, template := b.podInline()
				pods[i] = pod
				objects = append(objects, pod, template)
			}

			b.create(ctx, objects...)

			// We don't know the order. All that matters is that all of them get scheduled eventually.
			var wg sync.WaitGroup
			wg.Add(numPods)
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					b.testPod(ctx, f, pod, expectedEnv...)
					err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err, "delete pod")
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, time.Duration(numPods)*f.Timeouts.PodStartSlow))
				}()
			}
			wg.Wait()
		})

		ginkgo.It("supports sharing a claim concurrently", func(ctx context.Context) {
			var objects []klog.KMetadata
			objects = append(objects, b.externalClaim())
			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pod := b.podExternal()
				pods[i] = pod
				objects = append(objects, pod)
			}

			b.create(ctx, objects...)

			// We don't know the order. All that matters is that all of them get scheduled eventually.
			f.Timeouts.PodStartSlow *= time.Duration(numPods)
			var wg sync.WaitGroup
			wg.Add(numPods)
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					b.testPod(ctx, f, pod, expectedEnv...)
				}()
			}
			wg.Wait()
		})

		ginkgo.It("retries pod scheduling after creating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.podInline()
			deviceClassName := template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName
			class, err := f.ClientSet.ResourceV1beta2().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			deviceClassName += "-b"
			template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName = deviceClassName
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			class.UID = ""
			class.ResourceVersion = ""
			class.Name = deviceClassName
			b.create(ctx, class)

			b.testPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("retries pod scheduling after updating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.podInline()

			// First modify the class so that it matches no nodes (for classic DRA) and no devices (structured parameters).
			deviceClassName := template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName
			class, err := f.ClientSet.ResourceV1beta2().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			originalClass := class.DeepCopy()
			class.Spec.Selectors = []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{
					Expression: "false",
				},
			}}
			class, err = f.ClientSet.ResourceV1beta2().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// Now create the pod.
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			// Unblock the pod.
			class.Spec.Selectors = originalClass.Spec.Selectors
			_, err = f.ClientSet.ResourceV1beta2().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			b.testPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("runs a pod without a generated resource claim", func(ctx context.Context) {
			pod, _ /* template */ := b.podInline()
			created := b.create(ctx, pod)
			pod = created[0].(*v1.Pod)

			// Normally, this pod would be stuck because the
			// ResourceClaim cannot be created without the
			// template. We allow it to run by communicating
			// through the status that the ResourceClaim is not
			// needed.
			pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
				{Name: pod.Spec.ResourceClaims[0].Name, ResourceClaimName: nil},
			}
			_, err := f.ClientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
		})

		claimTests(b, driver)
	}

	// The following tests only make sense when there is more than one node.
	// They get skipped when there's only one node.
	multiNodeTests := func() {
		nodes := NewNodes(f, 3, 8)

		ginkgo.Context("with different ResourceSlices", func() {
			firstDevice := "pre-defined-device-01"
			secondDevice := "pre-defined-device-02"
			devicesPerNode := []map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				// First node:
				{
					firstDevice: {
						"healthy": {BoolValue: ptr.To(true)},
						"exists":  {BoolValue: ptr.To(true)},
					},
				},
				// Second node:
				{
					secondDevice: {
						"healthy": {BoolValue: ptr.To(false)},
						// Has no "exists" attribute!
					},
				},
			}
			driver := NewDriver(f, nodes, driverResources(-1, devicesPerNode...))
			b := newBuilder(f, driver)

			ginkgo.It("keeps pod pending because of CEL runtime errors", func(ctx context.Context) {
				// When pod scheduling encounters CEL runtime errors for some nodes, but not all,
				// it should still not schedule the pod because there is something wrong with it.
				// Scheduling it would make it harder to detect that there is a problem.
				//
				// This matches the "CEL-runtime-error-for-subset-of-nodes" unit test, except that
				// here we try it in combination with the actual scheduler and can extend it with
				// other checks, like event handling (future extension).

				gomega.Eventually(ctx, framework.ListObjects(f.ClientSet.ResourceV1beta2().ResourceSlices().List,
					metav1.ListOptions{
						FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driver.Name,
					},
				)).Should(gomega.HaveField("Items", gomega.ConsistOf(
					gomega.HaveField("Spec.Devices", gomega.ConsistOf(
						gomega.Equal(resourceapi.Device{
							Name:       firstDevice,
							Attributes: devicesPerNode[0][firstDevice],
						}))),
					gomega.HaveField("Spec.Devices", gomega.ConsistOf(
						gomega.Equal(resourceapi.Device{
							Name:       secondDevice,
							Attributes: devicesPerNode[1][secondDevice],
						}))),
				)))

				pod, template := b.podInline()
				template.Spec.Spec.Devices.Requests[0].Exactly.Selectors = append(template.Spec.Spec.Devices.Requests[0].Exactly.Selectors,
					resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							// Runtime error on one node, but not all.
							Expression: fmt.Sprintf(`device.attributes["%s"].exists`, driver.Name),
						},
					},
				)
				b.create(ctx, pod, template)

				framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "scheduling failure", f.Timeouts.PodStartShort, func(pod *v1.Pod) (bool, error) {
					for _, condition := range pod.Status.Conditions {
						if condition.Type == "PodScheduled" {
							if condition.Status != "False" {
								gomega.StopTrying("pod got scheduled unexpectedly").Now()
							}
							if strings.Contains(condition.Message, "CEL runtime error") {
								// This is what we are waiting for.
								return true, nil
							}
						}
					}
					return false, nil
				}), "pod must not get scheduled because of a CEL runtime error")
			})
		})

		ginkgo.Context("with node-local resources", func() {
			driver := NewDriver(f, nodes, driverResources(1))
			b := newBuilder(f, driver)

			ginkgo.It("uses all resources", func(ctx context.Context) {
				var objs []klog.KMetadata
				var pods []*v1.Pod
				for i := 0; i < len(nodes.NodeNames); i++ {
					pod, template := b.podInline()
					pods = append(pods, pod)
					objs = append(objs, pod, template)
				}
				b.create(ctx, objs...)

				for _, pod := range pods {
					err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
					framework.ExpectNoError(err, "start pod")
				}

				// The pods all should run on different
				// nodes because the maximum number of
				// claims per node was limited to 1 for
				// this test.
				//
				// We cannot know for sure why the pods
				// ran on two different nodes (could
				// also be a coincidence) but if they
				// don't cover all nodes, then we have
				// a problem.
				used := make(map[string]*v1.Pod)
				for _, pod := range pods {
					pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "get pod")
					nodeName := pod.Spec.NodeName
					if other, ok := used[nodeName]; ok {
						framework.Failf("Pod %s got started on the same node %s as pod %s although claim allocation should have been limited to one claim per node.", pod.Name, nodeName, other.Name)
					}
					used[nodeName] = pod
				}
			})
		})

		ginkgo.Context("with network-attached resources", func() {
			driver := NewDriver(f, nodes, networkResources(10, false))
			b := newBuilder(f, driver)

			f.It("supports sharing a claim sequentially", f.WithSlow(), func(ctx context.Context) {
				var objects []klog.KMetadata
				objects = append(objects, b.externalClaim())

				// This test used to test usage of the claim by one pod
				// at a time. After removing the "not sharable"
				// feature and bumping up the maximum number of
				// consumers this is now a stress test which runs
				// the maximum number of pods per claim in parallel.
				// This only works on clusters with >= 3 nodes.
				numMaxPods := resourceapi.ResourceClaimReservedForMaxSize
				ginkgo.By(fmt.Sprintf("Creating %d pods sharing the same claim", numMaxPods))
				pods := make([]*v1.Pod, numMaxPods)
				for i := 0; i < numMaxPods; i++ {
					pod := b.podExternal()
					pods[i] = pod
					objects = append(objects, pod)
				}
				b.create(ctx, objects...)

				timeout := f.Timeouts.PodStartSlow * time.Duration(numMaxPods)
				ensureDuration := f.Timeouts.PodStart // Don't check for too long, even if it is less precise.
				podIsPending := gomega.HaveField("Spec.NodeName", gomega.BeEmpty())
				waitForPodScheduled := func(pod *v1.Pod) {
					ginkgo.GinkgoHelper()
					gomega.Eventually(ctx, framework.GetObject(f.ClientSet.CoreV1().Pods(pod.Namespace).Get, pod.Name, metav1.GetOptions{})).
						WithTimeout(timeout).
						WithPolling(10*time.Second).
						ShouldNot(podIsPending, "Pod should get scheduled.")
				}
				ensurePodNotScheduled := func(pod *v1.Pod) {
					ginkgo.GinkgoHelper()
					gomega.Consistently(ctx, framework.GetObject(f.ClientSet.CoreV1().Pods(pod.Namespace).Get, pod.Name, metav1.GetOptions{})).
						WithTimeout(ensureDuration).
						WithPolling(10*time.Second).
						Should(podIsPending, "Pod should remain pending.")
				}

				// We don't know the order. All that matters is that all of them get scheduled eventually.
				ginkgo.By(fmt.Sprintf("Waiting for %d pods to be scheduled", numMaxPods))
				f.Timeouts.PodStartSlow *= time.Duration(numMaxPods)
				var wg sync.WaitGroup
				wg.Add(numMaxPods)
				for i := 0; i < numMaxPods; i++ {
					pod := pods[i]
					go func() {
						defer ginkgo.GinkgoRecover()
						defer wg.Done()
						waitForPodScheduled(pod)
					}()
				}
				wg.Wait()

				numMorePods := 10
				ginkgo.By(fmt.Sprintf("Creating %d additional pods for the same claim", numMorePods))
				morePods := make([]*v1.Pod, numMorePods)
				objects = nil
				for i := 0; i < numMorePods; i++ {
					pod := b.podExternal()
					morePods[i] = pod
					objects = append(objects, pod)
				}
				b.create(ctx, objects...)

				// None of the additional pods can run because of the ReservedFor limit.
				ginkgo.By(fmt.Sprintf("Check for %s that the additional pods don't get scheduled", ensureDuration))
				wg.Add(numMorePods)
				for i := 0; i < numMorePods; i++ {
					pod := morePods[i]
					go func() {
						defer ginkgo.GinkgoRecover()
						defer wg.Done()
						ensurePodNotScheduled(pod)
					}()
				}
				wg.Wait()

				// We need to delete each running pod, otherwise the new ones cannot use the claim.
				ginkgo.By(fmt.Sprintf("Deleting the initial %d pods", numMaxPods))
				wg.Add(numMaxPods)
				for i := 0; i < numMaxPods; i++ {
					pod := pods[i]
					go func() {
						defer ginkgo.GinkgoRecover()
						defer wg.Done()
						err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
						framework.ExpectNoError(err, "delete pod")
						framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow))
					}()
				}
				wg.Wait()

				// Now those should also run - eventually...
				ginkgo.By(fmt.Sprintf("Waiting for the additional %d pods to be scheduled", numMorePods))
				wg.Add(numMorePods)
				for i := 0; i < numMorePods; i++ {
					pod := morePods[i]
					go func() {
						defer ginkgo.GinkgoRecover()
						defer wg.Done()
						waitForPodScheduled(pod)
					}()
				}
				wg.Wait()
			})
		})
	}

	prioritizedListTests := func() {
		nodes := NewNodes(f, 1, 1)

		driver1Params, driver1Env := `{"driver":"1"}`, []string{"admin_driver", "1"}
		driver2Params, driver2Env := `{"driver":"2"}`, []string{"admin_driver", "2"}

		driver1 := NewDriver(f, nodes, driverResources(-1, []map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			{
				"device-1-1": {
					"dra.example.com/version":  {StringValue: ptr.To("1.0.0")},
					"dra.example.com/pcieRoot": {StringValue: ptr.To("bar")},
				},
				"device-1-2": {
					"dra.example.com/version":  {StringValue: ptr.To("2.0.0")},
					"dra.example.com/pcieRoot": {StringValue: ptr.To("foo")},
				},
			},
		}...))
		driver1.NameSuffix = "-1"
		b1 := newBuilder(f, driver1)
		b1.classParameters = driver1Params

		driver2 := NewDriver(f, nodes, driverResources(-1, []map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			{
				"device-2-1": {
					"dra.example.com/version":  {StringValue: ptr.To("1.0.0")},
					"dra.example.com/pcieRoot": {StringValue: ptr.To("foo")},
				},
			},
		}...))
		driver2.NameSuffix = "-2"
		b2 := newBuilder(f, driver2)
		b2.classParameters = driver2Params

		f.It("selects the first subrequest that can be satisfied", func(ctx context.Context) {
			name := "external-multiclaim"
			params := `{"a":"b"}`
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{{
							Name: "request-1",
							FirstAvailable: []resourceapi.DeviceSubRequest{
								{
									Name:            "sub-request-1",
									DeviceClassName: b1.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           3,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b1.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           2,
								},
								{
									Name:            "sub-request-3",
									DeviceClassName: b1.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						}},
						Config: []resourceapi.DeviceClaimConfiguration{
							{
								Requests: []string{"request-1"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(params),
										},
									},
								},
							},
						},
					},
				},
			}
			pod := b1.podExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.create(ctx, claim, pod)
			b1.testPod(ctx, f, pod)

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(2))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
			gomega.Expect(results[1].Request).To(gomega.Equal("request-1/sub-request-2"))
		})

		f.It("uses the config for the selected subrequest", func(ctx context.Context) {
			name := "external-multiclaim"
			parentReqParams, parentReqEnv := `{"a":"b"}`, []string{"user_a", "b"}
			subReq1Params := `{"c":"d"}`
			subReq2Params, subReq2Env := `{"e":"f"}`, []string{"user_e", "f"}
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{{
							Name: "request-1",
							FirstAvailable: []resourceapi.DeviceSubRequest{
								{
									Name:            "sub-request-1",
									DeviceClassName: b1.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           3,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b1.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           2,
								},
							},
						}},
						Config: []resourceapi.DeviceClaimConfiguration{
							{
								Requests: []string{"request-1"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(parentReqParams),
										},
									},
								},
							},
							{
								Requests: []string{"request-1/sub-request-1"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(subReq1Params),
										},
									},
								},
							},
							{
								Requests: []string{"request-1/sub-request-2"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(subReq2Params),
										},
									},
								},
							},
						},
					},
				},
			}
			pod := b1.podExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.create(ctx, claim, pod)
			var expectedEnv []string
			expectedEnv = append(expectedEnv, parentReqEnv...)
			expectedEnv = append(expectedEnv, subReq2Env...)
			b1.testPod(ctx, f, pod, expectedEnv...)
		})

		f.It("chooses the correct subrequest subject to constraints", func(ctx context.Context) {
			name := "external-multiclaim"
			params := `{"a":"b"}`
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Name: "request-1",
								FirstAvailable: []resourceapi.DeviceSubRequest{
									{
										Name:            "sub-request-1",
										DeviceClassName: b1.className(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
									{
										Name:            "sub-request-2",
										DeviceClassName: b1.className(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
								},
							},
							{
								Name: "request-2",
								Exactly: &resourceapi.ExactDeviceRequest{
									DeviceClassName: b2.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						},
						Constraints: []resourceapi.DeviceConstraint{
							{
								Requests:       []string{"request-1", "request-2"},
								MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("dra.example.com/version")),
							},
							{
								Requests:       []string{"request-1/sub-request-1", "request-2"},
								MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("dra.example.com/pcieRoot")),
							},
						},
						Config: []resourceapi.DeviceClaimConfiguration{
							{
								Requests: []string{},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(params),
										},
									},
								},
							},
						},
					},
				},
			}
			pod := b1.podExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.create(ctx, claim, pod)
			b1.testPod(ctx, f, pod)

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(2))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
			gomega.Expect(results[1].Request).To(gomega.Equal("request-2"))
		})

		f.It("filters config correctly for multiple devices", func(ctx context.Context) {
			name := "external-multiclaim"
			req1Params, req1Env := `{"a":"b"}`, []string{"user_a", "b"}
			req1subReq1Params, _ := `{"c":"d"}`, []string{"user_d", "d"}
			req1subReq2Params, req1subReq2Env := `{"e":"f"}`, []string{"user_e", "f"}
			req2Params, req2Env := `{"g":"h"}`, []string{"user_g", "h"}
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Name: "request-1",
								FirstAvailable: []resourceapi.DeviceSubRequest{
									{
										Name:            "sub-request-1",
										DeviceClassName: b1.className(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           20, // Requests more than are available.
									},
									{
										Name:            "sub-request-2",
										DeviceClassName: b1.className(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
								},
							},
							{
								Name: "request-2",
								Exactly: &resourceapi.ExactDeviceRequest{
									DeviceClassName: b2.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						},
						Config: []resourceapi.DeviceClaimConfiguration{
							{
								Requests: []string{"request-1"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(req1Params),
										},
									},
								},
							},
							{
								Requests: []string{"request-1/sub-request-1"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(req1subReq1Params),
										},
									},
								},
							},
							{
								Requests: []string{"request-1/sub-request-2"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b1.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(req1subReq2Params),
										},
									},
								},
							},
							{
								Requests: []string{"request-2"},
								DeviceConfiguration: resourceapi.DeviceConfiguration{
									Opaque: &resourceapi.OpaqueDeviceConfiguration{
										Driver: b2.driver.Name,
										Parameters: runtime.RawExtension{
											Raw: []byte(req2Params),
										},
									},
								},
							},
						},
					},
				},
			}
			pod := b1.pod()
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers[0].Name = "with-resource-0"
			pod.Spec.Containers[1].Name = "with-resource-1"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              name,
					ResourceClaimName: &name,
				},
			}
			pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: name, Request: "request-1"}}
			pod.Spec.Containers[1].Resources.Claims = []v1.ResourceClaim{{Name: name, Request: "request-2"}}

			b1.create(ctx, claim, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(2))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
			gomega.Expect(results[1].Request).To(gomega.Equal("request-2"))

			req1ExpectedEnv := []string{
				"claim_external_multiclaim_request_1",
				"true",
			}
			req1ExpectedEnv = append(req1ExpectedEnv, req1Env...)
			req1ExpectedEnv = append(req1ExpectedEnv, req1subReq2Env...)
			req1ExpectedEnv = append(req1ExpectedEnv, driver1Env...)
			testContainerEnv(ctx, f, pod, "with-resource-0", true, req1ExpectedEnv...)

			req2ExpectedEnv := []string{
				"claim_external_multiclaim_request_2",
				"true",
			}
			req2ExpectedEnv = append(req2ExpectedEnv, req2Env...)
			req2ExpectedEnv = append(req2ExpectedEnv, driver2Env...)
			testContainerEnv(ctx, f, pod, "with-resource-1", true, req2ExpectedEnv...)
		})
	}

	v1beta2Tests := func() {
		nodes := NewNodes(f, 1, 1)
		maxAllocations := 1
		driver := NewDriver(f, nodes, driverResources(maxAllocations))
		b := newBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.classParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.parametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports simple ResourceClaim", func(ctx context.Context) {
			pod, template := b.podInlineWithV1beta1()
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod, expectedEnv...)
		})

		f.It("supports requests with alternatives", f.WithFeatureGate(features.DRAPrioritizedList), func(ctx context.Context) {
			claimName := "external-multiclaim"
			parameters, _ := b.parametersEnv()
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: claimName,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{{
							Name: "request-1",
							FirstAvailable: []resourceapi.DeviceSubRequest{
								{
									Name:            "sub-request-1",
									DeviceClassName: b.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           2,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b.className(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						}},
						Config: []resourceapi.DeviceClaimConfiguration{{
							DeviceConfiguration: resourceapi.DeviceConfiguration{
								Opaque: &resourceapi.OpaqueDeviceConfiguration{
									Driver: b.driver.Name,
									Parameters: runtime.RawExtension{
										Raw: []byte(parameters),
									},
								},
							},
						}},
					},
				},
			}
			pod := b.podExternal()
			podClaimName := "resource-claim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &claimName,
				},
			}
			b.create(ctx, claim, pod)
			b.testPod(ctx, f, pod, expectedEnv...)

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(1))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
		})
	}

	partitionableDevicesTests := func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, toDriverResources(
			[]resourceapi.CounterSet{
				{
					Name: "counter-1",
					Counters: map[string]resourceapi.Counter{
						"memory": {
							Value: resource.MustParse("6Gi"),
						},
					},
				},
			},
			[]resourceapi.Device{
				{
					Name: "device-1",
					ConsumesCounters: []resourceapi.DeviceCounterConsumption{
						{
							CounterSet: "counter-1",
							Counters: map[string]resourceapi.Counter{
								"memory": {
									Value: resource.MustParse("4Gi"),
								},
							},
						},
					},
				},
				{
					Name: "device-2",
					ConsumesCounters: []resourceapi.DeviceCounterConsumption{
						{
							CounterSet: "counter-1",
							Counters: map[string]resourceapi.Counter{
								"memory": {
									Value: resource.MustParse("4Gi"),
								},
							},
						},
					},
				},
			}...,
		))
		b := newBuilder(f, driver)

		f.It("must consume and free up counters", feature.DRAPartitionableDevices, func(ctx context.Context) {
			// The first pod will use one of the devices. Since both devices are
			// available, there should be sufficient counters left to allocate
			// a device.
			claim := b.externalClaim()
			pod := b.podExternal()
			pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
			b.create(ctx, claim, pod)
			b.testPod(ctx, f, pod)

			// For the second pod, there should not be sufficient counters left, so
			// it should not succeed. This means the pod should remain in the pending state.
			claim2 := b.externalClaim()
			pod2 := b.podExternal()
			pod2.Spec.ResourceClaims[0].ResourceClaimName = &claim2.Name
			b.create(ctx, claim2, pod2)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				testPod, err := b.f.ClientSet.CoreV1().Pods(pod2.Namespace).Get(ctx, pod2.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the test pod %s to exist: %w", pod2.Name, err)
				}
				if testPod.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s: unexpected status %s, expected status: %s", pod2.Name, testPod.Status.Phase, v1.PodPending)
				}
				return nil
			}, 20*time.Second, 200*time.Millisecond).Should(gomega.Succeed())

			// Delete the first pod
			b.deletePodAndWaitForNotFound(ctx, pod)

			// There shoud not be available devices for pod2.
			b.testPod(ctx, f, pod2)
		})
	}

	ginkgo.Context("on single node", singleNodeTests)

	ginkgo.Context("on multiple nodes", multiNodeTests)

	framework.Context(f.WithFeatureGate(features.DRAPrioritizedList), prioritizedListTests)

	ginkgo.Context("with v1beta2 API", v1beta2Tests)

	ginkgo.Context("with partitionable devices", partitionableDevicesTests)

	framework.Context(f.WithFeatureGate(features.DRADeviceTaints), func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources(10, false), taintAllDevices(resourceapi.DeviceTaint{
			Key:    "example.com/taint",
			Value:  "tainted",
			Effect: resourceapi.DeviceTaintEffectNoSchedule,
		}))
		b := newBuilder(f, driver)

		f.It("DeviceTaint keeps pod pending", func(ctx context.Context) {
			pod, template := b.podInline()
			b.create(ctx, pod, template)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
		})

		f.It("DeviceToleration enables pod scheduling", func(ctx context.Context) {
			pod, template := b.podInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.Tolerations = []resourceapi.DeviceToleration{{
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
				Operator: resourceapi.DeviceTolerationOpExists,
				// No key: tolerate *all* taints with this effect.
			}}
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod)
		})

		f.It("DeviceTaintRule evicts pod", func(ctx context.Context) {
			pod, template := b.podInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.Tolerations = []resourceapi.DeviceToleration{{
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
				Operator: resourceapi.DeviceTolerationOpExists,
				// No key: tolerate *all* taints with this effect.
			}}
			// Add a finalizer to ensure that we get a chance to test the pod status after eviction (= deletion).
			pod.Finalizers = []string{"e2e-test/dont-delete-me"}
			b.create(ctx, pod, template)
			b.testPod(ctx, f, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				// Unblock shutdown by removing the finalizer.
				pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "get pod")
				pod.Finalizers = nil
				_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(ctx, pod, metav1.UpdateOptions{})
				framework.ExpectNoError(err, "remove finalizers from pod")
			})

			// Now evict it.
			ginkgo.By("Evicting pod...")
			taint := &resourcealphaapi.DeviceTaintRule{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "device-taint-rule-" + f.UniqueName + "-",
				},
				Spec: resourcealphaapi.DeviceTaintRuleSpec{
					// All devices of the current driver instance.
					DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
						Driver: &driver.Name,
					},
					Taint: resourcealphaapi.DeviceTaint{
						Effect: resourcealphaapi.DeviceTaintEffectNoExecute,
						Key:    "test.example.com/evict",
						Value:  "now",
						// No TimeAdded, gets defaulted.
					},
				},
			}
			createdTaint := b.create(ctx, taint)
			taint = createdTaint[0].(*resourcealphaapi.DeviceTaintRule)
			gomega.Expect(*taint).Should(gomega.HaveField("Spec.Taint.TimeAdded.Time", gomega.BeTemporally("~", time.Now(), time.Minute /* allow for some clock drift and delays */)))

			framework.ExpectNoError(e2epod.WaitForPodTerminatingInNamespaceTimeout(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart))
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get pod")
			gomega.Expect(pod).Should(gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				// LastTransitionTime is unknown.
				"Type":    gomega.Equal(v1.DisruptionTarget),
				"Status":  gomega.Equal(v1.ConditionTrue),
				"Reason":  gomega.Equal("DeletionByDeviceTaintManager"),
				"Message": gomega.Equal("Device Taint manager: deleting due to NoExecute taint"),
			}))))
		})
	})

	// TODO (https://github.com/kubernetes/kubernetes/issues/123699): move most of the test below into `testDriver` so that they get
	// executed with different parameters.

	ginkgo.Context("ResourceSlice Controller", func() {
		// This is a stress test for creating many large slices.
		// Each slice is as large as API limits allow.
		//
		// Could become a conformance test because it only depends
		// on the apiserver.
		f.It("creates slices", func(ctx context.Context) {
			// Define desired resource slices.
			driverName := f.Namespace.Name
			numSlices := 100
			devicePrefix := "dev-"
			domainSuffix := ".example.com"
			poolName := "network-attached"
			domain := strings.Repeat("x", 63 /* TODO(pohly): add to API */ -len(domainSuffix)) + domainSuffix
			stringValue := strings.Repeat("v", resourceapi.DeviceAttributeMaxValueLength)
			pool := resourceslice.Pool{
				Slices: make([]resourceslice.Slice, numSlices),
			}
			for i := 0; i < numSlices; i++ {
				devices := make([]resourceapi.Device, resourceapi.ResourceSliceMaxDevices)
				for e := 0; e < resourceapi.ResourceSliceMaxDevices; e++ {
					device := resourceapi.Device{
						Name:       devicePrefix + strings.Repeat("x", validation.DNS1035LabelMaxLength-len(devicePrefix)-4) + fmt.Sprintf("%04d", e),
						Attributes: make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice),
					}
					for j := 0; j < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; j++ {
						name := resourceapi.QualifiedName(domain + "/" + strings.Repeat("x", resourceapi.DeviceMaxIDLength-4) + fmt.Sprintf("%04d", j))
						device.Attributes[name] = resourceapi.DeviceAttribute{
							StringValue: &stringValue,
						}
					}
					devices[e] = device
				}
				pool.Slices[i].Devices = devices
			}
			resources := &resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{poolName: pool},
			}

			ginkgo.By("Creating slices")
			mutationCacheTTL := 10 * time.Second
			controller, err := resourceslice.StartController(ctx, resourceslice.Options{
				DriverName:       driverName,
				KubeClient:       f.ClientSet,
				Resources:        resources,
				MutationCacheTTL: &mutationCacheTTL,
			})
			framework.ExpectNoError(err, "start controller")
			ginkgo.DeferCleanup(func(ctx context.Context) {
				controller.Stop()
				err := f.ClientSet.ResourceV1beta2().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
					FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
				})
				framework.ExpectNoError(err, "delete resource slices")
			})

			// Eventually we should have all desired slices.
			listSlices := framework.ListObjects(f.ClientSet.ResourceV1beta2().ResourceSlices().List, metav1.ListOptions{
				FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
			})
			gomega.Eventually(ctx, listSlices).WithTimeout(time.Minute).Should(gomega.HaveField("Items", gomega.HaveLen(numSlices)))

			// Verify state.
			expectSlices, err := listSlices(ctx)
			framework.ExpectNoError(err)
			gomega.Expect(expectSlices.Items).ShouldNot(gomega.BeEmpty())
			framework.Logf("Protobuf size of one slice is %d bytes = %d KB.", expectSlices.Items[0].Size(), expectSlices.Items[0].Size()/1024)
			gomega.Expect(expectSlices.Items[0].Size()).Should(gomega.BeNumerically(">=", 600*1024), "ResourceSlice size")
			gomega.Expect(expectSlices.Items[0].Size()).Should(gomega.BeNumerically("<", 1024*1024), "ResourceSlice size")
			expectStats := resourceslice.Stats{NumCreates: int64(numSlices)}
			gomega.Expect(controller.GetStats()).Should(gomega.Equal(expectStats))

			// No further changes expected now, after after checking again.
			gomega.Consistently(ctx, controller.GetStats).WithTimeout(2 * mutationCacheTTL).Should(gomega.Equal(expectStats))

			// Ask the controller to delete all slices except for one empty slice.
			ginkgo.By("Deleting slices")
			resources = resources.DeepCopy()
			resources.Pools[poolName] = resourceslice.Pool{Slices: []resourceslice.Slice{{}}}
			controller.Update(resources)

			// One empty slice should remain, after removing the full ones and adding the empty one.
			emptySlice := gomega.HaveField("Spec.Devices", gomega.BeEmpty())
			gomega.Eventually(ctx, listSlices).WithTimeout(time.Minute).Should(gomega.HaveField("Items", gomega.ConsistOf(emptySlice)))
			expectStats = resourceslice.Stats{NumCreates: int64(numSlices) + 1, NumDeletes: int64(numSlices)}
			gomega.Consistently(ctx, controller.GetStats).WithTimeout(2 * mutationCacheTTL).Should(gomega.Equal(expectStats))
		})
	})

	ginkgo.Context("cluster", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources(10, false))
		b := newBuilder(f, driver)

		f.It("validate ResourceClaimTemplate and ResourceClaim for admin access", f.WithFeatureGate(features.DRAAdminAccess), func(ctx context.Context) {
			// Attempt to create claim and claim template with admin access. Must fail eventually.
			claim := b.externalClaim()
			claim.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			_, claimTemplate := b.podInline()
			claimTemplate.Spec.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			matchValidationError := gomega.MatchError(gomega.ContainSubstring("admin access to devices requires the `resource.k8s.io/admin-access: true` label on the containing namespace"))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(matchValidationError)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := b.f.ClientSet.ResourceV1beta2().ResourceClaimTemplates(b.f.Namespace.Name).Delete(ctx, claimTemplate.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := b.f.ClientSet.ResourceV1beta2().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(matchValidationError)

			// After labeling the namespace, creation must (eventually...) succeed.
			_, err := b.f.ClientSet.CoreV1().Namespaces().Apply(ctx,
				applyv1.Namespace(b.f.Namespace.Name).WithLabels(map[string]string{"resource.k8s.io/admin-access": "true"}),
				metav1.ApplyOptions{FieldManager: b.f.UniqueName})
			framework.ExpectNoError(err)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := b.f.ClientSet.ResourceV1beta2().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
		})

		ginkgo.It("truncates the name of a generated resource claim", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Name = strings.Repeat("p", 63)
			pod.Spec.ResourceClaims[0].Name = strings.Repeat("c", 63)
			pod.Spec.Containers[0].Resources.Claims[0].Name = pod.Spec.ResourceClaims[0].Name
			b.create(ctx, template, pod)

			b.testPod(ctx, f, pod)
		})

		ginkgo.It("supports count/resourceclaims.resource.k8s.io ResourceQuota", func(ctx context.Context) {
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "claim-0",
					Namespace: f.Namespace.Name,
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{{
							Name: "req-0",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: "my-class",
							},
						}},
					},
				},
			}
			_, err := f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create first claim")

			resourceName := "count/resourceclaims.resource.k8s.io"
			quota := &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "object-count",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{v1.ResourceName(resourceName): resource.MustParse("1")},
				},
			}
			quota, err = f.ClientSet.CoreV1().ResourceQuotas(f.Namespace.Name).Create(ctx, quota, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create resource quota")

			// Eventually the quota status should consider the existing claim.
			gomega.Eventually(ctx, framework.GetObject(f.ClientSet.CoreV1().ResourceQuotas(quota.Namespace).Get, quota.Name, metav1.GetOptions{})).
				Should(gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Status": gomega.Equal(v1.ResourceQuotaStatus{
						Hard: v1.ResourceList{v1.ResourceName(resourceName): resource.MustParse("1")},
						Used: v1.ResourceList{v1.ResourceName(resourceName): resource.MustParse("1")},
					})})))

			// Now creating another claim should eventually fail. The quota may not be enforced immediately if
			// it hasn't yet landed in the API server's cache.
			//
			// If creating a claim erroneously succeeds, we don't want to immediately fail on the next try
			// with an "already exists" error, so use a new name each time.
			claim.GenerateName = "claim-1-"
			claim.Name = ""
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(gomega.MatchError(gomega.ContainSubstring("exceeded quota: object-count, requested: count/resourceclaims.resource.k8s.io=1, used: count/resourceclaims.resource.k8s.io=1, limited: count/resourceclaims.resource.k8s.io=1")), "creating second claim not allowed")
		})

		f.It("DaemonSet with admin access", f.WithFeatureGate(features.DRAAdminAccess), func(ctx context.Context) {
			// Ensure namespace has the dra admin label.
			_, err := b.f.ClientSet.CoreV1().Namespaces().Apply(ctx,
				applyv1.Namespace(b.f.Namespace.Name).WithLabels(map[string]string{"resource.k8s.io/admin-access": "true"}),
				metav1.ApplyOptions{FieldManager: b.f.UniqueName})
			framework.ExpectNoError(err)

			pod, template := b.podInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			// Limit the daemon set to the one node where we have the driver.
			nodeName := nodes.NodeNames[0]
			pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": nodeName}
			pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			daemonSet := &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "monitoring-ds",
					Namespace: b.f.Namespace.Name,
				},
				Spec: appsv1.DaemonSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "monitoring"},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"app": "monitoring"},
						},
						Spec: pod.Spec,
					},
				},
			}

			created := b.create(ctx, template, daemonSet)
			if !ptr.Deref(created[0].(*resourceapi.ResourceClaimTemplate).Spec.Spec.Devices.Requests[0].Exactly.AdminAccess, false) {
				framework.Fail("AdminAccess field was cleared. This test depends on the DRAAdminAccess feature.")
			}
			ds := created[1].(*appsv1.DaemonSet)

			gomega.Eventually(ctx, func(ctx context.Context) (bool, error) {
				return e2edaemonset.CheckDaemonPodOnNodes(f, ds, []string{nodeName})(ctx)
			}).WithTimeout(f.Timeouts.PodStart).Should(gomega.BeTrueBecause("DaemonSet pod should be running on node %s but isn't", nodeName))
			framework.ExpectNoError(e2edaemonset.CheckDaemonStatus(ctx, f, daemonSet.Name))
		})
	})

	ginkgo.Context("cluster", func() {
		nodes := NewNodes(f, 1, 4)
		driver := NewDriver(f, nodes, driverResources(1))

		f.It("must apply per-node permission checks", func(ctx context.Context) {
			// All of the operations use the client set of a kubelet plugin for
			// a fictional node which both don't exist, so nothing interferes
			// when we actually manage to create a slice.
			fictionalNodeName := "dra-fictional-node"
			gomega.Expect(nodes.NodeNames).NotTo(gomega.ContainElement(fictionalNodeName))
			fictionalNodeClient := driver.impersonateKubeletPlugin(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fictionalNodeName + "-dra-plugin",
					Namespace: f.Namespace.Name,
					UID:       "12345",
				},
				Spec: v1.PodSpec{
					NodeName: fictionalNodeName,
				},
			})

			// This is for some actual node in the cluster.
			realNodeName := nodes.NodeNames[0]
			realNodeClient := driver.Nodes[realNodeName].ClientSet

			// This is the slice that we try to create. It needs to be deleted
			// after testing, if it still exists at that time.
			fictionalNodeSlice := &resourceapi.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name: fictionalNodeName + "-slice",
				},
				Spec: resourceapi.ResourceSliceSpec{
					NodeName: ptr.To(fictionalNodeName),
					Driver:   "dra.example.com",
					Pool: resourceapi.ResourcePool{
						Name:               "some-pool",
						ResourceSliceCount: 1,
					},
				},
			}
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.ClientSet.ResourceV1beta2().ResourceSlices().Delete(ctx, fictionalNodeSlice.Name, metav1.DeleteOptions{})
				if !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err)
				}
			})

			// Messages from test-driver/deploy/example/plugin-permissions.yaml
			matchVAPDeniedError := func(nodeName string, slice *resourceapi.ResourceSlice) types.GomegaMatcher {
				subStr := fmt.Sprintf("this user running on node '%s' may not modify ", nodeName)
				switch {
				case ptr.Deref(slice.Spec.NodeName, "") != "":
					subStr += fmt.Sprintf("resourceslices on node '%s'", *slice.Spec.NodeName)
				default:
					subStr += "cluster resourceslices"
				}
				return gomega.MatchError(gomega.ContainSubstring(subStr))
			}
			mustCreate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				ginkgo.GinkgoHelper()
				slice, err := clientSet.ResourceV1beta2().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				ginkgo.GinkgoHelper()
				slice, err := clientSet.ResourceV1beta2().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1beta2().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("DELETE: %s + %s", clientName, slice.Name))
			}
			mustCreateAndDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				slice = mustCreate(clientSet, clientName, slice)
				mustDelete(clientSet, clientName, slice)
			}
			mustFailToCreate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1beta2().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
			}
			mustFailToUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1beta2().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
			}
			mustFailToDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1beta2().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("DELETE: %s + %s", clientName, slice.Name))
			}

			// Create with different clients, keep it in the end.
			mustFailToCreate(realNodeClient, "real plugin", fictionalNodeSlice, matchVAPDeniedError(realNodeName, fictionalNodeSlice))
			mustCreateAndDelete(fictionalNodeClient, "fictional plugin", fictionalNodeSlice)
			createdFictionalNodeSlice := mustCreate(f.ClientSet, "admin", fictionalNodeSlice)

			// Update with different clients.
			mustFailToUpdate(realNodeClient, "real plugin", createdFictionalNodeSlice, matchVAPDeniedError(realNodeName, createdFictionalNodeSlice))
			createdFictionalNodeSlice = mustUpdate(fictionalNodeClient, "fictional plugin", createdFictionalNodeSlice)
			createdFictionalNodeSlice = mustUpdate(f.ClientSet, "admin", createdFictionalNodeSlice)

			// Delete with different clients.
			mustFailToDelete(realNodeClient, "real plugin", createdFictionalNodeSlice, matchVAPDeniedError(realNodeName, createdFictionalNodeSlice))
			mustDelete(fictionalNodeClient, "fictional plugin", createdFictionalNodeSlice)

			// Now the same for a slice which is not associated with a node.
			clusterSlice := &resourceapi.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name: "cluster-slice",
				},
				Spec: resourceapi.ResourceSliceSpec{
					AllNodes: ptr.To(true),
					Driver:   "another.example.com",
					Pool: resourceapi.ResourcePool{
						Name:               "cluster-pool",
						ResourceSliceCount: 1,
					},
				},
			}
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.ClientSet.ResourceV1beta2().ResourceSlices().Delete(ctx, clusterSlice.Name, metav1.DeleteOptions{})
				if !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err)
				}
			})

			// Create with different clients, keep it in the end.
			mustFailToCreate(realNodeClient, "real plugin", clusterSlice, matchVAPDeniedError(realNodeName, clusterSlice))
			mustFailToCreate(fictionalNodeClient, "fictional plugin", clusterSlice, matchVAPDeniedError(fictionalNodeName, clusterSlice))
			createdClusterSlice := mustCreate(f.ClientSet, "admin", clusterSlice)

			// Update with different clients.
			mustFailToUpdate(realNodeClient, "real plugin", createdClusterSlice, matchVAPDeniedError(realNodeName, createdClusterSlice))
			mustFailToUpdate(fictionalNodeClient, "fictional plugin", createdClusterSlice, matchVAPDeniedError(fictionalNodeName, createdClusterSlice))
			createdClusterSlice = mustUpdate(f.ClientSet, "admin", createdClusterSlice)

			// Delete with different clients.
			mustFailToDelete(realNodeClient, "real plugin", createdClusterSlice, matchVAPDeniedError(realNodeName, createdClusterSlice))
			mustFailToDelete(fictionalNodeClient, "fictional plugin", createdClusterSlice, matchVAPDeniedError(fictionalNodeName, createdClusterSlice))
			mustDelete(f.ClientSet, "admin", createdClusterSlice)
		})

		f.It("must manage ResourceSlices", func(ctx context.Context) {
			driverName := driver.Name

			// Now check for exactly the right set of objects for all nodes.
			ginkgo.By("check if ResourceSlice object(s) exist on the API server")
			resourceClient := f.ClientSet.ResourceV1beta2().ResourceSlices()
			var expectedObjects []any
			for _, nodeName := range nodes.NodeNames {
				node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
				framework.ExpectNoError(err, "get node")
				expectedObjects = append(expectedObjects,
					gstruct.MatchAllFields(gstruct.Fields{
						"TypeMeta": gstruct.Ignore(),
						"ObjectMeta": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"OwnerReferences": gomega.ContainElements(
								gstruct.MatchAllFields(gstruct.Fields{
									"APIVersion":         gomega.Equal("v1"),
									"Kind":               gomega.Equal("Node"),
									"Name":               gomega.Equal(nodeName),
									"UID":                gomega.Equal(node.UID),
									"Controller":         gomega.Equal(ptr.To(true)),
									"BlockOwnerDeletion": gomega.BeNil(),
								}),
							),
						}),
						// Ignoring some fields, like SharedCounters, because we don't run this test
						// for PRs (it's slow) and don't want CI breaks when fields get added or renamed.
						"Spec": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Driver":       gomega.Equal(driver.Name),
							"NodeName":     gomega.Equal(ptr.To(nodeName)),
							"NodeSelector": gomega.BeNil(),
							"AllNodes":     gomega.BeNil(),
							"Pool": gstruct.MatchAllFields(gstruct.Fields{
								"Name":               gomega.Equal(nodeName),
								"Generation":         gstruct.Ignore(),
								"ResourceSliceCount": gomega.Equal(int64(1)),
							}),
							"Devices": gomega.Equal([]resourceapi.Device{{Name: "device-00"}}),
						}),
					}),
				)
			}
			matchSlices := gomega.ContainElements(expectedObjects...)
			getSlices := func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
				slices, err := resourceClient.List(ctx, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName})
				if err != nil {
					return nil, err
				}
				return slices.Items, nil
			}
			gomega.Eventually(ctx, getSlices).WithTimeout(20 * time.Second).Should(matchSlices)
			gomega.Consistently(ctx, getSlices).WithTimeout(20 * time.Second).Should(matchSlices)

			// Removal of node resource slice is tested by the general driver removal code.
		})
	})

	multipleDrivers := func(nodeV1beta1 bool) {
		nodes := NewNodes(f, 1, 4)
		driver1 := NewDriver(f, nodes, driverResources(2))
		driver1.NodeV1beta1 = nodeV1beta1
		b1 := newBuilder(f, driver1)

		driver2 := NewDriver(f, nodes, driverResources(2))
		driver2.NodeV1beta1 = nodeV1beta1
		driver2.NameSuffix = "-other"
		b2 := newBuilder(f, driver2)

		ginkgo.It("work", func(ctx context.Context) {
			claim1 := b1.externalClaim()
			claim1b := b1.externalClaim()
			claim2 := b2.externalClaim()
			claim2b := b2.externalClaim()
			pod := b1.podExternal()
			for i, claim := range []*resourceapi.ResourceClaim{claim1b, claim2, claim2b} {
				claim := claim
				pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims,
					v1.PodResourceClaim{
						Name:              fmt.Sprintf("claim%d", i+1),
						ResourceClaimName: &claim.Name,
					},
				)
			}
			b1.create(ctx, claim1, claim1b, claim2, claim2b, pod)
			b1.testPod(ctx, f, pod)
		})
	}
	multipleDriversContext := func(prefix string, nodeV1beta1 bool) {
		ginkgo.Context(prefix, func() {
			multipleDrivers(nodeV1beta1)
		})
	}

	ginkgo.Context("multiple drivers", func() {
		multipleDriversContext("using only drapbv1beta1", true)
	})

	ginkgo.It("runs pod after driver starts", func(ctx context.Context) {
		nodes := NewNodesNow(ctx, f, 1, 4)
		driver := NewDriverInstance(f)
		b := newBuilderNow(ctx, f, driver)

		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)

		// Cannot run pod, no devices.
		framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

		// Set up driver, which makes devices available.
		driver.Run(nodes, driverResourcesNow(nodes, 1))

		// Now it should run.
		b.testPod(ctx, f, pod)

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		// framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
	})

	ginkgo.It("rolling update", func(ctx context.Context) {
		nodes := NewNodesNow(ctx, f, 1, 1)

		oldDriver := NewDriverInstance(f)
		oldDriver.InstanceSuffix = "-old"
		oldDriver.RollingUpdate = true
		oldDriver.Run(nodes, driverResourcesNow(nodes, 1))

		// We expect one ResourceSlice per node from the driver.
		getSlices := oldDriver.NewGetSlices()
		gomega.Eventually(ctx, getSlices).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))
		initialSlices, err := getSlices(ctx)
		framework.ExpectNoError(err)

		// Same driver name, different socket paths because of rolling update.
		newDriver := NewDriverInstance(f)
		newDriver.InstanceSuffix = "-new"
		newDriver.RollingUpdate = true
		newDriver.Run(nodes, driverResourcesNow(nodes, 1))

		// Stop old driver instance.
		oldDriver.TearDown(ctx)

		// Build behaves the same for both driver instances.
		b := newBuilderNow(ctx, f, oldDriver)
		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		// The exact same slices should still exist.
		finalSlices, err := getSlices(ctx)
		framework.ExpectNoError(err)
		gomega.Expect(finalSlices.Items).Should(gomega.Equal(initialSlices.Items))

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		// framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
	})

	ginkgo.It("failed update", func(ctx context.Context) {
		nodes := NewNodesNow(ctx, f, 1, 1)

		oldDriver := NewDriverInstance(f)
		oldDriver.InstanceSuffix = "-old"
		oldDriver.RollingUpdate = true
		oldDriver.Run(nodes, driverResourcesNow(nodes, 1))

		// We expect one ResourceSlice per node from the driver.
		getSlices := oldDriver.NewGetSlices()
		gomega.Eventually(ctx, getSlices).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))
		initialSlices, err := getSlices(ctx)
		framework.ExpectNoError(err)

		// Same driver name, different socket paths because of rolling update.
		newDriver := NewDriverInstance(f)
		newDriver.InstanceSuffix = "-new"
		newDriver.RollingUpdate = true
		newDriver.Run(nodes, driverResourcesNow(nodes, 1))

		// Stop new driver instance, simulating the failure of the new instance.
		// The kubelet should still have the old instance.
		newDriver.TearDown(ctx)

		// Build behaves the same for both driver instances.
		b := newBuilderNow(ctx, f, oldDriver)
		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		// The exact same slices should still exist.
		finalSlices, err := getSlices(ctx)
		framework.ExpectNoError(err)
		gomega.Expect(finalSlices.Items).Should(gomega.Equal(initialSlices.Items))

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		// framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
	})

	f.It("sequential update with pods replacing each other", framework.WithSlow(), func(ctx context.Context) {
		nodes := NewNodesNow(ctx, f, 1, 1)

		// Same driver name, same socket path.
		oldDriver := NewDriverInstance(f)
		oldDriver.InstanceSuffix = "-old"
		oldDriver.Run(nodes, driverResourcesNow(nodes, 1))

		// Collect set of resource slices for that driver.
		listSlices := framework.ListObjects(f.ClientSet.ResourceV1beta2().ResourceSlices().List, metav1.ListOptions{
			FieldSelector: "spec.driver=" + oldDriver.Name,
		})
		gomega.Eventually(ctx, listSlices).Should(gomega.HaveField("Items", gomega.Not(gomega.BeEmpty())), "driver should have published ResourceSlices, got none")
		oldSlices, err := listSlices(ctx)
		framework.ExpectNoError(err, "list slices published by old driver")
		if len(oldSlices.Items) == 0 {
			framework.Fail("driver should have published ResourceSlices, got none")
		}

		// "Update" the driver by taking it down and bringing up a new one.
		// Pods never run in parallel, similar to how a DaemonSet would update
		// its pods when maxSurge is zero.
		ginkgo.By("reinstall driver")
		start := time.Now()
		oldDriver.TearDown(ctx)
		newDriver := NewDriverInstance(f)
		newDriver.InstanceSuffix = "-new"
		newDriver.Run(nodes, driverResourcesNow(nodes, 1))
		updateDuration := time.Since(start)

		// Build behaves the same for both driver instances.
		b := newBuilderNow(ctx, f, oldDriver)
		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		// The slices should have survived the update, but only if it happened
		// quickly enough. If it took too long (= wipingDelay of 30 seconds in pkg/kubelet/cm/dra/manager.go,
		// https://github.com/kubernetes/kubernetes/blob/03763fd1abdf0f5d3dfceb3a6b138bb643e37411/pkg/kubelet/cm/dra/manager.go#L113),
		// the kubelet considered the driver gone and removed them.
		if updateDuration <= 25*time.Second {
			framework.Logf("Checking resource slices after downtime of %s.", updateDuration)
			newSlices, err := listSlices(ctx)
			framework.ExpectNoError(err, "list slices again")
			gomega.Expect(newSlices.Items).To(gomega.ConsistOf(oldSlices.Items), "Old slice should have survived a downtime of %s.", updateDuration)
		} else {
			framework.Logf("Not checking resource slices, downtime was too long with %s.", updateDuration)
		}

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		// framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))

		// Now shut down for good and wait for the kubelet to react.
		// This takes time...
		ginkgo.By("uninstalling driver and waiting for ResourceSlice wiping")
		newDriver.TearDown(ctx)
		newDriver.IsGone(ctx)
	})
})

// builder contains a running counter to make objects unique within thir
// namespace.
type builder struct {
	f      *framework.Framework
	driver *Driver

	podCounter      int
	claimCounter    int
	classParameters string // JSON
}

// className returns the default device class name.
func (b *builder) className() string {
	return b.f.UniqueName + b.driver.NameSuffix + "-class"
}

// class returns the device class that the builder's other objects
// reference.
func (b *builder) class() *resourceapi.DeviceClass {
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: b.className(),
		},
	}
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf(`device.driver == "%s"`, b.driver.Name),
		},
	}}
	if b.classParameters != "" {
		class.Spec.Config = []resourceapi.DeviceClassConfiguration{{
			DeviceConfiguration: resourceapi.DeviceConfiguration{
				Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver:     b.driver.Name,
					Parameters: runtime.RawExtension{Raw: []byte(b.classParameters)},
				},
			},
		}}
	}
	return class
}

// externalClaim returns external resource claim
// that test pods can reference
func (b *builder) externalClaim() *resourceapi.ResourceClaim {
	b.claimCounter++
	name := "external-claim" + b.driver.NameSuffix // This is what podExternal expects.
	if b.claimCounter > 1 {
		name += fmt.Sprintf("-%d", b.claimCounter)
	}
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: b.claimSpec(),
	}
}

// claimSpec returns the device request for a claim or claim template
// with the associated config using the v1beta1 API.
func (b *builder) claimSpecWithV1beta1() resourcev1beta1.ResourceClaimSpec {
	parameters, _ := b.parametersEnv()
	spec := resourcev1beta1.ResourceClaimSpec{
		Devices: resourcev1beta1.DeviceClaim{
			Requests: []resourcev1beta1.DeviceRequest{{
				Name:            "my-request",
				DeviceClassName: b.className(),
			}},
			Config: []resourcev1beta1.DeviceClaimConfiguration{{
				DeviceConfiguration: resourcev1beta1.DeviceConfiguration{
					Opaque: &resourcev1beta1.OpaqueDeviceConfiguration{
						Driver: b.driver.Name,
						Parameters: runtime.RawExtension{
							Raw: []byte(parameters),
						},
					},
				},
			}},
		},
	}

	return spec
}

// claimSpecWithV1beta2 returns the device request for a claim or claim template
// with the associated config using the latest API.
func (b *builder) claimSpec() resourceapi.ResourceClaimSpec {
	parameters, _ := b.parametersEnv()
	spec := resourceapi.ResourceClaimSpec{
		Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name: "my-request",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: b.className(),
				},
			}},
			Config: []resourceapi.DeviceClaimConfiguration{{
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver: b.driver.Name,
						Parameters: runtime.RawExtension{
							Raw: []byte(parameters),
						},
					},
				},
			}},
		},
	}

	return spec
}

// parametersEnv returns the default user env variables as JSON (config) and key/value list (pod env).
func (b *builder) parametersEnv() (string, []string) {
	return `{"a":"b"}`,
		[]string{"user_a", "b"}
}

// makePod returns a simple pod with no resource claims.
// The pod prints its env and waits.
func (b *builder) pod() *v1.Pod {
	// The e2epod.InfiniteSleepCommand was changed so that it reacts to SIGTERM,
	// causing the pod to shut down immediately. This is better than the previous approach
	// with `terminationGraceperiodseconds: 1` because that still caused a one second delay.
	//
	// It is tempting to use `terminationGraceperiodSeconds: 0`, but that is a very bad
	// idea because it removes the pod before the kubelet had a chance to react (https://github.com/kubernetes/kubernetes/issues/120671).
	pod := e2epod.MakePod(b.f.Namespace.Name, nil, nil, b.f.NamespacePodSecurityLevel, "" /* no command = pause */)
	pod.Labels = make(map[string]string)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	pod.ObjectMeta.GenerateName = ""
	b.podCounter++
	pod.ObjectMeta.Name = fmt.Sprintf("tester%s-%d", b.driver.NameSuffix, b.podCounter)
	return pod
}

// makePodInline adds an inline resource claim with default class name and parameters.
func (b *builder) podInline() (*v1.Pod, *resourceapi.ResourceClaimTemplate) {
	pod := b.pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "my-inline-claim"
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:                      podClaimName,
			ResourceClaimTemplateName: ptr.To(pod.Name),
		},
	}
	template := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: b.claimSpec(),
		},
	}
	return pod, template
}

func (b *builder) podInlineWithV1beta1() (*v1.Pod, *resourcev1beta1.ResourceClaimTemplate) {
	pod, _ := b.podInline()
	template := &resourcev1beta1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourcev1beta1.ResourceClaimTemplateSpec{
			Spec: b.claimSpecWithV1beta1(),
		},
	}
	return pod, template
}

// podInlineMultiple returns a pod with inline resource claim referenced by 3 containers
func (b *builder) podInlineMultiple() (*v1.Pod, *resourceapi.ResourceClaimTemplate) {
	pod, template := b.podInline()
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name = pod.Spec.Containers[1].Name + "-1"
	pod.Spec.Containers[2].Name = pod.Spec.Containers[1].Name + "-2"
	return pod, template
}

// podExternal adds a pod that references external resource claim with default class name and parameters.
func (b *builder) podExternal() *v1.Pod {
	pod := b.pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "resource-claim"
	externalClaimName := "external-claim" + b.driver.NameSuffix
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              podClaimName,
			ResourceClaimName: &externalClaimName,
		},
	}
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	return pod
}

// podShared returns a pod with 3 containers that reference external resource claim with default class name and parameters.
func (b *builder) podExternalMultiple() *v1.Pod {
	pod := b.podExternal()
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name = pod.Spec.Containers[1].Name + "-1"
	pod.Spec.Containers[2].Name = pod.Spec.Containers[1].Name + "-2"
	return pod
}

// create takes a bunch of objects and calls their Create function.
func (b *builder) create(ctx context.Context, objs ...klog.KMetadata) []klog.KMetadata {
	var createdObjs []klog.KMetadata
	for _, obj := range objs {
		ginkgo.By(fmt.Sprintf("creating %T %s", obj, obj.GetName()))
		var err error
		var createdObj klog.KMetadata
		switch obj := obj.(type) {
		case *resourceapi.DeviceClass:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().DeviceClasses().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1beta2().DeviceClasses().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete device class")
			})
		case *v1.Pod:
			createdObj, err = b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *v1.ConfigMap:
			createdObj, err = b.f.ClientSet.CoreV1().ConfigMaps(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1beta1().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1beta1().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceSlice:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().ResourceSlices().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1beta2().ResourceSlices().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete node resource slice")
			})
		case *resourcealphaapi.DeviceTaintRule:
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().DeviceTaintRules().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha3().DeviceTaintRules().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete DeviceTaintRule")
			})
		case *appsv1.DaemonSet:
			createdObj, err = b.f.ClientSet.AppsV1().DaemonSets(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			// Cleanup not really needed, but speeds up namespace shutdown.
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.AppsV1().DaemonSets(b.f.Namespace.Name).Delete(ctx, obj.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete daemonset")
			})
		default:
			framework.Fail(fmt.Sprintf("internal error, unsupported type %T", obj), 1)
		}
		framework.ExpectNoErrorWithOffset(1, err, "create %T", obj)
		createdObjs = append(createdObjs, createdObj)
	}
	return createdObjs
}

func (b *builder) deletePodAndWaitForNotFound(ctx context.Context, pod *v1.Pod) {
	err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	framework.ExpectNoErrorWithOffset(1, err, "delete %T", pod)
	err = e2epod.WaitForPodNotFoundInNamespace(ctx, b.f.ClientSet, pod.Name, pod.Namespace, b.f.Timeouts.PodDelete)
	framework.ExpectNoErrorWithOffset(1, err, "terminate %T", pod)
}

// testPod runs pod and checks if container logs contain expected environment variables
func (b *builder) testPod(ctx context.Context, f *framework.Framework, pod *v1.Pod, env ...string) {
	ginkgo.GinkgoHelper()
	err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "start pod")

	if len(env) == 0 {
		_, env = b.parametersEnv()
	}
	for _, container := range pod.Spec.Containers {
		testContainerEnv(ctx, f, pod, container.Name, false, env...)
	}
}

// envLineRE matches env output with variables set by test/e2e/dra/test-driver.
var envLineRE = regexp.MustCompile(`^(?:admin|user|claim)_[a-zA-Z0-9_]*=.*$`)

func testContainerEnv(ctx context.Context, f *framework.Framework, pod *v1.Pod, containerName string, fullMatch bool, env ...string) {
	ginkgo.GinkgoHelper()
	stdout, stderr, err := e2epod.ExecWithOptionsContext(ctx, f, e2epod.ExecOptions{
		Command:       []string{"env"},
		Namespace:     pod.Namespace,
		PodName:       pod.Name,
		ContainerName: containerName,
		CaptureStdout: true,
		CaptureStderr: true,
		Quiet:         true,
	})
	framework.ExpectNoError(err, fmt.Sprintf("get env output for container %s", containerName))
	gomega.Expect(stderr).To(gomega.BeEmpty(), fmt.Sprintf("env stderr for container %s", containerName))
	if fullMatch {
		// Find all env variables set by the test driver.
		var actualEnv, expectEnv []string
		for _, line := range strings.Split(stdout, "\n") {
			if envLineRE.MatchString(line) {
				actualEnv = append(actualEnv, line)
			}
		}
		for i := 0; i < len(env); i += 2 {
			expectEnv = append(expectEnv, env[i]+"="+env[i+1])
		}
		sort.Strings(actualEnv)
		sort.Strings(expectEnv)
		gomega.Expect(actualEnv).To(gomega.Equal(expectEnv), fmt.Sprintf("container %s env output:\n%s", containerName, stdout))
	} else {
		for i := 0; i < len(env); i += 2 {
			envStr := fmt.Sprintf("\n%s=%s\n", env[i], env[i+1])
			gomega.Expect(stdout).To(gomega.ContainSubstring(envStr), fmt.Sprintf("container %s env variables", containerName))
		}
	}
}

func newBuilder(f *framework.Framework, driver *Driver) *builder {
	b := &builder{f: f, driver: driver}
	ginkgo.BeforeEach(b.setUp)
	return b
}

func newBuilderNow(ctx context.Context, f *framework.Framework, driver *Driver) *builder {
	b := &builder{f: f, driver: driver}
	b.setUp(ctx)
	return b
}

func (b *builder) setUp(ctx context.Context) {
	b.podCounter = 0
	b.claimCounter = 0
	b.create(ctx, b.class())
	ginkgo.DeferCleanup(b.tearDown)
}

func (b *builder) tearDown(ctx context.Context) {
	// Before we allow the namespace and all objects in it do be deleted by
	// the framework, we must ensure that test pods and the claims that
	// they use are deleted. Otherwise the driver might get deleted first,
	// in which case deleting the claims won't work anymore.
	ginkgo.By("delete pods and claims")
	pods, err := b.listTestPods(ctx)
	framework.ExpectNoError(err, "list pods")
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &pod, klog.KObj(&pod)))
		err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete pod")
		}
	}
	gomega.Eventually(func() ([]v1.Pod, error) {
		return b.listTestPods(ctx)
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "remaining pods despite deletion")

	claims, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &claim, klog.KObj(&claim)))
		err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete claim")
		}
	}

	for host, plugin := range b.driver.Nodes {
		ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
		gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
	}

	ginkgo.By("waiting for claims to be deallocated and deleted")
	gomega.Eventually(func() ([]resourceapi.ResourceClaim, error) {
		claims, err := b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		return claims.Items, nil
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "claims in the namespaces")
}

func (b *builder) listTestPods(ctx context.Context) ([]v1.Pod, error) {
	pods, err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var testPods []v1.Pod
	for _, pod := range pods.Items {
		if pod.Labels["app.kubernetes.io/part-of"] == "dra-test-driver" {
			continue
		}
		testPods = append(testPods, pod)
	}
	return testPods, nil
}

func taintAllDevices(taints ...resourceapi.DeviceTaint) driverResourcesMutatorFunc {
	return func(resources map[string]resourceslice.DriverResources) {
		for i := range resources {
			for j := range resources[i].Pools {
				for k := range resources[i].Pools[j].Slices {
					for l := range resources[i].Pools[j].Slices[k].Devices {
						resources[i].Pools[j].Slices[k].Devices[l].Taints = append(resources[i].Pools[j].Slices[k].Devices[l].Taints, taints...)
					}
				}
			}
		}
	}
}

func networkResources(maxAllocations int, tainted bool) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		driverResources := make(map[string]resourceslice.DriverResources)
		devices := make([]resourceapi.Device, 0)
		for i := 0; i < maxAllocations; i++ {
			device := resourceapi.Device{
				Name: fmt.Sprintf("device-%d", i),
			}
			if tainted {
				device.Taints = []resourceapi.DeviceTaint{{
					Key:    "example.com/taint",
					Value:  "tainted",
					Effect: resourceapi.DeviceTaintEffectNoSchedule,
				}}
			}
			devices = append(devices, device)
		}
		driverResources[multiHostDriverResources] = resourceslice.DriverResources{
			Pools: map[string]resourceslice.Pool{
				"network": {
					Slices: []resourceslice.Slice{{
						Devices: devices,
					}},
					NodeSelector: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{{
							// MatchExpressions allow multiple values,
							// MatchFields don't.
							MatchExpressions: []v1.NodeSelectorRequirement{{
								Key:      "kubernetes.io/hostname",
								Operator: v1.NodeSelectorOpIn,
								Values:   nodes.NodeNames,
							}},
						}},
					},
					Generation: 1,
				},
			},
		}
		return driverResources
	}
}

func driverResources(maxAllocations int, devicesPerNode ...map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		return driverResourcesNow(nodes, maxAllocations, devicesPerNode...)
	}
}

func driverResourcesNow(nodes *Nodes, maxAllocations int, devicesPerNode ...map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) map[string]resourceslice.DriverResources {
	driverResources := make(map[string]resourceslice.DriverResources)
	for i, nodename := range nodes.NodeNames {
		if i < len(devicesPerNode) {
			devices := make([]resourceapi.Device, 0)
			for deviceName, attributes := range devicesPerNode[i] {
				devices = append(devices, resourceapi.Device{
					Name:       deviceName,
					Attributes: attributes,
				})
			}
			driverResources[nodename] = resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{{
							Devices: devices,
						}},
					},
				},
			}
		} else if maxAllocations >= 0 {
			devices := make([]resourceapi.Device, maxAllocations)
			for i := 0; i < maxAllocations; i++ {
				devices[i] = resourceapi.Device{
					Name: fmt.Sprintf("device-%02d", i),
				}
			}
			driverResources[nodename] = resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{{
							Devices: devices,
						}},
					},
				},
			}
		}
	}
	return driverResources
}

func toDriverResources(counters []resourceapi.CounterSet, devices ...resourceapi.Device) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		nodename := nodes.NodeNames[0]
		return map[string]resourceslice.DriverResources{
			nodename: {
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{
							{
								SharedCounters: counters,
								Devices:        devices,
							},
						},
					},
				},
			},
		}
	}
}
