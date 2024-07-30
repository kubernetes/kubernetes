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

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	applyv1 "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	// podStartTimeout is how long to wait for the pod to be started.
	podStartTimeout = 5 * time.Minute
)

//go:embed test-driver/deploy/example/admin-access-policy.yaml
var adminAccessPolicyYAML string

// networkResources can be passed to NewDriver directly.
func networkResources() app.Resources {
	return app.Resources{}
}

// perNode returns a function which can be passed to NewDriver. The nodes
// parameter has be instantiated, but not initialized yet, so the returned
// function has to capture it and use it when being called.
func perNode(maxAllocations int, nodes *Nodes) func() app.Resources {
	return func() app.Resources {
		return app.Resources{
			NodeLocal:      true,
			MaxAllocations: maxAllocations,
			Nodes:          nodes.NodeNames,
		}
	}
}

var _ = framework.SIGDescribe("node")("DRA", feature.DynamicResourceAllocation, func() {
	f := framework.NewDefaultFramework("dra")

	// The driver containers have to run with sufficient privileges to
	// modify /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("kubelet", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources)
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

			b.testPod(ctx, f.ClientSet, pod)

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

			testContainerEnv(ctx, f.ClientSet, pod, pod.Spec.Containers[0].Name, true, container0Env...)
			testContainerEnv(ctx, f.ClientSet, pod, pod.Spec.Containers[1].Name, true, container1Env...)
		})
	})

	// claimTests tries out several different combinations of pods with
	// claims, both inline and external.
	claimTests := func(b *builder, driver *Driver) {
		ginkgo.It("supports simple pod referencing inline resource claim", func(ctx context.Context) {
			pod, template := b.podInline()
			b.create(ctx, pod, template)
			b.testPod(ctx, f.ClientSet, pod)
		})

		ginkgo.It("supports inline claim referenced by multiple containers", func(ctx context.Context) {
			pod, template := b.podInlineMultiple()
			b.create(ctx, pod, template)
			b.testPod(ctx, f.ClientSet, pod)
		})

		ginkgo.It("supports simple pod referencing external resource claim", func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod)
			b.testPod(ctx, f.ClientSet, pod)
		})

		ginkgo.It("supports external claim referenced by multiple pods", func(ctx context.Context) {
			pod1 := b.podExternal()
			pod2 := b.podExternal()
			pod3 := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f.ClientSet, pod)
			}
		})

		ginkgo.It("supports external claim referenced by multiple containers of multiple pods", func(ctx context.Context) {
			pod1 := b.podExternalMultiple()
			pod2 := b.podExternalMultiple()
			pod3 := b.podExternalMultiple()
			claim := b.externalClaim()
			b.create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f.ClientSet, pod)
			}
		})

		ginkgo.It("supports init containers", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
			pod.Spec.InitContainers[0].Name += "-init"
			// This must succeed for the pod to start.
			pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
			b.create(ctx, pod, template)

			b.testPod(ctx, f.ClientSet, pod)
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
				return f.ClientSet.ResourceV1alpha3().ResourceClaims(pod.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
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
				claims, err := f.ClientSet.ResourceV1alpha3().ResourceClaims(pod.Namespace).List(ctx, metav1.ListOptions{})
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
			if driver.Controller != nil {
				gomega.Expect(driver.Controller.GetNumAllocations()).To(gomega.Equal(int64(1)), "number of allocations")
			}
		})

		ginkgo.It("must deallocate after use", func(ctx context.Context) {
			pod := b.podExternal()
			claim := b.externalClaim()
			b.create(ctx, claim, pod)

			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))

			b.testPod(ctx, f.ClientSet, pod)

			ginkgo.By(fmt.Sprintf("deleting pod %s", klog.KObj(pod)))
			framework.ExpectNoError(b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))

			ginkgo.By("waiting for claim to get deallocated")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
		})
	}

	singleNodeTests := func(parameterMode parameterMode) {
		nodes := NewNodes(f, 1, 1)
		maxAllocations := 1
		numPods := 10
		generateResources := func() app.Resources {
			resources := perNode(maxAllocations, nodes)()
			return resources
		}
		driver := NewDriver(f, nodes, generateResources) // All tests get their own driver instance.
		driver.parameterMode = parameterMode
		b := newBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.classParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.parametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports claim and class parameters", func(ctx context.Context) {
			pod, template := b.podInline()
			b.create(ctx, pod, template)
			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
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
					b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
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
					b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
				}()
			}
			wg.Wait()
		})

		f.It("supports sharing a claim sequentially", f.WithSlow(), func(ctx context.Context) {
			var objects []klog.KMetadata
			objects = append(objects, b.externalClaim())

			// This test used to test usage of the claim by one pod
			// at a time. After removing the "not sharable"
			// feature, we have to create more pods than supported
			// at the same time to get the same effect.
			numPods := resourceapi.ResourceClaimReservedForMaxSize + 10
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
					b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
					// We need to delete each running pod, otherwise the others cannot use the claim.
					err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err, "delete pod")
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow))
				}()
			}
			wg.Wait()
		})

		ginkgo.It("retries pod scheduling after creating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.podInline()
			deviceClassName := template.Spec.Spec.Devices.Requests[0].DeviceClassName
			class, err := f.ClientSet.ResourceV1alpha3().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			deviceClassName += "-b"
			template.Spec.Spec.Devices.Requests[0].DeviceClassName = deviceClassName
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			class.UID = ""
			class.ResourceVersion = ""
			class.Name = deviceClassName
			b.create(ctx, class)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("retries pod scheduling after updating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.podInline()

			// First modify the class so that it matches no nodes (for classic DRA) and no devices (structured parameters).
			deviceClassName := template.Spec.Spec.Devices.Requests[0].DeviceClassName
			class, err := f.ClientSet.ResourceV1alpha3().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			originalClass := class.DeepCopy()
			switch driver.parameterMode {
			case parameterModeClassicDRA:
				class.Spec.SuitableNodes = &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "no-such-label",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"no-such-value"},
								},
							},
						},
					},
				}
			case parameterModeStructured:
				class.Spec.Selectors = []resourceapi.DeviceSelector{{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: "false",
					},
				}}
			default:
				framework.Failf("unexpected mode: %s", driver.parameterMode)
			}
			class, err = f.ClientSet.ResourceV1alpha3().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// Now create the pod.
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			// Unblock the pod.
			class.Spec.SuitableNodes = originalClass.Spec.SuitableNodes
			class.Spec.Selectors = originalClass.Spec.Selectors
			_, err = f.ClientSet.ResourceV1alpha3().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
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

	// These tests depend on having more than one node and a DRA driver controller.
	multiNodeDRAControllerTests := func(nodes *Nodes) {
		driver := NewDriver(f, nodes, networkResources)
		driver.parameterMode = parameterModeClassicDRA
		b := newBuilder(f, driver)

		ginkgo.It("schedules onto different nodes", func(ctx context.Context) {
			label := "app.kubernetes.io/instance"
			instance := f.UniqueName + "-test-app"
			antiAffinity := &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							TopologyKey: "kubernetes.io/hostname",
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									label: instance,
								},
							},
						},
					},
				},
			}
			createPod := func() *v1.Pod {
				pod := b.podExternal()
				pod.Labels[label] = instance
				pod.Spec.Affinity = antiAffinity
				return pod
			}
			pod1 := createPod()
			pod2 := createPod()
			claim := b.externalClaim()
			b.create(ctx, claim, pod1, pod2)

			for _, pod := range []*v1.Pod{pod1, pod2} {
				err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
				framework.ExpectNoError(err, "start pod")
			}
		})

		// This test covers aspects of non graceful node shutdown by DRA controller
		// More details about this can be found in the KEP:
		// https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/2268-non-graceful-shutdown
		// NOTE: this test depends on kind. It will only work with kind cluster as it shuts down one of the
		// nodes by running `docker stop <node name>`, which is very kind-specific.
		f.It(f.WithSerial(), f.WithDisruptive(), f.WithSlow(), "must deallocate on non graceful node shutdown", func(ctx context.Context) {
			ginkgo.By("create test pod")
			label := "app.kubernetes.io/instance"
			instance := f.UniqueName + "-test-app"
			pod := b.podExternal()
			pod.Labels[label] = instance
			claim := b.externalClaim()
			b.create(ctx, claim, pod)

			ginkgo.By("wait for test pod " + pod.Name + " to run")
			labelSelector := labels.SelectorFromSet(labels.Set(pod.Labels))
			pods, err := e2epod.WaitForPodsWithLabelRunningReady(ctx, f.ClientSet, pod.Namespace, labelSelector, 1, framework.PodStartTimeout)
			framework.ExpectNoError(err, "start pod")
			runningPod := &pods.Items[0]

			nodeName := runningPod.Spec.NodeName
			// Prevent builder tearDown to fail waiting for unprepared resources
			delete(b.driver.Nodes, nodeName)
			ginkgo.By("stop node " + nodeName + " non gracefully")
			_, stderr, err := framework.RunCmd("docker", "stop", nodeName)
			gomega.Expect(stderr).To(gomega.BeEmpty())
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(framework.RunCmd, "docker", "start", nodeName)
			if ok := e2enode.WaitForNodeToBeNotReady(ctx, f.ClientSet, nodeName, f.Timeouts.NodeNotReady); !ok {
				framework.Failf("Node %s failed to enter NotReady state", nodeName)
			}

			ginkgo.By("apply out-of-service taint on node " + nodeName)
			taint := v1.Taint{
				Key:    v1.TaintNodeOutOfService,
				Effect: v1.TaintEffectNoExecute,
			}
			e2enode.AddOrUpdateTaintOnNode(ctx, f.ClientSet, nodeName, taint)
			e2enode.ExpectNodeHasTaint(ctx, f.ClientSet, nodeName, &taint)
			ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName, taint)

			ginkgo.By("waiting for claim to get deallocated")
			gomega.Eventually(ctx, framework.GetObject(b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Get, claim.Name, metav1.GetOptions{})).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", gomega.BeNil()))
		})
	}

	// The following tests only make sense when there is more than one node.
	// They get skipped when there's only one node.
	multiNodeTests := func(parameterMode parameterMode) {
		nodes := NewNodes(f, 2, 8)

		switch parameterMode {
		case parameterModeStructured:
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
				driver := NewDriver(f, nodes, perNode(-1, nodes), devicesPerNode...)
				b := newBuilder(f, driver)

				ginkgo.It("keeps pod pending because of CEL runtime errors", func(ctx context.Context) {
					// When pod scheduling encounters CEL runtime errors for some nodes, but not all,
					// it should still not schedule the pod because there is something wrong with it.
					// Scheduling it would make it harder to detect that there is a problem.
					//
					// This matches the "CEL-runtime-error-for-subset-of-nodes" unit test, except that
					// here we try it in combination with the actual scheduler and can extend it with
					// other checks, like event handling (future extension).

					gomega.Eventually(ctx, framework.ListObjects(f.ClientSet.ResourceV1alpha3().ResourceSlices().List,
						metav1.ListOptions{
							FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driver.Name,
						},
					)).Should(gomega.HaveField("Items", gomega.ConsistOf(
						gomega.HaveField("Spec.Devices", gomega.ConsistOf(
							gomega.Equal(resourceapi.Device{
								Name: firstDevice,
								Basic: &resourceapi.BasicDevice{
									Attributes: devicesPerNode[0][firstDevice],
								},
							}))),
						gomega.HaveField("Spec.Devices", gomega.ConsistOf(
							gomega.Equal(resourceapi.Device{
								Name: secondDevice,
								Basic: &resourceapi.BasicDevice{
									Attributes: devicesPerNode[1][secondDevice],
								},
							}))),
					)))

					pod, template := b.podInline()
					template.Spec.Spec.Devices.Requests[0].Selectors = append(template.Spec.Spec.Devices.Requests[0].Selectors,
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
		case parameterModeClassicDRA:
			ginkgo.Context("with network-attached resources", func() {
				multiNodeDRAControllerTests(nodes)
			})

			ginkgo.Context("reallocation", func() {
				var allocateWrapper2 app.AllocateWrapperType
				driver := NewDriver(f, nodes, perNode(1, nodes))
				driver.parameterMode = parameterModeClassicDRA
				driver2 := NewDriver(f, nodes, func() app.Resources {
					return app.Resources{
						NodeLocal:      true,
						MaxAllocations: 1,
						Nodes:          nodes.NodeNames,

						AllocateWrapper: func(
							ctx context.Context,
							claimAllocations []*controller.ClaimAllocation,
							selectedNode string,
							handler func(
								ctx context.Context,
								claimAllocations []*controller.ClaimAllocation,
								selectedNode string),
						) {
							allocateWrapper2(ctx, claimAllocations, selectedNode, handler)
						},
					}
				})
				driver2.NameSuffix = "-other"
				driver2.parameterMode = parameterModeClassicDRA

				b := newBuilder(f, driver)
				b2 := newBuilder(f, driver2)

				ginkgo.It("works", func(ctx context.Context) {
					// A pod with multiple claims can run on a node, but
					// only if allocation of all succeeds. This
					// test simulates the scenario where one claim
					// gets allocated from one driver, but the claims
					// from second driver fail allocation because of a
					// race with some other pod.
					//
					// To ensure the right timing, allocation of the
					// claims from second driver are delayed while
					// creating another pod that gets the remaining
					// resource on the node from second driver.
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()

					// Order is relevant here: each pod must be matched with its own claim.
					pod1claim1 := b.externalClaim()
					pod1 := b.podExternal()
					pod2claim1 := b2.externalClaim()
					pod2 := b2.podExternal()

					// Add another claim to pod1.
					pod1claim2 := b2.externalClaim()
					pod1.Spec.ResourceClaims = append(pod1.Spec.ResourceClaims,
						v1.PodResourceClaim{
							Name:              "claim-other",
							ResourceClaimName: &pod1claim2.Name,
						},
					)

					// Allocating the second claim in pod1 has to wait until pod2 has
					// consumed the available resources on the node.
					blockClaim, cancelBlockClaim := context.WithCancel(ctx)
					defer cancelBlockClaim()
					allocateWrapper2 = func(ctx context.Context,
						claimAllocations []*controller.ClaimAllocation,
						selectedNode string,
						handler func(ctx context.Context,
							claimAllocations []*controller.ClaimAllocation,
							selectedNode string),
					) {
						if claimAllocations[0].Claim.Name == pod1claim2.Name {
							<-blockClaim.Done()
						}
						handler(ctx, claimAllocations, selectedNode)
					}

					b.create(ctx, pod1claim1, pod1claim2, pod1)

					ginkgo.By("waiting for one claim from driver1 to be allocated")
					var nodeSelector *v1.NodeSelector
					gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
						claims, err := f.ClientSet.ResourceV1alpha3().ResourceClaims(f.Namespace.Name).List(ctx, metav1.ListOptions{})
						if err != nil {
							return 0, err
						}
						allocated := 0
						for _, claim := range claims.Items {
							if claim.Status.Allocation != nil {
								allocated++
								nodeSelector = claim.Status.Allocation.NodeSelector
							}
						}
						return allocated, nil
					}).WithTimeout(time.Minute).Should(gomega.Equal(1), "one claim allocated")

					// Now create a second pod which we force to
					// run on the same node that is currently being
					// considered for the first one. We know what
					// the node selector looks like and can
					// directly access the key and value from it.
					ginkgo.By(fmt.Sprintf("create second pod on the same node %s", nodeSelector))

					req := nodeSelector.NodeSelectorTerms[0].MatchExpressions[0]
					node := req.Values[0]
					pod2.Spec.NodeSelector = map[string]string{req.Key: node}

					b2.create(ctx, pod2claim1, pod2)
					framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod2), "start pod 2")

					// Allow allocation of second claim in pod1 to proceed. It should fail now
					// and the other node must be used instead, after deallocating
					// the first claim.
					ginkgo.By("move first pod to other node")
					cancelBlockClaim()

					framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod1), "start pod 1")
					pod1, err := f.ClientSet.CoreV1().Pods(pod1.Namespace).Get(ctx, pod1.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "get first pod")
					if pod1.Spec.NodeName == "" {
						framework.Fail("first pod should be running on node, was not scheduled")
					}
					gomega.Expect(pod1.Spec.NodeName).ToNot(gomega.Equal(node), "first pod should run on different node than second one")
					gomega.Expect(driver.Controller.GetNumDeallocations()).To(gomega.Equal(int64(1)), "number of deallocations")
				})
			})
		}

		ginkgo.Context("with node-local resources", func() {
			driver := NewDriver(f, nodes, perNode(1, nodes))
			driver.parameterMode = parameterMode
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
	}

	tests := func(parameterMode parameterMode) {
		ginkgo.Context("on single node", func() {
			singleNodeTests(parameterMode)
		})
		ginkgo.Context("on multiple nodes", func() {
			multiNodeTests(parameterMode)
		})
	}

	framework.Context("with classic DRA", feature.DRAControlPlaneController, func() { tests(parameterModeClassicDRA) })
	framework.Context("with structured parameters", func() { tests(parameterModeStructured) })

	// TODO (https://github.com/kubernetes/kubernetes/issues/123699): move most of the test below into `testDriver` so that they get
	// executed with different parameters.

	ginkgo.Context("cluster", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources)
		b := newBuilder(f, driver)

		ginkgo.It("support validating admission policy for admin access", func(ctx context.Context) {
			// Create VAP, after making it unique to the current test.
			adminAccessPolicyYAML := strings.ReplaceAll(adminAccessPolicyYAML, "dra.example.com", b.f.UniqueName)
			driver.createFromYAML(ctx, []byte(adminAccessPolicyYAML), "")

			// Wait for both VAPs to be processed. This ensures that there are no check errors in the status.
			matchStatus := gomega.Equal(admissionregistrationv1.ValidatingAdmissionPolicyStatus{ObservedGeneration: 1, TypeChecking: &admissionregistrationv1.TypeChecking{}})
			gomega.Eventually(ctx, framework.ListObjects(b.f.ClientSet.AdmissionregistrationV1().ValidatingAdmissionPolicies().List, metav1.ListOptions{})).Should(gomega.HaveField("Items", gomega.ContainElements(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"ObjectMeta": gomega.HaveField("Name", "resourceclaim-policy."+b.f.UniqueName),
					"Status":     matchStatus,
				}),
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"ObjectMeta": gomega.HaveField("Name", "resourceclaimtemplate-policy."+b.f.UniqueName),
					"Status":     matchStatus,
				}),
			)))

			// Attempt to create claim and claim template with admin access. Must fail eventually.
			claim := b.externalClaim()
			claim.Spec.Devices.Requests[0].AdminAccess = true
			_, claimTemplate := b.podInline()
			claimTemplate.Spec.Spec.Devices.Requests[0].AdminAccess = true
			matchVAPError := gomega.MatchError(gomega.ContainSubstring("admin access to devices not enabled" /* in namespace " + b.f.Namespace.Name */))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(matchVAPError)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := b.f.ClientSet.ResourceV1alpha3().ResourceClaimTemplates(b.f.Namespace.Name).Delete(ctx, claimTemplate.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(matchVAPError)

			// After labeling the namespace, creation must (eventually...) succeed.
			_, err := b.f.ClientSet.CoreV1().Namespaces().Apply(ctx,
				applyv1.Namespace(b.f.Namespace.Name).WithLabels(map[string]string{"admin-access." + b.f.UniqueName: "on"}),
				metav1.ApplyOptions{FieldManager: b.f.UniqueName})
			framework.ExpectNoError(err)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
		})

		ginkgo.It("truncates the name of a generated resource claim", func(ctx context.Context) {
			pod, template := b.podInline()
			pod.Name = strings.Repeat("p", 63)
			pod.Spec.ResourceClaims[0].Name = strings.Repeat("c", 63)
			pod.Spec.Containers[0].Resources.Claims[0].Name = pod.Spec.ResourceClaims[0].Name
			b.create(ctx, template, pod)

			b.testPod(ctx, f.ClientSet, pod)
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
							Name:            "req-0",
							DeviceClassName: "my-class",
						}},
					},
				},
			}
			_, err := f.ClientSet.ResourceV1alpha3().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
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

			// Now creating another claim should fail.
			claim2 := claim.DeepCopy()
			claim2.Name = "claim-1"
			_, err = f.ClientSet.ResourceV1alpha3().ResourceClaims(f.Namespace.Name).Create(ctx, claim2, metav1.CreateOptions{})
			gomega.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("exceeded quota: object-count, requested: count/resourceclaims.resource.k8s.io=1, used: count/resourceclaims.resource.k8s.io=1, limited: count/resourceclaims.resource.k8s.io=1")), "creating second claim not allowed")
		})
	})

	// The following tests are all about behavior in combination with a
	// control-plane DRA driver controller.
	framework.Context("cluster with classic DRA", feature.DRAControlPlaneController, func() {
		nodes := NewNodes(f, 1, 4)

		// kube-controller-manager can trigger delayed allocation for pods where the
		// node name was already selected when creating the pod. For immediate
		// allocation, the creator has to ensure that the node matches the claims.
		// This does not work for resource claim templates and only isn't
		// a problem here because the resource is network-attached and available
		// on all nodes.
		preScheduledTests := func(b *builder, driver *Driver) {
			ginkgo.It("supports scheduled pod referencing inline resource claim", func(ctx context.Context) {
				pod, template := b.podInline()
				pod.Spec.NodeName = nodes.NodeNames[0]
				b.create(ctx, pod, template)

				b.testPod(ctx, f.ClientSet, pod)
			})

			ginkgo.It("supports scheduled pod referencing external resource claim", func(ctx context.Context) {
				claim := b.externalClaim()
				pod := b.podExternal()
				pod.Spec.NodeName = nodes.NodeNames[0]
				b.create(ctx, claim, pod)

				b.testPod(ctx, f.ClientSet, pod)
			})
		}

		ginkgo.Context("with setting ReservedFor", func() {
			driver := NewDriver(f, nodes, networkResources)
			driver.parameterMode = parameterModeClassicDRA
			b := newBuilder(f, driver)
			preScheduledTests(b, driver)
			claimTests(b, driver)
		})

		ginkgo.Context("without setting ReservedFor", func() {
			driver := NewDriver(f, nodes, func() app.Resources {
				resources := networkResources()
				resources.DontSetReservedFor = true
				return resources
			})
			driver.parameterMode = parameterModeClassicDRA
			b := newBuilder(f, driver)
			preScheduledTests(b, driver)
			claimTests(b, driver)
		})
	})

	ginkgo.Context("cluster with structured parameters", func() {
		nodes := NewNodes(f, 1, 4)
		driver := NewDriver(f, nodes, perNode(1, nodes))

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
					NodeName: fictionalNodeName,
					Driver:   "dra.example.com",
					Pool: resourceapi.ResourcePool{
						Name:               "some-pool",
						ResourceSliceCount: 1,
					},
				},
			}
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.ClientSet.ResourceV1alpha3().ResourceSlices().Delete(ctx, fictionalNodeSlice.Name, metav1.DeleteOptions{})
				if !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err)
				}
			})

			// Messages from test-driver/deploy/example/plugin-permissions.yaml
			matchVAPDeniedError := func(nodeName string, slice *resourceapi.ResourceSlice) types.GomegaMatcher {
				subStr := fmt.Sprintf("this user running on node '%s' may not modify ", nodeName)
				switch {
				case slice.Spec.NodeName != "":
					subStr += fmt.Sprintf("resourceslices on node '%s'", slice.Spec.NodeName)
				default:
					subStr += "cluster resourceslices"
				}
				return gomega.MatchError(gomega.ContainSubstring(subStr))
			}
			mustCreate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				ginkgo.GinkgoHelper()
				slice, err := clientSet.ResourceV1alpha3().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				ginkgo.GinkgoHelper()
				slice, err := clientSet.ResourceV1alpha3().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1alpha3().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("DELETE: %s + %s", clientName, slice.Name))
			}
			mustCreateAndDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				slice = mustCreate(clientSet, clientName, slice)
				mustDelete(clientSet, clientName, slice)
			}
			mustFailToCreate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1alpha3().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
			}
			mustFailToUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1alpha3().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
			}
			mustFailToDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1alpha3().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
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
					AllNodes: true,
					Driver:   "another.example.com",
					Pool: resourceapi.ResourcePool{
						Name:               "cluster-pool",
						ResourceSliceCount: 1,
					},
				},
			}
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.ClientSet.ResourceV1alpha3().ResourceSlices().Delete(ctx, clusterSlice.Name, metav1.DeleteOptions{})
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

		f.It("must manage ResourceSlices", f.WithSlow(), func(ctx context.Context) {
			driverName := driver.Name

			// Now check for exactly the right set of objects for all nodes.
			ginkgo.By("check if ResourceSlice object(s) exist on the API server")
			resourceClient := f.ClientSet.ResourceV1alpha3().ResourceSlices()
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
						"Spec": gstruct.MatchAllFields(gstruct.Fields{
							"Driver":       gomega.Equal(driver.Name),
							"NodeName":     gomega.Equal(nodeName),
							"NodeSelector": gomega.BeNil(),
							"AllNodes":     gomega.BeFalseBecause("slice should be using NodeName"),
							"Pool": gstruct.MatchAllFields(gstruct.Fields{
								"Name":               gomega.Equal(nodeName),
								"Generation":         gstruct.Ignore(),
								"ResourceSliceCount": gomega.Equal(int64(1)),
							}),
							"Devices": gomega.Equal([]resourceapi.Device{{Name: "device-00", Basic: &resourceapi.BasicDevice{}}}),
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

	multipleDrivers := func(nodeV1alpha3 bool) {
		nodes := NewNodes(f, 1, 4)
		driver1 := NewDriver(f, nodes, perNode(2, nodes))
		driver1.NodeV1alpha3 = nodeV1alpha3
		b1 := newBuilder(f, driver1)

		driver2 := NewDriver(f, nodes, perNode(2, nodes))
		driver2.NameSuffix = "-other"
		driver2.NodeV1alpha3 = nodeV1alpha3
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
			b1.testPod(ctx, f.ClientSet, pod)
		})
	}
	multipleDriversContext := func(prefix string, nodeV1alpha3 bool) {
		ginkgo.Context(prefix, func() {
			multipleDrivers(nodeV1alpha3)
		})
	}

	ginkgo.Context("multiple drivers", func() {
		multipleDriversContext("using only drapbv1alpha3", true)
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
	switch b.driver.parameterMode {
	case parameterModeClassicDRA:
		class.Spec.SuitableNodes = b.nodeSelector()
	case parameterModeStructured:
		class.Spec.Selectors = []resourceapi.DeviceSelector{{
			CEL: &resourceapi.CELDeviceSelector{
				Expression: fmt.Sprintf(`device.driver == "%s"`, b.driver.Name),
			},
		}}
	}
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

// nodeSelector returns a node selector that matches all nodes on which the
// kubelet plugin was deployed.
func (b *builder) nodeSelector() *v1.NodeSelector {
	return &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      "kubernetes.io/hostname",
						Operator: v1.NodeSelectorOpIn,
						Values:   b.driver.Nodenames(),
					},
				},
			},
		},
	}
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
// with the associated config
func (b *builder) claimSpec() resourceapi.ResourceClaimSpec {
	parameters, _ := b.parametersEnv()
	spec := resourceapi.ResourceClaimSpec{
		Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name:            "my-request",
				DeviceClassName: b.className(),
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

	if b.driver.parameterMode == parameterModeClassicDRA {
		spec.Controller = b.driver.Name
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
	pod := e2epod.MakePod(b.f.Namespace.Name, nil, nil, b.f.NamespacePodSecurityLevel, "env && sleep 100000")
	pod.Labels = make(map[string]string)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	// Let kubelet kill the pods quickly. Setting
	// TerminationGracePeriodSeconds to zero would bypass kubelet
	// completely because then the apiserver enables a force-delete even
	// when DeleteOptions for the pod don't ask for it (see
	// https://github.com/kubernetes/kubernetes/blob/0f582f7c3f504e807550310d00f130cb5c18c0c3/pkg/registry/core/pod/strategy.go#L151-L171).
	//
	// We don't do that because it breaks tracking of claim usage: the
	// kube-controller-manager assumes that kubelet is done with the pod
	// once it got removed or has a grace period of 0. Setting the grace
	// period to zero directly in DeletionOptions or indirectly through
	// TerminationGracePeriodSeconds causes the controller to remove
	// the pod from ReservedFor before it actually has stopped on
	// the node.
	one := int64(1)
	pod.Spec.TerminationGracePeriodSeconds = &one
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
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().DeviceClasses().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha3().DeviceClasses().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete device class")
			})
		case *v1.Pod:
			createdObj, err = b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *v1.ConfigMap:
			createdObj, err = b.f.ClientSet.CoreV1().ConfigMaps(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceSlice:
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().ResourceSlices().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha3().ResourceSlices().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete node resource slice")
			})
		default:
			framework.Fail(fmt.Sprintf("internal error, unsupported type %T", obj), 1)
		}
		framework.ExpectNoErrorWithOffset(1, err, "create %T", obj)
		createdObjs = append(createdObjs, createdObj)
	}
	return createdObjs
}

// testPod runs pod and checks if container logs contain expected environment variables
func (b *builder) testPod(ctx context.Context, clientSet kubernetes.Interface, pod *v1.Pod, env ...string) {
	ginkgo.GinkgoHelper()
	err := e2epod.WaitForPodRunningInNamespace(ctx, clientSet, pod)
	framework.ExpectNoError(err, "start pod")

	if len(env) == 0 {
		_, env = b.parametersEnv()
	}
	for _, container := range pod.Spec.Containers {
		testContainerEnv(ctx, clientSet, pod, container.Name, false, env...)
	}
}

// envLineRE matches env output with variables set by test/e2e/dra/test-driver.
var envLineRE = regexp.MustCompile(`^(?:admin|user|claim)_[a-zA-Z0-9_]*=.*$`)

func testContainerEnv(ctx context.Context, clientSet kubernetes.Interface, pod *v1.Pod, containerName string, fullMatch bool, env ...string) {
	ginkgo.GinkgoHelper()
	log, err := e2epod.GetPodLogs(ctx, clientSet, pod.Namespace, pod.Name, containerName)
	framework.ExpectNoError(err, fmt.Sprintf("get logs for container %s", containerName))
	if fullMatch {
		// Find all env variables set by the test driver.
		var actualEnv, expectEnv []string
		for _, line := range strings.Split(log, "\n") {
			if envLineRE.MatchString(line) {
				actualEnv = append(actualEnv, line)
			}
		}
		for i := 0; i < len(env); i += 2 {
			expectEnv = append(expectEnv, env[i]+"="+env[i+1])
		}
		sort.Strings(actualEnv)
		sort.Strings(expectEnv)
		gomega.Expect(actualEnv).To(gomega.Equal(expectEnv), fmt.Sprintf("container %s log output:\n%s", containerName, log))
	} else {
		for i := 0; i < len(env); i += 2 {
			envStr := fmt.Sprintf("\n%s=%s\n", env[i], env[i+1])
			gomega.Expect(log).To(gomega.ContainSubstring(envStr), fmt.Sprintf("container %s env variables", containerName))
		}
	}
}

func newBuilder(f *framework.Framework, driver *Driver) *builder {
	b := &builder{f: f, driver: driver}

	ginkgo.BeforeEach(b.setUp)

	return b
}

func (b *builder) setUp() {
	b.podCounter = 0
	b.claimCounter = 0
	b.create(context.Background(), b.class())
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

	claims, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &claim, klog.KObj(&claim)))
		err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
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
		claims, err := b.f.ClientSet.ResourceV1alpha3().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
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
