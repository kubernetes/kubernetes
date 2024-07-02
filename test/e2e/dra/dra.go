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
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
	"github.com/onsi/gomega/gstruct"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
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

// networkResources can be passed to NewDriver directly.
func networkResources() app.Resources {
	return app.Resources{
		Shareable: true,
	}
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

		ginkgo.Context("with ConfigMap parameters", func() {
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
				parameters := b.parameters()
				pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)

				b.create(ctx, parameters, pod, template)

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

			ginkgo.It("must not run a pod if a claim is not reserved for it", func(ctx context.Context) {
				// Pretend that the resource is allocated and reserved for some other entity.
				// Until the resourceclaim controller learns to remove reservations for
				// arbitrary types we can simply fake somthing here.
				claim := b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
				b.create(ctx, claim)

				claim, err := f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "get claim")

				claim.Finalizers = append(claim.Finalizers, "e2e.test/delete-protection")
				claim, err = f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).Update(ctx, claim, metav1.UpdateOptions{})
				framework.ExpectNoError(err, "add claim finalizer")

				ginkgo.DeferCleanup(func(ctx context.Context) {
					claim.Status.Allocation = nil
					claim.Status.ReservedFor = nil
					claim, err = f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
					framework.ExpectNoError(err, "update claim")

					claim.Finalizers = nil
					_, err = f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).Update(ctx, claim, metav1.UpdateOptions{})
					framework.ExpectNoError(err, "remove claim finalizer")
				})

				claim.Status.Allocation = &resourcev1alpha2.AllocationResult{}
				claim.Status.DriverName = driver.Name
				claim.Status.ReservedFor = append(claim.Status.ReservedFor, resourcev1alpha2.ResourceClaimConsumerReference{
					APIGroup: "example.com",
					Resource: "some",
					Name:     "thing",
					UID:      "12345",
				})
				claim, err = f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
				framework.ExpectNoError(err, "update claim")

				pod := b.podExternal()

				// This bypasses scheduling and therefore the pod gets
				// to run on the node although it never gets added to
				// the `ReservedFor` field of the claim.
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
				parameters := b.parameters()
				claim := b.externalClaim(resourcev1alpha2.AllocationModeImmediate)
				pod := b.podExternal()
				zero := int64(0)
				pod.Spec.TerminationGracePeriodSeconds = &zero

				b.create(ctx, parameters, claim, pod)

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

			ginkgo.It("must skip NodePrepareResource if not used by any container", func(ctx context.Context) {
				parameters := b.parameters()
				pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
				for i := range pod.Spec.Containers {
					pod.Spec.Containers[i].Resources.Claims = nil
				}
				b.create(ctx, parameters, pod, template)
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "start pod")
				for host, plugin := range b.driver.Nodes {
					gomega.Expect(plugin.GetPreparedResources()).Should(gomega.BeEmpty(), "not claims should be prepared on host %s while pod is running", host)
				}
			})

		})
	})

	// claimTests tries out several different combinations of pods with
	// claims, both inline and external.
	claimTests := func(b *builder, driver *Driver, allocationMode resourcev1alpha2.AllocationMode) {
		ginkgo.It("supports simple pod referencing inline resource claim", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod, template := b.podInline(allocationMode)
			objects = append(objects, pod, template)
			b.create(ctx, objects...)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("supports inline claim referenced by multiple containers", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod, template := b.podInlineMultiple(allocationMode)
			objects = append(objects, pod, template)
			b.create(ctx, objects...)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("supports simple pod referencing external resource claim", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod := b.podExternal()
			claim := b.externalClaim(allocationMode)
			objects = append(objects, claim, pod)
			b.create(ctx, objects...)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("supports external claim referenced by multiple pods", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod1 := b.podExternal()
			pod2 := b.podExternal()
			pod3 := b.podExternal()
			claim := b.externalClaim(allocationMode)
			objects = append(objects, claim, pod1, pod2, pod3)
			b.create(ctx, objects...)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
			}
		})

		ginkgo.It("supports external claim referenced by multiple containers of multiple pods", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod1 := b.podExternalMultiple()
			pod2 := b.podExternalMultiple()
			pod3 := b.podExternalMultiple()
			claim := b.externalClaim(allocationMode)
			objects = append(objects, claim, pod1, pod2, pod3)
			b.create(ctx, objects...)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
			}
		})

		ginkgo.It("supports init containers", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod, template := b.podInline(allocationMode)
			pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
			pod.Spec.InitContainers[0].Name += "-init"
			// This must succeed for the pod to start.
			pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
			objects = append(objects, pod, template)
			b.create(ctx, objects...)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("removes reservation from claim when pod is done", func(ctx context.Context) {
			objects, _ := b.flexibleParameters()
			pod := b.podExternal()
			claim := b.externalClaim(allocationMode)
			pod.Spec.Containers[0].Command = []string{"true"}
			objects = append(objects, claim, pod)
			b.create(ctx, objects...)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be unreserved")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourcev1alpha2.ResourceClaim, error) {
				return f.ClientSet.ResourceV1alpha2().ResourceClaims(pod.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.ReservedFor", gomega.BeEmpty()), "reservation should have been removed")
		})

		ginkgo.It("deletes generated claims when pod is done", func(ctx context.Context) {
			objects, _ := b.flexibleParameters()
			pod, template := b.podInline(allocationMode)
			pod.Spec.Containers[0].Command = []string{"true"}
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be deleted")
			gomega.Eventually(ctx, func(ctx context.Context) ([]resourcev1alpha2.ResourceClaim, error) {
				claims, err := f.ClientSet.ResourceV1alpha2().ResourceClaims(pod.Namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return nil, err
				}
				return claims.Items, nil
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.BeEmpty(), "claim should have been deleted")
		})

		ginkgo.It("does not delete generated claims when pod is restarting", func(ctx context.Context) {
			objects, _ := b.flexibleParameters()
			pod, template := b.podInline(allocationMode)
			pod.Spec.Containers[0].Command = []string{"sh", "-c", "sleep 1; exit 1"}
			pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			ginkgo.By("waiting for pod to restart twice")
			gomega.Eventually(ctx, func(ctx context.Context) (*v1.Pod, error) {
				return f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodStartSlow).Should(gomega.HaveField("Status.ContainerStatuses", gomega.ContainElements(gomega.HaveField("RestartCount", gomega.BeNumerically(">=", 2)))))
			if driver.Controller != nil {
				gomega.Expect(driver.Controller.GetNumAllocations()).To(gomega.Equal(int64(1)), "number of allocations")
			}
		})

		ginkgo.It("must deallocate after use when using delayed allocation", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod := b.podExternal()
			claim := b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			objects = append(objects, claim, pod)
			b.create(ctx, objects...)

			gomega.Eventually(ctx, func(ctx context.Context) (*resourcev1alpha2.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourcev1alpha2.AllocationResult)(nil)))

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)

			ginkgo.By(fmt.Sprintf("deleting pod %s", klog.KObj(pod)))
			framework.ExpectNoError(b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))

			ginkgo.By("waiting for claim to get deallocated")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourcev1alpha2.ResourceClaim, error) {
				return b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", (*resourcev1alpha2.AllocationResult)(nil)))
		})
	}

	singleNodeTests := func(parameterMode parameterMode) {
		nodes := NewNodes(f, 1, 1)
		maxAllocations := 1
		numPods := 10
		generateResources := func() app.Resources {
			resources := perNode(maxAllocations, nodes)()
			resources.Shareable = true
			return resources
		}
		driver := NewDriver(f, nodes, generateResources) // All tests get their own driver instance.
		driver.parameterMode = parameterMode
		b := newBuilder(f, driver)
		// We need the parameters name *before* creating it.
		b.parametersCounter = 1
		b.classParametersName = b.parametersName()

		ginkgo.It("supports claim and class parameters", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()

			pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			objects = append(objects, pod, template)

			b.create(ctx, objects...)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("supports reusing resources", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
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
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow))
				}()
			}
			wg.Wait()
		})

		ginkgo.It("supports sharing a claim concurrently", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			objects = append(objects, b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer))

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
			objects, expectedEnv := b.flexibleParameters()
			numPods := numPods / 2

			// Change from "shareable" to "not shareable", if possible.
			switch parameterMode {
			case parameterModeConfigMap:
				ginkgo.Skip("cannot change the driver's controller behavior on-the-fly")
			case parameterModeTranslated, parameterModeStructured:
				objects[len(objects)-1].(*resourcev1alpha2.ResourceClaimParameters).Shareable = false
			}

			objects = append(objects, b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer))

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

		ginkgo.It("retries pod scheduling after creating resource class", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			class, err := f.ClientSet.ResourceV1alpha2().ResourceClasses().Get(ctx, template.Spec.Spec.ResourceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			template.Spec.Spec.ResourceClassName += "-b"
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			class.UID = ""
			class.ResourceVersion = ""
			class.Name = template.Spec.Spec.ResourceClassName
			b.create(ctx, class)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("retries pod scheduling after updating resource class", func(ctx context.Context) {
			objects, expectedEnv := b.flexibleParameters()
			pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)

			// First modify the class so that it matches no nodes.
			class, err := f.ClientSet.ResourceV1alpha2().ResourceClasses().Get(ctx, template.Spec.Spec.ResourceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			class.SuitableNodes = &v1.NodeSelector{
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
			class, err = f.ClientSet.ResourceV1alpha2().ResourceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// Now create the pod.
			objects = append(objects, template, pod)
			b.create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			// Unblock the pod.
			class.SuitableNodes = nil
			_, err = f.ClientSet.ResourceV1alpha2().ResourceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			b.testPod(ctx, f.ClientSet, pod, expectedEnv...)
		})

		ginkgo.It("runs a pod without a generated resource claim", func(ctx context.Context) {
			pod, _ /* template */ := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
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

		ginkgo.Context("with delayed allocation", func() {
			claimTests(b, driver, resourcev1alpha2.AllocationModeWaitForFirstConsumer)
		})

		ginkgo.Context("with immediate allocation", func() {
			claimTests(b, driver, resourcev1alpha2.AllocationModeImmediate)
		})
	}

	// These tests depend on having more than one node and a DRA driver controller.
	multiNodeDRAControllerTests := func(nodes *Nodes) {
		driver := NewDriver(f, nodes, networkResources)
		b := newBuilder(f, driver)

		ginkgo.It("schedules onto different nodes", func(ctx context.Context) {
			parameters := b.parameters()
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
			claim := b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			b.create(ctx, parameters, claim, pod1, pod2)

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
			parameters := b.parameters()
			label := "app.kubernetes.io/instance"
			instance := f.UniqueName + "-test-app"
			pod := b.podExternal()
			pod.Labels[label] = instance
			claim := b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			b.create(ctx, parameters, claim, pod)

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
			gomega.Eventually(ctx, framework.GetObject(b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).Get, claim.Name, metav1.GetOptions{})).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", gomega.BeNil()))
		})
	}

	// The following tests only make sense when there is more than one node.
	// They get skipped when there's only one node.
	multiNodeTests := func(parameterMode parameterMode) {
		nodes := NewNodes(f, 2, 8)

		if parameterMode == parameterModeConfigMap {
			ginkgo.Context("with network-attached resources", func() {
				multiNodeDRAControllerTests(nodes)
			})

			ginkgo.Context("reallocation", func() {
				var allocateWrapper2 app.AllocateWrapperType
				driver := NewDriver(f, nodes, perNode(1, nodes))
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

					parameters1 := b.parameters()
					parameters2 := b2.parameters()
					// Order is relevant here: each pod must be matched with its own claim.
					pod1claim1 := b.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
					pod1 := b.podExternal()
					pod2claim1 := b2.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
					pod2 := b2.podExternal()

					// Add another claim to pod1.
					pod1claim2 := b2.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
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

					b.create(ctx, parameters1, parameters2, pod1claim1, pod1claim2, pod1)

					ginkgo.By("waiting for one claim from driver1 to be allocated")
					var nodeSelector *v1.NodeSelector
					gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
						claims, err := f.ClientSet.ResourceV1alpha2().ResourceClaims(f.Namespace.Name).List(ctx, metav1.ListOptions{})
						if err != nil {
							return 0, err
						}
						allocated := 0
						for _, claim := range claims.Items {
							if claim.Status.Allocation != nil {
								allocated++
								nodeSelector = claim.Status.Allocation.AvailableOnNodes
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

			tests := func(allocationMode resourcev1alpha2.AllocationMode) {
				ginkgo.It("uses all resources", func(ctx context.Context) {
					objs, _ := b.flexibleParameters()
					var pods []*v1.Pod
					for i := 0; i < len(nodes.NodeNames); i++ {
						pod, template := b.podInline(allocationMode)
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
			}

			ginkgo.Context("with delayed allocation", func() {
				tests(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			})

			ginkgo.Context("with immediate allocation", func() {
				tests(resourcev1alpha2.AllocationModeImmediate)
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

	ginkgo.Context("with ConfigMap parameters", func() { tests(parameterModeConfigMap) })
	ginkgo.Context("with translated parameters", func() { tests(parameterModeTranslated) })
	ginkgo.Context("with structured parameters", func() { tests(parameterModeStructured) })

	// TODO (https://github.com/kubernetes/kubernetes/issues/123699): move most of the test below into `testDriver` so that they get
	// executed with different parameters.

	ginkgo.Context("cluster", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources)
		b := newBuilder(f, driver)

		ginkgo.It("truncates the name of a generated resource claim", func(ctx context.Context) {
			parameters := b.parameters()
			pod, template := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			pod.Name = strings.Repeat("p", 63)
			pod.Spec.ResourceClaims[0].Name = strings.Repeat("c", 63)
			pod.Spec.Containers[0].Resources.Claims[0].Name = pod.Spec.ResourceClaims[0].Name
			b.create(ctx, parameters, template, pod)

			b.testPod(ctx, f.ClientSet, pod)
		})
	})

	// The following tests are all about behavior in combination with a
	// control-plane DRA driver controller.
	ginkgo.Context("cluster with DRA driver controller", func() {
		nodes := NewNodes(f, 1, 4)

		ginkgo.Context("with structured parameters", func() {
			driver := NewDriver(f, nodes, perNode(1, nodes))
			driver.parameterMode = parameterModeStructured

			f.It("must manage ResourceSlices", f.WithSlow(), func(ctx context.Context) {
				driverName := driver.Name

				// Now check for exactly the right set of objects for all nodes.
				ginkgo.By("check if ResourceSlice object(s) exist on the API server")
				resourceClient := f.ClientSet.ResourceV1alpha2().ResourceSlices()
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
							"NodeName":   gomega.Equal(nodeName),
							"DriverName": gomega.Equal(driver.Name),
							"ResourceModel": gomega.Equal(resourcev1alpha2.ResourceModel{NamedResources: &resourcev1alpha2.NamedResourcesResources{
								Instances: []resourcev1alpha2.NamedResourcesInstance{{Name: "instance-00"}},
							}}),
						}),
					)
				}
				matchSlices := gomega.ContainElements(expectedObjects...)
				getSlices := func(ctx context.Context) ([]resourcev1alpha2.ResourceSlice, error) {
					slices, err := resourceClient.List(ctx, metav1.ListOptions{FieldSelector: fmt.Sprintf("driverName=%s", driverName)})
					if err != nil {
						return nil, err
					}
					return slices.Items, nil
				}
				gomega.Eventually(ctx, getSlices).WithTimeout(20 * time.Second).Should(matchSlices)
				gomega.Consistently(ctx, getSlices).WithTimeout(20 * time.Second).Should(matchSlices)

				// Removal of node resource slice is tested by the general driver removal code.
			})

			// TODO (https://github.com/kubernetes/kubernetes/issues/123699): more test scenarios:
			// - driver returns "unimplemented" as method response
			// - driver returns "Unimplemented" as part of stream
			// - driver returns EOF
			// - driver changes resources
			//
			// None of those matter if the publishing gets moved into the driver itself,
			// which is the goal for 1.31 to support version skew for kubelet.
		})

		ginkgo.Context("with local unshared resources", func() {
			driver := NewDriver(f, nodes, func() app.Resources {
				return app.Resources{
					NodeLocal:      true,
					MaxAllocations: 10,
					Nodes:          nodes.NodeNames,
				}
			})
			b := newBuilder(f, driver)

			// This test covers some special code paths in the scheduler:
			// - Patching the ReservedFor during PreBind because in contrast
			//   to claims specifically allocated for a pod, here the claim
			//   gets allocated without reserving it.
			// - Error handling when PreBind fails: multiple attempts to bind pods
			//   are started concurrently, only one attempt succeeds.
			// - Removing a ReservedFor entry because the first inline claim gets
			//   reserved during allocation.
			ginkgo.It("reuses an allocated immediate claim", func(ctx context.Context) {
				objects := []klog.KMetadata{
					b.parameters(),
					b.externalClaim(resourcev1alpha2.AllocationModeImmediate),
				}
				podExternal := b.podExternal()

				// Create many pods to increase the chance that the scheduler will
				// try to bind two pods at the same time.
				numPods := 5
				for i := 0; i < numPods; i++ {
					podInline, claimTemplate := b.podInline(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
					podInline.Spec.Containers[0].Resources.Claims = append(podInline.Spec.Containers[0].Resources.Claims, podExternal.Spec.Containers[0].Resources.Claims[0])
					podInline.Spec.ResourceClaims = append(podInline.Spec.ResourceClaims, podExternal.Spec.ResourceClaims[0])
					objects = append(objects, claimTemplate, podInline)
				}
				b.create(ctx, objects...)

				var runningPod *v1.Pod
				haveRunningPod := gcustom.MakeMatcher(func(pods []v1.Pod) (bool, error) {
					numRunning := 0
					runningPod = nil
					for _, pod := range pods {
						if pod.Status.Phase == v1.PodRunning {
							pod := pod // Don't keep pointer to loop variable...
							runningPod = &pod
							numRunning++
						}
					}
					return numRunning == 1, nil
				}).WithTemplate("Expected one running Pod.\nGot instead:\n{{.FormattedActual}}")

				for i := 0; i < numPods; i++ {
					ginkgo.By("waiting for exactly one pod to start")
					runningPod = nil
					gomega.Eventually(ctx, b.listTestPods).WithTimeout(f.Timeouts.PodStartSlow).Should(haveRunningPod)

					ginkgo.By("checking that no other pod gets scheduled")
					havePendingPods := gcustom.MakeMatcher(func(pods []v1.Pod) (bool, error) {
						numPending := 0
						for _, pod := range pods {
							if pod.Status.Phase == v1.PodPending {
								numPending++
							}
						}
						return numPending == numPods-1-i, nil
					}).WithTemplate("Expected only one running Pod.\nGot instead:\n{{.FormattedActual}}")
					gomega.Consistently(ctx, b.listTestPods).WithTimeout(time.Second).Should(havePendingPods)

					ginkgo.By(fmt.Sprintf("deleting pod %s", klog.KObj(runningPod)))
					framework.ExpectNoError(b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, runningPod.Name, metav1.DeleteOptions{}))

					ginkgo.By(fmt.Sprintf("waiting for pod %s to disappear", klog.KObj(runningPod)))
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, b.f.ClientSet, runningPod.Name, runningPod.Namespace, f.Timeouts.PodDelete))
				}
			})
		})

		ginkgo.Context("with shared network resources", func() {
			driver := NewDriver(f, nodes, networkResources)
			b := newBuilder(f, driver)

			// This test complements "reuses an allocated immediate claim" above:
			// because the claim can be shared, each PreBind attempt succeeds.
			ginkgo.It("shares an allocated immediate claim", func(ctx context.Context) {
				objects := []klog.KMetadata{
					b.parameters(),
					b.externalClaim(resourcev1alpha2.AllocationModeImmediate),
				}
				// Create many pods to increase the chance that the scheduler will
				// try to bind two pods at the same time.
				numPods := 5
				pods := make([]*v1.Pod, numPods)
				for i := 0; i < numPods; i++ {
					pods[i] = b.podExternal()
					objects = append(objects, pods[i])
				}
				b.create(ctx, objects...)

				ginkgo.By("waiting all pods to start")
				framework.ExpectNoError(e2epod.WaitForPodsRunning(ctx, b.f.ClientSet, f.Namespace.Name, numPods+len(nodes.NodeNames) /* driver(s) */, f.Timeouts.PodStartSlow))
			})
		})

		// kube-controller-manager can trigger delayed allocation for pods where the
		// node name was already selected when creating the pod. For immediate
		// allocation, the creator has to ensure that the node matches the claims.
		// This does not work for resource claim templates and only isn't
		// a problem here because the resource is network-attached and available
		// on all nodes.
		preScheduledTests := func(b *builder, driver *Driver, allocationMode resourcev1alpha2.AllocationMode) {
			ginkgo.It("supports scheduled pod referencing inline resource claim", func(ctx context.Context) {
				parameters := b.parameters()
				pod, template := b.podInline(allocationMode)
				pod.Spec.NodeName = nodes.NodeNames[0]
				b.create(ctx, parameters, pod, template)

				b.testPod(ctx, f.ClientSet, pod)
			})

			ginkgo.It("supports scheduled pod referencing external resource claim", func(ctx context.Context) {
				parameters := b.parameters()
				claim := b.externalClaim(allocationMode)
				pod := b.podExternal()
				pod.Spec.NodeName = nodes.NodeNames[0]
				b.create(ctx, parameters, claim, pod)

				b.testPod(ctx, f.ClientSet, pod)
			})
		}

		ginkgo.Context("with delayed allocation and setting ReservedFor", func() {
			driver := NewDriver(f, nodes, networkResources)
			b := newBuilder(f, driver)
			preScheduledTests(b, driver, resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			claimTests(b, driver, resourcev1alpha2.AllocationModeWaitForFirstConsumer)
		})

		ginkgo.Context("with delayed allocation and not setting ReservedFor", func() {
			driver := NewDriver(f, nodes, func() app.Resources {
				resources := networkResources()
				resources.DontSetReservedFor = true
				return resources
			})
			b := newBuilder(f, driver)
			preScheduledTests(b, driver, resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			claimTests(b, driver, resourcev1alpha2.AllocationModeWaitForFirstConsumer)
		})

		ginkgo.Context("with immediate allocation", func() {
			driver := NewDriver(f, nodes, networkResources)
			b := newBuilder(f, driver)
			preScheduledTests(b, driver, resourcev1alpha2.AllocationModeImmediate)
			claimTests(b, driver, resourcev1alpha2.AllocationModeImmediate)
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
			parameters1 := b1.parameters()
			parameters2 := b2.parameters()
			claim1 := b1.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			claim1b := b1.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			claim2 := b2.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			claim2b := b2.externalClaim(resourcev1alpha2.AllocationModeWaitForFirstConsumer)
			pod := b1.podExternal()
			for i, claim := range []*resourcev1alpha2.ResourceClaim{claim1b, claim2, claim2b} {
				claim := claim
				pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims,
					v1.PodResourceClaim{
						Name:              fmt.Sprintf("claim%d", i+1),
						ResourceClaimName: &claim.Name,
					},
				)
			}
			b1.create(ctx, parameters1, parameters2, claim1, claim1b, claim2, claim2b, pod)
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

	podCounter        int
	parametersCounter int
	claimCounter      int

	classParametersName string
}

// className returns the default resource class name.
func (b *builder) className() string {
	return b.f.UniqueName + b.driver.NameSuffix + "-class"
}

// class returns the resource class that the builder's other objects
// reference.
func (b *builder) class() *resourcev1alpha2.ResourceClass {
	class := &resourcev1alpha2.ResourceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: b.className(),
		},
		DriverName:           b.driver.Name,
		SuitableNodes:        b.nodeSelector(),
		StructuredParameters: ptr.To(b.driver.parameterMode != parameterModeConfigMap),
	}
	if b.classParametersName != "" {
		class.ParametersRef = &resourcev1alpha2.ResourceClassParametersReference{
			APIGroup:  b.driver.parameterAPIGroup,
			Kind:      b.driver.classParameterAPIKind,
			Name:      b.classParametersName,
			Namespace: b.f.Namespace.Name,
		}
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
func (b *builder) externalClaim(allocationMode resourcev1alpha2.AllocationMode) *resourcev1alpha2.ResourceClaim {
	b.claimCounter++
	name := "external-claim" + b.driver.NameSuffix // This is what podExternal expects.
	if b.claimCounter > 1 {
		name += fmt.Sprintf("-%d", b.claimCounter)
	}
	return &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourcev1alpha2.ResourceClaimSpec{
			ResourceClassName: b.className(),
			ParametersRef: &resourcev1alpha2.ResourceClaimParametersReference{
				APIGroup: b.driver.parameterAPIGroup,
				Kind:     b.driver.claimParameterAPIKind,
				Name:     b.parametersName(),
			},
			AllocationMode: allocationMode,
		},
	}
}

// flexibleParameters returns parameter objects for claims and
// class with their type depending on the current parameter mode.
// It also returns the expected environment in a pod using
// the corresponding resource.
func (b *builder) flexibleParameters() ([]klog.KMetadata, []string) {
	var objects []klog.KMetadata
	switch b.driver.parameterMode {
	case parameterModeConfigMap:
		objects = append(objects,
			b.parameters("x", "y"),
			b.parameters("a", "b", "request_foo", "bar"),
		)
	case parameterModeTranslated:
		objects = append(objects,
			b.parameters("x", "y"),
			b.classParameters(b.parametersName(), "x", "y"),
			b.parameters("a", "b", "request_foo", "bar"),
			b.claimParameters(b.parametersName(), []string{"a", "b"}, []string{"request_foo", "bar"}),
		)
		// The parameters object is not the last one but the second-last.
		b.parametersCounter--
	case parameterModeStructured:
		objects = append(objects,
			b.classParameters("", "x", "y"),
			b.claimParameters("", []string{"a", "b"}, []string{"request_foo", "bar"}),
		)
	}
	env := []string{"user_a", "b", "user_request_foo", "bar"}
	if b.classParametersName != "" {
		env = append(env, "admin_x", "y")
	}
	return objects, env
}

// parametersName returns the current ConfigMap name for resource
// claim or class parameters.
func (b *builder) parametersName() string {
	return fmt.Sprintf("parameters%s-%d", b.driver.NameSuffix, b.parametersCounter)
}

// parametersEnv returns the default env variables.
func (b *builder) parametersEnv() map[string]string {
	return map[string]string{
		"a":           "b",
		"request_foo": "bar",
	}
}

// parameters returns a config map with the default env variables.
func (b *builder) parameters(kv ...string) *v1.ConfigMap {
	data := b.parameterData(kv...)
	b.parametersCounter++
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.f.Namespace.Name,
			Name:      b.parametersName(),
		},
		Data: data,
	}
}

func (b *builder) classParameters(generatedFrom string, kv ...string) *resourcev1alpha2.ResourceClassParameters {
	raw := b.rawParameterData(kv...)
	b.parametersCounter++
	parameters := &resourcev1alpha2.ResourceClassParameters{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.f.Namespace.Name,
			Name:      b.parametersName(),
		},

		VendorParameters: []resourcev1alpha2.VendorParameters{
			{DriverName: b.driver.Name, Parameters: runtime.RawExtension{Raw: raw}},
		},
	}

	if generatedFrom != "" {
		parameters.GeneratedFrom = &resourcev1alpha2.ResourceClassParametersReference{
			Kind:      "ConfigMap",
			Namespace: b.f.Namespace.Name,
			Name:      generatedFrom,
		}
	}

	return parameters
}

func (b *builder) claimParameters(generatedFrom string, claimKV, requestKV []string) *resourcev1alpha2.ResourceClaimParameters {
	b.parametersCounter++
	parameters := &resourcev1alpha2.ResourceClaimParameters{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.f.Namespace.Name,
			Name:      b.parametersName(),
		},

		Shareable: true,

		// Without any request, nothing gets allocated and vendor
		// parameters are also not passed down because they get
		// attached to the allocation result.
		DriverRequests: []resourcev1alpha2.DriverRequests{
			{
				DriverName:       b.driver.Name,
				VendorParameters: runtime.RawExtension{Raw: b.rawParameterData(claimKV...)},
				Requests: []resourcev1alpha2.ResourceRequest{
					{
						VendorParameters: runtime.RawExtension{Raw: b.rawParameterData(requestKV...)},
						ResourceRequestModel: resourcev1alpha2.ResourceRequestModel{
							NamedResources: &resourcev1alpha2.NamedResourcesRequest{
								Selector: "true",
							},
						},
					},
				},
			},
		},
	}

	if generatedFrom != "" {
		parameters.GeneratedFrom = &resourcev1alpha2.ResourceClaimParametersReference{
			Kind: "ConfigMap",
			Name: generatedFrom,
		}
	}

	return parameters
}

func (b *builder) parameterData(kv ...string) map[string]string {
	data := map[string]string{}
	for i := 0; i < len(kv); i += 2 {
		data[kv[i]] = kv[i+1]
	}
	if len(data) == 0 {
		data = b.parametersEnv()
	}
	return data
}

func (b *builder) rawParameterData(kv ...string) []byte {
	data := b.parameterData(kv...)
	raw, err := json.Marshal(data)
	framework.ExpectNoError(err, "JSON encoding of parameter data")
	return raw
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
func (b *builder) podInline(allocationMode resourcev1alpha2.AllocationMode) (*v1.Pod, *resourcev1alpha2.ResourceClaimTemplate) {
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
	template := &resourcev1alpha2.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourcev1alpha2.ResourceClaimTemplateSpec{
			Spec: resourcev1alpha2.ResourceClaimSpec{
				ResourceClassName: b.className(),
				ParametersRef: &resourcev1alpha2.ResourceClaimParametersReference{
					APIGroup: b.driver.parameterAPIGroup,
					Kind:     b.driver.claimParameterAPIKind,
					Name:     b.parametersName(),
				},
				AllocationMode: allocationMode,
			},
		},
	}
	return pod, template
}

// podInlineMultiple returns a pod with inline resource claim referenced by 3 containers
func (b *builder) podInlineMultiple(allocationMode resourcev1alpha2.AllocationMode) (*v1.Pod, *resourcev1alpha2.ResourceClaimTemplate) {
	pod, template := b.podInline(allocationMode)
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
		case *resourcev1alpha2.ResourceClass:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceClasses().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha2().ResourceClasses().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete resource class")
			})
		case *v1.Pod:
			createdObj, err = b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *v1.ConfigMap:
			createdObj, err = b.f.ClientSet.CoreV1().ConfigMaps(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1alpha2.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1alpha2.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1alpha2.ResourceClassParameters:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceClassParameters(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1alpha2.ResourceClaimParameters:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceClaimParameters(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1alpha2.ResourceSlice:
			createdObj, err = b.f.ClientSet.ResourceV1alpha2().ResourceSlices().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha2().ResourceSlices().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
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
	err := e2epod.WaitForPodRunningInNamespace(ctx, clientSet, pod)
	framework.ExpectNoError(err, "start pod")

	for _, container := range pod.Spec.Containers {
		log, err := e2epod.GetPodLogs(ctx, clientSet, pod.Namespace, pod.Name, container.Name)
		framework.ExpectNoError(err, "get logs")
		if len(env) == 0 {
			for key, value := range b.parametersEnv() {
				envStr := fmt.Sprintf("\nuser_%s=%s\n", key, value)
				gomega.Expect(log).To(gomega.ContainSubstring(envStr), "container env variables")
			}
		} else {
			for i := 0; i < len(env); i += 2 {
				envStr := fmt.Sprintf("\n%s=%s\n", env[i], env[i+1])
				gomega.Expect(log).To(gomega.ContainSubstring(envStr), "container env variables")
			}
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
	b.parametersCounter = 0
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

	claims, err := b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &claim, klog.KObj(&claim)))
		err := b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete claim")
		}
	}

	for host, plugin := range b.driver.Nodes {
		ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
		gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
	}

	ginkgo.By("waiting for claims to be deallocated and deleted")
	gomega.Eventually(func() ([]resourcev1alpha2.ResourceClaim, error) {
		claims, err := b.f.ClientSet.ResourceV1alpha2().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
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
