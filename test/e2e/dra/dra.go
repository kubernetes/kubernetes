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
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	applyv1 "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/events"
	testdriverapp "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	// podStartTimeout is how long to wait for the pod to be started.
	podStartTimeout = 5 * time.Minute
)

// The "DRA" label is used to select tests related to DRA in a Ginkgo label filter.
//
// Sub-tests starting with "control plane" when testing only the control plane components, without depending
// on DRA support in the kubelet.
//
// Sub-tests starting with "kubelet" depend on DRA and plugin support in the kubelet.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), func() {
	f := framework.NewDefaultFramework("dra")

	// The driver containers have to run with sufficient privileges to
	// modify /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("kubelet", feature.DynamicResourceAllocation, func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, drautils.NetworkResources(10, false))
		b := drautils.NewBuilder(f, driver)

		ginkgo.It("registers plugin", func() {
			ginkgo.By("the driver is running")
		})

		ginkgo.It("must retry NodePrepareResources", func(ctx context.Context) {
			// We have exactly one host. The API version depends on the kubelet
			// we test with (version skew!), so we need to be a bit flexible.
			mV1Beta1 := drautils.MethodInstance{NodeName: driver.Nodenames()[0], FullMethod: "/k8s.io.kubelet.pkg.apis.dra.v1beta1.DRAPlugin/NodePrepareResources"}
			mV1 := drautils.MethodInstance{NodeName: driver.Nodenames()[0], FullMethod: "/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"}

			driver.Fail(mV1Beta1, true)
			driver.Fail(mV1, true)

			ginkgo.By("waiting for container startup to fail")
			pod, template := b.PodInline()

			b.Create(ctx, pod, template)

			ginkgo.By("wait for NodePrepareResources call")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				if driver.CallCount(mV1Beta1)+driver.CallCount(mV1) == 0 {
					return errors.New("NodePrepareResources not called yet")
				}
				return nil
			}).WithTimeout(podStartTimeout).Should(gomega.Succeed())

			ginkgo.By("allowing container startup to succeed")
			callCount := driver.CallCount(mV1Beta1) + driver.CallCount(mV1)
			driver.Fail(mV1, false)
			driver.Fail(mV1Beta1, false)
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
			framework.ExpectNoError(err, "start pod with inline resource claim")
			if driver.CallCount(mV1Beta1)+driver.CallCount(mV1) == callCount {
				framework.Fail("NodePrepareResources should have been called again")
			}
		})

		ginkgo.It("must not run a pod if a claim is not ready", func(ctx context.Context) {
			claim := b.ExternalClaim()
			b.Create(ctx, claim)
			pod := b.PodExternal()

			// This bypasses scheduling and therefore the pod gets
			// to run on the node although the claim is not ready.
			// Because the parameters are missing, the claim
			// also cannot be allocated later.
			pod.Spec.NodeName = nodes.NodeNames[0]
			b.Create(ctx, pod)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				testPod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
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
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			zero := int64(0)
			pod.Spec.TerminationGracePeriodSeconds = &zero

			b.Create(ctx, claim, pod)

			b.TestPod(ctx, f, pod)

			ginkgo.By(fmt.Sprintf("force delete test pod %s", pod.Name))
			err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &zero})
			if !apierrors.IsNotFound(err) {
				framework.ExpectNoError(err, "force delete test pod")
			}

			for host, plugin := range driver.Nodes {
				ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
				gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
			}
		})

		ginkgo.It("must call NodePrepareResources even if not used by any container", func(ctx context.Context) {
			pod, template := b.PodInline()
			for i := range pod.Spec.Containers {
				pod.Spec.Containers[i].Resources.Claims = nil
			}
			b.Create(ctx, pod, template)
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "start pod")
			for host, plugin := range driver.Nodes {
				gomega.Expect(plugin.GetPreparedResources()).ShouldNot(gomega.BeEmpty(), "claims should be prepared on host %s while pod is running", host)
			}
		})

		ginkgo.It("must map configs and devices to the right containers", func(ctx context.Context) {
			// Several claims, each with three requests and three configs.
			// One config applies to all requests, the other two only to one request each.
			claimForAllContainers := b.ExternalClaim()
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

			pod := b.PodExternal()
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

			b.Create(ctx, claimForAllContainers, claimForContainer0, claimForContainer1, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")

			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[0].Name, true, container0Env...)
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[1].Name, true, container1Env...)
		})

		// https://github.com/kubernetes/kubernetes/issues/131513 was fixed in master for 1.34 and not backported,
		// so this test only passes for kubelet >= 1.34.
		f.It("blocks new pod after force-delete", f.WithLabel("KubeletMinVersion:1.34"), func(ctx context.Context) {
			// The problem with a force-deleted pod is that kubelet
			// is not necessarily done yet with tearing down the
			// pod at the time when the pod and its claim are
			// already removed. The user can replace the claim and
			// pod with new instances under the same name.  The
			// kubelet then needs to detect that the new claim is
			// not the same as the one that kubelet currently works
			// on and that the new pod cannot start until the old
			// one is torn down.
			//
			// This test delays termination of the first pod to ensure
			// that the race goes bad (old pod pending shutdown when
			// new one arrives) and always schedules to the same node.
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			node := nodes.NodeNames[0]
			pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": node}
			oldClaim := b.Create(ctx, claim, pod)[0].(*resourceapi.ResourceClaim)
			b.TestPod(ctx, f, pod)

			ginkgo.By("Force-delete claim and pod")
			forceDelete := metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))}
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, pod.Name, forceDelete))

			// Fail NodeUnprepareResources to simulate long grace period
			unprepareResources := drautils.MethodInstance{NodeName: node, FullMethod: "/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"}
			driver.Fail(unprepareResources, true)

			// The pod should get deleted immediately.
			_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			if !apierrors.IsNotFound(err) {
				framework.Failf("Expected 'not found' error, got: %v", err)
			}

			// The claim may take a bit longer because of the allocation and finalizer.
			framework.ExpectNoError(f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Delete(ctx, claim.Name, forceDelete))
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				claim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return nil, nil
				}
				return claim, err
			}).Should(gomega.BeNil())
			gomega.Expect(driver.Nodes[node].GetPreparedResources()).Should(gomega.Equal([]testdriverapp.ClaimID{{Name: oldClaim.Name, UID: oldClaim.UID}}), "Old claim should still be prepared.")

			ginkgo.By("Re-creating the same claim and pod")
			newClaim := b.Create(ctx, claim, pod)[0].(*resourceapi.ResourceClaim)

			// Keep blocking NodeUnprepareResources for the old pod
			// until the new pod calls NodePrepareResources and fails.
			// This ensures that the race is triggered.
			expectedEvent := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      pod.Name,
				"involvedObject.namespace": pod.Namespace,
				"reason":                   events.FailedPrepareDynamicResources,
			}.AsSelector().String()

			// 10 min timeout (PodStartTimeout * 2) should be enough
			// for Kubelet to emit multiple events, so the test should
			// be able to catch at least one of them.
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(
				ctx,
				f.ClientSet,
				pod.Namespace,
				expectedEvent,
				fmt.Sprintf("old ResourceClaim with same name %s and different UID %s still exists", oldClaim.Name, oldClaim.UID),
				framework.PodStartTimeout*2))

			driver.Fail(unprepareResources, false)

			b.TestPod(ctx, f, pod)

			// The pod must not have started before NodeUnprepareResources was called for the old one,
			// i.e. what is prepared now must be the new claim.
			gomega.Expect(driver.Nodes[node].GetPreparedResources()).Should(gomega.Equal([]testdriverapp.ClaimID{{Name: newClaim.Name, UID: newClaim.UID}}), "Only new claim should be prepared now because new pod is running.")
		})

		f.It("DaemonSet with admin access", f.WithFeatureGate(features.DRAAdminAccess), func(ctx context.Context) {
			// Ensure namespace has the dra admin label.
			_, err := f.ClientSet.CoreV1().Namespaces().Apply(ctx,
				applyv1.Namespace(f.Namespace.Name).WithLabels(map[string]string{"resource.kubernetes.io/admin-access": "true"}),
				metav1.ApplyOptions{FieldManager: f.UniqueName})
			framework.ExpectNoError(err)

			pod, template := b.PodInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			// Limit the daemon set to the one node where we have the driver.
			nodeName := nodes.NodeNames[0]
			pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": nodeName}
			pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			daemonSet := &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "monitoring-ds",
					Namespace: f.Namespace.Name,
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

			created := b.Create(ctx, template, daemonSet)
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

	// Same "kubelet" context as above, but now with per-node resources.
	f.Context("kubelet", feature.DynamicResourceAllocation, func() {
		nodes := drautils.NewNodes(f, 1, 4)
		driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
		b := drautils.NewBuilder(f, driver)

		f.It("must manage ResourceSlices", func(ctx context.Context) {
			driverName := driver.Name

			// Now check for exactly the right set of objects for all nodes.
			ginkgo.By("check if ResourceSlice object(s) exist on the API server")
			resourceClient := f.ClientSet.ResourceV1().ResourceSlices()
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

		ginkgo.It("supports init containers with external claims", func(ctx context.Context) {
			pod := b.PodExternal()
			claim := b.ExternalClaim()
			pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
			pod.Spec.InitContainers[0].Name += "-init"
			// This must succeed for the pod to start.
			pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
			b.Create(ctx, pod, claim)

			b.TestPod(ctx, f, pod)
		})

		ginkgo.It("removes reservation from claim when pod is done", func(ctx context.Context) {
			pod := b.PodExternal()
			claim := b.ExternalClaim()
			pod.Spec.Containers[0].Command = []string{"true"}
			b.Create(ctx, claim, pod)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be unreserved")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return f.ClientSet.ResourceV1().ResourceClaims(pod.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.ReservedFor", gomega.BeEmpty()), "reservation should have been removed")
		})

		ginkgo.It("deletes generated claims when pod is done", func(ctx context.Context) {
			pod, template := b.PodInline()
			pod.Spec.Containers[0].Command = []string{"true"}
			b.Create(ctx, template, pod)

			ginkgo.By("waiting for pod to finish")
			framework.ExpectNoError(e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace), "wait for pod to finish")
			ginkgo.By("waiting for claim to be deleted")
			gomega.Eventually(ctx, func(ctx context.Context) ([]resourceapi.ResourceClaim, error) {
				claims, err := f.ClientSet.ResourceV1().ResourceClaims(pod.Namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return nil, err
				}
				return claims.Items, nil
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.BeEmpty(), "claim should have been deleted")
		})

		ginkgo.It("does not delete generated claims when pod is restarting", func(ctx context.Context) {
			pod, template := b.PodInline()
			pod.Spec.Containers[0].Command = []string{"sh", "-c", "sleep 1; exit 1"}
			pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			b.Create(ctx, template, pod)

			ginkgo.By("waiting for pod to restart twice")
			gomega.Eventually(ctx, func(ctx context.Context) (*v1.Pod, error) {
				return f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodStartSlow).Should(gomega.HaveField("Status.ContainerStatuses", gomega.ContainElements(gomega.HaveField("RestartCount", gomega.BeNumerically(">=", 2)))))
		})
	})

	// kubelet tests with individual configurations.
	f.Context("kubelet", feature.DynamicResourceAllocation, func() {
		ginkgo.It("runs pod after driver starts", func(ctx context.Context) {
			nodes := drautils.NewNodesNow(ctx, f, 1, 4)
			driver := drautils.NewDriverInstance(f)
			b := drautils.NewBuilderNow(ctx, f, driver)

			claim := b.ExternalClaim()
			pod := b.PodExternal()
			b.Create(ctx, claim, pod)

			// Cannot run pod, no devices.
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			// Set up driver, which makes devices available.
			driver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// Now it should run.
			b.TestPod(ctx, f, pod)

			// We need to clean up explicitly because the normal
			// cleanup doesn't work (driver shuts down first).
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
		})

		// Seamless upgrade support was added in Kubernetes 1.33.
		f.It("rolling update", f.WithLabel("KubeletMinVersion:1.33"), func(ctx context.Context) {
			nodes := drautils.NewNodesNow(ctx, f, 1, 1)

			oldDriver := drautils.NewDriverInstance(f)
			oldDriver.InstanceSuffix = "-old"
			oldDriver.RollingUpdate = true
			oldDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// We expect one ResourceSlice per node from the driver.
			getSlices := oldDriver.NewGetSlices()
			gomega.Eventually(ctx, getSlices).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))
			initialSlices, err := getSlices(ctx)
			framework.ExpectNoError(err)

			// Same driver name, different socket paths because of rolling update.
			newDriver := drautils.NewDriverInstance(f)
			newDriver.InstanceSuffix = "-new"
			newDriver.RollingUpdate = true
			newDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// Stop old driver instance.
			oldDriver.TearDown(ctx)

			// Build behaves the same for both driver instances.
			b := drautils.NewBuilderNow(ctx, f, oldDriver)
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)

			// The exact same slices should still exist.
			finalSlices, err := getSlices(ctx)
			framework.ExpectNoError(err)
			gomega.Expect(finalSlices.Items).Should(gomega.Equal(initialSlices.Items))

			// We need to clean up explicitly because the normal
			// cleanup doesn't work (driver shuts down first).
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
		})

		// Seamless upgrade support was added in Kubernetes 1.33.
		f.It("failed update", f.WithLabel("KubeletMinVersion:1.33"), func(ctx context.Context) {
			nodes := drautils.NewNodesNow(ctx, f, 1, 1)

			oldDriver := drautils.NewDriverInstance(f)
			oldDriver.InstanceSuffix = "-old"
			oldDriver.RollingUpdate = true
			oldDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// We expect one ResourceSlice per node from the driver.
			getSlices := oldDriver.NewGetSlices()
			gomega.Eventually(ctx, getSlices).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))
			initialSlices, err := getSlices(ctx)
			framework.ExpectNoError(err)

			// Same driver name, different socket paths because of rolling update.
			newDriver := drautils.NewDriverInstance(f)
			newDriver.InstanceSuffix = "-new"
			newDriver.RollingUpdate = true
			newDriver.ExpectResourceSliceRemoval = false
			newDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// Stop new driver instance, simulating the failure of the new instance.
			// The kubelet should still have the old instance.
			newDriver.TearDown(ctx)

			// Build behaves the same for both driver instances.
			b := drautils.NewBuilderNow(ctx, f, oldDriver)
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)

			// The exact same slices should still exist.
			finalSlices, err := getSlices(ctx)
			framework.ExpectNoError(err)
			gomega.Expect(finalSlices.Items).Should(gomega.Equal(initialSlices.Items))

			// We need to clean up explicitly because the normal
			// cleanup doesn't work (driver shuts down first).
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))
		})

		// Seamless upgrade support was added in Kubernetes 1.33.
		f.It("sequential update with pods replacing each other", f.WithLabel("KubeletMinVersion:1.33"), framework.WithSlow(), func(ctx context.Context) {
			nodes := drautils.NewNodesNow(ctx, f, 1, 1)

			// Same driver name, same socket path.
			oldDriver := drautils.NewDriverInstance(f)
			oldDriver.InstanceSuffix = "-old"
			oldDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))

			// Collect set of resource slices for that driver.
			listSlices := framework.ListObjects(f.ClientSet.ResourceV1().ResourceSlices().List, metav1.ListOptions{
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
			newDriver := drautils.NewDriverInstance(f)
			newDriver.InstanceSuffix = "-new"
			newDriver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))
			updateDuration := time.Since(start)

			// Build behaves the same for both driver instances.
			b := drautils.NewBuilderNow(ctx, f, oldDriver)
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)

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
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete))

			// Now shut down for good and wait for the kubelet to react.
			// This takes time...
			ginkgo.By("uninstalling driver and waiting for ResourceSlice wiping")
			newDriver.TearDown(ctx)
			newDriver.IsGone(ctx)
		})
	})

	// Tests that have the `withKubelet` argument can run with or without kubelet support for plugins and DRA, aka feature.DynamicResourceAllocation.
	// Without it, the test driver publishes ResourceSlices, but does not attempt to register itself.
	// Tests only expect pods to get scheduled, but not to become running.
	//
	// TODO before conformance promotion: add https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/conformance-tests.md#sample-conformance-test meta data

	singleNodeTests := func(withKubelet bool) {
		nodes := drautils.NewNodes(f, 1, 1)
		maxAllocations := 1
		numPods := 5
		driver := drautils.NewDriver(f, nodes, drautils.DriverResources(maxAllocations)) // All tests get their own driver instance.
		driver.WithKubelet = withKubelet
		b := drautils.NewBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.ClassParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.ParametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports claim and class parameters", func(ctx context.Context) {
			pod, template := b.PodInline()
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("supports reusing resources", func(ctx context.Context) {
			var objects []klog.KMetadata
			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pod, template := b.PodInline()
				pods[i] = pod
				objects = append(objects, pod, template)
			}

			b.Create(ctx, objects...)

			// We don't know the order. All that matters is that all of them get scheduled eventually.
			var wg sync.WaitGroup
			wg.Add(numPods)
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					b.TestPod(ctx, f, pod, expectedEnv...)
					err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err, "delete pod")
					framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, time.Duration(numPods)*f.Timeouts.PodStartSlow))
				}()
			}
			wg.Wait()
		})

		ginkgo.It("supports sharing a claim concurrently", func(ctx context.Context) {
			var objects []klog.KMetadata
			objects = append(objects, b.ExternalClaim())
			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pod := b.PodExternal()
				pods[i] = pod
				objects = append(objects, pod)
			}

			b.Create(ctx, objects...)

			// We don't know the order. All that matters is that all of them get scheduled eventually.
			f.Timeouts.PodStartSlow *= time.Duration(numPods)
			var wg sync.WaitGroup
			wg.Add(numPods)
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					b.TestPod(ctx, f, pod, expectedEnv...)
				}()
			}
			wg.Wait()
		})

		ginkgo.It("retries pod scheduling after creating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.PodInline()
			deviceClassName := template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName
			class, err := f.ClientSet.ResourceV1().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			deviceClassName += "-b"
			template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName = deviceClassName
			objects = append(objects, template, pod)
			b.Create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			class.UID = ""
			class.ResourceVersion = ""
			class.Name = deviceClassName
			b.Create(ctx, class)

			b.TestPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("retries pod scheduling after updating device class", func(ctx context.Context) {
			var objects []klog.KMetadata
			pod, template := b.PodInline()

			// First modify the class so that it matches no nodes (for classic DRA) and no devices (structured parameters).
			deviceClassName := template.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName
			class, err := f.ClientSet.ResourceV1().DeviceClasses().Get(ctx, deviceClassName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			originalClass := class.DeepCopy()
			class.Spec.Selectors = []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{
					Expression: "false",
				},
			}}
			class, err = f.ClientSet.ResourceV1().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// Now create the pod.
			objects = append(objects, template, pod)
			b.Create(ctx, objects...)

			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

			// Unblock the pod.
			class.Spec.Selectors = originalClass.Spec.Selectors
			_, err = f.ClientSet.ResourceV1().DeviceClasses().Update(ctx, class, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			b.TestPod(ctx, f, pod, expectedEnv...)
		})

		ginkgo.It("runs a pod without a generated resource claim", func(ctx context.Context) {
			pod, _ /* template */ := b.PodInline()
			created := b.Create(ctx, pod)
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

		ginkgo.It("supports simple pod referencing inline resource claim", func(ctx context.Context) {
			pod, template := b.PodInline()
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod)
		})

		ginkgo.It("supports inline claim referenced by multiple containers", func(ctx context.Context) {
			pod, template := b.PodInlineMultiple()
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod)
		})

		ginkgo.It("supports simple pod referencing external resource claim", func(ctx context.Context) {
			pod := b.PodExternal()
			claim := b.ExternalClaim()
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)
		})

		ginkgo.It("supports external claim referenced by multiple pods", func(ctx context.Context) {
			pod1 := b.PodExternal()
			pod2 := b.PodExternal()
			pod3 := b.PodExternal()
			claim := b.ExternalClaim()
			b.Create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.TestPod(ctx, f, pod)
			}
		})

		ginkgo.It("supports external claim referenced by multiple containers of multiple pods", func(ctx context.Context) {
			pod1 := b.PodExternalMultiple()
			pod2 := b.PodExternalMultiple()
			pod3 := b.PodExternalMultiple()
			claim := b.ExternalClaim()
			b.Create(ctx, claim, pod1, pod2, pod3)

			for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
				b.TestPod(ctx, f, pod)
			}
		})

		ginkgo.It("supports init containers", func(ctx context.Context) {
			pod, template := b.PodInline()
			pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
			pod.Spec.InitContainers[0].Name += "-init"
			// This must succeed for the pod to start.
			pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
			b.Create(ctx, pod, template)

			b.TestPod(ctx, f, pod)
		})

		ginkgo.It("must deallocate after use", func(ctx context.Context) {
			pod := b.PodExternal()
			claim := b.ExternalClaim()
			b.Create(ctx, claim, pod)

			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))

			b.TestPod(ctx, f, pod)

			ginkgo.By(fmt.Sprintf("deleting pod %s", klog.KObj(pod)))
			framework.ExpectNoError(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))

			ginkgo.By("waiting for claim to get deallocated")
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				return f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.HaveField("Status.Allocation", (*resourceapi.AllocationResult)(nil)))
		})

		// https://github.com/kubernetes/kubernetes/issues/133488
		// It conflicts with "must run pods with extended resource on dra nodes and device plugin nodes" test case,
		// because device plugin does not clean up the extended resource "example.com/resource", and kubelet still
		// keeps "example.com/resource" : 0 in node.status.Capacity.
		// add WithFlaky to filter out the following test until we can clean up the leaked "example.com/resource" in node.status.
		if withKubelet {
			// Serial because the example device plugin can only be deployed with one instance at a time.
			f.It("supports extended resources together with ResourceClaim", f.WithSerial(), f.WithFlaky(), func(ctx context.Context) {
				extendedResourceName := deployDevicePlugin(ctx, f, nodes.NodeNames[0:1])

				pod := b.PodExternal()
				resources := v1.ResourceList{extendedResourceName: resource.MustParse("1")}
				pod.Spec.Containers[0].Resources.Requests = resources
				pod.Spec.Containers[0].Resources.Limits = resources
				claim := b.ExternalClaim()
				b.Create(ctx, claim, pod)
				b.TestPod(ctx, f, pod)
			})
		}
	}

	// The following tests only make sense when there is more than one node.
	// They get skipped when there's only one node.
	multiNodeTests := func(withKubelet bool) {
		nodes := drautils.NewNodes(f, 2, 8)

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
			driver := drautils.NewDriver(f, nodes, drautils.DriverResources(-1, devicesPerNode...))
			driver.WithKubelet = withKubelet
			b := drautils.NewBuilder(f, driver)

			ginkgo.It("keeps pod pending because of CEL runtime errors", func(ctx context.Context) {
				// When pod scheduling encounters CEL runtime errors for some nodes, but not all,
				// it should still not schedule the pod because there is something wrong with it.
				// Scheduling it would make it harder to detect that there is a problem.
				//
				// This matches the "CEL-runtime-error-for-subset-of-nodes" unit test, except that
				// here we try it in combination with the actual scheduler and can extend it with
				// other checks, like event handling (future extension).

				gomega.Eventually(ctx, framework.ListObjects(f.ClientSet.ResourceV1().ResourceSlices().List,
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

				pod, template := b.PodInline()
				template.Spec.Spec.Devices.Requests[0].Exactly.Selectors = append(template.Spec.Spec.Devices.Requests[0].Exactly.Selectors,
					resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							// Runtime error on one node, but not all.
							Expression: fmt.Sprintf(`device.attributes["%s"].exists`, driver.Name),
						},
					},
				)
				b.Create(ctx, pod, template)

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
			driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
			driver.WithKubelet = withKubelet
			b := drautils.NewBuilder(f, driver)

			ginkgo.It("uses all resources", func(ctx context.Context) {
				var objs []klog.KMetadata
				var pods []*v1.Pod
				for i := 0; i < len(nodes.NodeNames); i++ {
					pod, template := b.PodInline()
					pods = append(pods, pod)
					objs = append(objs, pod, template)
				}
				b.Create(ctx, objs...)

				for _, pod := range pods {
					if withKubelet {
						err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
						framework.ExpectNoError(err, "start pod")
					} else {
						err := e2epod.WaitForPodScheduled(ctx, f.ClientSet, pod.Namespace, pod.Name)
						framework.ExpectNoError(err, "schedule pod")
					}
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

	prioritizedListTests := func() {
		nodes := drautils.NewNodes(f, 1, 1)

		driver1Params, driver1Env := `{"driver":"1"}`, []string{"admin_driver", "1"}
		driver2Params, driver2Env := `{"driver":"2"}`, []string{"admin_driver", "2"}

		driver1 := drautils.NewDriver(f, nodes, drautils.DriverResources(-1, []map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
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
		b1 := drautils.NewBuilder(f, driver1)
		b1.ClassParameters = driver1Params

		driver2 := drautils.NewDriver(f, nodes, drautils.DriverResources(-1, []map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			{
				"device-2-1": {
					"dra.example.com/version":  {StringValue: ptr.To("1.0.0")},
					"dra.example.com/pcieRoot": {StringValue: ptr.To("foo")},
				},
			},
		}...))
		driver2.NameSuffix = "-2"
		b2 := drautils.NewBuilder(f, driver2)
		b2.ClassParameters = driver2Params

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
									DeviceClassName: b1.ClassName(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           3,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b1.ClassName(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           2,
								},
								{
									Name:            "sub-request-3",
									DeviceClassName: b1.ClassName(),
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
										Driver: driver1.Name,
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
			pod := b1.PodExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.Create(ctx, claim, pod)
			b1.TestPod(ctx, f, pod)

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
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
									DeviceClassName: b1.ClassName(),
									AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
									Count:           3,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b1.ClassName(),
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
										Driver: driver1.Name,
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
										Driver: driver1.Name,
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
										Driver: driver1.Name,
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
			pod := b1.PodExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.Create(ctx, claim, pod)
			var expectedEnv []string
			expectedEnv = append(expectedEnv, parentReqEnv...)
			expectedEnv = append(expectedEnv, subReq2Env...)
			b1.TestPod(ctx, f, pod, expectedEnv...)
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
										DeviceClassName: b1.ClassName(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
									{
										Name:            "sub-request-2",
										DeviceClassName: b1.ClassName(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
								},
							},
							{
								Name: "request-2",
								Exactly: &resourceapi.ExactDeviceRequest{
									DeviceClassName: b2.ClassName(),
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
										Driver: driver1.Name,
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
			pod := b1.PodExternal()
			podClaimName := "resource-claim"
			externalClaimName := "external-multiclaim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &externalClaimName,
				},
			}
			b1.Create(ctx, claim, pod)
			b1.TestPod(ctx, f, pod)

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
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
										DeviceClassName: b1.ClassName(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           20, // Requests more than are available.
									},
									{
										Name:            "sub-request-2",
										DeviceClassName: b1.ClassName(),
										AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
										Count:           1,
									},
								},
							},
							{
								Name: "request-2",
								Exactly: &resourceapi.ExactDeviceRequest{
									DeviceClassName: b2.ClassName(),
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
										Driver: driver1.Name,
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
										Driver: driver1.Name,
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
										Driver: driver1.Name,
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
										Driver: driver2.Name,
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
			pod := b1.Pod()
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

			b1.Create(ctx, claim, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")

			var allocatedResourceClaim *resourceapi.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
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
			drautils.TestContainerEnv(ctx, f, pod, "with-resource-0", true, req1ExpectedEnv...)

			req2ExpectedEnv := []string{
				"claim_external_multiclaim_request_2",
				"true",
			}
			req2ExpectedEnv = append(req2ExpectedEnv, req2Env...)
			req2ExpectedEnv = append(req2ExpectedEnv, driver2Env...)
			drautils.TestContainerEnv(ctx, f, pod, "with-resource-1", true, req2ExpectedEnv...)
		})
	}

	v1beta1Tests := func() {
		nodes := drautils.NewNodes(f, 1, 1)
		maxAllocations := 1
		driver := drautils.NewDriver(f, nodes, drautils.DriverResources(maxAllocations))
		b := drautils.NewBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.ClassParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.ParametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports simple ResourceClaim", func(ctx context.Context) {
			pod, template := b.PodInlineWithV1beta1()
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod, expectedEnv...)
		})

		f.It("supports requests with alternatives", f.WithFeatureGate(features.DRAPrioritizedList), func(ctx context.Context) {
			claimName := "external-multiclaim"
			parameters, _ := b.ParametersEnv()
			claim := &resourcev1beta1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: claimName,
				},
				Spec: resourcev1beta1.ResourceClaimSpec{
					Devices: resourcev1beta1.DeviceClaim{
						Requests: []resourcev1beta1.DeviceRequest{{
							Name: "request-1",
							FirstAvailable: []resourcev1beta1.DeviceSubRequest{
								{
									Name:            "sub-request-1",
									DeviceClassName: b.ClassName(),
									AllocationMode:  resourcev1beta1.DeviceAllocationModeExactCount,
									Count:           2,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b.ClassName(),
									AllocationMode:  resourcev1beta1.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						}},
						Config: []resourcev1beta1.DeviceClaimConfiguration{{
							DeviceConfiguration: resourcev1beta1.DeviceConfiguration{
								Opaque: &resourcev1beta1.OpaqueDeviceConfiguration{
									Driver: driver.Name,
									Parameters: runtime.RawExtension{
										Raw: []byte(parameters),
									},
								},
							},
						}},
					},
				},
			}
			pod := b.PodExternal()
			podClaimName := "resource-claim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &claimName,
				},
			}
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod, expectedEnv...)

			var allocatedResourceClaim *resourcev1beta1.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourcev1beta1.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourcev1beta1.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(1))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
		})
	}

	v1beta2Tests := func() {
		nodes := drautils.NewNodes(f, 1, 1)
		maxAllocations := 1
		driver := drautils.NewDriver(f, nodes, drautils.DriverResources(maxAllocations))
		b := drautils.NewBuilder(f, driver)
		// We have to set the parameters *before* creating the class.
		b.ClassParameters = `{"x":"y"}`
		expectedEnv := []string{"admin_x", "y"}
		_, expected := b.ParametersEnv()
		expectedEnv = append(expectedEnv, expected...)

		ginkgo.It("supports simple ResourceClaim", func(ctx context.Context) {
			pod, template := b.PodInlineWithV1beta2()
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod, expectedEnv...)
		})

		f.It("supports requests with alternatives", f.WithFeatureGate(features.DRAPrioritizedList), func(ctx context.Context) {
			claimName := "external-multiclaim"
			parameters, _ := b.ParametersEnv()
			claim := &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: claimName,
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{{
							Name: "request-1",
							FirstAvailable: []resourcev1beta2.DeviceSubRequest{
								{
									Name:            "sub-request-1",
									DeviceClassName: b.ClassName(),
									AllocationMode:  resourcev1beta2.DeviceAllocationModeExactCount,
									Count:           2,
								},
								{
									Name:            "sub-request-2",
									DeviceClassName: b.ClassName(),
									AllocationMode:  resourcev1beta2.DeviceAllocationModeExactCount,
									Count:           1,
								},
							},
						}},
						Config: []resourcev1beta2.DeviceClaimConfiguration{{
							DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
								Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
									Driver: driver.Name,
									Parameters: runtime.RawExtension{
										Raw: []byte(parameters),
									},
								},
							},
						}},
					},
				},
			}
			pod := b.PodExternal()
			podClaimName := "resource-claim"
			pod.Spec.ResourceClaims = []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &claimName,
				},
			}
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod, expectedEnv...)

			var allocatedResourceClaim *resourcev1beta2.ResourceClaim
			gomega.Eventually(ctx, func(ctx context.Context) (*resourcev1beta2.ResourceClaim, error) {
				var err error
				allocatedResourceClaim, err = f.ClientSet.ResourceV1beta2().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
				return allocatedResourceClaim, err
			}).WithTimeout(f.Timeouts.PodDelete).ShouldNot(gomega.HaveField("Status.Allocation", (*resourcev1beta2.AllocationResult)(nil)))
			results := allocatedResourceClaim.Status.Allocation.Devices.Results
			gomega.Expect(results).To(gomega.HaveLen(1))
			gomega.Expect(results[0].Request).To(gomega.Equal("request-1/sub-request-2"))
		})
	}

	partitionableDevicesTests := func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, drautils.ToDriverResources(
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
		b := drautils.NewBuilder(f, driver)

		f.It("must consume and free up counters", func(ctx context.Context) {
			// The first pod will use one of the devices. Since both devices are
			// available, there should be sufficient counters left to allocate
			// a device.
			claim := b.ExternalClaim()
			pod := b.PodExternal()
			pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)

			// For the second pod, there should not be sufficient counters left, so
			// it should not succeed. This means the pod should remain in the pending state.
			claim2 := b.ExternalClaim()
			pod2 := b.PodExternal()
			pod2.Spec.ResourceClaims[0].ResourceClaimName = &claim2.Name
			b.Create(ctx, claim2, pod2)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				testPod, err := f.ClientSet.CoreV1().Pods(pod2.Namespace).Get(ctx, pod2.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the test pod %s to exist: %w", pod2.Name, err)
				}
				if testPod.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s: unexpected status %s, expected status: %s", pod2.Name, testPod.Status.Phase, v1.PodPending)
				}
				return nil
			}, 20*time.Second, 200*time.Millisecond).Should(gomega.Succeed())

			// Delete the first pod
			b.DeletePodAndWaitForNotFound(ctx, pod)

			// There shoud not be available devices for pod2.
			b.TestPod(ctx, f, pod2)
		})
	}

	consumableCapacityTests := func() {
		nodes := drautils.NewNodes(f, 1, 1)
		// single device which allows multiple allocations and has 80Gi consumable memory.
		driver := drautils.NewDriver(f, nodes, drautils.ToDriverResources(
			[]resourceapi.CounterSet{},
			[]resourceapi.Device{
				{
					Name:                     "consumable-device-1",
					AllowMultipleAllocations: ptr.To(true),
					Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
						"memory": {
							Value: resource.MustParse("8Gi"),
							RequestPolicy: &resourceapi.CapacityRequestPolicy{
								Default: ptr.To(resource.MustParse("1Gi")),
								ValidRange: &resourceapi.CapacityRequestPolicyRange{
									Min: ptr.To(resource.MustParse("1Gi")),
								},
							},
						},
					},
				},
			}...,
		))
		b := drautils.NewBuilder(f, driver)

		f.It("must allow multiple allocations and consume capacity", f.WithLabel("KubeletMinVersion:1.34"), func(ctx context.Context) {
			// The first pod will use 4Gi of the device.
			claim := b.ExternalClaim()
			claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
				Requests: map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("4Gi"),
				},
			}
			pod := b.PodExternal()
			pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
			b.Create(ctx, claim, pod)
			b.TestPod(ctx, f, pod)

			// The second pod will be failed to request 8Gi capacity.
			claim2 := b.ExternalClaim()
			claim2.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
				Requests: map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("8Gi"),
				},
			}
			pod2 := b.PodExternal()
			pod2.Spec.ResourceClaims[0].ResourceClaimName = &claim2.Name
			b.Create(ctx, claim2, pod2)

			// The third pod should be able to use the rest 4Gi of the device.
			claim3 := b.ExternalClaim()
			claim3.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
				Requests: map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("4Gi"),
				},
			}
			pod3 := b.PodExternal()
			pod3.Spec.ResourceClaims[0].ResourceClaimName = &claim3.Name
			b.Create(ctx, claim3, pod3)
			b.TestPod(ctx, f, pod3)

			gomega.Consistently(ctx, func(ctx context.Context) error {
				testPod2, err := f.ClientSet.CoreV1().Pods(pod2.Namespace).Get(ctx, pod2.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the test pod %s to exist: %w", pod2.Name, err)
				}
				if testPod2.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s: unexpected status %s, expected status: %s", pod2.Name, testPod2.Status.Phase, v1.PodPending)
				}
				return nil
			}, 20*time.Second, 200*time.Millisecond).Should(gomega.Succeed())

			// Delete the first and third pod
			b.DeletePodAndWaitForNotFound(ctx, pod)
			b.DeletePodAndWaitForNotFound(ctx, pod3)

			// There should be available capacity for pod2 now.
			b.TestPod(ctx, f, pod2)
		})
	}

	// It is okay to use the same context multiple times (like "control plane"),
	// as long as the test names the still remain unique overall.

	framework.Context("control plane", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func() { singleNodeTests(false) })
	framework.Context("kubelet", feature.DynamicResourceAllocation, "on single node", func() { singleNodeTests(true) })

	framework.Context("control plane", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func() { multiNodeTests(false) })
	framework.Context("kubelet", feature.DynamicResourceAllocation, "on multiple nodes", func() { multiNodeTests(true) })

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.DRAPrioritizedList), prioritizedListTests)

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.DRAConsumableCapacity), consumableCapacityTests)

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.DRAConsumableCapacity), consumableCapacityTests)

	framework.Context("kubelet", feature.DynamicResourceAllocation, "with v1beta1 API", v1beta1Tests)
	framework.Context("kubelet", feature.DynamicResourceAllocation, "with v1beta2 API", v1beta2Tests)

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.DRAPartitionableDevices), partitionableDevicesTests)

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.DRADeviceTaints), func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, drautils.NetworkResources(10, false), drautils.TaintAllDevices(resourceapi.DeviceTaint{
			Key:    "example.com/taint",
			Value:  "tainted",
			Effect: resourceapi.DeviceTaintEffectNoSchedule,
		}))
		b := drautils.NewBuilder(f, driver)

		f.It("DeviceTaint keeps pod pending", func(ctx context.Context) {
			pod, template := b.PodInline()
			b.Create(ctx, pod, template)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
		})

		f.It("DeviceToleration enables pod scheduling", func(ctx context.Context) {
			pod, template := b.PodInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.Tolerations = []resourceapi.DeviceToleration{{
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
				Operator: resourceapi.DeviceTolerationOpExists,
				// No key: tolerate *all* taints with this effect.
			}}
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod)
		})

		f.It("DeviceTaintRule evicts pod", func(ctx context.Context) {
			pod, template := b.PodInline()
			template.Spec.Spec.Devices.Requests[0].Exactly.Tolerations = []resourceapi.DeviceToleration{{
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
				Operator: resourceapi.DeviceTolerationOpExists,
				// No key: tolerate *all* taints with this effect.
			}}
			// Add a finalizer to ensure that we get a chance to test the pod status after eviction (= deletion).
			pod.Finalizers = []string{"e2e-test/dont-delete-me"}
			b.Create(ctx, pod, template)
			b.TestPod(ctx, f, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				gomega.Eventually(ctx, func(ctx context.Context) error {
					// Unblock shutdown by removing the finalizer.
					pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
					if err != nil {
						return fmt.Errorf("get pod: %w", err)
					}
					pod.Finalizers = nil
					_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(ctx, pod, metav1.UpdateOptions{})
					if err != nil {
						return fmt.Errorf("remove finalizers from pod: %w", err)
					}
					return nil
				}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Succeed(), "Failed to remove finalizers")
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
			createdTaint := b.Create(ctx, taint)
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

	framework.Context(f.WithFeatureGate(features.DRAExtendedResource), func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, drautils.NetworkResources(10, false))
		b := drautils.NewBuilder(f, driver)
		b.UseExtendedResourceName = true

		ginkgo.It("must run a pod with extended resource with one container one resource", func(ctx context.Context) {
			pod := b.Pod()
			res := v1.ResourceList{}
			res[v1.ResourceName(drautils.ExtendedResourceName(0))] = resource.MustParse("1")
			pod.Spec.Containers[0].Resources.Requests = res
			pod.Spec.Containers[0].Resources.Limits = res

			b.Create(ctx, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")
			containerEnv := []string{
				"container_0_request_0", "true",
			}
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[0].Name, false, containerEnv...)
		})

		ginkgo.It("must run a pod with extended resource with one container three resources", func(ctx context.Context) {
			pod := b.Pod()
			res := v1.ResourceList{}
			for i := range 3 {
				res[v1.ResourceName(drautils.ExtendedResourceName(i))] = resource.MustParse("1")
			}
			pod.Spec.Containers[0].Resources.Requests = res
			pod.Spec.Containers[0].Resources.Limits = res

			b.Create(ctx, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")
			containerEnv := []string{
				"container_0_request_0", "true",
				"container_0_request_1", "true",
				"container_0_request_2", "true",
			}
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[0].Name, false, containerEnv...)
		})
		ginkgo.It("must run a pod with extended resource with three containers one resource each", func(ctx context.Context) {
			pod := b.Pod()
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers[0].Name = "container0"
			pod.Spec.Containers[1].Name = "container1"
			pod.Spec.Containers[2].Name = "container2"

			for i := range 3 {
				res := v1.ResourceList{}
				res[v1.ResourceName(drautils.ExtendedResourceName(i))] = resource.MustParse("1")
				pod.Spec.Containers[i].Resources.Requests = res
				pod.Spec.Containers[i].Resources.Limits = res
			}

			b.Create(ctx, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")
			for i := range 3 {
				containerEnv := []string{
					fmt.Sprintf("container_%d_request_0", i), "true",
				}
				drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[i].Name, false, containerEnv...)
			}
		})
		ginkgo.It("must run a pod with extended resource with three containers multiple resources each", func(ctx context.Context) {
			pod := b.Pod()
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy())
			pod.Spec.Containers[0].Name = "container0"
			pod.Spec.Containers[1].Name = "container1"
			pod.Spec.Containers[2].Name = "container2"

			res := v1.ResourceList{}
			res[v1.ResourceName(drautils.ExtendedResourceName(0))] = resource.MustParse("1")
			pod.Spec.Containers[0].Resources.Requests = res
			pod.Spec.Containers[0].Resources.Limits = res
			res = v1.ResourceList{}
			res[v1.ResourceName(drautils.ExtendedResourceName(1))] = resource.MustParse("1")
			res[v1.ResourceName(drautils.ExtendedResourceName(2))] = resource.MustParse("1")
			pod.Spec.Containers[1].Resources.Requests = res
			pod.Spec.Containers[1].Resources.Limits = res
			res = v1.ResourceList{}
			res[v1.ResourceName(drautils.ExtendedResourceName(3))] = resource.MustParse("1")
			res[v1.ResourceName(drautils.ExtendedResourceName(4))] = resource.MustParse("1")
			res[v1.ResourceName(drautils.ExtendedResourceName(5))] = resource.MustParse("1")
			pod.Spec.Containers[2].Resources.Requests = res
			pod.Spec.Containers[2].Resources.Limits = res

			b.Create(ctx, pod)
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err, "start pod")
			containerEnv := []string{
				"container_0_request_0", "true",
			}
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[0].Name, false, containerEnv...)
			containerEnv = []string{
				"container_1_request_0", "true",
				"container_1_request_1", "true",
			}
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[1].Name, false, containerEnv...)
			containerEnv = []string{
				"container_2_request_0", "true",
				"container_2_request_1", "true",
				"container_2_request_2", "true",
			}
			drautils.TestContainerEnv(ctx, f, pod, pod.Spec.Containers[2].Name, false, containerEnv...)
		})
	})

	framework.Context(f.WithFeatureGate(features.DRAExtendedResource), func() {
		nodes := drautils.NewNodes(f, 2, 2)
		nodes.NumReservedNodes = 1
		driver := drautils.NewDriver(f, nodes, drautils.NetworkResources(2, false))
		b := drautils.NewBuilder(f, driver)
		b.UseExtendedResourceName = true

		// This test needs the entire test cluster for itself, one node in the cluster
		// is deployed device plugin for the test, therefore it is marked as serial.
		// The test runs two pods, one pod request extended resource backed by DRA,
		// the other pod requests extended resource by device plugin.
		f.It("must run pods with extended resource on dra nodes and device plugin nodes", f.WithSerial(), func(ctx context.Context) {
			extendedResourceName := deployDevicePlugin(ctx, f, nodes.ExtraNodeNames)
			// drautils.ExtendedResourceName(-1) must be the same as the returned extendedResourceName
			// drautils.ExtendedResourceName(-1) is used for DRA drivers
			// extendedResourceName is used for device plugin.
			gomega.Expect(string(extendedResourceName)).To(gomega.Equal(drautils.ExtendedResourceName(-1)))

			pod1 := b.Pod()
			res := v1.ResourceList{}
			res[v1.ResourceName(drautils.ExtendedResourceName(-1))] = resource.MustParse("2")
			pod1.Spec.Containers[0].Resources.Requests = res
			pod1.Spec.Containers[0].Resources.Limits = res
			b.Create(ctx, pod1)

			pod2 := b.Pod()
			pod2.Spec.Containers[0].Resources.Requests = res
			pod2.Spec.Containers[0].Resources.Limits = res
			b.Create(ctx, pod2)

			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod1)
			framework.ExpectNoError(err, "start pod1")
			err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod2)
			framework.ExpectNoError(err, "start pod2")

			scheduledPod1, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod1.Name, metav1.GetOptions{})
			gomega.Expect(scheduledPod1).ToNot(gomega.BeNil())
			framework.ExpectNoError(err)

			scheduledPod2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod2.Name, metav1.GetOptions{})
			gomega.Expect(scheduledPod2).ToNot(gomega.BeNil())
			framework.ExpectNoError(err)

			draPod := scheduledPod1
			devicePluginPod := scheduledPod2
			if scheduledPod1.Spec.NodeName == nodes.ExtraNodeNames[0] {
				draPod = scheduledPod2
				devicePluginPod = scheduledPod1
			}

			gomega.Expect(devicePluginPod.Spec.NodeName).To(gomega.Equal(nodes.ExtraNodeNames[0]))
			gomega.Expect(devicePluginPod.Status.ExtendedResourceClaimStatus).To(gomega.BeNil())
			gomega.Expect(draPod.Spec.NodeName).To(gomega.Equal(nodes.NodeNames[0]))
			containerEnv := []string{
				"container_0_request_0", "true",
			}
			drautils.TestContainerEnv(ctx, f, draPod, draPod.Spec.Containers[0].Name, false, containerEnv...)
		})
	})

	ginkgo.Context("ResourceSlice Controller", func() {
		// This is a stress test for creating many large slices.
		// Each slice is as large as API limits allow.
		//
		// Could become a conformance test because it only depends
		// on the apiserver.
		f.It("creates slices", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func(ctx context.Context) {
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
			numDevices := 0
			for i := 0; i < numSlices; i++ {
				devices := make([]resourceapi.Device, resourceapi.ResourceSliceMaxDevices)
				for e := 0; e < resourceapi.ResourceSliceMaxDevices; e++ {
					device := resourceapi.Device{
						Name:       devicePrefix + strings.Repeat("x", validation.DNS1035LabelMaxLength-len(devicePrefix)-6) + fmt.Sprintf("%06d", numDevices),
						Attributes: make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice),
					}
					numDevices++
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
			listSlices := framework.ListObjects(f.ClientSet.ResourceV1().ResourceSlices().List, metav1.ListOptions{
				FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
			})

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
				gomega.Eventually(ctx, func(ctx context.Context) (*resourceapi.ResourceSliceList, error) {
					err := f.ClientSet.ResourceV1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
						FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
					})
					if err != nil {
						return nil, fmt.Errorf("delete slices: %w", err)
					}
					return listSlices(ctx)
				}).Should(gomega.HaveField("Items", gomega.BeEmpty()))
			})

			// Eventually we should have all desired slices.
			gomega.Eventually(ctx, listSlices).WithTimeout(3 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveLen(numSlices)))

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
			gomega.Eventually(ctx, listSlices).WithTimeout(2 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveExactElements(emptySlice)))
			expectStats = resourceslice.Stats{NumCreates: int64(numSlices) + 1, NumDeletes: int64(numSlices)}

			// There is a window of time where the ResourceSlice exists and is
			// returned in a list but before that ResourceSlice is accounted for
			// in the controller's stats, consisting mostly of network latency
			// between this test process and the API server. Wait for the stats
			// to converge before asserting there are no further changes.
			gomega.Eventually(ctx, controller.GetStats).WithTimeout(30 * time.Second).Should(gomega.Equal(expectStats))

			gomega.Consistently(ctx, controller.GetStats).WithTimeout(2 * mutationCacheTTL).Should(gomega.Equal(expectStats))
		})
	})

	framework.Context("control plane", func() {
		nodes := drautils.NewNodes(f, 1, 1)
		driver := drautils.NewDriver(f, nodes, drautils.NetworkResources(10, false))
		driver.WithKubelet = false
		b := drautils.NewBuilder(f, driver)

		f.It("validate ResourceClaimTemplate and ResourceClaim for admin access", f.WithFeatureGate(features.DRAAdminAccess), func(ctx context.Context) {
			// Attempt to create claim and claim template with admin access. Must fail eventually.
			claim := b.ExternalClaim()
			claim.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			_, claimTemplate := b.PodInline()
			claimTemplate.Spec.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			matchValidationError := gomega.MatchError(gomega.ContainSubstring("admin access to devices requires the `resource.kubernetes.io/admin-access: true` label on the containing namespace"))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(matchValidationError)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				// First delete, in case that it succeeded earlier.
				if err := f.ClientSet.ResourceV1().ResourceClaimTemplates(f.Namespace.Name).Delete(ctx, claimTemplate.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				_, err := f.ClientSet.ResourceV1().ResourceClaimTemplates(f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(matchValidationError)

			// After labeling the namespace, creation must (eventually...) succeed.
			_, err := f.ClientSet.CoreV1().Namespaces().Apply(ctx,
				applyv1.Namespace(f.Namespace.Name).WithLabels(map[string]string{"resource.kubernetes.io/admin-access": "true"}),
				metav1.ApplyOptions{FieldManager: f.UniqueName})
			framework.ExpectNoError(err)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := f.ClientSet.ResourceV1().ResourceClaimTemplates(f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
				return err
			}).Should(gomega.Succeed())
		})

		f.It("truncates the name of a generated resource claim", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func(ctx context.Context) {
			pod, template := b.PodInline()
			pod.Name = strings.Repeat("p", 63)
			pod.Spec.ResourceClaims[0].Name = strings.Repeat("c", 63)
			pod.Spec.Containers[0].Resources.Claims[0].Name = pod.Spec.ResourceClaims[0].Name
			b.Create(ctx, template, pod)

			b.TestPod(ctx, f, pod)
		})

		f.It("supports count/resourceclaims.resource.k8s.io ResourceQuota", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func(ctx context.Context) {
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
								DeviceClassName: b.ClassName(),
							},
						}},
					},
				},
			}
			_, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
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
				_, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
				return err
			}).Should(gomega.MatchError(gomega.ContainSubstring("exceeded quota: object-count, requested: count/resourceclaims.resource.k8s.io=1, used: count/resourceclaims.resource.k8s.io=1, limited: count/resourceclaims.resource.k8s.io=1")), "creating second claim not allowed")
		})

		f.It("must be possible for the driver to update the ResourceClaim.Status.Devices once allocated", f.WithFeatureGate(features.DRAResourceClaimDeviceStatus), func(ctx context.Context) {
			pod := b.PodExternal()
			claim := b.ExternalClaim()
			b.Create(ctx, claim, pod)

			// Waits for the ResourceClaim to be allocated and the pod to be scheduled.
			b.TestPod(ctx, f, pod)

			allocatedResourceClaim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(allocatedResourceClaim).ToNot(gomega.BeNil())
			gomega.Expect(allocatedResourceClaim.Status.Allocation).ToNot(gomega.BeNil())
			gomega.Expect(allocatedResourceClaim.Status.Allocation.Devices.Results).To(gomega.HaveLen(1))

			scheduledPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(scheduledPod).ToNot(gomega.BeNil())

			var shareIDStr *string
			if shareID := allocatedResourceClaim.Status.Allocation.Devices.Results[0].ShareID; shareID != nil {
				shareIDStr = ptr.To(string(*shareID))
			}

			ginkgo.By("Setting the device status a first time")
			allocatedResourceClaim.Status.Devices = append(allocatedResourceClaim.Status.Devices,
				resourceapi.AllocatedDeviceStatus{
					Driver:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Driver,
					Pool:       allocatedResourceClaim.Status.Allocation.Devices.Results[0].Pool,
					Device:     allocatedResourceClaim.Status.Allocation.Devices.Results[0].Device,
					ShareID:    shareIDStr,
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
				ShareID:    shareIDStr,
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

			getResourceClaim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(getResourceClaim).ToNot(gomega.BeNil())
			gomega.Expect(getResourceClaim.Status.Devices).To(gomega.Equal(updatedResourceClaim.Status.Devices))
		})
	})

	framework.Context("control plane", func() {
		nodes := drautils.NewNodes(f, 1, 4)
		driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
		driver.WithKubelet = false

		f.It("must apply per-node permission checks", framework.WithLabel("ConformanceCandidate") /* TODO: replace with framework.WithConformance() */, func(ctx context.Context) {
			// All of the operations use the client set of a kubelet plugin for
			// a fictional node which both don't exist, so nothing interferes
			// when we actually manage to create a slice.
			fictionalNodeName := "dra-fictional-node"
			gomega.Expect(nodes.NodeNames).NotTo(gomega.ContainElement(fictionalNodeName))
			fictionalNodeClient := driver.ImpersonateKubeletPlugin(&v1.Pod{
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
				err := f.ClientSet.ResourceV1().ResourceSlices().Delete(ctx, fictionalNodeSlice.Name, metav1.DeleteOptions{})
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
				slice, err := clientSet.ResourceV1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				ginkgo.GinkgoHelper()
				slice, err := clientSet.ResourceV1().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
				return slice
			}
			mustDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, fmt.Sprintf("DELETE: %s + %s", clientName, slice.Name))
			}
			mustCreateAndDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice) {
				ginkgo.GinkgoHelper()
				slice = mustCreate(clientSet, clientName, slice)
				mustDelete(clientSet, clientName, slice)
			}
			mustFailToCreate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("CREATE: %s + %s", clientName, slice.Name))
			}
			mustFailToUpdate := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				_, err := clientSet.ResourceV1().ResourceSlices().Update(ctx, slice, metav1.UpdateOptions{})
				gomega.Expect(err).To(matchError, fmt.Sprintf("UPDATE: %s + %s", clientName, slice.Name))
			}
			mustFailToDelete := func(clientSet kubernetes.Interface, clientName string, slice *resourceapi.ResourceSlice, matchError types.GomegaMatcher) {
				ginkgo.GinkgoHelper()
				err := clientSet.ResourceV1().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
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
				err := f.ClientSet.ResourceV1().ResourceSlices().Delete(ctx, clusterSlice.Name, metav1.DeleteOptions{})
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
	})

	multipleDrivers := func(nodeV1beta1, nodeV1 bool) {
		nodes := drautils.NewNodes(f, 1, 4)
		driver1 := drautils.NewDriver(f, nodes, drautils.DriverResources(2))
		driver1.NodeV1beta1 = nodeV1beta1
		driver1.NodeV1 = nodeV1
		b1 := drautils.NewBuilder(f, driver1)

		driver2 := drautils.NewDriver(f, nodes, drautils.DriverResources(2))
		driver2.NodeV1beta1 = nodeV1beta1
		driver2.NodeV1 = nodeV1
		driver2.NameSuffix = "-other"
		b2 := drautils.NewBuilder(f, driver2)

		ginkgo.It("work", func(ctx context.Context) {
			claim1 := b1.ExternalClaim()
			claim1b := b1.ExternalClaim()
			claim2 := b2.ExternalClaim()
			claim2b := b2.ExternalClaim()
			pod := b1.PodExternal()
			for i, claim := range []*resourceapi.ResourceClaim{claim1b, claim2, claim2b} {
				claim := claim
				pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims,
					v1.PodResourceClaim{
						Name:              fmt.Sprintf("claim%d", i+1),
						ResourceClaimName: &claim.Name,
					},
				)
			}
			b1.Create(ctx, claim1, claim1b, claim2, claim2b, pod)
			b1.TestPod(ctx, f, pod)
		})
	}
	multipleDriversContext := func(prefix string, nodeV1beta1, nodeV1 bool) {
		args := []any{
			prefix,
			func() {
				multipleDrivers(nodeV1beta1, nodeV1)
			},
		}
		if !nodeV1beta1 {
			// If the v1beta1 gRPC API is disabled, then
			// kubelet from 1.34 is required because that is
			// when v1 was introduced.
			args = append(args, f.WithLabel("KubeletMinVersion:1.34"))
		}
		framework.Context(args...)
	}

	framework.Context("kubelet", feature.DynamicResourceAllocation, "with multiple drivers", func() {
		multipleDriversContext("using only drapbv1beta1", true, false)
		multipleDriversContext("using only drapbv1", false, true)
		multipleDriversContext("using drapbv1beta1 and drapbv1", true, true)
	})
})
