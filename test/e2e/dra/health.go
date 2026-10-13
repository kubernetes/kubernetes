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

package dra

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/kubernetes"
	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	testdriverapp "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// minKubeletVersionForHealth is the first kubelet release which publishes
	// device health in the pod status by default (ResourceHealthStatus became
	// beta). 1.36 kubelets only have the v1alpha1 DRAResourceHealth client.
	minKubeletVersionForHealth = "1.36"

	// minKubeletVersionForHealthV1 is the first kubelet release with the v1
	// DRAResourceHealth client. Strictly speaking the v1 client was merged
	// after v1.37.0-beta.0, so kubelets built from earlier 1.37 pre-release
	// artifacts still negotiate v1alpha1; that window closes with v1.37.0.
	//
	// TODO(harche): in 1.40, when kubelet 1.36 falls out of the supported
	// version skew, remove the kubelet's v1alpha1 client support and the
	// kubeletplugin helper's v1alpha1 serving support. In this file that
	// means removing the "v1alpha1 health API" context, the v1alpha1
	// fallback in expectedHealthMethod, and minKubeletVersionForHealth
	// (gating on minKubeletVersionForHealthV1 instead).
	minKubeletVersionForHealthV1 = "1.37"
)

// The gRPC methods on which the kubelet subscribes to device health updates,
// depending on the negotiated DRAResourceHealth API version.
const (
	nodeWatchResourcesV1Alpha1Method = drahealthv1alpha1.DRAResourceHealth_NodeWatchResources_FullMethodName
	nodeWatchResourcesV1Method       = drahealthv1.DRAResourceHealth_NodeWatchResources_FullMethodName
)

// Tests for the DRAResourceHealth gRPC API between the kubelet and a DRA
// driver and its effect on the pod status.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), func() {
	f := framework.NewDefaultFramework("dra-health")

	// The driver containers have to run with sufficient privileges to
	// modify /var/lib/kubelet/plugins.
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	framework.Context("kubelet", feature.DynamicResourceAllocation, f.WithFeatureGate(features.ResourceHealthStatus), f.WithKubeletMinVersion(minKubeletVersionForHealth), func() {
		nodes := drautils.NewNodes(f, 1, 1)

		// expectedHealthMethod determines the DRAResourceHealth gRPC method on
		// which the kubelet is expected to subscribe: the most recent API
		// version served by the driver which the kubelet supports.
		expectedHealthMethod := func(ctx context.Context, driver *drautils.Driver, nodeName string) string {
			if !driver.HealthV1 {
				return nodeWatchResourcesV1Alpha1Method
			}
			if !driver.HealthV1alpha1 {
				// Only choice. The KubeletMinVersion gate on the test
				// ensures that the kubelet has the v1 client.
				return nodeWatchResourcesV1Method
			}
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err, "get node")
			kubeletVersion, err := utilversion.ParseSemantic(node.Status.NodeInfo.KubeletVersion)
			framework.ExpectNoError(err, "parse kubelet version %q of node %s", node.Status.NodeInfo.KubeletVersion, nodeName)
			// The generic (major.minor) comparison intentionally treats all
			// 1.37 builds, including pre-releases, as v1-capable.
			if kubeletVersion.AtLeast(utilversion.MustParse(minKubeletVersionForHealthV1)) {
				return nodeWatchResourcesV1Method
			}
			return nodeWatchResourcesV1Alpha1Method
		}

		// testHealthTransitions runs a pod with an allocated device, then
		// verifies that device health transitions (Healthy -> Unhealthy ->
		// Healthy) reported by the DRA plugin are reflected in the pod's
		// status, and that the kubelet consumed them through the expected
		// DRAResourceHealth gRPC API version.
		testHealthTransitions := func(ctx context.Context, b *drautils.Builder) {
			driver := b.Driver
			claim := b.ExternalClaim()
			pod := b.PodExternal(claim.Name)
			b.Create(f.TContext(ctx), claim, pod)
			b.TestPod(f.TContext(ctx), pod)

			// Exactly one node was selected, so the pod must be running there.
			nodeName := nodes.NodeNames[0]
			plugin, ok := driver.Nodes[nodeName]
			if !ok {
				framework.Failf("no test driver plugin for node %s", nodeName)
			}

			allocatedClaim, err := b.ClientV1(f.TContext(ctx)).ResourceClaims(pod.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get allocated claim")
			gomega.Expect(allocatedClaim.Status.Allocation).ToNot(gomega.BeNil(), "claim allocation")
			result := allocatedClaim.Status.Allocation.Devices.Results[0]

			ginkgo.By("Verifying the negotiated health API version")
			expectedMethod := expectedHealthMethod(ctx, driver, nodeName)
			// CallCount reads an in-process counter, so polling can be tight.
			gomega.Eventually(ctx, func() int64 {
				return driver.CallCount(drautils.MethodInstance{NodeName: nodeName, FullMethod: expectedMethod})
			}).WithTimeout(60*time.Second).WithPolling(100*time.Millisecond).Should(gomega.BeNumerically(">", 0), "expected health stream on %s", expectedMethod)

			// The kubelet must subscribe on exactly one version.
			otherMethod := nodeWatchResourcesV1Alpha1Method
			if expectedMethod == nodeWatchResourcesV1Alpha1Method {
				otherMethod = nodeWatchResourcesV1Method
			}
			gomega.Expect(driver.CallCount(drautils.MethodInstance{NodeName: nodeName, FullMethod: otherMethod})).To(gomega.BeZero(), "unexpected health stream on %s", otherMethod)

			// The kubelet publishes the health under the name of the claim as
			// declared in the container, including the request name if one
			// was set there.
			claimRef := pod.Spec.Containers[0].Resources.Claims[0]
			statusName := v1.ResourceName("claim:" + claimRef.Name)
			if claimRef.Request != "" {
				statusName += v1.ResourceName("/" + claimRef.Request)
			}

			setHealth := func(health string) {
				plugin.HealthControlChan <- testdriverapp.DeviceHealthUpdate{
					PoolName:   result.Pool,
					DeviceName: result.Device,
					Health:     health,
				}
			}
			getHealth := func(ctx context.Context) (string, error) {
				return getDeviceHealth(ctx, f.ClientSet, pod.Namespace, pod.Name, statusName, driver.Name)
			}

			ginkgo.By("Setting device health to Healthy to establish a baseline")
			setHealth("Healthy")
			gomega.Eventually(ctx, getHealth).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Healthy"), "device health should be Healthy after explicit update")

			ginkgo.By("Setting device health to Unhealthy via control channel")
			setHealth("Unhealthy")
			gomega.Eventually(ctx, getHealth).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Unhealthy"), "device health should update to Unhealthy")

			ginkgo.By("Setting device health back to Healthy via control channel")
			setHealth("Healthy")
			gomega.Eventually(ctx, getHealth).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Healthy"), "device health should recover and update to Healthy")
		}

		ginkgo.Context("with a driver which serves all health API versions", func() {
			driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
			b := drautils.NewBuilder(f, driver)

			ginkgo.It("must reflect device health changes in the pod status", func(ctx context.Context) {
				testHealthTransitions(ctx, b)
			})
		})

		ginkgo.Context("with a driver which only serves the v1alpha1 health API", func() {
			// Like a driver which shipped before the v1 API existed.
			// TODO(harche): remove in 1.40, see minKubeletVersionForHealthV1.
			driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
			driver.HealthV1 = false
			b := drautils.NewBuilder(f, driver)

			ginkgo.It("must reflect device health changes in the pod status", func(ctx context.Context) {
				testHealthTransitions(ctx, b)
			})
		})

		ginkgo.Context("with a driver which only serves the v1 health API", func() {
			// Like a driver which has dropped v1alpha1 support.
			driver := drautils.NewDriver(f, nodes, drautils.DriverResources(1))
			driver.HealthV1alpha1 = false
			b := drautils.NewBuilder(f, driver)

			// Kubelets older than minKubeletVersionForHealthV1 only have the
			// v1alpha1 client and cannot consume health from such a driver.
			f.It("must reflect device health changes in the pod status", f.WithKubeletMinVersion(minKubeletVersionForHealthV1), func(ctx context.Context) {
				testHealthTransitions(ctx, b)
			})
		})
	})
})

// getDeviceHealth returns the health of the device of the given driver as
// published in the pod's AllocatedResourcesStatus under the given resource
// status name, or "NotFound" if there is no entry for it.
//
// The kubelet identifies the device by the first CDI device ID which the
// driver returned for it during NodePrepareResources, with
// "<driver>/<pool>/<device>" as fallback when there is none (see
// buildResourceHealth in pkg/kubelet/cm/dra/manager.go). The test driver
// generates CDI device IDs of the form "<driver>/test=...", so in both cases
// the ID starts with the driver name, which is sufficient to identify the
// device because the claims used in these tests allocate exactly one.
func getDeviceHealth(ctx context.Context, clientSet kubernetes.Interface, namespace, podName string, statusName v1.ResourceName, driverName string) (string, error) {
	pod, err := clientSet.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("get pod %s/%s: %w", namespace, podName, err)
	}

	for _, containerStatus := range pod.Status.ContainerStatuses {
		for _, resourceStatus := range containerStatus.AllocatedResourcesStatus {
			if resourceStatus.Name != statusName {
				continue
			}
			for _, resourceHealth := range resourceStatus.Resources {
				if strings.HasPrefix(string(resourceHealth.ResourceID), driverName+"/") {
					return string(resourceHealth.Health), nil
				}
			}
		}
	}

	return "NotFound", nil
}
