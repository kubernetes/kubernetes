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
	"fmt"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func testUsesAllResources(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")
	numDevicesPerNode := 100 // One pod gets scheduled per device.

	class, driverName := createTestClass(tCtx, namespace)
	for _, node := range nodes.Items {
		// Globally unique device names make debugging simpler...
		devices := make([]string, numDevicesPerNode)
		for i := range numDevicesPerNode {
			devices[i] = fmt.Sprintf("%s-device-%03d", node.Name, i)
		}
		slice := st.MakeResourceSlice(node.Name, driverName).Devices(devices...)
		createSlice(tCtx, slice.Obj())
	}

	var claims []*resourceapi.ResourceClaim
	var pods []*v1.Pod
	for i := range len(nodes.Items) * numDevicesPerNode {
		tCtx := ktesting.WithStep(tCtx, fmt.Sprintf("#%04d", i))

		claim := st.MakeResourceClaim().
			Name(fmt.Sprintf("claim-%04d", i)).
			Namespace(namespace).
			Request(class.Name).
			Obj()
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim")
		claims = append(claims, claim)

		pod := createPod(tCtx, namespace, fmt.Sprintf("-%04d", i), podWithClaimName, claim)
		pods = append(pods, pod)
	}

	startScheduler(tCtx)

	// Eventually, all pods should be scheduled and thus all claims allocated.
	allocated := make(map[structured.DeviceID]*resourceapi.ResourceClaim, len(claims))
	tCtx = ktesting.WithStep(tCtx, "check claim allocation")
	tCtx = ktesting.WithTimeout(tCtx, time.Duration(len(pods))*5*time.Second, "scheduling timeout for all pods")
	for _, claim := range claims {
		var actualClaim *resourceapi.ResourceClaim
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			actualClaim = c
			return c
		}).Should(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())))
		tCtx.Expect(actualClaim.Status.Allocation.Devices.Results).To(gomega.HaveLen(1))
		result := actualClaim.Status.Allocation.Devices.Results[0]
		id := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
		if otherClaim, ok := allocated[id]; ok {
			tCtx.Fatalf("device %s was allocated to claims %s and %s", id, klog.KObj(actualClaim), klog.KObj(otherClaim))
		}
		allocated[id] = actualClaim
	}
}
