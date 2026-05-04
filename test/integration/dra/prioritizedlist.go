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
	"math/rand/v2"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

func testPrioritizedList(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	class, _ := createTestClass(tCtx, namespace)
	// This is allowed to fail if the feature is disabled.
	// createClaim normally doesn't return errors because this is unusual, but we can get it indirectly.
	claim, err := func() (claim *resourceapi.ResourceClaim, finalError error) {
		tCtx, finalize := tCtx.WithError(&finalError)
		defer finalize()
		return createClaim(tCtx, namespace, "", class, claimPrioritizedList), nil
	}()

	if !enabled {
		require.Error(tCtx, err, "claim should have become invalid after dropping FirstAvailable")
		return
	}

	require.NotEmpty(tCtx, claim.Spec.Devices.Requests[0].FirstAvailable, "should store FirstAvailable")
	startScheduler(tCtx)

	// We could create ResourceSlices for some node with the right driver.
	// But failing during Filter is sufficient to determine that it did
	// not fail during PreFilter because of FirstAvailable.
	pod := createPod(tCtx, namespace, "", podWithClaimName, claim)
	schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
		gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Type":    gomega.Equal(v1.PodScheduled),
			"Status":  gomega.Equal(v1.ConditionFalse),
			"Reason":  gomega.Equal("Unschedulable"),
			"Message": gomega.Equal("0/8 nodes are available: 8 cannot allocate all claims. still not schedulable, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling."),
		}),
	))
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(schedulingAttempted)
}

func testPrioritizedListScoring(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")

	// We don't want to use more than 8 nodes since we limit the number
	// of subrequests to max 8.
	var nodesForTest []v1.Node
	if len(nodes.Items) > 8 {
		nodesForTest = nodes.Items[:8]
	} else {
		nodesForTest = nodes.Items
	}

	// Create a separate device class and driver for each node. This makes
	// it easy to create subrequests that can only be satisfied by specific
	// nodes.
	var nodeInfos []nodeInfo
	for _, node := range nodesForTest {
		driverName := fmt.Sprintf("%s-%s", namespace, node.Name)
		class, driverName := createTestClass(tCtx, driverName)
		slice := st.MakeResourceSlice(node.Name, driverName).Devices(device1, device2)
		createSlice(tCtx, slice.Obj())
		nodeInfos = append(nodeInfos, nodeInfo{
			name:       node.Name,
			driverName: driverName,
			class:      class,
			pool:       slice.Spec.Pool.Name,
		})
	}

	// Randomize the list of nodes so the selected node isn't always the first one.
	rand.Shuffle(len(nodeInfos), func(i, j int) {
		nodeInfos[i], nodeInfos[j] = nodeInfos[j], nodeInfos[i]
	})

	startScheduler(tCtx)

	runSubTest(tCtx, "single-claim", func(tCtx ktesting.TContext) {
		var firstAvailable []resourceapi.DeviceSubRequest
		for i := range nodeInfos {
			firstAvailable = append(firstAvailable, resourceapi.DeviceSubRequest{
				Name:            fmt.Sprintf("subreq-%d", i),
				DeviceClassName: nodeInfos[i].class.Name,
			})
		}
		claimPrioritizedList := st.MakeResourceClaim().
			Name(claimName + "-pl-single-claim").
			Namespace(namespace).
			RequestWithPrioritizedList(firstAvailable...).
			Obj()
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		_ = createPod(tCtx, namespace, "-pl-single-claim", podWithClaimName, claim)
		expectedSelectedRequest := fmt.Sprintf("%s/%s", claim.Spec.Devices.Requests[0].Name, claim.Spec.Devices.Requests[0].FirstAvailable[0].Name)
		tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
			return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
		}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[0]))
	})

	runSubTest(tCtx, "multi-claim", func(tCtx ktesting.TContext) {
		// Set up two claims where the node in nodeInfos[2] will be the best
		// option:
		// nodeInfos[1]: rank 1 in claim1 and rank 3 in claim2, so it will get a score of 14
		// nodeInfos[2]: rank 2 in claim1 and rank 1 in claim2, so it will get a score of 15
		// nodeInfos[3]: rank 3 in claim1 and rank 2 in claim2, so it will get a score of 13
		claimPrioritizedList1 := st.MakeResourceClaim().
			Name(claimName + "-pl-multiclaim-1").
			Namespace(namespace).
			RequestWithPrioritizedList([]resourceapi.DeviceSubRequest{
				{
					Name:            "subreq-0",
					DeviceClassName: nodeInfos[1].class.Name,
				},
				{
					Name:            "subreq-1",
					DeviceClassName: nodeInfos[2].class.Name,
				},
				{
					Name:            "subreq-2",
					DeviceClassName: nodeInfos[3].class.Name,
				},
			}...).
			Obj()
		claim1, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList1, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		claimPrioritizedList2 := st.MakeResourceClaim().
			Name(claimName + "-pl-multiclaim-2").
			Namespace(namespace).
			RequestWithPrioritizedList([]resourceapi.DeviceSubRequest{
				{
					Name:            "subreq-0",
					DeviceClassName: nodeInfos[2].class.Name,
				},
				{
					Name:            "subreq-1",
					DeviceClassName: nodeInfos[3].class.Name,
				},
				{
					Name:            "subreq-2",
					DeviceClassName: nodeInfos[1].class.Name,
				},
			}...).
			Obj()
		claim2, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList2, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		pod := st.MakePod().Name(podName).Namespace(namespace).
			Container("my-container").
			Obj()
		_ = createPod(tCtx, namespace, "-pl-multiclaim", pod, claim1, claim2)

		// The second subrequest in claim1 is for nodeInfos[2], so it should be chosen.
		expectedSelectedRequest := fmt.Sprintf("%s/%s", claim1.Spec.Devices.Requests[0].Name, claim1.Spec.Devices.Requests[0].FirstAvailable[1].Name)
		tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
			return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimPrioritizedList1.Name, metav1.GetOptions{})
		}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[2]))

		// The first subrequest in claim2 is for nodeInfos[2], so it should be chosen.
		expectedSelectedRequest = fmt.Sprintf("%s/%s", claim2.Spec.Devices.Requests[0].Name, claim2.Spec.Devices.Requests[0].FirstAvailable[0].Name)
		tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
			return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimPrioritizedList2.Name, metav1.GetOptions{})
		}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[2]))
	})
}
