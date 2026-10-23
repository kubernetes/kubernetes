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

package dynamicresources

import (
	"testing"

	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/ktesting"
)

func TestNominatedNodes(t *testing.T) {
	testNominatedNodes(ktesting.Init(t))
}

// testNominatedNodes checks that the plugin follows the nominations of the
// PodNominator for pods with claims.
func testNominatedNodes(tCtx ktesting.TContext) {
	pod := preemptionPod("pod", "claim").Priority(100).Obj()
	noClaims := preemptionPod("no-claims").Priority(100).Obj()
	testCtx := setupNominations(tCtx, []*resourceapi.ResourceClaim{preemptionClaim("claim").Obj()}, []apiruntime.Object{pod, noClaims})
	nominations := testCtx.p.nominations

	testCtx.p.AddAllocation(tCtx, pod, nodeName)
	require.Equal(tCtx, nodeName, nominations.nominatedNode(pod.UID), "nominated")
	testCtx.p.AddAllocation(tCtx, noClaims, nodeName)
	require.Empty(tCtx, nominations.nominatedNode(noClaims.UID), "nominated pod without claims")

	// A change of the nominated node is a removal followed by an addition.
	testCtx.p.RemoveAllocation(tCtx, pod, nodeName)
	testCtx.p.AddAllocation(tCtx, pod, node2Name)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "nominated to another node")

	// Assumed pods are not nominated.
	assumed := fromPod(pod).Node(node2Name).Obj()
	testCtx.p.AddAllocation(tCtx, assumed, node2Name)
	testCtx.p.RemoveAllocation(tCtx, assumed, node2Name)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "after the calls for the assumed pod")

	testCtx.p.RemoveAllocation(tCtx, pod, nodeName)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "after removing a nomination for another node")
	testCtx.p.RemoveAllocation(tCtx, pod, node2Name)
	require.Empty(tCtx, nominations.nominatedNode(pod.UID), "after the nomination ended")
}
