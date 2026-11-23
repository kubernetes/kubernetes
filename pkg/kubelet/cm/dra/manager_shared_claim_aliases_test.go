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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1"

	"k8s.io/ktesting"
	api "k8s.io/kubernetes/pkg/apis/core"
	apiv1 "k8s.io/kubernetes/pkg/apis/core/v1"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

// genTestPodWithSharedClaim builds a two-container pod. Both containers use
// DRA claims; with twoAliases, the containers reference two different
// spec.resourceClaims aliases ("first", "second") that both point at the same
// ResourceClaim. Without it, both containers share the single alias "first".
func genTestPodWithSharedClaim(twoAliases bool) *v1.Pod {
	sidecarClaims := []v1.ResourceClaim{{Name: "first"}}
	resourceClaims := []v1.PodResourceClaim{
		{Name: "first", ResourceClaimName: new(claimName)},
	}
	if twoAliases {
		sidecarClaims = []v1.ResourceClaim{{Name: "second"}}
		resourceClaims = append(resourceClaims, v1.PodResourceClaim{
			Name: "second", ResourceClaimName: new(claimName),
		})
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-shared-claim",
			Namespace: namespace,
			UID:       "test-pod-shared-claim-uid",
		},
		Spec: v1.PodSpec{
			ResourceClaims: resourceClaims,
			Containers: []v1.Container{
				{
					Name:  "main",
					Image: "busybox:1.36",
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "first"}},
					},
				},
				{
					Name:  "sidecar",
					Image: "busybox:1.36",
					Resources: v1.ResourceRequirements{
						Claims: sidecarClaims,
					},
				},
			},
		},
	}
}

// TestTwoAliasesForSameResourceClaimPassPodValidation runs the real API server
// pod validation on the dual-alias pod to establish that the configuration is
// legal: validatePodResourceClaim only enforces uniqueness of the alias names,
// not of the ResourceClaim they resolve to.
func TestTwoAliasesForSameResourceClaimPassPodValidation(t *testing.T) {
	for _, tc := range []struct {
		name       string
		twoAliases bool
	}{
		{name: "control: single alias", twoAliases: false},
		{name: "two aliases for the same ResourceClaim", twoAliases: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := genTestPodWithSharedClaim(tc.twoAliases)
			// The real apiserver path is: decode, apply API defaulting,
			// convert to the internal type, validate.
			apiv1.SetObjectDefaults_Pod(pod)
			internalPod := &api.Pod{}
			require.NoError(t, apiv1.Convert_v1_Pod_To_core_Pod(pod, internalPod, nil))
			errs := corevalidation.ValidatePodCreate(internalPod, corevalidation.PodValidationOptions{})
			assert.Empty(t, errs, "pod must pass real API validation, got: %v", errs)
		})
	}
}

// TestPrepareResourcesWithTwoAliasesForSameClaim exercises the real
// Manager.PrepareResources against the test gRPC driver. The driver follows
// the protocol: it processes every claim in the request and returns a
// successful result for each unique claim UID, which is all the UID-keyed
// response map can hold.
func TestPrepareResourcesWithTwoAliasesForSameClaim(t *testing.T) {
	for _, tc := range []struct {
		name       string
		twoAliases bool
	}{
		{name: "control: single alias", twoAliases: false},
		{name: "two aliases for the same ResourceClaim", twoAliases: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger, tCtx := ktesting.NewTestContext(t)
			fakeKubeClient := fake.NewClientset()

			pod := genTestPodWithSharedClaim(tc.twoAliases)
			claim := genTestClaim(claimName, driverName, deviceName, string(pod.UID))
			_, err := fakeKubeClient.ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
			require.NoError(t, err)

			draServerInfo, err := setupFakeDRADriverGRPCServer(tCtx, false, nil, genPrepareResourcesResponse(claim.UID), nil, nil)
			require.NoError(t, err)
			defer draServerInfo.teardownFn()

			manager, err := NewManager(logger, fakeKubeClient, t.TempDir())
			require.NoError(t, err)
			defer manager.Stop()
			manager.initDRAPluginManager(tCtx, getFakeNode, time.Second)
			plg := manager.GetWatcherHandler()
			require.NoError(t, plg.RegisterPlugin(tCtx, driverName, draServerInfo.socketName, []string{drapb.DRAPluginService}, nil))

			err = manager.PrepareResources(tCtx, pod)

			requestUIDs := draServerInfo.server.receivedPrepareRequestUIDs()
			claimInfo, exists := manager.cache.get(claimName, namespace)
			require.True(t, exists, "claim info must be in the cache")

			require.NoError(t, err)
			assert.Equal(t, [][]string{{string(claim.UID)}}, requestUIDs,
				"driver must receive the claim exactly once")
			assert.True(t, claimInfo.prepared)

			require.NoError(t, manager.PrepareResources(tCtx, pod))
			assert.Len(t, draServerInfo.server.receivedPrepareRequestUIDs(), 1)
			require.NoError(t, manager.UnprepareResources(tCtx, pod))
			assert.Equal(t, uint32(1), draServerInfo.server.unprepareResourceCalls.Load())
			_, exists = manager.cache.get(claimName, namespace)
			assert.False(t, exists)
		})
	}
}
