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

package kubelet

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodOperationStatusProviderRetainsCheckpointResources(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = nil
	kl.podManager.SetPods(nil)
	runtime.PodList = nil
	provider := &podOperationAwareStatusProvider{PodStatusProvider: kl.statusManager, kubelet: kl}
	_, found := provider.GetPodStatus(pod.UID)
	require.False(t, found)
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: "first", PodUID: pod.UID}))
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: "second", PodUID: pod.UID}))
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
	require.True(t, provider.PodOperationMayUsePodResources(pod.UID), "gate disable and a cache miss cannot release resources")
	require.False(t, provider.PodOperationMayUsePodResources("unrelated"))
	acknowledged := false
	runtime.CancelPodOperationFunc = func(_ context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		if request.OperationId == "second" && !acknowledged {
			return errors.New("runtime work is still active")
		}
		return nil
	}
	require.ErrorIs(t, kl.reconcilePodCheckpointOperation(ctx, pod), errPodCheckpointCompletionPending)
	require.True(t, provider.PodOperationMayUsePodResources(pod.UID), "one pending operation still owns the allocations")
	acknowledged = true
	require.NoError(t, kl.reconcilePodCheckpointOperation(ctx, pod))
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
	kl.statusManager.SetPodStatus(ctx.Logger(), pod, v1.PodStatus{Phase: v1.PodRunning})
	got, found := provider.GetPodStatus(pod.UID)
	require.True(t, found)
	require.Equal(t, v1.PodRunning, got.Phase)
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID), "a cached status must not prolong completed runtime ownership")
}

func TestPodOperationStatusProviderRetainsUncertainCheckpointRecords(t *testing.T) {
	kl, _, pod, _ := checkpointOperationFixture(t)
	kl.podManager.SetPods(nil)
	operation := podCheckpointOperation{CheckpointUID: "checkpoint", PodUID: pod.UID}
	require.NoError(t, kl.writePodCheckpointOperation(operation))
	dir, key, err := kl.podCheckpointOperationPath(pod.UID, operation.CheckpointUID)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, key), []byte("invalid JSON"), 0600))
	provider := &podOperationAwareStatusProvider{PodStatusProvider: kl.statusManager, kubelet: kl}
	require.True(t, provider.PodOperationMayUsePodResources(pod.UID))
}

func TestPodOperationStatusProviderRetainsCheckpointPreparation(t *testing.T) {
	kl, _, pod, _ := checkpointOperationFixture(t)
	kl.podManager.SetPods(nil)
	provider := &podOperationAwareStatusProvider{PodStatusProvider: kl.statusManager, kubelet: kl}
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
	kl.checkpointOperationOwners.Store(pod.UID, struct{}{})
	require.True(t, provider.PodOperationMayUsePodResources(pod.UID), "preparation can own allocations before the runtime record exists")
	require.False(t, provider.PodOperationMayUsePodResources("unrelated"))
	kl.checkpointOperationOwners.Delete(pod.UID)
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
}
