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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestPodOperationStatusProviderRetainsRestoreResources(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   kubecontainer.PodRestoreState
		corrupt bool
		retain  bool
	}{
		{name: "not started"},
		{name: "started", state: kubecontainer.PodRestoreStarted, retain: true},
		{name: "cleanup pending", state: kubecontainer.PodRestoreCleanupPending, retain: true},
		{name: "corrupt", state: kubecontainer.PodRestoreStarted, corrupt: true, retain: true},
		{name: "adopted", state: kubecontainer.PodRestoreAdopted},
		{name: "cleanup completed", state: kubecontainer.PodRestoreCleanupCompleted},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kl, runtime, pod, _ := checkpointOperationFixture(t)
			kl.podManager.SetPods(nil)
			runtime.PodList = nil
			if tc.state != kubecontainer.PodRestoreNotStarted {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, tc.state))
			}
			if tc.corrupt {
				require.NoError(t, os.WriteFile(filepath.Join(kl.getPodDir(pod.UID), podRestoreStartedKey), []byte("invalid JSON"), 0600))
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
			provider := &podOperationAwareStatusProvider{PodStatusProvider: kl.statusManager, kubelet: kl}
			_, found := provider.GetPodStatus(pod.UID)
			require.False(t, found)
			require.Equal(t, tc.retain, provider.PodOperationMayUsePodResources(pod.UID))
			require.False(t, provider.PodOperationMayUsePodResources("unrelated"))
			if tc.corrupt {
				require.Error(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
				return
			}
			if tc.retain {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
				require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
			}
		})
	}
}

func TestPodOperationStatusProviderRetainsRestorePreparation(t *testing.T) {
	kl, _, pod, _ := checkpointOperationFixture(t)
	kl.podManager.SetPods(nil)
	provider := &podOperationAwareStatusProvider{PodStatusProvider: kl.statusManager, kubelet: kl}
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
	kl.restoreOperationOwners.Store(pod.UID, struct{}{})
	require.True(t, provider.PodOperationMayUsePodResources(pod.UID), "preparation can own allocations before the runtime record exists")
	require.False(t, provider.PodOperationMayUsePodResources("unrelated"))
	kl.restoreOperationOwners.Delete(pod.UID)
	require.False(t, provider.PodOperationMayUsePodResources(pod.UID))
}
