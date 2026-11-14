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
	"testing"

	"github.com/stretchr/testify/require"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/ktesting"
)

func TestPodCheckpointSweepRechecksRestoreOwnership(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, _ := checkpointOperationFixture(t)
	seedCheckpointCache(t, kl)
	uid := types.UID("deleted-checkpoint")
	_, data, err := kl.podCheckpointOutputPath(uid)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(data, 0700))
	client := fake.NewClientset()
	kl.kubeClient = client
	// Publish ownership after the sweeper took its first snapshot but before it
	// removes data. The live API list cannot report this local handoff.
	client.PrependReactor("list", "podcheckpoints", func(clienttesting.Action) (bool, apiruntime.Object, error) {
		kl.restoreCheckpointUIDs.Store(pod.UID, uid)
		return false, nil, nil
	})
	require.NoError(t, kl.sweepPodCheckpointData(ctx))
	require.DirExists(t, data, "a restore acquired the checkpoint while the live read was pending")
	kl.restoreCheckpointUIDs.Delete(pod.UID)
	client.ReactionChain = client.ReactionChain[1:]
	require.NoError(t, kl.sweepPodCheckpointData(ctx))
	require.NoDirExists(t, data, "settled ownership must permit eventual collection")
}
