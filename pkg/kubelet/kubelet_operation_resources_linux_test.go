//go:build linux

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
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

type operationResourceCgroups struct {
	cm.PodContainerManager
	destroyed chan cm.CgroupName
}

func (m *operationResourceCgroups) Destroy(_ klog.Logger, name cm.CgroupName) error {
	m.destroyed <- name
	return nil
}

func TestOrphanPodResourcesWaitForCheckpointCompletion(t *testing.T) {
	testOrphanPodResourcesWaitForOperation(t, func(kl *Kubelet, uid types.UID, completed bool) error {
		return kl.writePodCheckpointOperation(podCheckpointOperation{PodUID: uid, CheckpointUID: "checkpoint-uid", Complete: completed})
	})
}

func testOrphanPodResourcesWaitForOperation(t *testing.T, writeState func(*Kubelet, types.UID, bool) error) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
	for _, resource := range []string{"paths", "cgroups"} {
		t.Run(resource, func(t *testing.T) {
			ctx := ktesting.Init(t)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			uid := types.UID("orphan-operation-uid")
			require.NoError(t, writeState(kl, uid, false))
			logger := klog.FromContext(ctx)
			if resource == "paths" {
				prepared := filepath.Join(kl.getPodDir(uid), "prepared", "hosts")
				require.NoError(t, os.MkdirAll(filepath.Dir(prepared), 0700))
				require.NoError(t, os.WriteFile(prepared, []byte("runtime-owned"), 0600))
				require.NoError(t, kl.cleanupOrphanedPodDirs(logger, nil, nil))
				require.FileExists(t, prepared, "absence from the API and runtime does not prove the operation released its paths")
				require.NoError(t, writeState(kl, uid, true))
				require.NoError(t, kl.cleanupOrphanedPodDirs(logger, nil, nil))
				require.NoFileExists(t, prepared)
				return
			}
			groups := &operationResourceCgroups{destroyed: make(chan cm.CgroupName, 2)}
			cgroup := cm.CgroupName{"kubepods", "orphan-operation-uid"}
			found := map[types.UID]cm.CgroupName{uid: cgroup}
			kl.cleanupOrphanedPodCgroups(logger, groups, found, map[types.UID]sets.Empty{})
			select {
			case <-groups.destroyed:
				t.Fatal("orphan cleanup destroyed cgroups before operation completion")
			case <-time.After(100 * time.Millisecond):
			}
			require.NoError(t, writeState(kl, uid, true))
			kl.cleanupOrphanedPodCgroups(logger, groups, found, map[types.UID]sets.Empty{})
			select {
			case actual := <-groups.destroyed:
				require.Equal(t, cgroup, actual)
			case <-time.After(5 * time.Second):
				t.Fatal("cgroups were not released after operation completion")
			}
		})
	}
}
