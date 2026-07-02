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

package reconciler

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
)

func TestReconstructedVolumeWaitsForPodOperation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	root := t.TempDir()
	volumePath := filepath.Join(root, "pods", "orphan-uid", "volumes", "fake-plugin", "volume")
	require.NoError(t, os.MkdirAll(volumePath, 0700))
	r, plugin := getReconciler(root, t, []string{volumePath}, nil)
	rc := r.(*reconciler)
	retained := true
	rc.podOperationMayUsePodResources = func(uid types.UID) bool { require.Equal(t, types.UID("orphan-uid"), uid); return retained }
	rc.reconstructVolumes(logger)
	mounted := rc.actualStateOfWorld.GetAllMountedVolumes()
	require.Len(t, mounted, 1)
	require.Empty(t, rc.desiredStateOfWorld.GetVolumesToMount())
	rc.unmountVolumes(logger)
	require.False(t, rc.operationExecutor.IsOperationPending(mounted[0].VolumeName, mounted[0].PodName, nestedpendingoperations.EmptyNodeName), "reconstruction must not authorize unmount while runtime operation owns the Pod")
	require.NoError(t, volumetesting.VerifyZeroTearDownCallCount(plugin))
	retained = false
	rc.unmountVolumes(logger)
	require.Eventually(t, func() bool {
		return len(rc.actualStateOfWorld.GetAllMountedVolumes()) == 0 && verifyTearDownCalls(plugin, 1) == nil
	}, 5*time.Second, 10*time.Millisecond)
}

func TestFailedReconstructionWaitsForPodOperation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	r, plugin := getReconciler(t.TempDir(), t, nil, nil)
	rc := r.(*reconciler)
	rc.volumesFailedReconstruction = []podVolume{{podName: "orphan-uid", volumeSpecName: "volume", pluginName: "fake-plugin", volumeMode: v1.PersistentVolumeFilesystem}}
	retained := true
	rc.podOperationMayUsePodResources = func(uid types.UID) bool { require.Equal(t, types.UID("orphan-uid"), uid); return retained }
	rc.cleanOrphanVolumes(logger)
	require.Len(t, rc.volumesFailedReconstruction, 1, "retained failures must stay queued for cleanup after acknowledgment")
	require.NoError(t, volumetesting.VerifyZeroTearDownCallCount(plugin))
	retained = false
	rc.cleanOrphanVolumes(logger)
	require.Empty(t, rc.volumesFailedReconstruction)
	require.Eventually(t, func() bool { return verifyTearDownCalls(plugin, 1) == nil }, 5*time.Second, 10*time.Millisecond)
}

func TestFailedReconstructionRetainsPluginDevices(t *testing.T) {
	for _, tc := range []struct {
		name         string
		mounted      bool
		failedPlugin string
		hold         bool
	}{
		{name: "globally mounted device", mounted: true, failedPlugin: "fake-plugin", hold: true},
		{name: "attached device", failedPlugin: "fake-plugin", hold: true},
		{name: "different plugin", failedPlugin: "other-plugin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			r, plugin := getReconciler(t.TempDir(), t, nil, nil)
			rc := r.(*reconciler)
			rc.controllerAttachDetachEnabled = false
			pod := getInlineFakePod("previous-pod", "previous-uid", "outer-name", "global-device")
			spec := &volumepkg.Spec{Volume: &pod.Spec.Volumes[0]}
			name := v1.UniqueVolumeName("fake-plugin/global-device")
			attacher, err := plugin.NewAttacher()
			require.NoError(t, err)
			_, err = attacher.Attach(spec, nodeName)
			require.NoError(t, err)
			require.NoError(t, rc.actualStateOfWorld.MarkVolumeAsAttached(logger, name, spec, nodeName, "device-path"))
			if tc.mounted {
				require.NoError(t, rc.actualStateOfWorld.MarkDeviceAsMounted(name, "device-path", t.TempDir(), ""))
			}
			rc.volumesFailedReconstruction = []podVolume{{podName: "orphan-uid", pluginName: tc.failedPlugin, volumeSpecName: "unknown-global-identity"}}
			retained := true
			rc.podOperationMayUsePodResources = func(uid types.UID) bool { require.Equal(t, types.UID("orphan-uid"), uid); return retained }
			require.Empty(t, rc.actualStateOfWorld.GetAllMountedVolumes(), "the failed reconstruction has no ASW Pod reference")
			rc.unmountDetachDevices(logger)
			if tc.hold {
				require.False(t, rc.operationExecutor.IsOperationPending(name, nestedpendingoperations.EmptyUniquePodName, nestedpendingoperations.EmptyNodeName))
				require.Len(t, rc.actualStateOfWorld.GetUnmountedVolumes(), 1)
				require.NoError(t, volumetesting.VerifyUnmountDeviceCallCount(0, plugin))
				require.NoError(t, volumetesting.VerifyZeroDetachCallCount(plugin))
				retained = false
				rc.unmountDetachDevices(logger)
			}
			if tc.mounted {
				require.Eventually(t, func() bool {
					attached, ok := rc.actualStateOfWorld.GetAttachedVolume(name)
					return !ok || !attached.DeviceMayBeMounted()
				}, 5*time.Second, 10*time.Millisecond)
				rc.unmountDetachDevices(logger)
			}
			require.Eventually(t, func() bool { _, ok := rc.actualStateOfWorld.GetAttachedVolume(name); return !ok }, 5*time.Second, 10*time.Millisecond)
			require.NoError(t, volumetesting.VerifyDetachCallCount(1, plugin))
		})
	}
}
