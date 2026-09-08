/*
Copyright 2026 The Kubernetes Authors.

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
	"errors"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

var errNotListable = errors.New("cannot list global volumes")

// globalListerPlugin is a plugin that reports global mounts, the shape
// reconstruction relies on to find a volume no pod directory names any more.
type globalListerPlugin struct {
	*volumetesting.FakeVolumePlugin
	globalVolumes []volume.GlobalVolume
	err           error
}

func (p *globalListerPlugin) ListGlobalVolumes() ([]volume.GlobalVolume, error) {
	return p.globalVolumes, p.err
}

var _ volume.GlobalVolumeListerPlugin = &globalListerPlugin{}

// TestReconstructGlobalVolume covers what reconstruction does with a volume that
// is still staged on the node but has no pod directory left to be found through,
// the state a node drain and reboot leaves behind (issue #121937). Registering
// it is what keeps it in node.status.volumesInUse until the unstage completes.
func TestReconstructGlobalVolume(t *testing.T) {
	setup := func(t *testing.T) (*reconciler, *globalListerPlugin, volume.GlobalVolume) {
		t.Helper()
		rc, fakePlugin := getReconciler(t.TempDir(), t, nil, nil)
		rcInstance := rc.(*reconciler)
		plugin := &globalListerPlugin{FakeVolumePlugin: fakePlugin}
		globalVolume := volume.GlobalVolume{
			ReconstructedVolume: volume.ReconstructedVolume{
				Spec: volume.NewSpecFromPersistentVolume(&v1.PersistentVolume{
					ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "staged-pv"},
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: "fake-device1"},
						},
					},
				}, false),
			},
			DeviceMountPath: "/var/lib/kubelet/plugins/kubernetes.io/csi/driver/somehash/globalmount",
		}
		return rcInstance, plugin, globalVolume
	}

	uniqueName := func(t *testing.T, plugin *globalListerPlugin, gv volume.GlobalVolume) v1.UniqueVolumeName {
		t.Helper()
		name, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, gv.Spec)
		if err != nil {
			t.Fatalf("GetUniqueVolumeNameFromSpec: %v", err)
		}
		return name
	}

	t.Run("registers a pod-less global mount as an uncertain device", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		rc, plugin, globalVolume := setup(t)

		rc.reconstructGlobalVolume(logger, plugin, globalVolume)

		volumeName := uniqueName(t, plugin, globalVolume)
		if !rc.actualStateOfWorld.VolumeExists(volumeName) {
			t.Fatalf("volume %q is not in the actual state of world", volumeName)
		}
		// It has to be uncertain rather than mounted: kubelet has not verified
		// this mount, it only found it staged on disk. Uncertain is also what
		// makes DeviceMayBeMounted true, which is what gets it to UnmountDevice.
		if got, want := rc.actualStateOfWorld.GetDeviceMountState(volumeName), operationexecutor.DeviceMountUncertain; got != want {
			t.Errorf("device mount state: got %q, want %q", got, want)
		}
		// The device path is filled in later from node.status.volumesAttached,
		// exactly as it is for volumes reconstructed from a pod directory.
		found := false
		for _, name := range rc.volumesNeedUpdateFromNodeStatus {
			if name == volumeName {
				found = true
			}
		}
		if !found {
			t.Errorf("volume %q was not queued for a device path update, got %v", volumeName, rc.volumesNeedUpdateFromNodeStatus)
		}
	})

	t.Run("leaves a volume a pod directory already accounted for", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		rc, plugin, globalVolume := setup(t)

		// First pass stands in for the pod directory walk having found it.
		rc.reconstructGlobalVolume(logger, plugin, globalVolume)
		before := len(rc.volumesNeedUpdateFromNodeStatus)

		rc.reconstructGlobalVolume(logger, plugin, globalVolume)

		if got := len(rc.volumesNeedUpdateFromNodeStatus); got != before {
			t.Errorf("volume was registered twice: %d entries, want %d", got, before)
		}
	})

	t.Run("a plugin that cannot list its volumes does not stop the others", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		rc, plugin, _ := setup(t)
		plugin.err = errNotListable

		// Nothing to assert beyond it returning: a plugin failing to list is
		// logged and skipped, never fatal to reconstruction.
		rc.reconstructGlobalVolumes(logger)

		if len(rc.volumesNeedUpdateFromNodeStatus) != 0 {
			t.Errorf("expected no volumes registered, got %v", rc.volumesNeedUpdateFromNodeStatus)
		}
	})
}
