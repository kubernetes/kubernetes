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
	"errors"
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// stagedVolume is what a plugin reports for a volume that is still staged on the
// node with no pod directory left to be found through.
func stagedVolume(name, pdName string) volume.GlobalVolume {
	return volume.GlobalVolume{
		ReconstructedVolume: volume.ReconstructedVolume{
			Spec: volume.NewSpecFromPersistentVolume(&v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{UID: "001", Name: name},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: pdName},
					},
				},
			}, false),
		},
		DeviceMountPath: "/var/lib/kubelet/plugins/kubernetes.io/csi/driver/somehash/globalmount",
		VolumeMode:      v1.PersistentVolumeFilesystem,
	}
}

// stagedBlockVolume is what a plugin reports for a raw block volume still
// staged on the node. The mode travels both on the entry, which is what decides
// how it is registered, and on the spec, which is what UnmountDevice later
// picks its branch from.
func stagedBlockVolume(name, pdName string) volume.GlobalVolume {
	gv := stagedVolume(name, pdName)
	blockMode := v1.PersistentVolumeBlock
	gv.Spec.PersistentVolume.Spec.VolumeMode = &blockMode
	gv.DeviceMountPath = "/var/lib/kubelet/plugins/kubernetes.io/csi/volumeDevices/" + name + "/dev"
	gv.VolumeMode = v1.PersistentVolumeBlock
	return gv
}

// listerWithoutMapper keeps global mounts but has no block mapper. The
// interface a plugin implements to be listed requires device mounting, not
// mapping, so a block entry from a plugin like this one has nothing to tear it
// down and must not be registered.
type listerWithoutMapper struct {
	volume.DeviceMountableVolumePlugin
	staged []volume.GlobalVolume
}

func (p *listerWithoutMapper) ListGlobalVolumes() ([]volume.GlobalVolume, error) {
	return p.staged, nil
}

// TestReconstructGlobalVolumes covers reconstruction of a volume that is still
// staged on the node but has no pod directory naming it, the state a node drain
// and reboot leaves behind (issue #121937). Kubelet finds it by asking each
// plugin rather than by walking /var/lib/kubelet/pods.
func TestReconstructGlobalVolumes(t *testing.T) {
	setup := func(t *testing.T, staged ...volume.GlobalVolume) (*reconciler, *volumetesting.FakeVolumePlugin) {
		t.Helper()
		rc, fakePlugin := getReconciler(t.TempDir(), t, nil, nil)
		fakePlugin.GlobalVolumes = staged
		return rc.(*reconciler), fakePlugin
	}

	uniqueName := func(t *testing.T, plugin *volumetesting.FakeVolumePlugin, gv volume.GlobalVolume) v1.UniqueVolumeName {
		t.Helper()
		name, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, gv.Spec)
		if err != nil {
			t.Fatalf("GetUniqueVolumeNameFromSpec: %v", err)
		}
		return name
	}

	t.Run("registers a pod-less staged volume as an uncertain device", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		staged := stagedVolume("staged-pv", "fake-device1")
		staged.SELinuxMountContext = "system_u:object_r:container_file_t:s0:c1,c2"
		rc, plugin := setup(t, staged)

		rc.reconstructGlobalVolumes(logger)

		volumeName := uniqueName(t, plugin, staged)
		if !rc.actualStateOfWorld.VolumeExists(volumeName) {
			t.Fatalf("volume %q is not in the actual state of world", volumeName)
		}
		// Uncertain rather than mounted: kubelet has not verified this mount,
		// it only found it staged. Uncertain is what makes DeviceMayBeMounted
		// true, which is what gets the volume to UnmountDevice.
		if got, want := rc.actualStateOfWorld.GetDeviceMountState(volumeName), operationexecutor.DeviceMountUncertain; got != want {
			t.Errorf("device mount state: got %q, want %q", got, want)
		}
		// The path the plugin reported has to reach the actual state of world:
		// it is what UnmountDevice falls back to when it cannot recompute one.
		found := false
		for _, vol := range rc.actualStateOfWorld.GetAttachedVolumes() {
			if vol.VolumeName != volumeName {
				continue
			}
			found = true
			if got, want := vol.DeviceMountPath, staged.DeviceMountPath; got != want {
				t.Errorf("device mount path: got %q, want %q", got, want)
			}
			if got, want := vol.SELinuxMountContext, staged.SELinuxMountContext; got != want {
				t.Errorf("selinux mount context: got %q, want %q", got, want)
			}
		}
		if !found {
			t.Fatalf("volume %q is not among the attached volumes", volumeName)
		}
		// Attachability stays uncertain until node.status.volumesAttached is
		// read, which is why the volume has to be queued for that update: it is
		// what decides whether it is reported in node.status.volumesInUse.
		if !containsVolume(rc.volumesNeedUpdateFromNodeStatus, volumeName) {
			t.Errorf("volume %q was not queued for a device path update, got %v", volumeName, rc.volumesNeedUpdateFromNodeStatus)
		}
	})

	t.Run("leaves a volume a pod directory already accounted for", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		staged := stagedVolume("staged-pv", "fake-device1")
		rc, plugin := setup(t, staged)

		// The first pass stands in for the pod directory walk having found it.
		rc.reconstructGlobalVolumes(logger)
		volumeName := uniqueName(t, plugin, staged)
		before := len(rc.volumesNeedUpdateFromNodeStatus)

		rc.reconstructGlobalVolumes(logger)

		if got := len(rc.volumesNeedUpdateFromNodeStatus); got != before {
			t.Errorf("volume %q was registered twice: %d entries, want %d", volumeName, got, before)
		}
	})

	t.Run("a plugin that cannot list its volumes is not fatal", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		rc, plugin := setup(t, stagedVolume("staged-pv", "fake-device1"))
		plugin.ListGlobalVolumesErr = errors.New("cannot read the plugin directory")

		rc.reconstructGlobalVolumes(logger)

		if len(rc.volumesNeedUpdateFromNodeStatus) != 0 {
			t.Errorf("expected nothing registered, got %v", rc.volumesNeedUpdateFromNodeStatus)
		}
	})

	t.Run("registers a raw block volume the same way", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		staged := stagedBlockVolume("block-pv", "fake-device1")
		rc, plugin := setup(t, staged)

		rc.reconstructGlobalVolumes(logger)

		volumeName := uniqueName(t, plugin, staged)
		if !rc.actualStateOfWorld.VolumeExists(volumeName) {
			t.Fatalf("block volume %q is not in the actual state of world", volumeName)
		}
		// Uncertain is what makes DeviceMayBeMounted true, which is what gets
		// the volume to UnmountDevice, which sends a block volume to its unmap.
		if got, want := rc.actualStateOfWorld.GetDeviceMountState(volumeName), operationexecutor.DeviceMountUncertain; got != want {
			t.Errorf("device mount state: got %q, want %q", got, want)
		}
		// GenerateUnmapDeviceFunc takes this path as the global map path, so it
		// has to be the one the plugin reported, not a device mount path.
		found := false
		for _, vol := range rc.actualStateOfWorld.GetAttachedVolumes() {
			if vol.VolumeName != volumeName {
				continue
			}
			found = true
			if got, want := vol.DeviceMountPath, staged.DeviceMountPath; got != want {
				t.Errorf("device mount path: got %q, want %q", got, want)
			}
		}
		if !found {
			t.Fatalf("block volume %q is not among the attached volumes", volumeName)
		}
		if !containsVolume(rc.volumesNeedUpdateFromNodeStatus, volumeName) {
			t.Errorf("block volume %q was not queued for a device path update, got %v", volumeName, rc.volumesNeedUpdateFromNodeStatus)
		}
	})

	t.Run("skips a block volume whose plugin has no mapper", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		staged := stagedBlockVolume("block-pv", "fake-device1")
		rc, fakePlugin := setup(t)

		// Reported by a plugin that can be listed and device mounted but has no
		// mapper, so nothing would tear the volume down if it were registered.
		rc.reconstructGlobalVolume(logger, &listerWithoutMapper{DeviceMountableVolumePlugin: fakePlugin}, staged)

		if rc.actualStateOfWorld.VolumeExists(uniqueName(t, fakePlugin, staged)) {
			t.Errorf("a block volume was registered by a plugin with no mapper")
		}
	})

	t.Run("a volume found both ways is queued once", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIGlobalMountReconstruction, true)

		// A volume that is staged and whose pod directory also survived is
		// reported by both sources: the listing runs first, then the pod
		// directory walk reconstructs the same volume for its pod.
		kubeletDir := t.TempDir()
		podVolumeDir := filepath.Join(kubeletDir, "pods", "pod1", "volumes", "fake-plugin", "fake-device1")
		if err := os.MkdirAll(podVolumeDir, 0o755); err != nil {
			t.Fatalf("setup pod volume dir: %v", err)
		}
		rc, fakePlugin := getReconciler(kubeletDir, t, []string{podVolumeDir}, nil)
		rcInstance := rc.(*reconciler)
		staged := stagedVolume("staged-pv", "fake-device1")
		fakePlugin.GlobalVolumes = []volume.GlobalVolume{staged}

		rcInstance.reconstructVolumes(logger)

		volumeName := uniqueName(t, fakePlugin, staged)
		count := 0
		for _, name := range rcInstance.volumesNeedUpdateFromNodeStatus {
			if name == volumeName {
				count++
			}
		}
		if count != 1 {
			t.Errorf("volume %q queued %d times, want 1, got %v", volumeName, count, rcInstance.volumesNeedUpdateFromNodeStatus)
		}
	})

	t.Run("nothing is listed with the feature gate off", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIGlobalMountReconstruction, false)
		staged := stagedVolume("staged-pv", "fake-device1")
		rc, plugin := setup(t, staged)

		rc.reconstructVolumes(logger)

		volumeName := uniqueName(t, plugin, staged)
		if rc.actualStateOfWorld.VolumeExists(volumeName) {
			t.Errorf("volume %q reached the actual state of world with the gate off", volumeName)
		}
	})

	t.Run("a staged volume is listed with the feature gate on", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIGlobalMountReconstruction, true)
		staged := stagedVolume("staged-pv", "fake-device1")
		rc, plugin := setup(t, staged)

		rc.reconstructVolumes(logger)

		volumeName := uniqueName(t, plugin, staged)
		if !rc.actualStateOfWorld.VolumeExists(volumeName) {
			t.Errorf("volume %q did not reach the actual state of world with the gate on", volumeName)
		}
	})
}

func containsVolume(names []v1.UniqueVolumeName, name v1.UniqueVolumeName) bool {
	for _, n := range names {
		if n == name {
			return true
		}
	}
	return false
}
