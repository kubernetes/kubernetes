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

package csi

import (
	"os"
	"path/filepath"
	"testing"

	storagev1 "k8s.io/api/storage/v1"
)

// TestListGlobalVolumes covers the source of candidates reconstruction has when
// no pod directory names the volume any more, the case a node drain and reboot
// leaves behind (issue #121937). Every skip here is a directory the walk must
// step over without hiding the volumes around it.
func TestListGlobalVolumes(t *testing.T) {
	const driver = "test-driver"

	// stage writes what MountDevice leaves on disk for one volume. MountDevice
	// always names the directory sha256(volumeHandle), and reconstruction
	// checks that, so the fixture has to use the real name.
	stage := func(t *testing.T, pluginDir, driverName, dirName string, data map[string]string) string {
		t.Helper()
		if handle := data[volDataKey.volHandle]; handle != "" && dirName == "" {
			dirName = generateSha(handle)
		}
		volDir := filepath.Join(pluginDir, driverName, dirName)
		if err := os.MkdirAll(filepath.Join(volDir, globalMountInGlobalPath), 0o755); err != nil {
			t.Fatalf("stage %s: %v", dirName, err)
		}
		if data != nil {
			if err := saveVolumeData(volDir, volDataFileName, data); err != nil {
				t.Fatalf("save volume data for %s: %v", dirName, err)
			}
		}
		return volDir
	}

	volumeData := func(specVolID, handle string) map[string]string {
		return map[string]string{
			volDataKey.specVolID:           specVolID,
			volDataKey.volHandle:           handle,
			volDataKey.driverName:          driver,
			volDataKey.volumeLifecycleMode: string(storagev1.VolumeLifecyclePersistent),
		}
	}

	t.Run("reports a staged volume with the handle its unstage needs", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())
		volDir := stage(t, pluginDir, driver, "", volumeData("staged-pv", "handle-of-the-staged-pv"))

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want 1", len(found))
		}
		if got, want := found[0].DeviceMountPath, filepath.Join(volDir, globalMountInGlobalPath); got != want {
			t.Errorf("DeviceMountPath: got %q, want %q", got, want)
		}
		if got, want := found[0].Spec.Name(), "staged-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
		// GenerateUnmountDeviceFunc recomputes the device mount path from the
		// spec, so a spec carrying the wrong handle would unstage another
		// volume's directory.
		source, err := getPVSourceFromSpec(found[0].Spec)
		if err != nil {
			t.Fatalf("getPVSourceFromSpec: %v", err)
		}
		if got, want := source.VolumeHandle, "handle-of-the-staged-pv"; got != want {
			t.Errorf("volume handle: got %q, want %q", got, want)
		}
		if got, want := source.Driver, driver; got != want {
			t.Errorf("driver: got %q, want %q", got, want)
		}
	})

	t.Run("skips volumeDevices and keeps walking", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		// The raw block subtree is a sibling of the per-driver directories, not a
		// driver, and a block volume is not a filesystem device mount. Nothing
		// under it is reported whatever it holds, so this stages a directory
		// there that would otherwise be described in full.
		blockDir := filepath.Base(plug.host.GetVolumeDevicePluginDir(CSIPluginName))
		stage(t, pluginDir, blockDir, "", volumeData("block-pv", "handle-of-a-block-volume"))
		stage(t, pluginDir, driver, "", volumeData("staged-pv", "handle-of-the-staged-pv"))

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want only the filesystem one", len(found))
		}
		if got, want := found[0].Spec.Name(), "staged-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
	})

	t.Run("skips what it cannot describe without hiding the rest", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		// No vol_data.json at all: nothing names the driver or the handle.
		stage(t, pluginDir, driver, "no-data", nil)
		// Volume data that names no driver.
		stage(t, pluginDir, driver, "no-driver", map[string]string{
			volDataKey.specVolID: "nameless",
			volDataKey.volHandle: "handle-without-a-driver",
		})
		// A volume directory caught before MountDevice staged anything.
		unstaged := filepath.Join(pluginDir, driver, "not-staged")
		if err := os.MkdirAll(unstaged, 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := saveVolumeData(unstaged, volDataFileName, volumeData("not-staged", "handle-of-nothing")); err != nil {
			t.Fatalf("save volume data: %v", err)
		}
		// The one good volume, last so that a walk aborting early fails here.
		stage(t, pluginDir, driver, "", volumeData("staged-pv", "handle-of-the-staged-pv"))

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want only the one that can be described", len(found))
		}
		if got, want := found[0].Spec.Name(), "staged-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
	})

	t.Run("recovers a volume staged before this feature existed", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		// A kubelet older than this feature wrote no specVolID. Those are the
		// volumes already staged when the gate is turned on, and the unique
		// volume name comes from the driver and handle, not from this name, so
		// they are recoverable.
		stage(t, pluginDir, driver, "", map[string]string{
			volDataKey.volHandle:  "handle-of-an-older-volume",
			volDataKey.driverName: driver,
		})

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want 1", len(found))
		}
		name, err := plug.GetVolumeName(found[0].Spec)
		if err != nil {
			t.Fatalf("GetVolumeName: %v", err)
		}
		if want := driver + volNameSep + "handle-of-an-older-volume"; name != want {
			t.Errorf("volume name: got %q, want %q", name, want)
		}
		// With no specVolID the spec is named from the handle, so the volume is
		// still identifiable in a log rather than nameless.
		if got, want := found[0].Spec.Name(), "handle-of-an-older-volume"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
	})

	t.Run("skips a directory whose volume data names another volume", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		// MountDevice stages under sha256(volumeHandle). A directory holding
		// volume data for a different handle would be unstaged at a path that
		// is not this one, reporting success and leaving this mount in place.
		stage(t, pluginDir, driver, generateSha("some-other-handle"),
			volumeData("staged-pv", "handle-of-the-staged-pv"))

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})

	t.Run("reports nothing when no volume was ever staged", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})
}
