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

	api "k8s.io/api/core/v1"
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

	t.Run("keeps volumeDevices out of the driver walk", func(t *testing.T) {
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		// The raw block subtree is a sibling of the per-driver directories, not
		// a driver, and it is read by a walk of its own. This stages a
		// directory there in the filesystem layout, which passes every check
		// the driver walk makes: naming the driver after the subtree is what
		// makes MountDevice's own path land inside it, so only keeping that
		// subtree out of this walk can keep it from being reported as a
		// filesystem volume. The block walk passes it over in turn, since
		// nothing under staging names it.
		blockDir := filepath.Base(plug.host.GetVolumeDevicePluginDir(CSIPluginName))
		stage(t, pluginDir, blockDir, "", map[string]string{
			volDataKey.specVolID:           "block-pv",
			volDataKey.volHandle:           "handle-of-a-block-volume",
			volDataKey.driverName:          blockDir,
			volDataKey.volumeLifecycleMode: string(storagev1.VolumeLifecyclePersistent),
		})
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

// TestListGlobalBlockVolumes covers the raw block half of the listing. Block
// volumes leak their staging the same way filesystem volumes leak their global
// mount, but they stage under a layout of their own, so the walk, the checks
// and the paths reported are all separate.
func TestListGlobalBlockVolumes(t *testing.T) {
	const driver = "test-driver"

	// stageBlock writes what a raw block volume leaves on disk. The volume
	// directory and its identity come from NewBlockVolumeMapper; the staging
	// directory comes later, from the node level staging step, and is the one
	// that says the volume is still staged.
	stageBlock := func(t *testing.T, blockDir, dirName string, data map[string]string, staged bool) string {
		t.Helper()
		volDir := filepath.Join(blockDir, dirName)
		dataDir := filepath.Join(volDir, blockVolumeDataDirName)
		if err := os.MkdirAll(dataDir, 0o755); err != nil {
			t.Fatalf("stage block %s: %v", dirName, err)
		}
		if data != nil {
			if err := saveVolumeData(dataDir, volDataFileName, data); err != nil {
				t.Fatalf("save volume data for %s: %v", dirName, err)
			}
		}
		if staged {
			if err := os.MkdirAll(filepath.Join(blockDir, blockStagingDirName, dirName), 0o755); err != nil {
				t.Fatalf("stage block %s: %v", dirName, err)
			}
		}
		return volDir
	}

	blockData := func(specVolID, handle string) map[string]string {
		return map[string]string{
			volDataKey.specVolID:  specVolID,
			volDataKey.volHandle:  handle,
			volDataKey.driverName: driver,
		}
	}

	setup := func(t *testing.T) (*csiPlugin, string) {
		t.Helper()
		plug, tmpDir := newTestPlugin(t, nil)
		t.Cleanup(func() { _ = os.RemoveAll(tmpDir) })
		return plug, plug.host.GetVolumeDevicePluginDir(CSIPluginName)
	}

	t.Run("reports a staged block volume with the identity its teardown needs", func(t *testing.T) {
		plug, blockDir := setup(t)
		volDir := stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want 1", len(found))
		}
		// NewBlockVolumeUnmapper takes the spec name and rebuilds every path
		// from it, so the name has to be the one the directory is called after.
		if got, want := found[0].Spec.Name(), "block-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
		// The mode has to travel with the entry and on the spec: reconstruction
		// picks the registration path from the first, and UnmountDevice picks
		// its branch from the second.
		if got, want := found[0].VolumeMode, api.PersistentVolumeBlock; got != want {
			t.Errorf("volume mode: got %q, want %q", got, want)
		}
		if found[0].Spec.PersistentVolume.Spec.VolumeMode == nil {
			t.Fatalf("spec carries no volume mode")
		}
		if got, want := *found[0].Spec.PersistentVolume.Spec.VolumeMode, api.PersistentVolumeBlock; got != want {
			t.Errorf("spec volume mode: got %q, want %q", got, want)
		}
		// GenerateUnmapDeviceFunc takes the reported path as the global map
		// path, which is the dev directory, not the staging one.
		if got, want := found[0].DeviceMountPath, filepath.Join(volDir, "dev"); got != want {
			t.Errorf("DeviceMountPath: got %q, want %q", got, want)
		}
		source, err := getPVSourceFromSpec(found[0].Spec)
		if err != nil {
			t.Fatalf("getPVSourceFromSpec: %v", err)
		}
		if got, want := source.VolumeHandle, "handle-of-the-block-pv"; got != want {
			t.Errorf("volume handle: got %q, want %q", got, want)
		}
		if got, want := source.Driver, driver; got != want {
			t.Errorf("driver: got %q, want %q", got, want)
		}
	})

	t.Run("reports a volume staged for a pod that never ran", func(t *testing.T) {
		plug, blockDir := setup(t)
		volDir := stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), true)

		// The dev directory is not created by staging. It arrives later, when
		// the volume is mapped into a pod, so a volume staged for a pod that
		// never got that far has none. It leaks exactly like the rest, and
		// looking for that path rather than for the staging one would pass it
		// over.
		if _, err := os.Stat(filepath.Join(volDir, "dev")); !os.IsNotExist(err) {
			t.Fatalf("the fixture was expected to have no dev directory, stat said: %v", err)
		}

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want the one staged with no pod mapping", len(found))
		}
		if got, want := found[0].DeviceMountPath, filepath.Join(volDir, "dev"); got != want {
			t.Errorf("DeviceMountPath: got %q, want %q", got, want)
		}
	})

	t.Run("skips a volume directory with nothing staged", func(t *testing.T) {
		plug, blockDir := setup(t)
		// Identity written, staging gone: this is what is left after
		// NodeUnstageVolume succeeded and the cleanup did not finish. There is
		// nothing to unstage.
		stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), false)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})

	t.Run("leaves the staging and publish siblings out", func(t *testing.T) {
		plug, blockDir := setup(t)

		// staging and publish sit beside the volume directories. This dresses
		// each of them as a volume of that name, identity file and all, so that
		// only the check on the name can keep them out of the listing.
		for _, sibling := range []string{blockStagingDirName, blockPublishDirName} {
			stageBlock(t, blockDir, sibling, blockData(sibling, "handle-of-"+sibling), true)
		}
		stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			names := make([]string, 0, len(found))
			for _, gv := range found {
				names = append(names, gv.Spec.Name())
			}
			t.Fatalf("got %v, want only the real volume", names)
		}
		if got, want := found[0].Spec.Name(), "block-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
	})

	t.Run("skips a directory whose volume data names another volume", func(t *testing.T) {
		plug, blockDir := setup(t)
		// Every path a block teardown touches is built from the spec name, so a
		// directory holding another volume's data would be torn down under a
		// name that is not its own.
		stageBlock(t, blockDir, "block-pv", blockData("another-pv", "handle-of-another-pv"), true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})

	t.Run("skips a directory whose volume data carries no name", func(t *testing.T) {
		plug, blockDir := setup(t)
		// There is no hash to recompute here, so a file with no specVolID
		// leaves nothing to check the directory against but itself.
		stageBlock(t, blockDir, "block-pv", map[string]string{
			volDataKey.volHandle:  "handle-of-the-block-pv",
			volDataKey.driverName: driver,
		}, true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})

	t.Run("skips what it cannot describe without hiding the rest", func(t *testing.T) {
		plug, blockDir := setup(t)
		stageBlock(t, blockDir, "no-data", nil, true)
		stageBlock(t, blockDir, "no-driver", map[string]string{
			volDataKey.specVolID: "no-driver",
			volDataKey.volHandle: "handle-without-a-driver",
		}, true)
		// The one good volume last, so that a walk aborting early fails here.
		stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 1 {
			t.Fatalf("got %d volumes, want only the one that can be described", len(found))
		}
		if got, want := found[0].Spec.Name(), "block-pv"; got != want {
			t.Errorf("spec name: got %q, want %q", got, want)
		}
	})

	t.Run("reports block and filesystem volumes together", func(t *testing.T) {
		plug, blockDir := setup(t)
		pluginDir := plug.host.GetPluginDir(plug.GetPluginName())

		fsDir := filepath.Join(pluginDir, driver, generateSha("handle-of-the-fs-pv"))
		if err := os.MkdirAll(filepath.Join(fsDir, globalMountInGlobalPath), 0o755); err != nil {
			t.Fatalf("stage filesystem volume: %v", err)
		}
		if err := saveVolumeData(fsDir, volDataFileName, map[string]string{
			volDataKey.specVolID:  "fs-pv",
			volDataKey.volHandle:  "handle-of-the-fs-pv",
			volDataKey.driverName: driver,
		}); err != nil {
			t.Fatalf("save volume data: %v", err)
		}
		stageBlock(t, blockDir, "block-pv", blockData("block-pv", "handle-of-the-block-pv"), true)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 2 {
			t.Fatalf("got %d volumes, want both", len(found))
		}
		modes := map[string]api.PersistentVolumeMode{}
		for _, gv := range found {
			modes[gv.Spec.Name()] = gv.VolumeMode
		}
		if got, want := modes["fs-pv"], api.PersistentVolumeFilesystem; got != want {
			t.Errorf("fs-pv mode: got %q, want %q", got, want)
		}
		if got, want := modes["block-pv"], api.PersistentVolumeBlock; got != want {
			t.Errorf("block-pv mode: got %q, want %q", got, want)
		}
	})

	t.Run("reports nothing when no block volume was ever staged", func(t *testing.T) {
		plug, _ := setup(t)

		found, err := plug.ListGlobalVolumes()
		if err != nil {
			t.Fatalf("ListGlobalVolumes: %v", err)
		}
		if len(found) != 0 {
			t.Fatalf("got %d volumes, want none", len(found))
		}
	})
}
