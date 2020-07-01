// +build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package volume

import (
	"fmt"
	"path/filepath"
	"syscall"

	"os"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/selinux"
)

const (
	rwMask   = os.FileMode(0660)
	roMask   = os.FileMode(0440)
	execMask = os.FileMode(0110)
)

// SetVolumeOwnership modifies the given volume to be owned by
// fsGroup, sets SetGid so that newly created files are owned by
// fsGroup and sets SELinux context of the volumes in a single
// recursive sweep through the volume.
// If fsGroup is nil, no ownership change is done.
// If seLinuxOptions is nil or the filesystem does not support
// SELinux or the SELinux context is not known, no label change is
// done.
func SetVolumeOwnership(mounter Mounter, fsGroup *int64, seLinuxOptions *v1.SELinuxOptions, seLinuxSupported bool, volumeChangePolicy *v1.PodVolumeChangePolicy) error {
	var needFSGroup, needSELinux bool
	var seLinuxLabel string
	var err error

	volumeChangePolicyEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ConfigurableVolumeChangePolicy)

	// Quick checks for skipping FSGroup / SELinux relabeling
	if fsGroup != nil {
		needFSGroup = true
	}
	if volumeChangePolicyEnabled {
		if seLinuxSupported && seLinuxOptions != nil && seLinuxOptions.Level != "" && selinux.SELinuxEnabled() {
			needSELinux = true
		}
	}

	if !needFSGroup && !needSELinux {
		// Nothing to do
		return nil
	}

	klog.Warningf("Setting volume ownership for %s and fsGroup set. If the volume has a lot of files then setting volume ownership could be slow, see https://github.com/kubernetes/kubernetes/issues/69699", mounter.GetPath())

	// This code exists for legacy purposes, so as old behaviour is entirely preserved when feature gate is disabled
	// TODO: remove this when ConfigurableVolumeChangePolicy turns GA.
	if !volumeChangePolicyEnabled {
		return legacyOwnershipChange(mounter, fsGroup)
	}

	// Slow checks for skipping FSGroup / SELinux relabeling
	if needFSGroup {
		if skipPermissionChange(mounter, fsGroup, volumeChangePolicy) {
			klog.V(3).Infof("skipping permission and ownership change for volume %s", mounter.GetPath())
			needFSGroup = false
		}
	}

	if needSELinux {
		seLinuxLabel, needSELinux, err = computeSELinuxLabel(seLinuxOptions)
		if err != nil {
			return fmt.Errorf("failed get SELinux label for pod: %s", err)
		}
		if skipLabelChange(mounter, seLinuxLabel, volumeChangePolicy) {
			klog.V(3).Infof("skipping SELinux label change for volume %s", mounter.GetPath())
			needSELinux = false
		}
	}

	if !needSELinux && !needFSGroup {
		// Nothing to do
		return nil
	}

	// Do not change SELinux label if the root already has the right label.
	if !needSELinux {
		seLinuxLabel = ""
	}
	// Do not change FSGroup if the root already has the right owner.
	if !needFSGroup {
		fsGroup = nil
	}

	return walkDeep(mounter.GetPath(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		return changeFilePermission(path, fsGroup, seLinuxLabel, mounter.GetAttributes().ReadOnly, info)
	})

}

// VolumeNeedsRelabel returns true if a pod's volume with given security
// context can be assumed to have already the right SELinux context.
func VolumeNeedsRelabel(ctx *v1.PodSecurityContext) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ConfigurableVolumeChangePolicy) {
		return true
	}
	if ctx.VolumeChangePolicy == nil || *ctx.VolumeChangePolicy != v1.VolumeChangeOnRootMismatch {
		// VolumeChangePolicy requires relabeling in the container runtime
		return true
	}
	if ctx == nil || ctx.SELinuxOptions == nil || ctx.SELinuxOptions.Level == "" {
		// Kubelet does not know the context, so the container runtime must
		// allocate a new random one and do relabeling there.
		return true
	}
	return false
}

func computeSELinuxLabel(seLinuxOptions *v1.SELinuxOptions) (string, bool, error) {
	if seLinuxOptions == nil {
		return "", false, nil
	}
	if !selinux.SELinuxEnabled() {
		return "", false, nil
	}

	if seLinuxOptions.Level == "" {
		// The container runtime will assign a random MCS level for this pod,
		// kubelet can't relabel the pod's volumes.
		return "", false, nil
	}

	label, err := selinux.GetSELinuxLabelString(seLinuxOptions.User, seLinuxOptions.Role, seLinuxOptions.Type, seLinuxOptions.Level)
	if err != nil {
		return "", false, err
	}
	if label == "" {
		// In theory unreachable, because SELinux must be supported at this point.
		return "", false, nil
	}
	return label, true, nil
}

func legacyOwnershipChange(mounter Mounter, fsGroup *int64) error {
	return filepath.Walk(mounter.GetPath(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		return changeFilePermission(path, fsGroup, "", mounter.GetAttributes().ReadOnly, info)
	})
}

func changeFilePermission(filename string, fsGroup *int64, seLinuxLabel string, readonly bool, info os.FileInfo) error {
	// Apply SELinux label if needed
	if seLinuxLabel != "" {
		if err := selinux.SetFileLabel(filename, seLinuxLabel); err != nil {
			return err
		}
	}

	// Apply FSGroup ownership if needed
	if fsGroup == nil {
		return nil
	}

	// chown and chmod pass through to the underlying file for symlinks.
	// Symlinks have a mode of 777 but this really doesn't mean anything.
	// The permissions of the underlying file are what matter.
	// However, if one reads the mode of a symlink then chmods the symlink
	// with that mode, it changes the mode of the underlying file, overridden
	// the defaultMode and permissions initialized by the volume plugin, which
	// is not what we want; thus, we skip chown/chmod for symlinks.
	if info.Mode()&os.ModeSymlink != 0 {
		return nil
	}

	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return nil
	}

	if stat == nil {
		klog.Errorf("Got nil stat_t for path %v while setting ownership of volume", filename)
		return nil
	}

	err := os.Chown(filename, int(stat.Uid), int(*fsGroup))
	if err != nil {
		klog.Errorf("Chown failed on %v: %v", filename, err)
	}

	mask := rwMask
	if readonly {
		mask = roMask
	}

	if info.IsDir() {
		mask |= os.ModeSetgid
		mask |= execMask
	}

	err = os.Chmod(filename, info.Mode()|mask)
	if err != nil {
		klog.Errorf("Chmod failed on %v: %v", filename, err)
	}

	return nil
}

func skipPermissionChange(mounter Mounter, fsGroup *int64, volumeChangePolicy *v1.PodVolumeChangePolicy) bool {
	dir := mounter.GetPath()

	if volumeChangePolicy == nil || *volumeChangePolicy != v1.VolumeChangeOnRootMismatch {
		klog.V(4).Infof("perform recursive ownership change for %s", dir)
		return false
	}
	return !requiresPermissionChange(mounter.GetPath(), fsGroup, mounter.GetAttributes().ReadOnly)
}

func skipLabelChange(mounter Mounter, seLinuxLabel string, volumeChangePolicy *v1.PodVolumeChangePolicy) bool {
	dir := mounter.GetPath()

	if volumeChangePolicy == nil || *volumeChangePolicy != v1.VolumeChangeOnRootMismatch {
		klog.V(4).Infof("skipping recursive SELinux label change for %s, it will be relabeled by the container runtime", dir)
		return true
	}
	if seLinuxLabel != "" {
		return !requireLabelChange(mounter.GetPath(), seLinuxLabel)
	}
	return false

}
func requireLabelChange(rootDir string, label string) bool {
	existingLabel, err := selinux.NewSELinuxRunner().Getfilecon(rootDir)
	if err != nil {
		klog.Errorf("performing recursive SELinux label change on %s because reading label of root volume failed: %v", rootDir, err)
		return true
	}
	if existingLabel != label {
		klog.V(4).Infof("performing recursive SELinux label change on %s because the volume label %q does not match %q", rootDir, existingLabel, label)
		return true
	}
	klog.V(4).Infof("skipping recursive SELinux label change")
	return false
}

func requiresPermissionChange(rootDir string, fsGroup *int64, readonly bool) bool {
	fsInfo, err := os.Stat(rootDir)
	if err != nil {
		klog.Errorf("performing recursive ownership change on %s because reading permissions of root volume failed: %v", rootDir, err)
		return true
	}
	stat, ok := fsInfo.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		klog.Errorf("performing recursive ownership change on %s because reading permissions of root volume failed", rootDir)
		return true
	}

	if int(stat.Gid) != int(*fsGroup) {
		klog.V(4).Infof("expected group ownership of volume %s did not match with: %d", rootDir, stat.Gid)
		return true
	}
	unixPerms := rwMask

	if readonly {
		unixPerms = roMask
	}

	// if rootDir is not a directory then we should apply permission change anyways
	if !fsInfo.IsDir() {
		return true
	}
	unixPerms |= execMask
	filePerm := fsInfo.Mode().Perm()

	// We need to check if actual permissions of root directory is a superset of permissions required by unixPerms.
	// This is done by checking if permission bits expected in unixPerms is set in actual permissions of the directory.
	// We use bitwise AND operation to check set bits. For example:
	//     unixPerms: 770, filePerms: 775 : 770&775 = 770 (perms on directory is a superset)
	//     unixPerms: 770, filePerms: 770 : 770&770 = 770 (perms on directory is a superset)
	//     unixPerms: 770, filePerms: 750 : 770&750 = 750 (perms on directory is NOT a superset)
	// We also need to check if setgid bits are set in permissions of the directory.
	if (unixPerms&filePerm != unixPerms) || (fsInfo.Mode()&os.ModeSetgid == 0) {
		klog.V(4).Infof("performing recursive ownership change on %s because of mismatching mode", rootDir)
		return true
	}
	return false
}

// readDirNames reads the directory named by dirname and returns
// a list of directory entries.
// We are not using filepath.readDirNames because we do not want to sort files found in a directory before changing
// permissions for performance reasons.
func readDirNames(dirname string) ([]string, error) {
	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	names, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		return nil, err
	}
	return names, nil
}

// walkDeep can be used to traverse directories and has two minor differences
// from filepath.Walk:
//   - List of files/dirs is not sorted for performance reasons
//   - callback walkFunc is invoked on root directory after visiting children dirs and files
func walkDeep(root string, walkFunc filepath.WalkFunc) error {
	info, err := os.Lstat(root)
	if err != nil {
		return walkFunc(root, nil, err)
	}
	return walk(root, info, walkFunc)
}

func walk(path string, info os.FileInfo, walkFunc filepath.WalkFunc) error {
	if !info.IsDir() {
		return walkFunc(path, info, nil)
	}
	names, err := readDirNames(path)
	if err != nil {
		return err
	}
	for _, name := range names {
		filename := filepath.Join(path, name)
		fileInfo, err := os.Lstat(filename)
		if err != nil {
			if err := walkFunc(filename, fileInfo, err); err != nil {
				return err
			}
		} else {
			err = walk(filename, fileInfo, walkFunc)
			if err != nil {
				return err
			}
		}
	}
	return walkFunc(path, info, nil)
}
