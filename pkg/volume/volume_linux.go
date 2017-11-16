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
	"path/filepath"
	"syscall"

	"os"

	"github.com/golang/glog"
)

const (
	rwMask = os.FileMode(0660)
	roMask = os.FileMode(0440)
)

// SetVolumeOwnership modifies the given volume to be owned by
// fsGroup, and sets SetGid so that newly created files are owned by
// fsGroup. If fsGroup is nil nothing is done.
func SetVolumeOwnership(mounter Mounter, fsGroup *int64) error {

	if fsGroup == nil {
		return nil
	}

	return filepath.Walk(mounter.GetPath(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
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
			glog.Errorf("Got nil stat_t for path %v while setting ownership of volume", path)
			return nil
		}

		err = os.Chown(path, int(stat.Uid), int(*fsGroup))
		if err != nil {
			glog.Errorf("Chown failed on %v: %v", path, err)
		}

		mask := rwMask
		if mounter.GetAttributes().ReadOnly {
			mask = roMask
		}

		if info.IsDir() {
			mask |= os.ModeSetgid
		}

		err = os.Chmod(path, info.Mode()|mask)
		if err != nil {
			glog.Errorf("Chmod failed on %v: %v", path, err)
		}

		return nil
	})
}

// IsSameFSGroup is called only for requests to mount an already mounted
// volume. It checks if fsGroup of new mount request is the same or not.
// It returns false if it not the same. It also returns current Gid of a path
// provided for dir variable.
func IsSameFSGroup(dir string, fsGroup int64) (bool, int, error) {
	info, err := os.Stat(dir)
	if err != nil {
		glog.Errorf("Error getting stats for %s (%v)", dir, err)
		return false, 0, err
	}
	s := info.Sys().(*syscall.Stat_t)
	return int(s.Gid) == int(fsGroup), int(s.Gid), nil
}
