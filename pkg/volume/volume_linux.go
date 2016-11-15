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

	"k8s.io/kubernetes/pkg/util/chmod"
	"k8s.io/kubernetes/pkg/util/chown"

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

	chownRunner := chown.New()
	chmodRunner := chmod.New()
	return filepath.Walk(mounter.GetPath(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		stat, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return nil
		}

		if stat == nil {
			glog.Errorf("Got nil stat_t for path %v while setting ownership of volume", path)
			return nil
		}

		err = chownRunner.Chown(path, int(stat.Uid), int(*fsGroup))
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

		err = chmodRunner.Chmod(path, info.Mode()|mask)
		if err != nil {
			glog.Errorf("Chmod failed on %v: %v", path, err)
		}

		return nil
	})
}
