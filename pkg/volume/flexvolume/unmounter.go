/*
Copyright 2017 The Kubernetes Authors.

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

package flexvolume

import (
	"fmt"
	"os"

	"k8s.io/klog"
	"k8s.io/utils/exec"
	"k8s.io/utils/mount"

	"k8s.io/kubernetes/pkg/volume"
)

// FlexVolumeUnmounter is the disk that will be cleaned by this plugin.
type flexVolumeUnmounter struct {
	*flexVolume
	// Runner used to teardown the volume.
	runner exec.Interface
}

var _ volume.Unmounter = &flexVolumeUnmounter{}

// Unmounter interface
func (f *flexVolumeUnmounter) TearDown() error {
	path := f.GetPath()
	return f.TearDownAt(path)
}

func (f *flexVolumeUnmounter) TearDownAt(dir string) error {
	pathExists, pathErr := mount.PathExists(dir)
	if pathErr != nil {
		// only log warning here since plugins should anyways have to deal with errors
		klog.Warningf("Error checking path: %v", pathErr)
	} else {
		if !pathExists {
			klog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
			return nil
		}
	}

	call := f.plugin.NewDriverCall(unmountCmd)
	call.Append(dir)
	_, err := call.Run()
	if isCmdNotSupportedErr(err) {
		err = (*unmounterDefaults)(f).TearDownAt(dir)
	}
	if err != nil {
		return err
	}

	// Flexvolume driver may remove the directory. Ignore if it does.
	if pathExists, pathErr := mount.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		return nil
	}
	return os.Remove(dir)
}
