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

package flexvolume

import (
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// FlexVolumeMounter is the disk that will be exposed by this plugin.
type flexVolumeMounter struct {
	*flexVolume
	// Runner used to setup the volume.
	runner exec.Interface
	// blockDeviceMounter provides the interface to create filesystem if the
	// filesystem doesn't exist.
	blockDeviceMounter mount.Interface
	// the considered volume spec
	spec     *volume.Spec
	readOnly bool
	volume.MetricsNil
}

var _ volume.Mounter = &flexVolumeMounter{}

// Mounter interface

// SetUp creates new directory.
func (f *flexVolumeMounter) SetUp(fsGroup *int64) error {
	return f.SetUpAt(f.GetPath(), fsGroup)
}

// SetUpAt creates new directory.
func (f *flexVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	// Mount only once.
	alreadyMounted, err := prepareForMount(f.mounter, dir)
	if err != nil {
		return err
	}
	if alreadyMounted {
		return nil
	}

	call := f.plugin.NewDriverCall(mountCmd)
	// Implicit parameters
	if fsGroup == nil {
		call.AppendSpec(f.spec, f.plugin.host, nil)
	} else {
		call.AppendSpec(f.spec, f.plugin.host, map[string]string{
			optionFSGroup: strconv.FormatInt(*fsGroup, 10),
		})
	}

	a, err := f.plugin.NewAttacher()
	if err != nil {
		return fmt.Errorf("NewAttacher failed: %v", err)
	}
	src, err := a.GetDeviceMountPath(f.spec)
	if err != nil {
		return fmt.Errorf("GetDeviceMountPath failed: %v", err)
	}
	call.Append(src)

	// Interface parameters
	call.Append(dir)

	_, err = call.Run()
	if isCmdNotSupportedErr(err) {
		return (*mounterDefaults)(f).SetUpAt(src, dir, fsGroup)
	}
	return err
}

// GetAttributes get the flex volume attributes. The attributes will be queried
// using plugin callout after we finalize the callout syntax.
func (f *flexVolumeMounter) GetAttributes() volume.Attributes {
	return (*mounterDefaults)(f).GetAttributes()
}
