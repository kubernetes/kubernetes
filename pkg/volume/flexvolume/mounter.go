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
	"os"
	"strconv"

	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

// FlexVolumeMounter is the disk that will be exposed by this plugin.
type flexVolumeMounter struct {
	*flexVolume
	// Runner used to setup the volume.
	runner exec.Interface
	// the considered volume spec
	spec     *volume.Spec
	readOnly bool
}

var _ volume.Mounter = &flexVolumeMounter{}

// Mounter interface

// SetUp creates new directory.
func (f *flexVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return f.SetUpAt(f.GetPath(), mounterArgs)
}

// SetUpAt creates new directory.
func (f *flexVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	// Mount only once.
	alreadyMounted, err := prepareForMount(f.mounter, dir)
	if err != nil {
		return err
	}
	if alreadyMounted {
		return nil
	}

	call := f.plugin.NewDriverCall(mountCmd)

	// Interface parameters
	call.Append(dir)

	extraOptions := make(map[string]string)

	// pod metadata
	extraOptions[optionKeyPodName] = f.podName
	extraOptions[optionKeyPodNamespace] = f.podNamespace
	extraOptions[optionKeyPodUID] = string(f.podUID)
	// service account metadata
	extraOptions[optionKeyServiceAccountName] = f.podServiceAccountName

	// Extract secret and pass it as options.
	if err := addSecretsToOptions(extraOptions, f.spec, f.podNamespace, f.driverName, f.plugin.host); err != nil {
		os.Remove(dir)
		return err
	}

	// Implicit parameters
	if mounterArgs.FsGroup != nil {
		extraOptions[optionFSGroup] = strconv.FormatInt(int64(*mounterArgs.FsGroup), 10)
	}

	call.AppendSpec(f.spec, f.plugin.host, extraOptions)

	_, err = call.Run()
	if isCmdNotSupportedErr(err) {
		err = (*mounterDefaults)(f).SetUpAt(dir, mounterArgs)
	}

	if err != nil {
		os.Remove(dir)
		return err
	}

	if !f.readOnly {
		if f.plugin.capabilities.FSGroup {
			// fullPluginName helps to distinguish different driver from flex volume plugin
			volume.SetVolumeOwnership(f, dir, mounterArgs.FsGroup, mounterArgs.FSGroupChangePolicy, util.FSGroupCompleteHook(f.plugin, f.spec))
		}
	}

	return nil
}

// GetAttributes get the flex volume attributes. The attributes will be queried
// using plugin callout after we finalize the callout syntax.
func (f *flexVolumeMounter) GetAttributes() volume.Attributes {
	return (*mounterDefaults)(f).GetAttributes()
}
