/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"
	"io/ioutil"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
)

var errUnsupportedVolumeType = fmt.Errorf("unsupported volume type")

// This just exports required functions from kubelet proper, for use by volume
// plugins.
type volumeHost struct {
	kubelet *Kubelet
}

func (vh *volumeHost) GetPluginDir(pluginName string) string {
	return vh.kubelet.getPluginDir(pluginName)
}

func (vh *volumeHost) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return vh.kubelet.getPodVolumeDir(podUID, pluginName, volumeName)
}

func (vh *volumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return vh.kubelet.getPodPluginDir(podUID, pluginName)
}

func (kl *Kubelet) newVolumeBuilderFromPlugins(spec *api.Volume, podUID types.UID) volume.Builder {
	plugin, err := kl.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		glog.Warningf("Can't use volume plugins for %s: %v", spew.Sprintf("%#v", *spec), err)
		return nil
	}
	if plugin == nil {
		glog.Errorf("No error, but nil volume plugin for %s", spew.Sprintf("%#v", *spec))
		return nil
	}
	builder, err := plugin.NewBuilder(spec, podUID)
	if err != nil {
		glog.Warningf("Error instantiating volume plugin for %s: %v", spew.Sprintf("%#v", *spec), err)
		return nil
	}
	glog.V(3).Infof("Used volume plugin %q for %s", plugin.Name(), spew.Sprintf("%#v", *spec))
	return builder
}

func (kl *Kubelet) mountExternalVolumes(pod *api.BoundPod) (volumeMap, error) {
	podVolumes := make(volumeMap)
	for i := range pod.Spec.Volumes {
		volSpec := &pod.Spec.Volumes[i]

		// Try to use a plugin for this volume.
		builder := kl.newVolumeBuilderFromPlugins(volSpec, pod.UID)
		if builder == nil {
			return nil, errUnsupportedVolumeType
		}
		err := builder.SetUp()
		if err != nil {
			return nil, err
		}
		podVolumes[volSpec.Name] = builder
	}
	return podVolumes, nil
}

// getPodVolumesFromDisk examines directory structure to determine volumes that
// are presently active and mounted. Returns a map of volume.Cleaner types.
func (kl *Kubelet) getPodVolumesFromDisk() map[string]volume.Cleaner {
	currentVolumes := make(map[string]volume.Cleaner)

	podUIDs, err := kl.listPodsFromDisk()
	if err != nil {
		glog.Errorf("Could not get pods from disk: %v", err)
		return map[string]volume.Cleaner{}
	}

	// Find the volumes for each on-disk pod.
	for _, podUID := range podUIDs {
		podVolDir := kl.getPodVolumesDir(podUID)
		volumeKindDirs, err := ioutil.ReadDir(podVolDir)
		if err != nil {
			glog.Errorf("Could not read directory %s: %v", podVolDir, err)
		}
		for _, volumeKindDir := range volumeKindDirs {
			volumeKind := volumeKindDir.Name()
			volumeKindPath := path.Join(podVolDir, volumeKind)
			volumeNameDirs, err := ioutil.ReadDir(volumeKindPath)
			if err != nil {
				glog.Errorf("Could not read directory %s: %v", volumeKindPath, err)
			}
			for _, volumeNameDir := range volumeNameDirs {
				volumeName := volumeNameDir.Name()
				identifier := fmt.Sprintf("%s/%s", podUID, volumeName)
				glog.V(4).Infof("Making a volume.Cleaner for %s", volumeKindPath)
				// TODO(thockin) This should instead return a reference to an extant
				// volume object, except that we don't actually hold on to pod specs
				// or volume objects.

				// Try to use a plugin for this volume.
				cleaner := kl.newVolumeCleanerFromPlugins(volumeKind, volumeName, podUID)
				if cleaner == nil {
					glog.Errorf("Could not create volume cleaner for %s: %v", volumeNameDir.Name(), errUnsupportedVolumeType)
					continue
				}
				currentVolumes[identifier] = cleaner
			}
		}
	}
	return currentVolumes
}

func (kl *Kubelet) newVolumeCleanerFromPlugins(kind string, name string, podUID types.UID) volume.Cleaner {
	plugName := volume.UnescapePluginName(kind)
	plugin, err := kl.volumePluginMgr.FindPluginByName(plugName)
	if err != nil {
		// TODO: Maybe we should launch a cleanup of this dir?
		glog.Warningf("Can't use volume plugins for %s/%s: %v", podUID, kind, err)
		return nil
	}
	if plugin == nil {
		glog.Errorf("No error, but nil volume plugin for %s/%s", podUID, kind)
		return nil
	}
	cleaner, err := plugin.NewCleaner(name, podUID)
	if err != nil {
		glog.Warningf("Error instantiating volume plugin for %s/%s: %v", podUID, kind, err)
		return nil
	}
	glog.V(3).Infof("Used volume plugin %q for %s/%s", plugin.Name(), podUID, kind)
	return cleaner
}
