/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
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

func (vh *volumeHost) GetKubeClient() client.Interface {
	return vh.kubelet.kubeClient
}

func (vh *volumeHost) NewWrapperBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	b, err := vh.kubelet.newVolumeBuilderFromPlugins(spec, pod, opts, mounter)
	if err == nil && b == nil {
		return nil, errUnsupportedVolumeType
	}
	return b, nil
}

func (vh *volumeHost) NewWrapperCleaner(spec *volume.Spec, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	plugin, err := vh.kubelet.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if plugin == nil {
		// Not found but not an error
		return nil, nil
	}
	c, err := plugin.NewCleaner(spec.Name(), podUID, mounter)
	if err == nil && c == nil {
		return nil, errUnsupportedVolumeType
	}
	return c, nil
}

func (vh *volumeHost) GetCloudProvider() cloudprovider.Interface {
	return vh.kubelet.cloud
}

func (kl *Kubelet) newVolumeBuilderFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	plugin, err := kl.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	if plugin == nil {
		// Not found but not an error
		return nil, nil
	}
	builder, err := plugin.NewBuilder(spec, pod, opts, mounter)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s: %v", spec.Name(), err)
	}
	glog.V(3).Infof("Used volume plugin %q for %s", plugin.Name(), spec.Name())
	return builder, nil
}

func (kl *Kubelet) mountExternalVolumes(pod *api.Pod) (kubecontainer.VolumeMap, error) {
	podVolumes := make(kubecontainer.VolumeMap)
	for i := range pod.Spec.Volumes {
		volSpec := &pod.Spec.Volumes[i]

		rootContext, err := kl.getRootDirContext()
		if err != nil {
			return nil, err
		}

		// Try to use a plugin for this volume.
		internal := volume.NewSpecFromVolume(volSpec)
		builder, err := kl.newVolumeBuilderFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext}, kl.mounter)
		if err != nil {
			glog.Errorf("Could not create volume builder for pod %s: %v", pod.UID, err)
			return nil, err
		}
		if builder == nil {
			return nil, errUnsupportedVolumeType
		}
		err = builder.SetUp()
		if err != nil {
			return nil, err
		}
		podVolumes[volSpec.Name] = builder
	}
	return podVolumes, nil
}

type volumeTuple struct {
	Kind string
	Name string
}

func (kl *Kubelet) getPodVolumes(podUID types.UID) ([]*volumeTuple, error) {
	var volumes []*volumeTuple
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
			return []*volumeTuple{}, fmt.Errorf("could not read directory %s: %v", volumeKindPath, err)
		}
		for _, volumeNameDir := range volumeNameDirs {
			volumes = append(volumes, &volumeTuple{Kind: volumeKind, Name: volumeNameDir.Name()})
		}
	}
	return volumes, nil
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
		volumes, err := kl.getPodVolumes(podUID)
		if err != nil {
			glog.Errorf("%v", err)
			continue
		}
		for _, volume := range volumes {
			identifier := fmt.Sprintf("%s/%s", podUID, volume.Name)
			glog.V(4).Infof("Making a volume.Cleaner for volume %s/%s of pod %s", volume.Kind, volume.Name, podUID)
			// TODO(thockin) This should instead return a reference to an extant
			// volume object, except that we don't actually hold on to pod specs
			// or volume objects.

			// Try to use a plugin for this volume.
			cleaner, err := kl.newVolumeCleanerFromPlugins(volume.Kind, volume.Name, podUID, kl.mounter)
			if err != nil {
				glog.Errorf("Could not create volume cleaner for %s: %v", volume.Name, err)
				continue
			}
			if cleaner == nil {
				glog.Errorf("Could not create volume cleaner for %s: %v", volume.Name, errUnsupportedVolumeType)
				continue
			}
			currentVolumes[identifier] = cleaner
		}
	}
	return currentVolumes
}

func (kl *Kubelet) newVolumeCleanerFromPlugins(kind string, name string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	plugName := util.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindPluginByName(plugName)
	if err != nil {
		// TODO: Maybe we should launch a cleanup of this dir?
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}
	cleaner, err := plugin.NewCleaner(name, podUID, mounter)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(3).Infof("Used volume plugin %q for %s/%s", plugin.Name(), podUID, kind)
	return cleaner, nil
}
