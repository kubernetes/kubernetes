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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
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

func (vh *volumeHost) GetKubeClient() clientset.Interface {
	return vh.kubelet.kubeClient
}

func (vh *volumeHost) NewWrapperBuilder(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	b, err := vh.kubelet.newVolumeBuilderFromPlugins(&spec, pod, opts)
	if err == nil && b == nil {
		return nil, errUnsupportedVolumeType
	}
	return b, nil
}

func (vh *volumeHost) NewWrapperCleaner(volName string, spec volume.Spec, podUID types.UID) (volume.Cleaner, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	plugin, err := vh.kubelet.volumePluginMgr.FindPluginBySpec(&spec)
	if err != nil {
		return nil, err
	}
	if plugin == nil {
		// Not found but not an error
		return nil, nil
	}
	c, err := plugin.NewCleaner(spec.Name(), podUID)
	if err == nil && c == nil {
		return nil, errUnsupportedVolumeType
	}
	return c, nil
}

func (vh *volumeHost) GetCloudProvider() cloudprovider.Interface {
	return vh.kubelet.cloud
}

func (vh *volumeHost) GetMounter() mount.Interface {
	return vh.kubelet.mounter
}

func (vh *volumeHost) GetWriter() io.Writer {
	return vh.kubelet.writer
}

// Returns the hostname of the host kubelet is running on
func (vh *volumeHost) GetHostName() string {
	return vh.kubelet.hostname
}

func (kl *Kubelet) mountExternalVolumes(pod *api.Pod) (kubecontainer.VolumeMap, error) {
	podVolumes := make(kubecontainer.VolumeMap)
	for i := range pod.Spec.Volumes {
		volSpec := &pod.Spec.Volumes[i]
		var fsGroup *int64
		if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroup != nil {
			fsGroup = pod.Spec.SecurityContext.FSGroup
		}

		rootContext, err := kl.getRootDirContext()
		if err != nil {
			return nil, err
		}

		// Try to use a plugin for this volume.
		internal := volume.NewSpecFromVolume(volSpec)
		builder, err := kl.newVolumeBuilderFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume builder for pod %s: %v", pod.UID, err)
			return nil, err
		}
		if builder == nil {
			return nil, errUnsupportedVolumeType
		}

		// some volumes require attachment before builder's setup.
		// The plugin can be nil, but non-nil errors are legitimate errors.
		// For non-nil plugins, Attachment to a node is required before Builder's setup.
		attacher, err := kl.newVolumeAttacherFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume attacher for pod %s: %v", pod.UID, err)
			return nil, err
		}
		if attacher != nil {
			err = attacher.Attach()
			if err != nil {
				return nil, err
			}
		}

		err = builder.SetUp(fsGroup)
		if err != nil {
			return nil, err
		}
		podVolumes[volSpec.Name] = kubecontainer.VolumeInfo{Builder: builder}
	}
	return podVolumes, nil
}

type volumeTuple struct {
	Kind string
	Name string
}

// ListVolumesForPod returns a map of the volumes associated with the given pod
func (kl *Kubelet) ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool) {
	result := map[string]volume.Volume{}
	vm, ok := kl.volumeManager.GetVolumes(podUID)
	if !ok {
		return result, false
	}
	for name, info := range vm {
		result[name] = info.Builder
	}
	return result, true
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
		// ioutil.ReadDir exits without returning any healthy dir when encountering the first lstat error
		// but skipping dirs means no cleanup for healthy volumes. switching to a no-exit api solves this problem
		volumeNameDirs, volumeNameDirsStat, err := util.ReadDirNoExit(volumeKindPath)
		if err != nil {
			return []*volumeTuple{}, fmt.Errorf("could not read directory %s: %v", volumeKindPath, err)
		}
		for i, volumeNameDir := range volumeNameDirs {
			if volumeNameDir != nil {
				volumes = append(volumes, &volumeTuple{Kind: volumeKind, Name: volumeNameDir.Name()})
			} else {
				glog.Errorf("Could not read directory %s: %v", podVolDir, volumeNameDirsStat[i])
			}
		}
	}
	return volumes, nil
}

// cleanerTuple is a union struct to allow separating detaching from the cleaner.
// some volumes require detachment but not all.  Cleaner cannot be nil but Detacher is optional.
type cleanerTuple struct {
	Cleaner  volume.Cleaner
	Detacher *volume.Detacher
}

// getPodVolumesFromDisk examines directory structure to determine volumes that
// are presently active and mounted. Returns a union struct containing a volume.Cleaner
// and potentially a volume.Detacher.
func (kl *Kubelet) getPodVolumesFromDisk() map[string]cleanerTuple {
	currentVolumes := make(map[string]cleanerTuple)
	podUIDs, err := kl.listPodsFromDisk()
	if err != nil {
		glog.Errorf("Could not get pods from disk: %v", err)
		return map[string]cleanerTuple{}
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
			cleaner, err := kl.newVolumeCleanerFromPlugins(volume.Kind, volume.Name, podUID)
			if err != nil {
				glog.Errorf("Could not create volume cleaner for %s: %v", volume.Name, err)
				continue
			}
			if cleaner == nil {
				glog.Errorf("Could not create volume cleaner for %s: %v", volume.Name, errUnsupportedVolumeType)
				continue
			}

			tuple := cleanerTuple{Cleaner: cleaner}
			detacher, err := kl.newVolumeDetacherFromPlugins(volume.Kind, volume.Name, podUID)
			// plugin can be nil but a non-nil error is a legitimate error
			if err != nil {
				glog.Errorf("Could not create volume detacher for %s: %v", volume.Name, err)
				continue
			}
			if detacher != nil {
				tuple.Detacher = &detacher
			}
			currentVolumes[identifier] = tuple
		}
	}
	return currentVolumes
}

func (kl *Kubelet) newVolumeBuilderFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	plugin, err := kl.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	if plugin == nil {
		// Not found but not an error
		return nil, nil
	}
	builder, err := plugin.NewBuilder(spec, pod, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume builder for %s: %v", spec.Name(), err)
	}
	glog.V(3).Infof("Used volume plugin %q to mount %s", plugin.Name(), spec.Name())
	return builder, nil
}

func (kl *Kubelet) newVolumeAttacherFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Attacher, error) {
	plugin, err := kl.volumePluginMgr.FindAttachablePluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}

	attacher, err := plugin.NewAttacher(spec)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume attacher for %s: %v", spec.Name(), err)
	}
	glog.V(3).Infof("Used volume plugin %q to attach %s/%s", plugin.Name(), spec.Name())
	return attacher, nil
}

func (kl *Kubelet) newVolumeCleanerFromPlugins(kind string, name string, podUID types.UID) (volume.Cleaner, error) {
	plugName := strings.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindPluginByName(plugName)
	if err != nil {
		// TODO: Maybe we should launch a cleanup of this dir?
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}
	cleaner, err := plugin.NewCleaner(name, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(3).Infof("Used volume plugin %q to unmount %s/%s", plugin.Name(), podUID, kind)
	return cleaner, nil
}

func (kl *Kubelet) newVolumeDetacherFromPlugins(kind string, name string, podUID types.UID) (volume.Detacher, error) {
	plugName := strings.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindAttachablePluginByName(plugName)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}

	detacher, err := plugin.NewDetacher(name, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(3).Infof("Used volume plugin %q to detach %s/%s", plugin.Name(), podUID, kind)
	return detacher, nil
}
