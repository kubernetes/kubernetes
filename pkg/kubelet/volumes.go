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

// NewWrapperMounter attempts to create a volume mounter
// from a volume Spec, pod and volume options.
// Returns a new volume Mounter or an error.
func (vh *volumeHost) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	return vh.kubelet.newVolumeMounterFromPlugins(&spec, pod, opts)
}

// NewWrapperUnmounter attempts to create a volume unmounter
// from a volume name and pod uid.
// Returns a new volume Unmounter or an error.
func (vh *volumeHost) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	plugin, err := vh.kubelet.volumePluginMgr.FindPluginBySpec(&spec)
	if err != nil {
		return nil, err
	}

	return plugin.NewUnmounter(spec.Name(), podUID)
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

// mountExternalVolumes mounts the volumes declared in a pod, attaching them
// to the host if necessary, and returns a map containing information about
// the volumes for the pod or an error.  This method is run multiple times,
// and requires that implementations of Attach() and SetUp() be idempotent.
//
// Note, in the future, the attach-detach controller will handle attaching and
// detaching volumes; this call site will be maintained for backward-
// compatibility with current behavior of static pods and pods created via the
// Kubelet's http API.
func (kl *Kubelet) mountExternalVolumes(pod *api.Pod) (kubecontainer.VolumeMap, error) {
	podVolumes := make(kubecontainer.VolumeMap)
	for i := range pod.Spec.Volumes {
		var fsGroup *int64
		if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroup != nil {
			fsGroup = pod.Spec.SecurityContext.FSGroup
		}

		rootContext, err := kl.getRootDirContext()
		if err != nil {
			return nil, err
		}

		// Try to use a plugin for this volume.
		volSpec := volume.NewSpecFromVolume(&pod.Spec.Volumes[i])
		mounter, err := kl.newVolumeMounterFromPlugins(volSpec, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume mounter for pod %s: %v", pod.UID, err)
			return nil, err
		}

		// some volumes require attachment before mounter's setup.
		// The plugin can be nil, but non-nil errors are legitimate errors.
		// For non-nil plugins, Attachment to a node is required before Mounter's setup.
		attacher, err := kl.newVolumeAttacherFromPlugins(volSpec, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume attacher for pod %s: %v", pod.UID, err)
			return nil, err
		}
		if attacher != nil {
			err = attacher.Attach(volSpec, kl.hostname)
			if err != nil {
				return nil, err
			}

			devicePath, err := attacher.WaitForAttach(volSpec, maxWaitForVolumeOps)
			if err != nil {
				return nil, err
			}

			deviceMountPath := attacher.GetDeviceMountPath(volSpec)
			if err = attacher.MountDevice(devicePath, deviceMountPath, kl.mounter); err != nil {
				return nil, err
			}
		}

		err = mounter.SetUp(fsGroup)
		if err != nil {
			return nil, err
		}
		podVolumes[volSpec.Volume.Name] = kubecontainer.VolumeInfo{Mounter: mounter}
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
		result[name] = info.Mounter
	}
	return result, true
}

// getPodVolumes examines the directory structure for a pod and returns
// information about the name and kind of each presently mounted volume, or an
// error.
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

// cleaner is a union struct to allow separating detaching from the cleaner.
// some volumes require detachment but not all.  Unmounter cannot be nil but Detacher is optional.
type cleaner struct {
	Unmounter volume.Unmounter
	Detacher  *volume.Detacher
}

// getPodVolumesFromDisk examines directory structure to determine volumes that
// are presently active and mounted. Returns a union struct containing a volume.Unmounter
// and potentially a volume.Detacher.
func (kl *Kubelet) getPodVolumesFromDisk() map[string]cleaner {
	currentVolumes := make(map[string]cleaner)
	podUIDs, err := kl.listPodsFromDisk()
	if err != nil {
		glog.Errorf("Could not get pods from disk: %v", err)
		return map[string]cleaner{}
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
			glog.V(4).Infof("Making a volume.Unmounter for volume %s/%s of pod %s", volume.Kind, volume.Name, podUID)
			// TODO(thockin) This should instead return a reference to an extant
			// volume object, except that we don't actually hold on to pod specs
			// or volume objects.

			// Try to use a plugin for this volume.
			unmounter, err := kl.newVolumeUnmounterFromPlugins(volume.Kind, volume.Name, podUID)
			if err != nil {
				glog.Errorf("Could not create volume unmounter for %s: %v", volume.Name, err)
				continue
			}

			tuple := cleaner{Unmounter: unmounter}
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

// newVolumeMounterFromPlugins attempts to find a plugin by volume spec, pod
// and volume options and then creates a Mounter.
// Returns a valid Unmounter or an error.
func (kl *Kubelet) newVolumeMounterFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	plugin, err := kl.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	physicalMounter, err := plugin.NewMounter(spec, pod, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate mounter for volume: %s using plugin: %s with a root cause: %v", spec.Name(), plugin.Name(), err)
	}
	glog.V(10).Infof("Used volume plugin %q to mount %s", plugin.Name(), spec.Name())
	return physicalMounter, nil
}

// newVolumeAttacherFromPlugins attempts to find a plugin from a volume spec
// and then create an Attacher.
// Returns:
//  - an attacher if one exists
//  - an error if no plugin was found for the volume
//    or the attacher failed to instantiate
//  - nil if there is no appropriate attacher for this volume
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

// newVolumeUnmounterFromPlugins attempts to find a plugin by name and then
// create an Unmounter.
// Returns a valid Unmounter or an error.
func (kl *Kubelet) newVolumeUnmounterFromPlugins(kind string, name string, podUID types.UID) (volume.Unmounter, error) {
	plugName := strings.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindPluginByName(plugName)
	if err != nil {
		// TODO: Maybe we should launch a cleanup of this dir?
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}

	unmounter, err := plugin.NewUnmounter(name, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(5).Infof("Used volume plugin %q to unmount %s/%s", plugin.Name(), podUID, kind)
	return unmounter, nil
}

// newVolumeDetacherFromPlugins attempts to find a plugin by a name and then
// create a Detacher.
// Returns:
//  - a detacher if one exists
//  - an error if no plugin was found for the volume
//    or the detacher failed to instantiate
//  - nil if there is no appropriate detacher for this volume
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
