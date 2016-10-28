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

package kubelet

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/selinux"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// ListVolumesForPod returns a map of the mounted volumes for the given pod.
// The key in the map is the OuterVolumeSpecName (i.e. pod.Spec.Volumes[x].Name)
func (kl *Kubelet) ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool) {
	volumesToReturn := make(map[string]volume.Volume)
	podVolumes := kl.volumeManager.GetMountedVolumesForPod(
		volumetypes.UniquePodName(podUID))
	for outerVolumeSpecName, volume := range podVolumes {
		// TODO: volume.Mounter could be nil if volume object is recovered
		// from reconciler's sync state process. PR 33616 will fix this problem
		// to create Mounter object when recovering volume state.
		if volume.Mounter == nil {
			continue
		}
		volumesToReturn[outerVolumeSpecName] = volume.Mounter
	}

	return volumesToReturn, len(volumesToReturn) > 0
}

// podVolumesExist checks with the volume manager and returns true any of the
// pods for the specified volume are mounted.
func (kl *Kubelet) podVolumesExist(podUID types.UID) bool {
	if mountedVolumes :=
		kl.volumeManager.GetMountedVolumesForPod(
			volumetypes.UniquePodName(podUID)); len(mountedVolumes) > 0 {
		return true
	}

	return false
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
		return nil, fmt.Errorf("failed to instantiate mounter for volume: %s using plugin: %s with a root cause: %v", spec.Name(), plugin.GetPluginName(), err)
	}
	glog.V(10).Infof("Using volume plugin %q to mount %s", plugin.GetPluginName(), spec.Name())
	return physicalMounter, nil
}

// relabelVolumes relabels SELinux volumes to match the pod's
// SELinuxOptions specification. This is only needed if the pod uses
// hostPID or hostIPC. Otherwise relabeling is delegated to docker.
func (kl *Kubelet) relabelVolumes(pod *api.Pod, volumes kubecontainer.VolumeMap) error {
	if pod.Spec.SecurityContext.SELinuxOptions == nil {
		return nil
	}

	rootDirContext, err := kl.getRootDirContext()
	if err != nil {
		return err
	}

	selinuxRunner := selinux.NewSelinuxContextRunner()
	// Apply the pod's Level to the rootDirContext
	rootDirSELinuxOptions, err := securitycontext.ParseSELinuxOptions(rootDirContext)
	if err != nil {
		return err
	}

	rootDirSELinuxOptions.Level = pod.Spec.SecurityContext.SELinuxOptions.Level
	volumeContext := fmt.Sprintf("%s:%s:%s:%s", rootDirSELinuxOptions.User, rootDirSELinuxOptions.Role, rootDirSELinuxOptions.Type, rootDirSELinuxOptions.Level)

	for _, vol := range volumes {
		if vol.Mounter.GetAttributes().Managed && vol.Mounter.GetAttributes().SupportsSELinux {
			// Relabel the volume and its content to match the 'Level' of the pod
			path, err := volume.GetPath(vol.Mounter)
			if err != nil {
				return err
			}
			err = filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				return selinuxRunner.SetContext(path, volumeContext)
			})
			if err != nil {
				return err
			}
			vol.SELinuxLabeled = true
		}
	}
	return nil
}

// cleanupOrphanedPodDirs removes the volumes of pods that should not be
// running and that have no containers running.
func (kl *Kubelet) cleanupOrphanedPodDirs(
	pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	allPods := sets.NewString()
	for _, pod := range pods {
		allPods.Insert(string(pod.UID))
	}
	for _, pod := range runningPods {
		allPods.Insert(string(pod.ID))
	}

	found, err := kl.listPodsFromDisk()
	if err != nil {
		return err
	}
	errlist := []error{}
	for _, uid := range found {
		if allPods.Has(string(uid)) {
			continue
		}
		// If volumes have not been unmounted/detached, do not delete directory.
		// Doing so may result in corruption of data.
		if podVolumesExist := kl.podVolumesExist(uid); podVolumesExist {
			glog.V(3).Infof("Orphaned pod %q found, but volumes are not cleaned up; err: %v", uid, err)
			continue
		}
		// Check whether volume is still mounted on disk. If so, do not delete directory
		if volumeNames, err := kl.getPodVolumeNameListFromDisk(uid); err != nil || len(volumeNames) != 0 {
			glog.V(3).Infof("Orphaned pod %q found, but volumes are still mounted; err: %v, volumes: %v ", uid, err, volumeNames)
			continue
		}

		glog.V(3).Infof("Orphaned pod %q found, removing", uid)
		if err := os.RemoveAll(kl.getPodDir(uid)); err != nil {
			glog.Errorf("Failed to remove orphaned pod %q dir; err: %v", uid, err)
			errlist = append(errlist, err)
		}
	}
	return utilerrors.NewAggregate(errlist)
}
