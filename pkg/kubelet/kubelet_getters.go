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
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"path/filepath"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	"k8s.io/klog"
	"k8s.io/utils/mount"
	utilpath "k8s.io/utils/path"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// getRootDir returns the full path to the directory under which kubelet can
// store data.  These functions are useful to pass interfaces to other modules
// that may need to know where to write data without getting a whole kubelet
// instance.
func (kl *Kubelet) getRootDir() string {
	return kl.rootDirectory
}

// getPodsDir returns the full path to the directory under which pod
// directories are created.
func (kl *Kubelet) getPodsDir() string {
	return filepath.Join(kl.getRootDir(), config.DefaultKubeletPodsDirName)
}

// getPluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
func (kl *Kubelet) getPluginsDir() string {
	return filepath.Join(kl.getRootDir(), config.DefaultKubeletPluginsDirName)
}

// getPluginsRegistrationDir returns the full path to the directory under which
// plugins socket should be placed to be registered.
// More information is available about plugin registration in the pluginwatcher
// module
func (kl *Kubelet) getPluginsRegistrationDir() string {
	return filepath.Join(kl.getRootDir(), config.DefaultKubeletPluginsRegistrationDirName)
}

// getPluginDir returns a data directory name for a given plugin name.
// Plugins can use these directories to store data that they need to persist.
// For per-pod plugin data, see getPodPluginDir.
func (kl *Kubelet) getPluginDir(pluginName string) string {
	return filepath.Join(kl.getPluginsDir(), pluginName)
}

// getVolumeDevicePluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
func (kl *Kubelet) getVolumeDevicePluginsDir() string {
	return filepath.Join(kl.getRootDir(), config.DefaultKubeletPluginsDirName)
}

// getVolumeDevicePluginDir returns a data directory name for a given plugin name.
// Plugins can use these directories to store data that they need to persist.
// For per-pod plugin data, see getVolumeDevicePluginsDir.
func (kl *Kubelet) getVolumeDevicePluginDir(pluginName string) string {
	return filepath.Join(kl.getVolumeDevicePluginsDir(), pluginName, config.DefaultKubeletVolumeDevicesDirName)
}

// GetPodDir returns the full path to the per-pod data directory for the
// specified pod. This directory may not exist if the pod does not exist.
func (kl *Kubelet) GetPodDir(podUID types.UID) string {
	return kl.getPodDir(podUID)
}

// getPodDir returns the full path to the per-pod directory for the pod with
// the given UID.
func (kl *Kubelet) getPodDir(podUID types.UID) string {
	return filepath.Join(kl.getPodsDir(), string(podUID))
}

// getPodVolumesSubpathsDir returns the full path to the per-pod subpaths directory under
// which subpath volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist or subpaths are not specified.
func (kl *Kubelet) getPodVolumeSubpathsDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), config.DefaultKubeletVolumeSubpathsDirName)
}

// getPodVolumesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumesDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), config.DefaultKubeletVolumesDirName)
}

// getPodVolumeDir returns the full path to the directory which represents the
// named volume under the named plugin for specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return filepath.Join(kl.getPodVolumesDir(podUID), pluginName, volumeName)
}

// getPodVolumeDevicesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod. This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumeDevicesDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), config.DefaultKubeletVolumeDevicesDirName)
}

// getPodVolumeDeviceDir returns the full path to the directory which represents the
// named plugin for specified pod. This directory may not exist if the pod does not exist.
func (kl *Kubelet) getPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return filepath.Join(kl.getPodVolumeDevicesDir(podUID), pluginName)
}

// getPodPluginsDir returns the full path to the per-pod data directory under
// which plugins may store data for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodPluginsDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), config.DefaultKubeletPluginsDirName)
}

// getPodPluginDir returns a data directory name for a given plugin name for a
// given pod UID.  Plugins can use these directories to store data that they
// need to persist.  For non-per-pod plugin data, see getPluginDir.
func (kl *Kubelet) getPodPluginDir(podUID types.UID, pluginName string) string {
	return filepath.Join(kl.getPodPluginsDir(podUID), pluginName)
}

// getPodContainerDir returns the full path to the per-pod data directory under
// which container data is held for the specified pod.  This directory may not
// exist if the pod or container does not exist.
func (kl *Kubelet) getPodContainerDir(podUID types.UID, ctrName string) string {
	return filepath.Join(kl.getPodDir(podUID), config.DefaultKubeletContainersDirName, ctrName)
}

// getPodResourcesSocket returns the full path to the directory containing the pod resources socket
func (kl *Kubelet) getPodResourcesDir() string {
	return filepath.Join(kl.getRootDir(), config.DefaultKubeletPodResourcesDirName)
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pods.
func (kl *Kubelet) GetPods() []*v1.Pod {
	pods := kl.podManager.GetPods()
	// a kubelet running without apiserver requires an additional
	// update of the static pod status. See #57106
	for _, p := range pods {
		if kubelettypes.IsStaticPod(p) {
			if status, ok := kl.statusManager.GetPodStatus(p.UID); ok {
				klog.V(2).Infof("status for pod %v updated to %v", p.Name, status)
				p.Status = status
			}
		}
	}
	return pods
}

// GetRunningPods returns all pods running on kubelet from looking at the
// container runtime cache. This function converts kubecontainer.Pod to
// v1.Pod, so only the fields that exist in both kubecontainer.Pod and
// v1.Pod are considered meaningful.
func (kl *Kubelet) GetRunningPods() ([]*v1.Pod, error) {
	pods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}

	apiPods := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		apiPods = append(apiPods, pod.ToAPIPod())
	}
	return apiPods, nil
}

// GetPodByFullName gets the pod with the given 'full' name, which
// incorporates the namespace as well as whether the pod was found.
func (kl *Kubelet) GetPodByFullName(podFullName string) (*v1.Pod, bool) {
	return kl.podManager.GetPodByFullName(podFullName)
}

// GetPodByName provides the first pod that matches namespace and name, as well
// as whether the pod was found.
func (kl *Kubelet) GetPodByName(namespace, name string) (*v1.Pod, bool) {
	return kl.podManager.GetPodByName(namespace, name)
}

// GetPodByCgroupfs provides the pod that maps to the specified cgroup, as well
// as whether the pod was found.
func (kl *Kubelet) GetPodByCgroupfs(cgroupfs string) (*v1.Pod, bool) {
	pcm := kl.containerManager.NewPodContainerManager()
	if result, podUID := pcm.IsPodCgroup(cgroupfs); result {
		return kl.podManager.GetPodByUID(podUID)
	}
	return nil, false
}

// GetHostname Returns the hostname as the kubelet sees it.
func (kl *Kubelet) GetHostname() string {
	return kl.hostname
}

// getRuntime returns the current Runtime implementation in use by the kubelet.
func (kl *Kubelet) getRuntime() kubecontainer.Runtime {
	return kl.containerRuntime
}

// GetNode returns the node info for the configured node name of this Kubelet.
func (kl *Kubelet) GetNode() (*v1.Node, error) {
	if kl.kubeClient == nil {
		return kl.initialNode(context.TODO())
	}
	return kl.nodeLister.Get(string(kl.nodeName))
}

// getNodeAnyWay() must return a *v1.Node which is required by RunGeneralPredicates().
// The *v1.Node is obtained as follows:
// Return kubelet's nodeInfo for this node, except on error or if in standalone mode,
// in which case return a manufactured nodeInfo representing a node with no pods,
// zero capacity, and the default labels.
func (kl *Kubelet) getNodeAnyWay() (*v1.Node, error) {
	if kl.kubeClient != nil {
		if n, err := kl.nodeLister.Get(string(kl.nodeName)); err == nil {
			return n, nil
		}
	}
	return kl.initialNode(context.TODO())
}

// GetNodeConfig returns the container manager node config.
func (kl *Kubelet) GetNodeConfig() cm.NodeConfig {
	return kl.containerManager.GetNodeConfig()
}

// GetPodCgroupRoot returns the listeral cgroupfs value for the cgroup containing all pods
func (kl *Kubelet) GetPodCgroupRoot() string {
	return kl.containerManager.GetPodCgroupRoot()
}

// GetHostIP returns host IP or nil in case of error.
func (kl *Kubelet) GetHostIP() (net.IP, error) {
	node, err := kl.GetNode()
	if err != nil {
		return nil, fmt.Errorf("cannot get node: %v", err)
	}
	return utilnode.GetNodeHostIP(node)
}

// getHostIPAnyway attempts to return the host IP from kubelet's nodeInfo, or
// the initialNode.
func (kl *Kubelet) getHostIPAnyWay() (net.IP, error) {
	node, err := kl.getNodeAnyWay()
	if err != nil {
		return nil, err
	}
	return utilnode.GetNodeHostIP(node)
}

// GetExtraSupplementalGroupsForPod returns a list of the extra
// supplemental groups for the Pod. These extra supplemental groups come
// from annotations on persistent volumes that the pod depends on.
func (kl *Kubelet) GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64 {
	return kl.volumeManager.GetExtraSupplementalGroupsForPod(pod)
}

// getPodVolumePathListFromDisk returns a list of the volume paths by reading the
// volume directories for the given pod from the disk.
func (kl *Kubelet) getPodVolumePathListFromDisk(podUID types.UID) ([]string, error) {
	volumes := []string{}
	podVolDir := kl.getPodVolumesDir(podUID)

	if pathExists, pathErr := mount.PathExists(podVolDir); pathErr != nil {
		return volumes, fmt.Errorf("error checking if path %q exists: %v", podVolDir, pathErr)
	} else if !pathExists {
		klog.Warningf("Path %q does not exist", podVolDir)
		return volumes, nil
	}

	volumePluginDirs, err := ioutil.ReadDir(podVolDir)
	if err != nil {
		klog.Errorf("Could not read directory %s: %v", podVolDir, err)
		return volumes, err
	}
	for _, volumePluginDir := range volumePluginDirs {
		volumePluginName := volumePluginDir.Name()
		volumePluginPath := filepath.Join(podVolDir, volumePluginName)
		volumeDirs, err := utilpath.ReadDirNoStat(volumePluginPath)
		if err != nil {
			return volumes, fmt.Errorf("could not read directory %s: %v", volumePluginPath, err)
		}
		for _, volumeDir := range volumeDirs {
			volumes = append(volumes, filepath.Join(volumePluginPath, volumeDir))
		}
	}
	return volumes, nil
}

func (kl *Kubelet) getMountedVolumePathListFromDisk(podUID types.UID) ([]string, error) {
	mountedVolumes := []string{}
	volumePaths, err := kl.getPodVolumePathListFromDisk(podUID)
	if err != nil {
		return mountedVolumes, err
	}
	for _, volumePath := range volumePaths {
		isNotMount, err := kl.mounter.IsLikelyNotMountPoint(volumePath)
		if err != nil {
			return mountedVolumes, err
		}
		if !isNotMount {
			mountedVolumes = append(mountedVolumes, volumePath)
		}
	}
	return mountedVolumes, nil
}

// podVolumesSubpathsDirExists returns true if the pod volume-subpaths directory for
// a given pod exists
func (kl *Kubelet) podVolumeSubpathsDirExists(podUID types.UID) (bool, error) {
	podVolDir := kl.getPodVolumeSubpathsDir(podUID)

	if pathExists, pathErr := mount.PathExists(podVolDir); pathErr != nil {
		return true, fmt.Errorf("error checking if path %q exists: %v", podVolDir, pathErr)
	} else if !pathExists {
		return false, nil
	}
	return true, nil
}

// GetVersionInfo returns information about the version of cAdvisor in use.
func (kl *Kubelet) GetVersionInfo() (*cadvisorapiv1.VersionInfo, error) {
	return kl.cadvisor.VersionInfo()
}

// GetCachedMachineInfo assumes that the machine info can't change without a reboot
func (kl *Kubelet) GetCachedMachineInfo() (*cadvisorapiv1.MachineInfo, error) {
	return kl.machineInfo, nil
}
