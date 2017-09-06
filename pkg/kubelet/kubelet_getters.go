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
	"io/ioutil"
	"net"
	"path/filepath"

	"github.com/golang/glog"
	cadvisorapiv1 "github.com/google/cadvisor/info/v1"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilfile "k8s.io/kubernetes/pkg/util/file"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
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
	return filepath.Join(kl.getRootDir(), options.DefaultKubeletPodsDirName)
}

// getPluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
func (kl *Kubelet) getPluginsDir() string {
	return filepath.Join(kl.getRootDir(), options.DefaultKubeletPluginsDirName)
}

// getPluginDir returns a data directory name for a given plugin name.
// Plugins can use these directories to store data that they need to persist.
// For per-pod plugin data, see getPodPluginDir.
func (kl *Kubelet) getPluginDir(pluginName string) string {
	return filepath.Join(kl.getPluginsDir(), pluginName)
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

// getPodVolumesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumesDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), options.DefaultKubeletVolumesDirName)
}

// getPodVolumeDir returns the full path to the directory which represents the
// named volume under the named plugin for specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return filepath.Join(kl.getPodVolumesDir(podUID), pluginName, volumeName)
}

// getPodPluginsDir returns the full path to the per-pod data directory under
// which plugins may store data for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodPluginsDir(podUID types.UID) string {
	return filepath.Join(kl.getPodDir(podUID), options.DefaultKubeletPluginsDirName)
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
	return filepath.Join(kl.getPodDir(podUID), options.DefaultKubeletContainersDirName, ctrName)
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pods.
func (kl *Kubelet) GetPods() []*v1.Pod {
	return kl.podManager.GetPods()
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

// GetHostname Returns the hostname as the kubelet sees it.
func (kl *Kubelet) GetHostname() string {
	return kl.hostname
}

// GetRuntime returns the current Runtime implementation in use by the kubelet. This func
// is exported to simplify integration with third party kubelet extensions (e.g. kubernetes-mesos).
func (kl *Kubelet) GetRuntime() kubecontainer.Runtime {
	return kl.containerRuntime
}

// GetNode returns the node info for the configured node name of this Kubelet.
func (kl *Kubelet) GetNode() (*v1.Node, error) {
	if kl.kubeClient == nil {
		return kl.initialNode()
	}
	return kl.nodeInfo.GetNodeInfo(string(kl.nodeName))
}

// getNodeAnyWay() must return a *v1.Node which is required by RunGeneralPredicates().
// The *v1.Node is obtained as follows:
// Return kubelet's nodeInfo for this node, except on error or if in standalone mode,
// in which case return a manufactured nodeInfo representing a node with no pods,
// zero capacity, and the default labels.
func (kl *Kubelet) getNodeAnyWay() (*v1.Node, error) {
	if kl.kubeClient != nil {
		if n, err := kl.nodeInfo.GetNodeInfo(string(kl.nodeName)); err == nil {
			return n, nil
		}
	}
	return kl.initialNode()
}

// GetNodeConfig returns the container manager node config.
func (kl *Kubelet) GetNodeConfig() cm.NodeConfig {
	return kl.containerManager.GetNodeConfig()
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

	if pathExists, pathErr := volumeutil.PathExists(podVolDir); pathErr != nil {
		return volumes, fmt.Errorf("Error checking if path %q exists: %v", podVolDir, pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: path %q does not exist: %q", podVolDir)
		return volumes, nil
	}

	volumePluginDirs, err := ioutil.ReadDir(podVolDir)
	if err != nil {
		glog.Errorf("Could not read directory %s: %v", podVolDir, err)
		return volumes, err
	}
	for _, volumePluginDir := range volumePluginDirs {
		volumePluginName := volumePluginDir.Name()
		volumePluginPath := filepath.Join(podVolDir, volumePluginName)
		volumeDirs, err := utilfile.ReadDirNoStat(volumePluginPath)
		if err != nil {
			return volumes, fmt.Errorf("Could not read directory %s: %v", volumePluginPath, err)
		}
		for _, volumeDir := range volumeDirs {
			volumes = append(volumes, filepath.Join(volumePluginPath, volumeDir))
		}
	}
	return volumes, nil
}

// GetVersionInfo returns information about the version of cAdvisor in use.
func (kl *Kubelet) GetVersionInfo() (*cadvisorapiv1.VersionInfo, error) {
	return kl.cadvisor.VersionInfo()
}

// GetCachedMachineInfo assumes that the machine info can't change without a reboot
func (kl *Kubelet) GetCachedMachineInfo() (*cadvisorapiv1.MachineInfo, error) {
	if kl.machineInfo == nil {
		info, err := kl.cadvisor.MachineInfo()
		if err != nil {
			return nil, err
		}
		kl.machineInfo = info
	}
	return kl.machineInfo, nil
}
