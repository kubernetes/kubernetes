/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package admission

import (
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// RealAdmissionPluginHost is implementation of AdmissionPluginHost.
// To save users from implementing / providing a VolumeHost interface to
// instantiate VolumePluginMgr, it implements the interface and initializes
// VolumePluginMgr internally.
type realAdmissionPluginHost struct {
	// For VolumeHost, only functions that are relevant to APIServer were implemented.
	cloudProvider   cloudprovider.Interface
	volumePluginMgr *volume.VolumePluginMgr
}

var _ AdmissionPluginHost = &realAdmissionPluginHost{}
var _ volume.VolumeHost = &realAdmissionPluginHost{}

func NewAdmissionPluginHost(cloudProvider cloudprovider.Interface, volumePlugins []volume.VolumePlugin) AdmissionPluginHost {
	host := &realAdmissionPluginHost{
		cloudProvider:   cloudProvider,
		volumePluginMgr: &volume.VolumePluginMgr{},
	}
	host.volumePluginMgr.InitPlugins(volumePlugins, host)
	return host
}

func (host *realAdmissionPluginHost) GetCloudProvider() cloudprovider.Interface {
	return host.cloudProvider
}

func (host *realAdmissionPluginHost) GetVolumePluginMgr() *volume.VolumePluginMgr {
	return host.volumePluginMgr
}

// VolumeHost

func (host *realAdmissionPluginHost) GetPluginDir(pluginName string) string {
	// used only in kubelet
	return ""
}

func (host *realAdmissionPluginHost) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	// used only in kubelet
	return ""
}

func (host *realAdmissionPluginHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	// used only in kubelet
	return ""
}

func (host *realAdmissionPluginHost) GetKubeClient() clientset.Interface {
	// used only in kubelet
	return nil
}

func (host *realAdmissionPluginHost) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	// used only in kubelet
	return nil, nil
}

func (host *realAdmissionPluginHost) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	// used only in kubelet
	return nil, nil
}

func (host *realAdmissionPluginHost) GetMounter() mount.Interface {
	// used only in kubelet
	return nil
}

func (host *realAdmissionPluginHost) GetWriter() io.Writer {
	// used only in kubelet
	return nil
}

func (host *realAdmissionPluginHost) GetHostName() string {
	// used only in kubelet
	return ""
}
