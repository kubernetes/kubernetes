/*
Copyright 2014 The Kubernetes Authors.

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

package openstack

import (
	"fmt"
	"net"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

type OpenStackHost interface {
	Run(stopCh <-chan struct{})
}

type volumeHost struct {
	// kubeClient is the kube API client used by volumehost to communicate with
	// the API server.
	kubeClient clientset.Interface

	// pvcLister is the shared PVC lister used to fetch and store PVC
	// objects from the API server. It is shared with other controllers and
	// therefore the PVC objects in its store should be treated as immutable.
	pvcLister  corelisters.PersistentVolumeClaimLister
	pvcsSynced kcache.InformerSynced

	pvLister corelisters.PersistentVolumeLister
	pvSynced kcache.InformerSynced

	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// volumePluginMgr used to initialize and fetch volume plugins
	volumePluginMgr volume.VolumePluginMgr

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	// Volume resize map of volumes that needs resizing
	resizeMap cache.VolumeResizeMap

	// Operation executor
	opExecutor operationexecutor.OperationExecutor
}

func NewVolumeHost(
	kubeClient clientset.Interface, cloud cloudprovider.Interface, plugins []volume.VolumePlugin) (volumeHost, error) {

	opst := volumeHost{
		kubeClient: kubeClient,
		cloud:      cloud,
	}

	if err := opst.volumePluginMgr.InitPlugins(plugins, nil, opst); err != nil {
		return opst, fmt.Errorf("Could not initialize volume plugins for openstack : %+v", err)
	}

	return opst, nil
}

// Implementing VolumeHost interface
func (openstack *volumeHost) GetPluginDir(pluginName string) string {
	return ""
}

func (openstack *volumeHost) GetVolumeDevicePluginDir(pluginName string) string {
	return ""
}

func (openstack *volumeHost) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return ""
}

func (openstack *volumeHost) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return ""
}

func (openstack *volumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (openstack *volumeHost) GetKubeClient() clientset.Interface {
	return openstack.kubeClient
}

func (openstack *volumeHost) NewWrapperMounter(volName string, spec volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by expand controller's VolumeHost implementation")
}

func (openstack *volumeHost) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by expand controller's VolumeHost implementation")
}

func (openstack *volumeHost) GetCloudProvider() cloudprovider.Interface {
	return openstack.cloud
}

func (openstack *volumeHost) GetMounter(pluginName string) mount.Interface {
	return nil
}

func (openstack *volumeHost) GetExec(pluginName string) mount.Exec {
	return mount.NewOsExec()
}

func (openstack *volumeHost) GetWriter() io.Writer {
	return nil
}

func (openstack *volumeHost) GetHostName() string {
	return ""
}

func (openstack *volumeHost) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP not supported by openstack VolumeHost implementation")
}

func (openstack *volumeHost) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (openstack *volumeHost) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(_, _ string) (*v1.Secret, error) {
		return nil, fmt.Errorf("GetSecret unsupported in openstack")
	}
}

func (openstack *volumeHost) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(_, _ string) (*v1.ConfigMap, error) {
		return nil, fmt.Errorf("GetConfigMap unsupported in openstack")
	}
}

func (openstack *volumeHost) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels unsupported in openstack")
}

func (openstack *volumeHost) GetNodeName() types.NodeName {
	return ""
}
