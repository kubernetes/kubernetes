/*
Copyright 2018 The Kubernetes Authors.

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

package snapshot

import (
	"fmt"
	"net"

	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// Implementing VolumeHost interface
func (ctrl *SnapshotController) GetPluginDir(pluginName string) string {
	return ""
}

func (ctrl *SnapshotController) GetVolumeDevicePluginDir(pluginName string) string {
	return ""
}

func (ctrl *SnapshotController) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return ""
}

func (ctrl *SnapshotController) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return ""
}

func (ctrl *SnapshotController) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (ctrl *SnapshotController) GetKubeClient() clientset.Interface {
	return ctrl.client
}

func (ctrl *SnapshotController) NewWrapperMounter(volName string, spec volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by expand controller's VolumeHost implementation")
}

func (ctrl *SnapshotController) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by expand controller's VolumeHost implementation")
}

func (ctrl *SnapshotController) GetCloudProvider() cloudprovider.Interface {
	return ctrl.cloud
}

func (ctrl *SnapshotController) GetMounter(pluginName string) mount.Interface {
	return nil
}

func (ctrl *SnapshotController) GetExec(pluginName string) mount.Exec {
	return mount.NewOsExec()
}

func (ctrl *SnapshotController) GetWriter() io.Writer {
	return nil
}

func (ctrl *SnapshotController) GetHostName() string {
	return ""
}

func (ctrl *SnapshotController) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP not supported by expand controller's VolumeHost implementation")
}

func (ctrl *SnapshotController) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (ctrl *SnapshotController) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(_, _ string) (*v1.Secret, error) {
		return nil, fmt.Errorf("GetSecret unsupported in Controller")
	}
}

func (ctrl *SnapshotController) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(_, _ string) (*v1.ConfigMap, error) {
		return nil, fmt.Errorf("GetConfigMap unsupported in Controller")
	}
}

func (ctrl *SnapshotController) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels unsupported in Controller")
}

func (ctrl *SnapshotController) GetNodeName() types.NodeName {
	return ""
}

func (ctrl *SnapshotController) GetEventRecorder() record.EventRecorder {
	return ctrl.eventRecorder
}

func (ctrl *SnapshotController) GetPodsDir() string {
	return ""
}

func (ctrl *SnapshotController) GetServiceAccountTokenFunc() func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
	return func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
		return nil, fmt.Errorf("GetServiceAccountToken unsupported in SnapshotController")
	}
}
