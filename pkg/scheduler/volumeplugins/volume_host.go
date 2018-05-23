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

// Package volumeplugins defines a dummy VolumeHost interface to initialize the
// Volume Plugin manager
package volumeplugins

import (
	"fmt"
	"net"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// SchedulerVolumeHost implements filler VolumeHost interface so as
// it can be used for initializing volume plugins.
type SchedulerVolumeHost struct{}

var _ volume.VolumeHost = &SchedulerVolumeHost{}

func (svh *SchedulerVolumeHost) GetPluginDir(podUID string) string {
	return ""
}

func (svh *SchedulerVolumeHost) GetVolumeDevicePluginDir(podUID string) string {
	return ""
}

func (svh *SchedulerVolumeHost) GetPodsDir() string {
	return ""
}

func (svh *SchedulerVolumeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (svh *SchedulerVolumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (svh *SchedulerVolumeHost) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return ""
}

func (svh *SchedulerVolumeHost) GetKubeClient() clientset.Interface {
	return nil
}

func (svh *SchedulerVolumeHost) NewWrapperMounter(volName string, spec volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (svh *SchedulerVolumeHost) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (svh *SchedulerVolumeHost) GetCloudProvider() cloudprovider.Interface {
	return nil
}

func (svh *SchedulerVolumeHost) GetMounter(pluginName string) mount.Interface {
	return nil
}

func (svh *SchedulerVolumeHost) GetWriter() io.Writer {
	return nil
}

func (svh *SchedulerVolumeHost) GetHostName() string {
	return ""
}

func (svh *SchedulerVolumeHost) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP() not supported by Attach/Detach controller's VolumeHost implementation")
}

func (svh *SchedulerVolumeHost) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (svh *SchedulerVolumeHost) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(_, _ string) (*v1.Secret, error) {
		return nil, fmt.Errorf("GetSecret unsupported in SchedulerVolumeHost")
	}
}

func (svh *SchedulerVolumeHost) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(_, _ string) (*v1.ConfigMap, error) {
		return nil, fmt.Errorf("GetConfigMap unsupported in SchedulerVolumeHost")
	}
}

func (svh *SchedulerVolumeHost) GetExec(pluginName string) mount.Exec {
	return nil
}

func (svh *SchedulerVolumeHost) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels is unsupported in SchedulerVolumeHost")
}

func (svh *SchedulerVolumeHost) GetNodeName() types.NodeName {
	return types.NodeName("")
}

func (svh *SchedulerVolumeHost) GetEventRecorder() record.EventRecorder {
	return nil
}
