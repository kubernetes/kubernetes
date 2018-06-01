package snapshot

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
