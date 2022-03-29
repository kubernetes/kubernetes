//go:build windows
// +build windows

package kubelet

import (
	"k8s.io/kubernetes/pkg/volume/util"
)

func (kvh *kubeletVolumeHost) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return util.GetWindowsPath(dir)
}