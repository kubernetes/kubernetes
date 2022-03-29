//go:build !windows
// +build !windows

package kubelet

import (
	"k8s.io/apimachinery/pkg/types"
)

func getPodHostsPath(kvh *kubeletVolumeHost, podUID types.UID, pluginName string, volumeName string) string {
	return kvh.kubelet.getPodVolumeDir(podUID, pluginName, volumeName)
}