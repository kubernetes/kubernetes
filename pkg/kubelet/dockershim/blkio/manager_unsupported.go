// +build !linux

package blkio

import (
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func UpdateBlkio(containerId string, docker libdocker.Interface) (err error) {
	return nil
}
