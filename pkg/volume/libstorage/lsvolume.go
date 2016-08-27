package libstorage

import (
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type lsVolume struct {
	volName   string
	readOnly  bool
	fsType    string
	mountPath string
	plugin    *lsPlugin
	options   volume.VolumeOptions

	podUID  types.UID
	mounter mount.Interface
	k8mtx   keymutex.KeyMutex

	volume.MetricsNil
}
