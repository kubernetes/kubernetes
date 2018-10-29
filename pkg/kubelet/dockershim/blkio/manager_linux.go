// +build linux

package blkio

import (
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func UpdateBlkio(containerId string, docker libdocker.Interface) (err error) {
	info, err := docker.InspectContainer(containerId)
	if err != nil {
		return fmt.Errorf("failed to inspect container %q: %v", containerId, err)
	}
	containerType, ok := info.Config.Labels[containerTypeLabelKey]
	if !ok {
		return fmt.Errorf("the container: %s is neither a regular container is a sandbox.", containerId)
	}
	if containerType != containerTypeLabelContainer {
		// Limit regular containers only
		return nil
	}
	sandboxID, ok := info.Config.Labels[sandboxIDLabelKey]
	if !ok {
		return fmt.Errorf("the container: %s is not a regular container,the %s could not be found", containerId, sandboxIDLabelKey)
	}

	sandboxInfo, err := docker.InspectContainer(sandboxID)
	if err != nil {
		return fmt.Errorf("failed to inspect sandbox %q: %v", sandboxID, err)
	}
	blkiolable, ok := sandboxInfo.Config.Labels[BlkioKey]
	if !ok {
		glog.V(4).Infof("the sandbox is not set %s, sandboxID:%s, containerId:%s", BlkioKey, sandboxID, containerId)
		return nil
	}
	blkio := Blkio{}
	err = json.Unmarshal([]byte(blkiolable), &blkio)
	if err != nil {
		return fmt.Errorf("failed to unmarshal blkio config,%s, sandboxID:%s, containerId:%s", err.Error(), sandboxID, containerId)
	}

	driverName := info.GraphDriver.Name
	if driverName != GraphDriverName {
		glog.V(4).Infof("the container driver is %v, sandboxID:%s, containerId:%s", driverName, sandboxID, containerId)
		return nil
	}
	deviceName, ok := info.GraphDriver.Data[GraphDriverDeviceNameKey]
	if !ok {
		glog.V(4).Infof("the container GraphDriverDeviceName not found. sandboxID:%s, containerId:%s", sandboxID, containerId)
		return nil
	}
	containerRoot := filepath.Join(DeviceRoot, deviceName)
	blkioResource, err := getBlkioResource(&blkio, containerRoot)
	if err != nil {
		return fmt.Errorf("getBlkioResource failed. sandboxID:%s, containerId:%s, %v", sandboxID, containerId, err.Error())
	}

	cpath, err := getBlkioCgroupPath(BlkioSubsystemName, info.HostConfig.CgroupParent, containerId)
	if err != nil {
		return fmt.Errorf("getBlkioCgroupPath failed. sandboxID:%s, containerId:%s, %v", sandboxID, containerId, err.Error())
	}
	cg := &configs.Cgroup{
		Path:      cpath,
		Resources: &blkioResource,
	}
	err = blkioSubsystem.Set(cpath, cg)
	if err != nil {
		return fmt.Errorf("blkioSubsystem.Set failed. sandboxID:%s, containerId:%s, %v", sandboxID, containerId, err.Error())
	}
	glog.V(4).Infof("set Blkio cgroup success. sandboxID:%s, containerId:%s, cgroup path:%v, cgroup:%+v", sandboxID, containerId, cpath, cgroupToString(cg))
	return nil
}
