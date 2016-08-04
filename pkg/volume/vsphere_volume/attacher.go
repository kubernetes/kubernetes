package vsphere_volume

import (
	"fmt"
	"os"
	"path"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type vsphereVMDKAttacher struct {
	host           volume.VolumeHost
	vsphereVolumes vsphere.Volumes
}

var _ volume.Attacher = &vsphereVMDKAttacher{}
var _ volume.AttachableVolumePlugin = &vsphereVolumePlugin{}

// Singleton key mutex for keeping attach operations for the same host atomic
var attachMutex = keymutex.NewKeyMutex()

func (plugin *vsphereVolumePlugin) NewAttacher() (volume.Attacher, error) {
	vsphereCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &vsphereVMDKAttacher{
		host:           plugin.host,
		vsphereVolumes: vsphereCloud,
	}, nil
}

// Attaches the volume specified by the given spec to the given host.
// On success, returns the device path where the device was attached on the
// node.
// Callers are responsible for retryinging on failure.
// Callers are responsible for thread safety between concurrent attach and
// detach operations.
func (attacher *vsphereVMDKAttacher) Attach(spec *volume.Spec, hostName string) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volPath := volumeSource.VolumePath
	glog.V(4).Infof("vSphere: Attach disk called for host %s", hostName)

	// Keeps concurrent attach operations to same host atomic
	attachMutex.LockKey(hostName)
	defer attachMutex.UnlockKey(hostName)

	// vsphereCloud.AttachDisk checks if disk is already attached to host and
	// succeeds in that case, so no need to do that separately.
	_, diskUUID, err := attacher.vsphereVolumes.AttachDisk(volPath, hostName)
	if err != nil {
		glog.Errorf("Error attaching volume %q: %+v", volPath, err)
		return "", err
	}

	return path.Join(diskByIDPath, diskSCSIPrefix+diskUUID), nil
}

func (attacher *vsphereVMDKAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volPath := volumeSource.VolumePath

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for VMDK %q: devicePath is empty.", volPath)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking VMDK %q is attached", volPath)
			path, err := verifyDevicePath(devicePath)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				glog.Errorf("Error verifying VMDK (%q) is attached: %v", volPath, err)
			} else if path != "" {
				// A device path has successfully been created for the VMDK
				glog.Infof("Successfully found attached VMDK %q.", volPath)
				return path, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached VMDK %q. Timeout waiting for mount paths to be created.", volPath)
		}
	}
}

func (attacher *vsphereVMDKAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDPath(attacher.host, volumeSource.VolumePath), nil
}

func (attacher *vsphereVMDKAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}

	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		glog.V(4).Infof("formatting spec %v devicePath %v deviceMountPath %v fs %v with options %+v", spec.Name(), devicePath, deviceMountPath, volumeSource.FSType, options)
	}
	return nil
}

type vsphereVMDKDetacher struct {
	mounter        mount.Interface
	vsphereVolumes vsphere.Volumes
}

var _ volume.Detacher = &vsphereVMDKDetacher{}

// Singleton key mutex for keeping attach operations for the same host atomic
var detachMutex = keymutex.NewKeyMutex()

func (plugin *vsphereVolumePlugin) NewDetacher() (volume.Detacher, error) {
	vsphereCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &vsphereVMDKDetacher{
		mounter:        plugin.host.GetMounter(),
		vsphereVolumes: vsphereCloud,
	}, nil
}

// Detach the given device from the given host.
func (detacher *vsphereVMDKDetacher) Detach(deviceMountPath string, hostName string) error {

	volPath := getVolPathfromDeviceMountPath(deviceMountPath)
	attached, err := detacher.vsphereVolumes.DiskIsAttached(volPath, hostName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volPath, hostName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %q.", volPath, hostName)
		return nil
	}

	detachMutex.LockKey(hostName)
	defer detachMutex.UnlockKey(hostName)
	if err := detacher.vsphereVolumes.DetachDisk(volPath, hostName); err != nil {
		glog.Errorf("Error detaching volume %q: %v", volPath, err)
		return err
	}
	return nil
}

func (detacher *vsphereVMDKDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := pathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; Device %v is still attached", devicePath)
		}
	}
}

func (detacher *vsphereVMDKDetacher) UnmountDevice(deviceMountPath string) error {
	volume := path.Base(deviceMountPath)
	if err := unmountPDAndRemoveGlobalPath(deviceMountPath, detacher.mounter); err != nil {
		glog.Errorf("Error unmounting %q: %v", volume, err)
		return err
	}
	return nil
}
