/*
Copyright 2015 The Kubernetes Authors.

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

package scaleio

import (
	"fmt"
	"os"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

type sioVolume struct {
	sioMgr     *sioMgr
	plugin     *sioPlugin
	pod        *api.Pod
	podUID     types.UID
	spec       *volume.Spec
	source     *api.ScaleIOVolumeSource
	volName    string
	readOnly   bool
	options    volume.VolumeOptions
	configData map[string]string

	volume.MetricsNil
}

// *******************
// volume.Volume Impl
// *******************
var _ volume.Volume = &sioVolume{}

// GetPath returns the volume directory for the pod
func (v *sioVolume) GetPath() string {
	return v.plugin.host.GetPodVolumeDir(
		v.podUID,
		strings.EscapeQualifiedNameForDisk(sioPluginName),
		v.volName)
}

// *************
// Mounter Impl
// *************
var _ volume.Mounter = &sioVolume{}

// CanMount checks to verify that the volume can be mounted prior to Setup.
// A nil error indicates that the volume is ready for mounitnig.
func (v *sioVolume) CanMount() error {
	return nil
}

func (v *sioVolume) SetUp(fsGroup *int64) error {
	return v.SetUpAt(v.GetPath(), fsGroup)
}

// SetUp bind mounts the disk global mount to the volume path.
func (v *sioVolume) SetUpAt(dir string, fsGroup *int64) error {
	glog.Info("sio: setting up volume")

	options := []string{}
	if v.source.ReadOnly {
		options = append(options, "ro")
	}
	pdPath := getNodeVolumeDir(v.plugin, sioName, v.source.VolumeName)

	// attach the volume and mount
	glog.V(4).Infof("sio: attaching volume %s", v.source.VolumeName)
	devicePath, err := v.sioMgr.AttachVolume(v.source.VolumeName)
	if err != nil {
		glog.Errorf("sio: failed to attach volume:  %v", err)
		return err
	}

	notDevMnt, err := v.plugin.mounter.IsLikelyNotMountPoint(pdPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("sio: IsLikelyNotMountPoint test failed for dir %v", pdPath)
		return err
	}

	if notDevMnt {
		glog.V(4).Infof("sio: mounting device %s -> %s", devicePath, pdPath)

		if err := os.MkdirAll(pdPath, 0750); err != nil {
			glog.Errorf("sio: failed to create dir %#v:  %v", pdPath, err)
			return err
		}
		glog.V(4).Infof("sio: created directory %s", pdPath)

		diskMounter := &mount.SafeFormatAndMount{
			Interface: v.plugin.mounter,
			Runner:    exec.New(),
		}
		err = diskMounter.FormatAndMount(
			devicePath,
			pdPath,
			v.source.FSType,
			options,
		)

		if err != nil {
			os.Remove(pdPath)
			return err
		}
		glog.V(4).Infof(
			"sio: formatted %s:%s [%s,%+v], mounted as %s",
			v.source.VolumeName, devicePath, v.source.FSType, options, pdPath)
	} else {
		glog.Warningf("sio: already mounted: %s", pdPath)
	}

	// make sure we can bind-mount before even continuing
	notMntPoint, err := v.plugin.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.V(4).Infof("sio: IsLikelyNotMountPoint failed: %s", err)
		return err
	}
	if !notMntPoint {
		glog.Warningf("sio: volume %s already mounted at %s", v.source.VolumeName, dir)
		return nil
	}

	// bind-mount for pod
	options = append(options, "bind")
	glog.V(4).Infof("sio: bind-mount %s ->  %s", pdPath, dir)
	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("sio: mkdir failed: %v", err)
		return err
	}
	glog.V(4).Infof("sio: created bind-mout target dir %s", dir)

	if _, err := os.Stat(dir); err != nil {
		glog.Errorf("libStorage Error creating dir %v: %v", dir, err)
	} else {
		glog.V(4).Infof("sio: mount dir created ok %v", dir)
	}

	// bind-mount libstorage mountpoint to k8s dir
	glog.V(4).Infof("sio: bind-mounting %s:%s to %s", v.volName, pdPath, dir)
	err = v.plugin.mounter.Mount(pdPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := v.plugin.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("sio: IsLikelyNotMountPoint failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = v.plugin.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("libStoage: failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := v.plugin.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("sio: IsLikelyNotMountPoint failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("sio: %s is still mounted.  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("sio: bind-mount %s failed: %v", dir, err)
		return err
	}

	glog.Infof("sio: successfully bind-mounted %s:%s as %s",
		v.source.VolumeName, pdPath, dir)
	return nil
}

func (v *sioVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        v.source.ReadOnly,
		Managed:         false,
		SupportsSELinux: true,
	}
}

// **********************
// volume.Unmounter Impl
// *********************
var _ volume.Unmounter = &sioVolume{}

// TearDownAt unmounts the bind mount
func (v *sioVolume) TearDown() error {
	return v.TearDownAt(v.GetPath())
}

// Unmounts the bind mount, and remove the volume only if the libstorage
// resource was the last reference to that volume on the kubelet.
func (v *sioVolume) TearDownAt(dir string) error {
	glog.Infof("sio: tearing down bind-mount to dir %s", dir)
	notMnt, err := v.plugin.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("sio: checking mount point %s failed: %v", dir, err)
		return err
	}

	dirRemoved := false
	if notMnt {
		if err := os.Remove(dir); err != nil && !os.IsNotExist(err) {
			return err
		}
		dirRemoved = true
		glog.V(4).Infof("sio: dir %s removed", dir)
	}

	// Unmount the bind-mount inside this pod
	// only do this if dir is still around and it's a mnt point
	if !dirRemoved {
		if err := v.plugin.mounter.Unmount(dir); err != nil {
			glog.V(2).Infof("sio: error unmounting dir %s: %v ", dir, err)
			return err
		}
		glog.V(4).Infof("sio: dir %s unmounted successfully", dir)

		// check again on dir
		notMnt, mntErr := v.plugin.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil && !os.IsNotExist(mntErr) {
			glog.Errorf("sio: mount point check failed for %s: %v", dir, mntErr)
			return err
		}
		if notMnt {
			if err := os.Remove(dir); err != nil && !os.IsNotExist(err) {
				glog.V(2).Infof("sio: error removing bind-mount dir %s: %v", dir, err)
				return err
			}
			glog.V(4).Infof("sio: removed dir %s", dir)
		}
	}

	//unmount device
	if v.sioMgr != nil {
		pdPathRemoved := false
		pdPath := getNodeVolumeDir(v.plugin, sioName, v.volName)
		glog.V(4).Info("sio: attempting to unmout device diretory %s", pdPath)

		notMnt, err = v.plugin.mounter.IsLikelyNotMountPoint(pdPath)
		if err != nil && !os.IsNotExist(err) {
			glog.Errorf("sio: checking mount point %s failed: %v", dir, err)
			return err
		}

		if notMnt {
			if err := os.Remove(pdPath); err != nil && !os.IsNotExist(err) {
				return err
			}
			pdPathRemoved = true
			glog.V(4).Infof("sio: dir %s removed", pdPath)
		}

		if !pdPathRemoved {
			if err := v.plugin.mounter.Unmount(pdPath); err != nil {
				glog.V(2).Infof("sio: error unmounting dir %s: %v ", dir, err)
				return err
			}
			glog.V(4).Infof("sio: dir %s unmounted successfully", dir)

			// check mount point again
			notMnt, err = v.plugin.mounter.IsLikelyNotMountPoint(pdPath)
			if err != nil && !os.IsNotExist(err) {
				glog.Errorf("sio: checking mount point %s failed: %v", dir, err)
				return err
			}

			if notMnt {
				if err := os.Remove(pdPath); err != nil && !os.IsNotExist(err) {
					return err
				}
				pdPathRemoved = true
				glog.V(4).Infof("sio: dir %s removed", pdPath)
			}
		}

		if pdPathRemoved {
			if err := v.sioMgr.DetachVolume(v.volName); err != nil {
				glog.Errorf("sio: failed detaching volume %s  %v", v.volName, err)
				return err
			}
			glog.V(4).Infof("sio: volume %v detached successfully", v.volName)
		}
	} else {
		glog.Warningf("sio: did not receive lsclient settings, volume %s may not have been detached", v.volName)
	}

	glog.V(4).Infof("sio: teardown successful")
	return nil
}

// ********************
// volume.Deleter Impl
// ********************
var _ volume.Deleter = &sioVolume{}

func (v *sioVolume) Delete() error {
	err := v.sioMgr.DeleteVolume(v.volName)
	if err != nil {
		glog.Errorf("sio: failed to delete volume %s: %v", v.volName, err)
		return err
	}

	glog.V(4).Infof("sio: successfully deleted %s", v.volName)
	return nil
}

// ************************
// volume.Provisioner Impl
// ************************
var _ volume.Provisioner = &sioVolume{}

func (v *sioVolume) Provision() (*api.PersistentVolume, error) {
	glog.V(4).Info("sio: attempting to automatically provision volume")

	// create volume and returns a libStorage Volume value
	name := volume.GenerateVolumeName(v.options.ClusterName, v.options.PVName, 255)
	capacity := v.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	volSizeBytes := capacity.Value()
	volSizeGB := int64(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))

	vol, err := v.sioMgr.CreateVolume(name, volSizeGB)
	if err != nil {
		return nil, err
	}

	sslEnabled, _ := strconv.ParseBool(v.configData[confKey.sslEnabled])
	readOnly, err := strconv.ParseBool(v.configData[confKey.readOnly])
	if err != nil {
		readOnly = true
	}

	// describe created pv
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   v.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "scaleio-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   v.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", volSizeGB),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				ScaleIO: &api.ScaleIOVolumeSource{
					Gateway:          v.configData[confKey.gateway],
					SSLEnabled:       sslEnabled,
					SecretRef:        &api.LocalObjectReference{Name: v.configData[confKey.secretRef]},
					System:           v.configData[confKey.system],
					ProtectionDomain: v.configData[confKey.protectionDomain],
					StoragePool:      v.configData[confKey.storagePool],
					StorageMode:      v.configData[confKey.storageMode],
					SDCRootPath:      v.configData[confKey.sdcRootPath],
					VolumeName:       name,
					FSType:           v.configData[confKey.fsType],
					ReadOnly:         readOnly,
				},
			},
		},
	}
	if len(v.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = v.plugin.GetAccessModes()
	}

	glog.V(4).Infof("sio: dynamically provisioned volume %s for PV: %v", vol.Name, v.options.PVName)
	return pv, nil
}
