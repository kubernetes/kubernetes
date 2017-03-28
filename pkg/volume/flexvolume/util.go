/*
Copyright 2017 The Kubernetes Authors.

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

package flexvolume

import (
	"encoding/base64"
	"fmt"
	"os"

	"github.com/golang/glog"
	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

func addSecretsToOptions(options map[string]string, spec *volume.Spec, namespace string, driverName string, host volume.VolumeHost) error {
	fv, _ := getVolumeSource(spec)
	if fv.SecretRef == nil {
		return nil
	}

	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot get kube client")
	}

	secrets, err := util.GetSecretForPV(namespace, fv.SecretRef.Name, driverName, host.GetKubeClient())
	if err != nil {
		err = fmt.Errorf("Couldn't get secret %v/%v err: %v", namespace, fv.SecretRef.Name, err)
		return err
	}
	for name, data := range secrets {
		options[optionKeySecret+"/"+name] = base64.StdEncoding.EncodeToString([]byte(data))
		glog.V(1).Infof("found flex volume secret info: %s", name)
	}

	return nil
}

func getVolumeSource(spec *volume.Spec) (volumeSource *api.FlexVolumeSource, readOnly bool) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		volumeSource = spec.Volume.FlexVolume
		readOnly = volumeSource.ReadOnly
	} else if spec.PersistentVolume != nil {
		volumeSource = spec.PersistentVolume.Spec.FlexVolume
		readOnly = spec.ReadOnly
	}
	return
}

func prepareForMount(mounter mount.Interface, deviceMountPath string) (bool, error) {

	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return false, err
			}
			notMnt = true
		} else {
			return false, err
		}
	}

	return !notMnt, nil
}

// Mounts the device at the given path.
// It is expected that prepareForMount has been called before.
func doMount(mounter mount.Interface, devicePath, deviceMountPath, fsType string, options []string) error {
	err := mounter.Mount(devicePath, deviceMountPath, fsType, options)
	if err != nil {
		glog.Errorf("Failed to mount the volume at %s, device: %s, error: %s", deviceMountPath, devicePath, err.Error())
		return err
	}
	return nil
}

func isNotMounted(mounter mount.Interface, deviceMountPath string) (bool, error) {
	notmnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		glog.Errorf("Error checking mount point %s, error: %v", deviceMountPath, err)
		return false, err
	}
	return notmnt, nil
}
