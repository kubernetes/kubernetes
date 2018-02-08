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
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

func addSecretsToOptions(options map[string]string, spec *volume.Spec, namespace string, driverName string, host volume.VolumeHost) error {
	secretName, secretNamespace, err := getSecretNameAndNamespace(spec, namespace)
	if err != nil {
		return err
	}

	if len(secretName) == 0 || len(secretNamespace) == 0 {
		return nil
	}

	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot get kube client")
	}

	secrets, err := util.GetSecretForPV(secretNamespace, secretName, driverName, host.GetKubeClient())
	if err != nil {
		err = fmt.Errorf("Couldn't get secret %v/%v err: %v", secretNamespace, secretName, err)
		return err
	}
	for name, data := range secrets {
		options[optionKeySecret+"/"+name] = base64.StdEncoding.EncodeToString([]byte(data))
		glog.V(1).Infof("found flex volume secret info: %s", name)
	}

	return nil
}

var notFlexVolume = fmt.Errorf("not a flex volume")

func getDriver(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		return spec.Volume.FlexVolume.Driver, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil {
		return spec.PersistentVolume.Spec.FlexVolume.Driver, nil
	}
	return "", notFlexVolume
}

func getFSType(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		return spec.Volume.FlexVolume.FSType, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil {
		return spec.PersistentVolume.Spec.FlexVolume.FSType, nil
	}
	return "", notFlexVolume
}

func getSecretNameAndNamespace(spec *volume.Spec, podNamespace string) (string, string, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		if spec.Volume.FlexVolume.SecretRef == nil {
			return "", "", nil
		}
		return spec.Volume.FlexVolume.SecretRef.Name, podNamespace, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil {
		if spec.PersistentVolume.Spec.FlexVolume.SecretRef == nil {
			return "", "", nil
		}
		secretName := spec.PersistentVolume.Spec.FlexVolume.SecretRef.Name
		secretNamespace := spec.PersistentVolume.Spec.FlexVolume.SecretRef.Namespace
		if len(secretNamespace) == 0 {
			secretNamespace = podNamespace
		}
		return secretName, secretNamespace, nil
	}
	return "", "", notFlexVolume
}

func getReadOnly(spec *volume.Spec) (bool, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		return spec.Volume.FlexVolume.ReadOnly, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil {
		// ReadOnly is specified at the PV level
		return spec.ReadOnly, nil
	}
	return false, notFlexVolume
}

func getOptions(spec *volume.Spec) (map[string]string, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		return spec.Volume.FlexVolume.Options, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil {
		return spec.PersistentVolume.Spec.FlexVolume.Options, nil
	}
	return nil, notFlexVolume
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
