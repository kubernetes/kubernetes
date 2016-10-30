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

package util

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/util/mount"
)

const readyFileName = "ready"

// IsReady checks for the existence of a regular file
// called 'ready' in the given directory and returns
// true if that file exists.
func IsReady(dir string) bool {
	readyFile := path.Join(dir, readyFileName)
	s, err := os.Stat(readyFile)
	if err != nil {
		return false
	}

	if !s.Mode().IsRegular() {
		glog.Errorf("ready-file is not a file: %s", readyFile)
		return false
	}

	return true
}

// SetReady creates a file called 'ready' in the given
// directory.  It logs an error if the file cannot be
// created.
func SetReady(dir string) {
	if err := os.MkdirAll(dir, 0750); err != nil && !os.IsExist(err) {
		glog.Errorf("Can't mkdir %s: %v", dir, err)
		return
	}

	readyFile := path.Join(dir, readyFileName)
	file, err := os.Create(readyFile)
	if err != nil {
		glog.Errorf("Can't touch %s: %v", readyFile, err)
		return
	}
	file.Close()
}

// UnmountPath is a common unmount routine that unmounts the given path and
// deletes the remaining directory if successful.
func UnmountPath(mountPath string, mounter mount.Interface) error {
	if pathExists, pathErr := PathExists(mountPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", mountPath)
		return nil
	}

	notMnt, err := mounter.IsLikelyNotMountPoint(mountPath)
	if err != nil {
		return err
	}
	if notMnt {
		glog.Warningf("Warning: %q is not a mountpoint, deleting", mountPath)
		return os.Remove(mountPath)
	}

	// Unmount the mount path
	if err := mounter.Unmount(mountPath); err != nil {
		return err
	}
	notMnt, mntErr := mounter.IsLikelyNotMountPoint(mountPath)
	if mntErr != nil {
		return err
	}
	if notMnt {
		glog.V(4).Info("%q is unmounted, deleting the directory", mountPath)
		return os.Remove(mountPath)
	}
	return nil
}

// PathExists returns true if the specified path exists.
func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// GetSecretForPod locates secret by name in the pod's namespace and returns secret map
func GetSecretForPod(pod *api.Pod, secretName string, kubeClient clientset.Interface) (map[string]string, error) {
	secret := make(map[string]string)
	if kubeClient == nil {
		return secret, fmt.Errorf("Cannot get kube client")
	}
	secrets, err := kubeClient.Core().Secrets(pod.Namespace).Get(secretName)
	if err != nil {
		return secret, err
	}
	for name, data := range secrets.Data {
		secret[name] = string(data)
	}
	return secret, nil
}

// GetSecretForPV locates secret by name and namespace, verifies the secret type, and returns secret map
func GetSecretForPV(secretNamespace, secretName, volumePluginName string, kubeClient clientset.Interface) (map[string]string, error) {
	secret := make(map[string]string)
	if kubeClient == nil {
		return secret, fmt.Errorf("Cannot get kube client")
	}
	secrets, err := kubeClient.Core().Secrets(secretNamespace).Get(secretName)
	if err != nil {
		return secret, err
	}
	if secrets.Type != api.SecretType(volumePluginName) {
		return secret, fmt.Errorf("Cannot get secret of type %s", volumePluginName)
	}
	for name, data := range secrets.Data {
		secret[name] = string(data)
	}
	return secret, nil
}

// AddVolumeAnnotations adds a golang Map as annotation to a PersistentVolume
func AddVolumeAnnotations(pv *api.PersistentVolume, annotations map[string]string) {
	if pv.Annotations == nil {
		pv.Annotations = map[string]string{}
	}

	for k, v := range annotations {
		pv.Annotations[k] = v
	}
}

// ParseVolumeAnnotations reads the defined annoations from a PersistentVolume
func ParseVolumeAnnotations(pv *api.PersistentVolume, parseAnnotations []string) (map[string]string, error) {
	result := map[string]string{}

	if pv.Annotations == nil {
		return result, fmt.Errorf("cannot parse volume annotations: no annotations found")
	}

	for _, annotation := range parseAnnotations {
		if val, ok := pv.Annotations[annotation]; ok {
			result[annotation] = val
		} else {
			return result, fmt.Errorf("cannot parse volume annotations: annotation %s not found", annotation)
		}
	}

	return result, nil
}

func GetClassForVolume(kubeClient clientset.Interface, pv *api.PersistentVolume) (*storage.StorageClass, error) {
	// TODO: replace with a real attribute after beta
	className, found := pv.Annotations["volume.beta.kubernetes.io/storage-class"]
	if !found {
		return nil, fmt.Errorf("Volume has no class annotation")
	}

	class, err := kubeClient.Storage().StorageClasses().Get(className)
	if err != nil {
		return nil, err
	}
	return class, nil
}
