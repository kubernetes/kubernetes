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
	"io/ioutil"
	"os"
	"path"

	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
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
	return UnmountMountPoint(mountPath, mounter, false /* extensiveMountPointCheck */)
}

// UnmountMountPoint is a common unmount routine that unmounts the given path and
// deletes the remaining directory if successful.
// if extensiveMountPointCheck is true
// IsNotMountPoint will be called instead of IsLikelyNotMountPoint.
// IsNotMountPoint is more expensive but properly handles bind mounts.
func UnmountMountPoint(mountPath string, mounter mount.Interface, extensiveMountPointCheck bool) error {
	if pathExists, pathErr := PathExists(mountPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", mountPath)
		return nil
	}

	var notMnt bool
	var err error

	if extensiveMountPointCheck {
		notMnt, err = mount.IsNotMountPoint(mounter, mountPath)
	} else {
		notMnt, err = mounter.IsLikelyNotMountPoint(mountPath)
	}

	if err != nil {
		return err
	}

	if notMnt {
		glog.Warningf("Warning: %q is not a mountpoint, deleting", mountPath)
		return os.Remove(mountPath)
	}

	// Unmount the mount path
	glog.V(4).Infof("%q is a mountpoint, unmounting", mountPath)
	if err := mounter.Unmount(mountPath); err != nil {
		return err
	}
	notMnt, mntErr := mounter.IsLikelyNotMountPoint(mountPath)
	if mntErr != nil {
		return err
	}
	if notMnt {
		glog.V(4).Infof("%q is unmounted, deleting the directory", mountPath)
		return os.Remove(mountPath)
	}
	return fmt.Errorf("Failed to unmount path %v", mountPath)
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
func GetSecretForPod(pod *v1.Pod, secretName string, kubeClient clientset.Interface) (map[string]string, error) {
	secret := make(map[string]string)
	if kubeClient == nil {
		return secret, fmt.Errorf("Cannot get kube client")
	}
	secrets, err := kubeClient.Core().Secrets(pod.Namespace).Get(secretName, metav1.GetOptions{})
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
	secrets, err := kubeClient.Core().Secrets(secretNamespace).Get(secretName, metav1.GetOptions{})
	if err != nil {
		return secret, err
	}
	if secrets.Type != v1.SecretType(volumePluginName) {
		return secret, fmt.Errorf("Cannot get secret of type %s", volumePluginName)
	}
	for name, data := range secrets.Data {
		secret[name] = string(data)
	}
	return secret, nil
}

func GetClassForVolume(kubeClient clientset.Interface, pv *v1.PersistentVolume) (*storage.StorageClass, error) {
	if kubeClient == nil {
		return nil, fmt.Errorf("Cannot get kube client")
	}
	className := v1helper.GetPersistentVolumeClass(pv)
	if className == "" {
		return nil, fmt.Errorf("Volume has no storage class")
	}

	class, err := kubeClient.StorageV1().StorageClasses().Get(className, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return class, nil
}

// CheckNodeAffinity looks at the PV node affinity, and checks if the node has the same corresponding labels
// This ensures that we don't mount a volume that doesn't belong to this node
func CheckNodeAffinity(pv *v1.PersistentVolume, nodeLabels map[string]string) error {
	affinity, err := v1helper.GetStorageNodeAffinityFromAnnotation(pv.Annotations)
	if err != nil {
		return fmt.Errorf("Error getting storage node affinity: %v", err)
	}
	if affinity == nil {
		return nil
	}

	if affinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		terms := affinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
		glog.V(10).Infof("Match for RequiredDuringSchedulingIgnoredDuringExecution node selector terms %+v", terms)
		for _, term := range terms {
			selector, err := v1helper.NodeSelectorRequirementsAsSelector(term.MatchExpressions)
			if err != nil {
				return fmt.Errorf("Failed to parse MatchExpressions: %v", err)
			}
			if !selector.Matches(labels.Set(nodeLabels)) {
				return fmt.Errorf("NodeSelectorTerm %+v does not match node labels", term.MatchExpressions)
			}
		}
	}
	return nil
}

// LoadPodFromFile will read, decode, and return a Pod from a file.
func LoadPodFromFile(filePath string) (*v1.Pod, error) {
	if filePath == "" {
		return nil, fmt.Errorf("file path not specified")
	}
	podDef, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file path %s: %+v", filePath, err)
	}
	if len(podDef) == 0 {
		return nil, fmt.Errorf("file was empty: %s", filePath)
	}
	pod := &v1.Pod{}

	codec := api.Codecs.LegacyCodec(api.Registry.GroupOrDie(v1.GroupName).GroupVersion)
	if err := runtime.DecodeInto(codec, podDef, pod); err != nil {
		return nil, fmt.Errorf("failed decoding file: %v", err)
	}
	return pod, nil
}

func ZonesSetToLabelValue(strSet sets.String) string {
	return strings.Join(strSet.UnsortedList(), kubeletapis.LabelMultiZoneDelimiter)
}

// ZonesToSet converts a string containing a comma separated list of zones to set
func ZonesToSet(zonesString string) (sets.String, error) {
	return stringToSet(zonesString, ",")
}

// LabelZonesToSet converts a PV label value from string containing a delimited list of zones to set
func LabelZonesToSet(labelZonesValue string) (sets.String, error) {
	return stringToSet(labelZonesValue, kubeletapis.LabelMultiZoneDelimiter)
}

// StringToSet converts a string containing list separated by specified delimiter to to a set
func stringToSet(str, delimiter string) (sets.String, error) {
	zonesSlice := strings.Split(str, delimiter)
	zonesSet := make(sets.String)
	for _, zone := range zonesSlice {
		trimmedZone := strings.TrimSpace(zone)
		if trimmedZone == "" {
			return make(sets.String), fmt.Errorf(
				"%q separated list (%q) must not contain an empty string",
				delimiter,
				str)
		}
		zonesSet.Insert(trimmedZone)
	}
	return zonesSet, nil
}
