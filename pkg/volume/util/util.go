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
	"path/filepath"
	"strings"
	"syscall"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/util/mount"
)

const (
	readyFileName = "ready"
	losetupPath   = "losetup"

	ErrDeviceNotFound     = "device not found"
	ErrDeviceNotSupported = "device not supported"
	ErrNotAvailable       = "not available"
)

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
	pathExists, pathErr := PathExists(mountPath)
	if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", mountPath)
		return nil
	}
	corruptedMnt := isCorruptedMnt(pathErr)
	if pathErr != nil && !corruptedMnt {
		return fmt.Errorf("Error checking path: %v", pathErr)
	}
	return doUnmountMountPoint(mountPath, mounter, extensiveMountPointCheck, corruptedMnt)
}

// doUnmountMountPoint is a common unmount routine that unmounts the given path and
// deletes the remaining directory if successful.
// if extensiveMountPointCheck is true
// IsNotMountPoint will be called instead of IsLikelyNotMountPoint.
// IsNotMountPoint is more expensive but properly handles bind mounts.
// if corruptedMnt is true, it means that the mountPath is a corrupted mountpoint, Take it as an argument for convenience of testing
func doUnmountMountPoint(mountPath string, mounter mount.Interface, extensiveMountPointCheck bool, corruptedMnt bool) error {
	if !corruptedMnt {
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
	}

	// Unmount the mount path
	glog.V(4).Infof("%q is a mountpoint, unmounting", mountPath)
	if err := mounter.Unmount(mountPath); err != nil {
		return err
	}
	notMnt, mntErr := mounter.IsLikelyNotMountPoint(mountPath)
	if mntErr != nil {
		return mntErr
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
	} else if isCorruptedMnt(err) {
		return true, err
	} else {
		return false, err
	}
}

// isCorruptedMnt return true if err is about corrupted mount point
func isCorruptedMnt(err error) bool {
	if err == nil {
		return false
	}
	var underlyingError error
	switch pe := err.(type) {
	case nil:
		return false
	case *os.PathError:
		underlyingError = pe.Err
	case *os.LinkError:
		underlyingError = pe.Err
	case *os.SyscallError:
		underlyingError = pe.Err
	}
	return underlyingError == syscall.ENOTCONN || underlyingError == syscall.ESTALE
}

// GetSecretForPod locates secret by name in the pod's namespace and returns secret map
func GetSecretForPod(pod *v1.Pod, secretName string, kubeClient clientset.Interface) (map[string]string, error) {
	secret := make(map[string]string)
	if kubeClient == nil {
		return secret, fmt.Errorf("Cannot get kube client")
	}
	secrets, err := kubeClient.CoreV1().Secrets(pod.Namespace).Get(secretName, metav1.GetOptions{})
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
	secrets, err := kubeClient.CoreV1().Secrets(secretNamespace).Get(secretName, metav1.GetOptions{})
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

	codec := legacyscheme.Codecs.UniversalDecoder()
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

// BlockVolumePathHandler defines a set of operations for handling block volume-related operations
type BlockVolumePathHandler interface {
	// MapDevice creates a symbolic link to block device under specified map path
	MapDevice(devicePath string, mapPath string, linkName string) error
	// UnmapDevice removes a symbolic link to block device under specified map path
	UnmapDevice(mapPath string, linkName string) error
	// RemovePath removes a file or directory on specified map path
	RemoveMapPath(mapPath string) error
	// IsSymlinkExist retruns true if specified symbolic link exists
	IsSymlinkExist(mapPath string) (bool, error)
	// GetDeviceSymlinkRefs searches symbolic links under global map path
	GetDeviceSymlinkRefs(devPath string, mapPath string) ([]string, error)
	// FindGlobalMapPathUUIDFromPod finds {pod uuid} symbolic link under globalMapPath
	// corresponding to map path symlink, and then return global map path with pod uuid.
	FindGlobalMapPathUUIDFromPod(pluginDir, mapPath string, podUID types.UID) (string, error)
	// AttachFileDevice takes a path to a regular file and makes it available as an
	// attached block device.
	AttachFileDevice(path string) (string, error)
	// GetLoopDevice returns the full path to the loop device associated with the given path.
	GetLoopDevice(path string) (string, error)
	// RemoveLoopDevice removes specified loopback device
	RemoveLoopDevice(device string) error
}

// NewBlockVolumePathHandler returns a new instance of BlockVolumeHandler.
func NewBlockVolumePathHandler() BlockVolumePathHandler {
	var volumePathHandler VolumePathHandler
	return volumePathHandler
}

// VolumePathHandler is path related operation handlers for block volume
type VolumePathHandler struct {
}

// MapDevice creates a symbolic link to block device under specified map path
func (v VolumePathHandler) MapDevice(devicePath string, mapPath string, linkName string) error {
	// Example of global map path:
	//   globalMapPath/linkName: plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/{podUid}
	//   linkName: {podUid}
	//
	// Example of pod device map path:
	//   podDeviceMapPath/linkName: pods/{podUid}/{DefaultKubeletVolumeDevicesDirName}/{escapeQualifiedPluginName}/{volumeName}
	//   linkName: {volumeName}
	if len(devicePath) == 0 {
		return fmt.Errorf("Failed to map device to map path. devicePath is empty")
	}
	if len(mapPath) == 0 {
		return fmt.Errorf("Failed to map device to map path. mapPath is empty")
	}
	if !filepath.IsAbs(mapPath) {
		return fmt.Errorf("The map path should be absolute: map path: %s", mapPath)
	}
	glog.V(5).Infof("MapDevice: devicePath %s", devicePath)
	glog.V(5).Infof("MapDevice: mapPath %s", mapPath)
	glog.V(5).Infof("MapDevice: linkName %s", linkName)

	// Check and create mapPath
	_, err := os.Stat(mapPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate map path: %s", mapPath)
		return err
	}
	if err = os.MkdirAll(mapPath, 0750); err != nil {
		return fmt.Errorf("Failed to mkdir %s, error %v", mapPath, err)
	}
	// Remove old symbolic link(or file) then create new one.
	// This should be done because current symbolic link is
	// stale accross node reboot.
	linkPath := path.Join(mapPath, string(linkName))
	if err = os.Remove(linkPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	err = os.Symlink(devicePath, linkPath)
	return err
}

// UnmapDevice removes a symbolic link associated to block device under specified map path
func (v VolumePathHandler) UnmapDevice(mapPath string, linkName string) error {
	if len(mapPath) == 0 {
		return fmt.Errorf("Failed to unmap device from map path. mapPath is empty")
	}
	glog.V(5).Infof("UnmapDevice: mapPath %s", mapPath)
	glog.V(5).Infof("UnmapDevice: linkName %s", linkName)

	// Check symbolic link exists
	linkPath := path.Join(mapPath, string(linkName))
	if islinkExist, checkErr := v.IsSymlinkExist(linkPath); checkErr != nil {
		return checkErr
	} else if !islinkExist {
		glog.Warningf("Warning: Unmap skipped because symlink does not exist on the path: %v", linkPath)
		return nil
	}
	err := os.Remove(linkPath)
	return err
}

// RemoveMapPath removes a file or directory on specified map path
func (v VolumePathHandler) RemoveMapPath(mapPath string) error {
	if len(mapPath) == 0 {
		return fmt.Errorf("Failed to remove map path. mapPath is empty")
	}
	glog.V(5).Infof("RemoveMapPath: mapPath %s", mapPath)
	err := os.RemoveAll(mapPath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// IsSymlinkExist returns true if specified file exists and the type is symbolik link.
// If file doesn't exist, or file exists but not symbolick link, return false with no error.
// On other cases, return false with error from Lstat().
func (v VolumePathHandler) IsSymlinkExist(mapPath string) (bool, error) {
	fi, err := os.Lstat(mapPath)
	if err == nil {
		// If file exits and it's symbolick link, return true and no error
		if fi.Mode()&os.ModeSymlink == os.ModeSymlink {
			return true, nil
		}
		// If file exits but it's not symbolick link, return fale and no error
		return false, nil
	}
	// If file doesn't exist, return false and no error
	if os.IsNotExist(err) {
		return false, nil
	}
	// Return error from Lstat()
	return false, err
}

// GetDeviceSymlinkRefs searches symbolic links under global map path
func (v VolumePathHandler) GetDeviceSymlinkRefs(devPath string, mapPath string) ([]string, error) {
	var refs []string
	files, err := ioutil.ReadDir(mapPath)
	if err != nil {
		return nil, fmt.Errorf("Directory cannot read %v", err)
	}
	for _, file := range files {
		if file.Mode()&os.ModeSymlink != os.ModeSymlink {
			continue
		}
		filename := file.Name()
		filepath, err := os.Readlink(path.Join(mapPath, filename))
		if err != nil {
			return nil, fmt.Errorf("Symbolic link cannot be retrieved %v", err)
		}
		glog.V(5).Infof("GetDeviceSymlinkRefs: filepath: %v, devPath: %v", filepath, devPath)
		if filepath == devPath {
			refs = append(refs, path.Join(mapPath, filename))
		}
	}
	glog.V(5).Infof("GetDeviceSymlinkRefs: refs %v", refs)
	return refs, nil
}

// FindGlobalMapPathUUIDFromPod finds {pod uuid} symbolic link under globalMapPath
// corresponding to map path symlink, and then return global map path with pod uuid.
// ex. mapPath symlink: pods/{podUid}}/{DefaultKubeletVolumeDevicesDirName}/{escapeQualifiedPluginName}/{volumeName} -> /dev/sdX
//     globalMapPath/{pod uuid}: plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/{pod uuid} -> /dev/sdX
func (v VolumePathHandler) FindGlobalMapPathUUIDFromPod(pluginDir, mapPath string, podUID types.UID) (string, error) {
	var globalMapPathUUID string
	// Find symbolic link named pod uuid under plugin dir
	err := filepath.Walk(pluginDir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if (fi.Mode()&os.ModeSymlink == os.ModeSymlink) && (fi.Name() == string(podUID)) {
			glog.V(5).Infof("FindGlobalMapPathFromPod: path %s, mapPath %s", path, mapPath)
			if res, err := compareSymlinks(path, mapPath); err == nil && res {
				globalMapPathUUID = path
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	glog.V(5).Infof("FindGlobalMapPathFromPod: globalMapPathUUID %s", globalMapPathUUID)
	// Return path contains global map path + {pod uuid}
	return globalMapPathUUID, nil
}

func compareSymlinks(global, pod string) (bool, error) {
	devGlobal, err := os.Readlink(global)
	if err != nil {
		return false, err
	}
	devPod, err := os.Readlink(pod)
	if err != nil {
		return false, err
	}
	glog.V(5).Infof("CompareSymlinks: devGloBal %s, devPod %s", devGlobal, devPod)
	if devGlobal == devPod {
		return true, nil
	}
	return false, nil
}
