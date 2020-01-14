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
	"path/filepath"
	"reflect"
	"runtime"
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	utypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
	utilstrings "k8s.io/utils/strings"
)

const (
	readyFileName = "ready"

	// ControllerManagedAttachAnnotation is the key of the annotation on Node
	// objects that indicates attach/detach operations for the node should be
	// managed by the attach/detach controller
	ControllerManagedAttachAnnotation string = "volumes.kubernetes.io/controller-managed-attach-detach"

	// KeepTerminatedPodVolumesAnnotation is the key of the annotation on Node
	// that decides if pod volumes are unmounted when pod is terminated
	KeepTerminatedPodVolumesAnnotation string = "volumes.kubernetes.io/keep-terminated-pod-volumes"

	// MountsInGlobalPDPath is name of the directory appended to a volume plugin
	// name to create the place for volume mounts in the global PD path.
	MountsInGlobalPDPath = "mounts"

	// VolumeGidAnnotationKey is the of the annotation on the PersistentVolume
	// object that specifies a supplemental GID.
	VolumeGidAnnotationKey = "pv.beta.kubernetes.io/gid"

	// VolumeDynamicallyCreatedByKey is the key of the annotation on PersistentVolume
	// object created dynamically
	VolumeDynamicallyCreatedByKey = "kubernetes.io/createdby"
)

// IsReady checks for the existence of a regular file
// called 'ready' in the given directory and returns
// true if that file exists.
func IsReady(dir string) bool {
	readyFile := filepath.Join(dir, readyFileName)
	s, err := os.Stat(readyFile)
	if err != nil {
		return false
	}

	if !s.Mode().IsRegular() {
		klog.Errorf("ready-file is not a file: %s", readyFile)
		return false
	}

	return true
}

// SetReady creates a file called 'ready' in the given
// directory.  It logs an error if the file cannot be
// created.
func SetReady(dir string) {
	if err := os.MkdirAll(dir, 0750); err != nil && !os.IsExist(err) {
		klog.Errorf("Can't mkdir %s: %v", dir, err)
		return
	}

	readyFile := filepath.Join(dir, readyFileName)
	file, err := os.Create(readyFile)
	if err != nil {
		klog.Errorf("Can't touch %s: %v", readyFile, err)
		return
	}
	file.Close()
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

// GetClassForVolume locates storage class by persistent volume
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
	return checkVolumeNodeAffinity(pv, nodeLabels)
}

func checkVolumeNodeAffinity(pv *v1.PersistentVolume, nodeLabels map[string]string) error {
	if pv.Spec.NodeAffinity == nil {
		return nil
	}

	if pv.Spec.NodeAffinity.Required != nil {
		terms := pv.Spec.NodeAffinity.Required.NodeSelectorTerms
		klog.V(10).Infof("Match for Required node selector terms %+v", terms)
		if !v1helper.MatchNodeSelectorTerms(terms, labels.Set(nodeLabels), nil) {
			return fmt.Errorf("No matching NodeSelectorTerms")
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
	if err := apiruntime.DecodeInto(codec, podDef, pod); err != nil {
		return nil, fmt.Errorf("failed decoding file: %v", err)
	}
	return pod, nil
}

// CalculateTimeoutForVolume calculates time for a Recycler pod to complete a
// recycle operation. The calculation and return value is either the
// minimumTimeout or the timeoutIncrement per Gi of storage size, whichever is
// greater.
func CalculateTimeoutForVolume(minimumTimeout, timeoutIncrement int, pv *v1.PersistentVolume) int64 {
	giQty := resource.MustParse("1Gi")
	pvQty := pv.Spec.Capacity[v1.ResourceStorage]
	giSize := giQty.Value()
	pvSize := pvQty.Value()
	timeout := (pvSize / giSize) * int64(timeoutIncrement)
	if timeout < int64(minimumTimeout) {
		return int64(minimumTimeout)
	}
	return timeout
}

// GenerateVolumeName returns a PV name with clusterName prefix. The function
// should be used to generate a name of GCE PD or Cinder volume. It basically
// adds "<clusterName>-dynamic-" before the PV name, making sure the resulting
// string fits given length and cuts "dynamic" if not.
func GenerateVolumeName(clusterName, pvName string, maxLength int) string {
	prefix := clusterName + "-dynamic"
	pvLen := len(pvName)

	// cut the "<clusterName>-dynamic" to fit full pvName into maxLength
	// +1 for the '-' dash
	if pvLen+1+len(prefix) > maxLength {
		prefix = prefix[:maxLength-pvLen-1]
	}
	return prefix + "-" + pvName
}

// GetPath checks if the path from the mounter is empty.
func GetPath(mounter volume.Mounter) (string, error) {
	path := mounter.GetPath()
	if path == "" {
		return "", fmt.Errorf("Path is empty %s", reflect.TypeOf(mounter).String())
	}
	return path, nil
}

// UnmountViaEmptyDir delegates the tear down operation for secret, configmap, git_repo and downwardapi
// to empty_dir
func UnmountViaEmptyDir(dir string, host volume.VolumeHost, volName string, volSpec volume.Spec, podUID utypes.UID) error {
	klog.V(3).Infof("Tearing down volume %v for pod %v at %v", volName, podUID, dir)

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := host.NewWrapperUnmounter(volName, volSpec, podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

// MountOptionFromSpec extracts and joins mount options from volume spec with supplied options
func MountOptionFromSpec(spec *volume.Spec, options ...string) []string {
	pv := spec.PersistentVolume

	if pv != nil {
		// Use beta annotation first
		if mo, ok := pv.Annotations[v1.MountOptionAnnotation]; ok {
			moList := strings.Split(mo, ",")
			return JoinMountOptions(moList, options)
		}

		if len(pv.Spec.MountOptions) > 0 {
			return JoinMountOptions(pv.Spec.MountOptions, options)
		}
	}

	return options
}

// JoinMountOptions joins mount options eliminating duplicates
func JoinMountOptions(userOptions []string, systemOptions []string) []string {
	allMountOptions := sets.NewString()

	for _, mountOption := range userOptions {
		if len(mountOption) > 0 {
			allMountOptions.Insert(mountOption)
		}
	}

	for _, mountOption := range systemOptions {
		allMountOptions.Insert(mountOption)
	}
	return allMountOptions.List()
}

// AccessModesContains returns whether the requested mode is contained by modes
func AccessModesContains(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// AccessModesContainedInAll returns whether all of the requested modes are contained by modes
func AccessModesContainedInAll(indexedModes []v1.PersistentVolumeAccessMode, requestedModes []v1.PersistentVolumeAccessMode) bool {
	for _, mode := range requestedModes {
		if !AccessModesContains(indexedModes, mode) {
			return false
		}
	}
	return true
}

// GetWindowsPath get a windows path
func GetWindowsPath(path string) string {
	windowsPath := strings.Replace(path, "/", "\\", -1)
	if strings.HasPrefix(windowsPath, "\\") {
		windowsPath = "c:" + windowsPath
	}
	return windowsPath
}

// GetUniquePodName returns a unique identifier to reference a pod by
func GetUniquePodName(pod *v1.Pod) types.UniquePodName {
	return types.UniquePodName(pod.UID)
}

// GetUniqueVolumeName returns a unique name representing the volume/plugin.
// Caller should ensure that volumeName is a name/ID uniquely identifying the
// actual backing device, directory, path, etc. for a particular volume.
// The returned name can be used to uniquely reference the volume, for example,
// to prevent operations (attach/detach or mount/unmount) from being triggered
// on the same volume.
func GetUniqueVolumeName(pluginName, volumeName string) v1.UniqueVolumeName {
	return v1.UniqueVolumeName(fmt.Sprintf("%s/%s", pluginName, volumeName))
}

// GetUniqueVolumeNameFromSpecWithPod returns a unique volume name with pod
// name included. This is useful to generate different names for different pods
// on same volume.
func GetUniqueVolumeNameFromSpecWithPod(
	podName types.UniquePodName, volumePlugin volume.VolumePlugin, volumeSpec *volume.Spec) v1.UniqueVolumeName {
	return v1.UniqueVolumeName(
		fmt.Sprintf("%s/%v-%s", volumePlugin.GetPluginName(), podName, volumeSpec.Name()))
}

// GetUniqueVolumeNameFromSpec uses the given VolumePlugin to generate a unique
// name representing the volume defined in the specified volume spec.
// This returned name can be used to uniquely reference the actual backing
// device, directory, path, etc. referenced by the given volumeSpec.
// If the given plugin does not support the volume spec, this returns an error.
func GetUniqueVolumeNameFromSpec(
	volumePlugin volume.VolumePlugin,
	volumeSpec *volume.Spec) (v1.UniqueVolumeName, error) {
	if volumePlugin == nil {
		return "", fmt.Errorf(
			"volumePlugin should not be nil. volumeSpec.Name=%q",
			volumeSpec.Name())
	}

	volumeName, err := volumePlugin.GetVolumeName(volumeSpec)
	if err != nil || volumeName == "" {
		return "", fmt.Errorf(
			"failed to GetVolumeName from volumePlugin for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	return GetUniqueVolumeName(
			volumePlugin.GetPluginName(),
			volumeName),
		nil
}

// IsPodTerminated checks if pod is terminated
func IsPodTerminated(pod *v1.Pod, podStatus v1.PodStatus) bool {
	return podStatus.Phase == v1.PodFailed || podStatus.Phase == v1.PodSucceeded || (pod.DeletionTimestamp != nil && notRunning(podStatus.ContainerStatuses))
}

// notRunning returns true if every status is terminated or waiting, or the status list
// is empty.
func notRunning(statuses []v1.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil && status.State.Waiting == nil {
			return false
		}
	}
	return true
}

// SplitUniqueName splits the unique name to plugin name and volume name strings. It expects the uniqueName to follow
// the format plugin_name/volume_name and the plugin name must be namespaced as described by the plugin interface,
// i.e. namespace/plugin containing exactly one '/'. This means the unique name will always be in the form of
// plugin_namespace/plugin/volume_name, see k8s.io/kubernetes/pkg/volume/plugins.go VolumePlugin interface
// description and pkg/volume/util/volumehelper/volumehelper.go GetUniqueVolumeNameFromSpec that constructs
// the unique volume names.
func SplitUniqueName(uniqueName v1.UniqueVolumeName) (string, string, error) {
	components := strings.SplitN(string(uniqueName), "/", 3)
	if len(components) != 3 {
		return "", "", fmt.Errorf("cannot split volume unique name %s to plugin/volume components", uniqueName)
	}
	pluginName := fmt.Sprintf("%s/%s", components[0], components[1])
	return pluginName, components[2], nil
}

// NewSafeFormatAndMountFromHost creates a new SafeFormatAndMount with Mounter
// and Exec taken from given VolumeHost.
func NewSafeFormatAndMountFromHost(pluginName string, host volume.VolumeHost) *mount.SafeFormatAndMount {
	mounter := host.GetMounter(pluginName)
	exec := host.GetExec(pluginName)
	return &mount.SafeFormatAndMount{Interface: mounter, Exec: exec}
}

// GetVolumeMode retrieves VolumeMode from pv.
// If the volume doesn't have PersistentVolume, it's an inline volume,
// should return volumeMode as filesystem to keep existing behavior.
func GetVolumeMode(volumeSpec *volume.Spec) (v1.PersistentVolumeMode, error) {
	if volumeSpec == nil || volumeSpec.PersistentVolume == nil {
		return v1.PersistentVolumeFilesystem, nil
	}
	if volumeSpec.PersistentVolume.Spec.VolumeMode != nil {
		return *volumeSpec.PersistentVolume.Spec.VolumeMode, nil
	}
	return "", fmt.Errorf("cannot get volumeMode for volume: %v", volumeSpec.Name())
}

// GetPersistentVolumeClaimVolumeMode retrieves VolumeMode from pvc.
func GetPersistentVolumeClaimVolumeMode(claim *v1.PersistentVolumeClaim) (v1.PersistentVolumeMode, error) {
	if claim.Spec.VolumeMode != nil {
		return *claim.Spec.VolumeMode, nil
	}
	return "", fmt.Errorf("cannot get volumeMode from pvc: %v", claim.Name)
}

// GetPersistentVolumeClaimQualifiedName returns a qualified name for pvc.
func GetPersistentVolumeClaimQualifiedName(claim *v1.PersistentVolumeClaim) string {
	return utilstrings.JoinQualifiedName(claim.GetNamespace(), claim.GetName())
}

// CheckVolumeModeFilesystem checks VolumeMode.
// If the mode is Filesystem, return true otherwise return false.
func CheckVolumeModeFilesystem(volumeSpec *volume.Spec) (bool, error) {
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		volumeMode, err := GetVolumeMode(volumeSpec)
		if err != nil {
			return true, err
		}
		if volumeMode == v1.PersistentVolumeBlock {
			return false, nil
		}
	}
	return true, nil
}

// CheckPersistentVolumeClaimModeBlock checks VolumeMode.
// If the mode is Block, return true otherwise return false.
func CheckPersistentVolumeClaimModeBlock(pvc *v1.PersistentVolumeClaim) bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) && pvc.Spec.VolumeMode != nil && *pvc.Spec.VolumeMode == v1.PersistentVolumeBlock
}

// IsWindowsUNCPath checks if path is prefixed with \\
// This can be used to skip any processing of paths
// that point to SMB shares, local named pipes and local UNC path
func IsWindowsUNCPath(goos, path string) bool {
	if goos != "windows" {
		return false
	}
	// Check for UNC prefix \\
	if strings.HasPrefix(path, `\\`) {
		return true
	}
	return false
}

// IsWindowsLocalPath checks if path is a local path
// prefixed with "/" or "\" like "/foo/bar" or "\foo\bar"
func IsWindowsLocalPath(goos, path string) bool {
	if goos != "windows" {
		return false
	}
	if IsWindowsUNCPath(goos, path) {
		return false
	}
	if strings.Contains(path, ":") {
		return false
	}
	if !(strings.HasPrefix(path, `/`) || strings.HasPrefix(path, `\`)) {
		return false
	}
	return true
}

// MakeAbsolutePath convert path to absolute path according to GOOS
func MakeAbsolutePath(goos, path string) string {
	if goos != "windows" {
		return filepath.Clean("/" + path)
	}
	// These are all for windows
	// If there is a colon, give up.
	if strings.Contains(path, ":") {
		return path
	}
	// If there is a slash, but no drive, add 'c:'
	if strings.HasPrefix(path, "/") || strings.HasPrefix(path, "\\") {
		return "c:" + path
	}
	// Otherwise, add 'c:\'
	return "c:\\" + path
}

// MapBlockVolume is a utility function to provide a common way of mounting
// block device path for a specified volume and pod.  This function should be
// called by volume plugins that implements volume.BlockVolumeMapper.Map() method.
func MapBlockVolume(
	devicePath,
	globalMapPath,
	podVolumeMapPath,
	volumeMapName string,
	podUID utypes.UID,
) error {
	blkUtil := volumepathhandler.NewBlockVolumePathHandler()

	// map devicePath to global node path
	mapErr := blkUtil.MapDevice(devicePath, globalMapPath, string(podUID))
	if mapErr != nil {
		return mapErr
	}

	// map devicePath to pod volume path
	mapErr = blkUtil.MapDevice(devicePath, podVolumeMapPath, volumeMapName)
	if mapErr != nil {
		return mapErr
	}

	return nil
}

// GetPluginMountDir returns the global mount directory name appended
// to the given plugin name's plugin directory
func GetPluginMountDir(host volume.VolumeHost, name string) string {
	mntDir := filepath.Join(host.GetPluginDir(name), MountsInGlobalPDPath)
	return mntDir
}

// IsLocalEphemeralVolume determines whether the argument is a local ephemeral
// volume vs. some other type
func IsLocalEphemeralVolume(volume v1.Volume) bool {
	return volume.GitRepo != nil ||
		(volume.EmptyDir != nil && volume.EmptyDir.Medium != v1.StorageMediumMemory) ||
		volume.ConfigMap != nil || volume.DownwardAPI != nil
}

//WriteVolumeCache flush disk data given the spcified mount path
func WriteVolumeCache(deviceMountPath string, exec mount.Exec) error {
	// If runtime os is windows, execute Write-VolumeCache powershell command on the disk
	if runtime.GOOS == "windows" {
		cmd := fmt.Sprintf("Get-Volume -FilePath %s | Write-Volumecache", deviceMountPath)
		output, err := exec.Run("powershell", "/c", cmd)
		klog.Infof("command (%q) execeuted: %v, output: %q", cmd, err, string(output))
		if err != nil {
			return fmt.Errorf("command (%q) failed: %v, output: %q", cmd, err, string(output))
		}
	}
	// For linux runtime, it skips because unmount will automatically flush disk data
	return nil
}

// IsMultiAttachAllowed checks if attaching this volume to multiple nodes is definitely not allowed/possible.
// In its current form, this function can only reliably say for which volumes it's definitely forbidden. If it returns
// false, it is not guaranteed that multi-attach is actually supported by the volume type and we must rely on the
// attacher to fail fast in such cases.
// Please see https://github.com/kubernetes/kubernetes/issues/40669 and https://github.com/kubernetes/kubernetes/pull/40148#discussion_r98055047
func IsMultiAttachAllowed(volumeSpec *volume.Spec) bool {
	if volumeSpec == nil {
		// we don't know if it's supported or not and let the attacher fail later in cases it's not supported
		return true
	}

	if volumeSpec.Volume != nil {
		// Check for volume types which are known to fail slow or cause trouble when trying to multi-attach
		if volumeSpec.Volume.AzureDisk != nil ||
			volumeSpec.Volume.Cinder != nil {
			return false
		}
	}

	// Only if this volume is a persistent volume, we have reliable information on whether it's allowed or not to
	// multi-attach. We trust in the individual volume implementations to not allow unsupported access modes
	if volumeSpec.PersistentVolume != nil {
		// Check for persistent volume types which do not fail when trying to multi-attach
		if len(volumeSpec.PersistentVolume.Spec.AccessModes) == 0 {
			// No access mode specified so we don't know for sure. Let the attacher fail if needed
			return true
		}

		// check if this volume is allowed to be attached to multiple PODs/nodes, if yes, return false
		for _, accessMode := range volumeSpec.PersistentVolume.Spec.AccessModes {
			if accessMode == v1.ReadWriteMany || accessMode == v1.ReadOnlyMany {
				return true
			}
		}
		return false
	}

	// we don't know if it's supported or not and let the attacher fail later in cases it's not supported
	return true
}
