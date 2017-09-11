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

package storageos

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&storageosPlugin{nil}}
}

type storageosPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &storageosPlugin{}
var _ volume.PersistentVolumePlugin = &storageosPlugin{}
var _ volume.DeletableVolumePlugin = &storageosPlugin{}
var _ volume.ProvisionableVolumePlugin = &storageosPlugin{}

const (
	storageosPluginName = "kubernetes.io/storageos"
	storageosDevicePath = "/var/lib/storageos/volumes"
	defaultAPIAddress   = "tcp://localhost:5705"
	defaultAPIUser      = "storageos"
	defaultAPIPassword  = "storageos"
	defaultAPIVersion   = "1"
	defaultFSType       = "ext4"
	defaultNamespace    = "default"
)

func getPath(uid types.UID, volNamespace string, volName string, pvName string, host volume.VolumeHost) string {
	if len(volNamespace) != 0 && len(volName) != 0 && strings.Count(volName, ".") == 0 {
		return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(storageosPluginName), pvName+"."+volNamespace+"."+volName)
	}
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(storageosPluginName), pvName)
}

func (plugin *storageosPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *storageosPlugin) GetPluginName() string {
	return storageosPluginName
}

func (plugin *storageosPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s/%s", volumeSource.VolumeNamespace, volumeSource.VolumeName), nil
}

func (plugin *storageosPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.StorageOS != nil) ||
		(spec.Volume != nil && spec.Volume.StorageOS != nil)
}

func (plugin *storageosPlugin) RequiresRemount() bool {
	return false
}

func (plugin *storageosPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *storageosPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {

	apiCfg, err := getAPICfg(spec, pod, plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}

	return plugin.newMounterInternal(spec, pod, apiCfg, &storageosUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *storageosPlugin) newMounterInternal(spec *volume.Spec, pod *v1.Pod, apiCfg *storageosAPIConfig, manager storageosManager, mounter mount.Interface, exec mount.Exec) (volume.Mounter, error) {

	volName, volNamespace, fsType, readOnly, err := getVolumeInfoFromSpec(spec)
	if err != nil {
		return nil, err
	}

	return &storageosMounter{
		storageos: &storageos{
			podUID:          pod.UID,
			podNamespace:    pod.GetNamespace(),
			pvName:          spec.Name(),
			volName:         volName,
			volNamespace:    volNamespace,
			fsType:          fsType,
			readOnly:        readOnly,
			apiCfg:          apiCfg,
			manager:         manager,
			mounter:         mounter,
			exec:            exec,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(pod.UID, volNamespace, volName, spec.Name(), plugin.host)),
		},
		devicePath:  storageosDevicePath,
		diskMounter: &mount.SafeFormatAndMount{Interface: mounter, Exec: exec},
	}, nil
}

func (plugin *storageosPlugin) NewUnmounter(pvName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(pvName, podUID, &storageosUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *storageosPlugin) newUnmounterInternal(pvName string, podUID types.UID, manager storageosManager, mounter mount.Interface, exec mount.Exec) (volume.Unmounter, error) {

	// Parse volume namespace & name from mountpoint if mounted
	volNamespace, volName, err := getVolumeInfo(pvName, podUID, plugin.host)
	if err != nil {
		return nil, err
	}

	return &storageosUnmounter{
		storageos: &storageos{
			podUID:          podUID,
			pvName:          pvName,
			volName:         volName,
			volNamespace:    volNamespace,
			manager:         manager,
			mounter:         mounter,
			exec:            exec,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volNamespace, volName, pvName, plugin.host)),
		},
	}, nil
}

func (plugin *storageosPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.StorageOS == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.StorageOS is nil")
	}

	class, err := util.GetClassForVolume(plugin.host.GetKubeClient(), spec.PersistentVolume)
	if err != nil {
		return nil, err
	}

	var adminSecretName, adminSecretNamespace string

	for k, v := range class.Parameters {
		switch strings.ToLower(k) {
		case "adminsecretname":
			adminSecretName = v
		case "adminsecretnamespace":
			adminSecretNamespace = v
		}
	}

	apiCfg, err := parsePVSecret(adminSecretNamespace, adminSecretName, plugin.host.GetKubeClient())
	if err != nil {
		return nil, fmt.Errorf("failed to get admin secret from [%q/%q]: %v", adminSecretNamespace, adminSecretName, err)
	}

	return plugin.newDeleterInternal(spec, apiCfg, &storageosUtil{})
}

func (plugin *storageosPlugin) newDeleterInternal(spec *volume.Spec, apiCfg *storageosAPIConfig, manager storageosManager) (volume.Deleter, error) {

	return &storageosDeleter{
		storageosMounter: &storageosMounter{
			storageos: &storageos{
				pvName:       spec.Name(),
				volName:      spec.PersistentVolume.Spec.StorageOS.VolumeName,
				volNamespace: spec.PersistentVolume.Spec.StorageOS.VolumeNamespace,
				apiCfg:       apiCfg,
				manager:      manager,
				plugin:       plugin,
			},
		},
		pvUID: spec.PersistentVolume.UID,
	}, nil
}

func (plugin *storageosPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &storageosUtil{})
}

func (plugin *storageosPlugin) newProvisionerInternal(options volume.VolumeOptions, manager storageosManager) (volume.Provisioner, error) {
	return &storageosProvisioner{
		storageosMounter: &storageosMounter{
			storageos: &storageos{
				manager: manager,
				plugin:  plugin,
			},
		},
		options: options,
	}, nil
}

func (plugin *storageosPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	volNamespace, volName, err := getVolumeFromRef(volumeName)
	if err != nil {
		volNamespace = defaultNamespace
		volName = volumeName
	}
	storageosVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			StorageOS: &v1.StorageOSVolumeSource{
				VolumeName:      volName,
				VolumeNamespace: volNamespace,
			},
		},
	}
	return volume.NewSpecFromVolume(storageosVolume), nil
}

func (plugin *storageosPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *storageosPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func getVolumeSource(spec *volume.Spec) (*v1.StorageOSVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.StorageOS != nil {
		return spec.Volume.StorageOS, spec.Volume.StorageOS.ReadOnly, nil
	}
	return nil, false, fmt.Errorf("Spec does not reference a StorageOS volume type")
}

func getPersistentVolumeSource(spec *volume.Spec) (*v1.StorageOSPersistentVolumeSource, bool, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.StorageOS != nil {
		return spec.PersistentVolume.Spec.StorageOS, spec.ReadOnly, nil
	}
	return nil, false, fmt.Errorf("Spec does not reference a StorageOS persistent volume type")
}

// storageosManager is the abstract interface to StorageOS volume ops.
type storageosManager interface {
	// Connects to the StorageOS API using the supplied configuration.
	NewAPI(apiCfg *storageosAPIConfig) error
	// Creates a StorageOS volume.
	CreateVolume(provisioner *storageosProvisioner) (*storageosVolume, error)
	// Attaches the disk to the kubelet's host machine.
	AttachVolume(mounter *storageosMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	DetachVolume(unmounter *storageosUnmounter, dir string) error
	// Mounts the disk on the Kubelet's host machine.
	MountVolume(mounter *storageosMounter, mnt, dir string) error
	// Unmounts the disk from the Kubelet's host machine.
	UnmountVolume(unounter *storageosUnmounter) error
	// Deletes the storageos volume.  All data will be lost.
	DeleteVolume(deleter *storageosDeleter) error
}

// storageos volumes represent a bare host directory mount of an StorageOS export.
type storageos struct {
	podUID       types.UID
	podNamespace string
	pvName       string
	volName      string
	volNamespace string
	secretName   string
	readOnly     bool
	description  string
	pool         string
	fsType       string
	sizeGB       int
	labels       map[string]string
	apiCfg       *storageosAPIConfig
	manager      storageosManager
	mounter      mount.Interface
	exec         mount.Exec
	plugin       *storageosPlugin
	volume.MetricsProvider
}

type storageosMounter struct {
	*storageos
	devicePath string
	// Interface used to mount the file or block device
	diskMounter *mount.SafeFormatAndMount
}

var _ volume.Mounter = &storageosMounter{}

func (b *storageosMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *storageosMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *storageosMounter) SetUp(fsGroup *int64) error {
	// Need a namespace to find the volume, try pod's namespace if not set.
	if b.volNamespace == "" {
		glog.V(2).Infof("Setting StorageOS volume namespace to pod namespace: %s", b.podNamespace)
		b.volNamespace = b.podNamespace
	}

	// Attach the StorageOS volume as a block device
	devicePath, err := b.manager.AttachVolume(b)
	if err != nil {
		glog.Errorf("Failed to attach StorageOS volume %s: %s", b.volName, err.Error())
		return err
	}

	// Mount the loop device into the plugin's disk global mount dir.
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pvName, b.podNamespace, b.volName)
	err = b.manager.MountVolume(b, devicePath, globalPDPath)
	if err != nil {
		return err
	}
	glog.V(4).Infof("Successfully mounted StorageOS volume %s into global mount directory", b.volName)

	// Bind mount the volume into the pod
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp bind mounts the disk global mount to the give volume path.
func (b *storageosMounter) SetUpAt(dir string, fsGroup *int64) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("StorageOS volume set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("Cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	if err = os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("mkdir failed on disk %s (%v)", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	globalPDPath := makeGlobalPDName(b.plugin.host, b.pvName, b.volNamespace, b.volName)
	glog.V(4).Infof("Attempting to bind mount to pod volume at %s", dir)

	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("Mount of disk %s failed: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}
	glog.V(4).Infof("StorageOS volume setup complete on %s", dir)
	return nil
}

func makeGlobalPDName(host volume.VolumeHost, pvName, volNamespace, volName string) string {
	return path.Join(host.GetPluginDir(kstrings.EscapeQualifiedNameForDisk(storageosPluginName)), mount.MountsInGlobalPDPath, pvName+"."+volNamespace+"."+volName)
}

// Given the pod id and PV name, finds the volume's namespace and name from the
// name or volume mount.  We mount as volNamespace.pvName, but k8s will specify
// only the pvName to unmount.
// Will return empty volNamespace/pvName if the volume is not mounted.
func getVolumeInfo(pvName string, podUID types.UID, host volume.VolumeHost) (string, string, error) {
	if volNamespace, volName, err := getVolumeFromRef(pvName); err == nil {
		return volNamespace, volName, nil
	}

	volumeDir := filepath.Dir(host.GetPodVolumeDir(podUID, kstrings.EscapeQualifiedNameForDisk(storageosPluginName), pvName))
	files, err := ioutil.ReadDir(volumeDir)
	if err != nil {
		return "", "", fmt.Errorf("Could not read mounts from pod volume dir: %s", err)
	}
	for _, f := range files {
		if f.Mode().IsDir() && strings.HasPrefix(f.Name(), pvName+".") {
			if volNamespace, volName, err := getVolumeFromRef(f.Name()); err == nil {
				return volNamespace, volName, nil
			}
		}
	}
	return "", "", fmt.Errorf("Could not get info from unmounted pv %q at %q", pvName, volumeDir)
}

// Splits the volume ref on "." to return the volNamespace and pvName.  Neither
// namespaces nor service names allow "." in their names.
func getVolumeFromRef(ref string) (volNamespace string, volName string, err error) {
	refParts := strings.Split(ref, ".")
	switch len(refParts) {
	case 2:
		return refParts[0], refParts[1], nil
	case 3:
		return refParts[1], refParts[2], nil
	}
	return "", "", fmt.Errorf("ref not in format volNamespace.volName or pvName.volNamespace.volName")
}

// GetPath returns the path to the user specific mount of a StorageOS volume
func (storageosVolume *storageos) GetPath() string {
	return getPath(storageosVolume.podUID, storageosVolume.volNamespace, storageosVolume.volName, storageosVolume.pvName, storageosVolume.plugin.host)
}

type storageosUnmounter struct {
	*storageos
}

var _ volume.Unmounter = &storageosUnmounter{}

func (b *storageosUnmounter) GetPath() string {
	return getPath(b.podUID, b.volNamespace, b.volName, b.pvName, b.plugin.host)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (b *storageosUnmounter) TearDown() error {
	if len(b.volNamespace) == 0 || len(b.volName) == 0 {
		glog.Warningf("volNamespace: %q, volName: %q not set, skipping TearDown", b.volNamespace, b.volName)
		return fmt.Errorf("pvName not specified for TearDown, waiting for next sync loop")
	}
	// Unmount from pod
	mountPath := b.GetPath()

	err := b.TearDownAt(mountPath)
	if err != nil {
		glog.Errorf("Unmount from pod failed: %v", err)
		return err
	}

	// Find device name from global mount
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pvName, b.volNamespace, b.volName)
	devicePath, _, err := mount.GetDeviceNameFromMount(b.mounter, globalPDPath)
	if err != nil {
		glog.Errorf("Detach failed when getting device from global mount: %v", err)
		return err
	}

	// Unmount from plugin's disk global mount dir.
	err = b.TearDownAt(globalPDPath)
	if err != nil {
		glog.Errorf("Detach failed during unmount: %v", err)
		return err
	}

	// Detach loop device
	err = b.manager.DetachVolume(b, devicePath)
	if err != nil {
		glog.Errorf("Detach device %s failed for volume %s: %v", devicePath, b.pvName, err)
		return err
	}

	glog.V(4).Infof("Successfully unmounted StorageOS volume %s and detached devices", b.pvName)

	return nil
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (b *storageosUnmounter) TearDownAt(dir string) error {
	if err := util.UnmountPath(dir, b.mounter); err != nil {
		glog.V(4).Infof("Unmounted StorageOS volume %s failed with: %v", b.pvName, err)
	}
	if err := b.manager.UnmountVolume(b); err != nil {
		glog.V(4).Infof("Mount reference for volume %s could not be removed from StorageOS: %v", b.pvName, err)
	}
	return nil
}

type storageosDeleter struct {
	*storageosMounter
	pvUID types.UID
}

var _ volume.Deleter = &storageosDeleter{}

func (d *storageosDeleter) GetPath() string {
	return getPath(d.podUID, d.volNamespace, d.volName, d.pvName, d.plugin.host)
}

func (d *storageosDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type storageosProvisioner struct {
	*storageosMounter
	options volume.VolumeOptions
}

var _ volume.Provisioner = &storageosProvisioner{}

func (c *storageosProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(c.plugin.GetAccessModes(), c.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", c.options.PVC.Spec.AccessModes, c.plugin.GetAccessModes())
	}

	var adminSecretName, adminSecretNamespace string

	// Apply ProvisionerParameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "adminsecretname":
			adminSecretName = v
		case "adminsecretnamespace":
			adminSecretNamespace = v
		case "volumenamespace":
			c.volNamespace = v
		case "description":
			c.description = v
		case "pool":
			c.pool = v
		case "fstype":
			c.fsType = v
		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}

	// Set from PVC
	c.podNamespace = c.options.PVC.Namespace
	c.volName = c.options.PVName
	if c.volNamespace == "" {
		c.volNamespace = c.options.PVC.Namespace
	}
	c.labels = make(map[string]string)
	for k, v := range c.options.PVC.Labels {
		c.labels[k] = v
	}
	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	c.sizeGB = int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024))

	apiCfg, err := parsePVSecret(adminSecretNamespace, adminSecretName, c.plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}
	c.apiCfg = apiCfg

	vol, err := c.manager.CreateVolume(c)
	if err != nil {
		glog.Errorf("failed to create volume: %v", err)
		return nil, err
	}
	if vol.FSType == "" {
		vol.FSType = defaultFSType
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   vol.Name,
			Labels: map[string]string{},
			Annotations: map[string]string{
				volumehelper.VolumeDynamicallyCreatedByKey: "storageos-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", vol.SizeGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				StorageOS: &v1.StorageOSPersistentVolumeSource{
					VolumeName:      vol.Name,
					VolumeNamespace: vol.Namespace,
					FSType:          vol.FSType,
					ReadOnly:        false,
					SecretRef: &v1.ObjectReference{
						Name:      adminSecretName,
						Namespace: adminSecretNamespace,
					},
				},
			},
		},
	}
	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}
	if len(vol.Labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range vol.Labels {
			pv.Labels[k] = v
		}
	}
	return pv, nil
}

// Returns StorageOS volume name, namespace, fstype and readonly from spec
func getVolumeInfoFromSpec(spec *volume.Spec) (string, string, string, bool, error) {
	if spec.PersistentVolume != nil {
		source, readOnly, err := getPersistentVolumeSource(spec)
		if err != nil {
			return "", "", "", false, err
		}
		return source.VolumeName, source.VolumeNamespace, source.FSType, readOnly, nil
	}

	if spec.Volume != nil {
		source, readOnly, err := getVolumeSource(spec)
		if err != nil {
			return "", "", "", false, err
		}
		return source.VolumeName, source.VolumeNamespace, source.FSType, readOnly, nil
	}
	return "", "", "", false, fmt.Errorf("spec not Volume or PersistentVolume")
}

// Returns API config if secret set, otherwise empty struct so defaults can be
// attempted.
func getAPICfg(spec *volume.Spec, pod *v1.Pod, kubeClient clientset.Interface) (*storageosAPIConfig, error) {
	if spec.PersistentVolume != nil {
		source, _, err := getPersistentVolumeSource(spec)
		if err != nil {
			return nil, err
		}
		if source.SecretRef == nil {
			return nil, nil
		}
		return parsePVSecret(source.SecretRef.Namespace, source.SecretRef.Name, kubeClient)
	}

	if spec.Volume != nil {
		source, _, err := getVolumeSource(spec)
		if err != nil {
			return nil, err
		}
		if source.SecretRef == nil {
			return nil, nil
		}
		return parsePodSecret(pod, source.SecretRef.Name, kubeClient)
	}

	return nil, fmt.Errorf("spec not Volume or PersistentVolume")
}

func parsePodSecret(pod *v1.Pod, secretName string, kubeClient clientset.Interface) (*storageosAPIConfig, error) {
	secret, err := util.GetSecretForPod(pod, secretName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret from [%q/%q]", pod.Namespace, secretName)
		return nil, fmt.Errorf("failed to get secret from [%q/%q]", pod.Namespace, secretName)
	}
	return parseAPIConfig(secret)
}

// Important: Only to be called with data from a PV to avoid secrets being
// loaded from a user-suppler namespace.
func parsePVSecret(namespace, secretName string, kubeClient clientset.Interface) (*storageosAPIConfig, error) {
	secret, err := util.GetSecretForPV(namespace, secretName, storageosPluginName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
		return nil, fmt.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
	}
	return parseAPIConfig(secret)
}

// Parse API configuration from parameters or secret
func parseAPIConfig(params map[string]string) (*storageosAPIConfig, error) {

	if len(params) == 0 {
		return nil, fmt.Errorf("empty API config")
	}

	c := &storageosAPIConfig{}

	for name, data := range params {
		switch strings.ToLower(name) {
		case "apiaddress":
			c.apiAddr = string(data)
		case "apiusername":
			c.apiUser = string(data)
		case "apipassword":
			c.apiPass = string(data)
		case "apiversion":
			c.apiVersion = string(data)
		}
	}

	return c, nil
}
