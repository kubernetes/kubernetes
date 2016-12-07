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

package flexvolume

import (
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins(pluginDir string) []volume.VolumePlugin {
	plugins := []volume.VolumePlugin{}

	files, _ := ioutil.ReadDir(pluginDir)
	for _, f := range files {
		// only directories are counted as plugins
		// and pluginDir/dirname/dirname should be an executable
		// unless dirname contains '~' for escaping namespace
		// e.g. dirname = vendor~cifs
		// then, executable will be pluginDir/dirname/cifs
		if f.IsDir() {
			execPath := path.Join(pluginDir, f.Name())
			plugins = append(plugins, &flexVolumePlugin{driverName: utilstrings.UnescapePluginName(f.Name()), execPath: execPath})
		}
	}
	return plugins
}

// FlexVolumePlugin object.
type flexVolumePlugin struct {
	driverName string
	execPath   string
	host       volume.VolumeHost
}

// Init initializes the plugin.
func (plugin *flexVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	// call the init script
	u := &flexVolumeUtil{}
	return u.init(plugin)
}

func (plugin *flexVolumePlugin) getExecutable() string {
	parts := strings.Split(plugin.driverName, "/")
	execName := parts[len(parts)-1]
	return path.Join(plugin.execPath, execName)
}

func (plugin *flexVolumePlugin) GetPluginName() string {
	return plugin.driverName
}

func (plugin *flexVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.Driver, nil
}

// CanSupport checks whether the plugin can support the input volume spec.
func (plugin *flexVolumePlugin) CanSupport(spec *volume.Spec) bool {
	source, _, _ := getVolumeSource(spec)
	return (source != nil) && (source.Driver == plugin.driverName)
}

func (plugin *flexVolumePlugin) RequiresRemount() bool {
	return false
}

// GetAccessModes gets the allowed access modes for this plugin.
func (plugin *flexVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

// NewMounter is the mounter routine to build the volume.
func (plugin *flexVolumePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	fv, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	secrets := make(map[string]string)
	if fv.SecretRef != nil {
		kubeClient := plugin.host.GetKubeClient()
		if kubeClient == nil {
			return nil, fmt.Errorf("Cannot get kube client")
		}

		secretName, err := kubeClient.Core().Secrets(pod.Namespace).Get(fv.SecretRef.Name)
		if err != nil {
			err = fmt.Errorf("Couldn't get secret %v/%v err: %v", pod.Namespace, fv.SecretRef, err)
			return nil, err
		}
		for name, data := range secretName.Data {
			secrets[name] = base64.StdEncoding.EncodeToString(data)
			glog.V(1).Infof("found flex volume secret info: %s", name)
		}
	}
	return plugin.newMounterInternal(spec, pod, &flexVolumeUtil{}, plugin.host.GetMounter(), exec.New(), secrets)
}

// newMounterInternal is the internal mounter routine to build the volume.
func (plugin *flexVolumePlugin) newMounterInternal(spec *volume.Spec, pod *api.Pod, manager flexVolumeManager, mounter mount.Interface, runner exec.Interface, secrets map[string]string) (volume.Mounter, error) {
	source, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &flexVolumeMounter{
		flexVolumeDisk: &flexVolumeDisk{
			podUID:       pod.UID,
			podNamespace: pod.Namespace,
			podName:      pod.Name,
			volName:      spec.Name(),
			driverName:   source.Driver,
			execPath:     plugin.getExecutable(),
			mounter:      mounter,
			plugin:       plugin,
			secrets:      secrets,
		},
		fsType:             source.FSType,
		readOnly:           source.ReadOnly,
		options:            source.Options,
		runner:             runner,
		manager:            manager,
		blockDeviceMounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: runner},
	}, nil
}

// NewUnmounter is the unmounter routine to clean the volume.
func (plugin *flexVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &flexVolumeUtil{}, plugin.host.GetMounter(), exec.New())
}

// newUnmounterInternal is the internal unmounter routine to clean the volume.
func (plugin *flexVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager flexVolumeManager, mounter mount.Interface, runner exec.Interface) (volume.Unmounter, error) {
	return &flexVolumeUnmounter{
		flexVolumeDisk: &flexVolumeDisk{
			podUID:     podUID,
			volName:    volName,
			driverName: plugin.driverName,
			execPath:   plugin.getExecutable(),
			mounter:    mounter,
			plugin:     plugin,
		},
		runner:  runner,
		manager: manager,
	}, nil
}

func (plugin *flexVolumePlugin) ConstructVolumeSpec(volumeName, sourceName string) (*volume.Spec, error) {
	flexVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			FlexVolume: &api.FlexVolumeSource{
				Driver: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(flexVolume), nil
}

// flexVolume is the disk resource provided by this plugin.
type flexVolumeDisk struct {
	// podUID is the UID of the pod.
	podUID types.UID
	// podNamespace is the namespace of the pod.
	podNamespace string
	// podName is the name of the pod.
	podName string
	// volName is the name of the pod volume.
	volName string
	// driverName is the name of the plugin driverName.
	driverName string
	// Driver executable used to setup the volume.
	execPath string
	// mounter provides the interface that is used to mount the actual
	// block device.
	mounter mount.Interface
	// secret for the volume.
	secrets map[string]string
	plugin  *flexVolumePlugin
}

// FlexVolumeUnmounter is the disk that will be cleaned by this plugin.
type flexVolumeUnmounter struct {
	*flexVolumeDisk
	// Runner used to teardown the volume.
	runner exec.Interface
	// manager is the utility interface that provides API calls to the
	// driverName to setup & teardown disks
	manager flexVolumeManager
	volume.MetricsNil
}

// FlexVolumeMounter is the disk that will be exposed by this plugin.
type flexVolumeMounter struct {
	*flexVolumeDisk
	// fsType is the type of the filesystem to create on the volume.
	fsType string
	// readOnly specifies whether the disk will be setup as read-only.
	readOnly bool
	// options are the extra params that will be passed to the plugin
	// driverName.
	options map[string]string
	// Runner used to setup the volume.
	runner exec.Interface
	// manager is the utility interface that provides API calls to the
	// driverName to setup & teardown disks
	manager flexVolumeManager
	// blockDeviceMounter provides the interface to create filesystem if the
	// filesystem doesn't exist.
	blockDeviceMounter mount.Interface
	volume.MetricsNil
}

// SetUp creates new directory.
func (f *flexVolumeMounter) SetUp(fsGroup *int64) error {
	return f.SetUpAt(f.GetPath(), fsGroup)
}

// GetAttributes get the flex volume attributes. The attributes will be queried
// using plugin callout after we finalize the callout syntax.
func (f flexVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        f.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (f *flexVolumeMounter) CanMount() error {
	return nil
}

// flexVolumeManager is the abstract interface to flex volume ops.
type flexVolumeManager interface {
	// Attaches the disk to the kubelet's host machine.
	attach(mounter *flexVolumeMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	detach(unmounter *flexVolumeUnmounter, dir string) error
	// Mounts the disk on the Kubelet's host machine.
	mount(mounter *flexVolumeMounter, mnt, dir string) error
	// Unmounts the disk from the Kubelet's host machine.
	unmount(unounter *flexVolumeUnmounter, dir string) error
}

// SetUpAt creates new directory.
func (f *flexVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {

	notmnt, err := f.blockDeviceMounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("Cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notmnt {
		return nil
	}

	if f.options == nil {
		f.options = make(map[string]string)
	}

	f.options[optionFSType] = f.fsType

	// Read write mount options.
	if f.readOnly {
		f.options[optionReadWrite] = "ro"
	} else {
		f.options[optionReadWrite] = "rw"
	}

	// Extract secret and pass it as options.
	for name, secret := range f.secrets {
		f.options[optionKeySecret+"/"+name] = secret
	}

	glog.V(4).Infof("attempting to attach volume: %s with options %v", f.volName, f.options)
	device, err := f.manager.attach(f)
	if err != nil {
		if !isCmdNotSupportedErr(err) {
			glog.Errorf("failed to attach volume: %s", f.volName)
			return err
		}
		// Attach not supported or required. Continue to mount.
	}

	glog.V(4).Infof("attempting to mount volume: %s", f.volName)
	if err := f.manager.mount(f, device, dir); err != nil {
		if !isCmdNotSupportedErr(err) {
			glog.Errorf("failed to mount volume: %s", f.volName)
			return err
		}
		options := make([]string, 0)

		if f.readOnly {
			options = append(options, "ro")
		} else {
			options = append(options, "rw")
		}
		// Extract secret and pass it as options.
		for name, secret := range f.secrets {
			f.options[optionKeySecret+"/"+name] = secret
		}

		os.MkdirAll(dir, 0750)
		// Mount not supported by driver. Use core mounting logic.
		glog.V(4).Infof("attempting to mount the volume: %s to device: %s", f.volName, device)
		err = f.blockDeviceMounter.Mount(string(device), dir, f.fsType, options)
		if err != nil {
			glog.Errorf("failed to mount the volume: %s to device: %s, error: %v", f.volName, device, err)
			return err
		}
	}

	glog.V(4).Infof("Successfully mounted volume: %s on device: %s", f.volName, device)
	return nil
}

// IsReadOnly returns true if the volume is read only.
func (f *flexVolumeMounter) IsReadOnly() bool {
	return f.readOnly
}

// GetPathFromPlugin gets the actual volume mount directory based on plugin.
func (f *flexVolumeDisk) GetPath() string {
	name := f.driverName
	return f.plugin.host.GetPodVolumeDir(f.podUID, utilstrings.EscapeQualifiedNameForDisk(name), f.volName)
}

// TearDown simply deletes everything in the directory.
func (f *flexVolumeUnmounter) TearDown() error {
	path := f.GetPath()
	return f.TearDownAt(path)
}

// TearDownAt simply deletes everything in the directory.
func (f *flexVolumeUnmounter) TearDownAt(dir string) error {

	notmnt, err := f.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking mount point %s, error: %v", dir, err)
		return err
	}
	if notmnt {
		return os.Remove(dir)
	}

	device, refCount, err := mount.GetDeviceNameFromMount(f.mounter, dir)
	if err != nil {
		glog.Errorf("Failed to get reference count for volume: %s", dir)
		return err
	}

	if err := f.manager.unmount(f, dir); err != nil {
		if !isCmdNotSupportedErr(err) {
			glog.Errorf("Failed to unmount volume %s", f.volName)
			return err
		}
		// Unmount not supported by the driver. Use core unmount logic.
		if err := f.mounter.Unmount(dir); err != nil {
			glog.Errorf("Failed to unmount volume: %s, error: %v", dir, err)
			return err
		}
	}

	if refCount == 1 {
		if err := f.manager.detach(f, device); err != nil {
			if !isCmdNotSupportedErr(err) {
				glog.Errorf("Failed to teardown volume: %s, error: %v", dir, err)
				return err
			}
			// Teardown not supported by driver. Unmount is good enough.
		}
	}

	notmnt, err = f.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking mount point %s, error: %v", dir, err)
		return err
	}
	if notmnt {
		return os.Remove(dir)
	}

	return nil
}

func getVolumeSource(spec *volume.Spec) (*api.FlexVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.FlexVolume != nil {
		return spec.Volume.FlexVolume, spec.Volume.FlexVolume.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.FlexVolume != nil {
		return spec.PersistentVolume.Spec.FlexVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Flex volume type")
}
