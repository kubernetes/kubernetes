/*
Copyright 2014 The Kubernetes Authors.

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

package volume

import (
	"fmt"
	"net"
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/validation"
)

// VolumeOptions contains option information about a volume.
type VolumeOptions struct {
	// The attributes below are required by volume.Provisioner
	// TODO: refactor all of this out of volumes when an admin can configure
	// many kinds of provisioners.

	// Reclamation policy for a persistent volume
	PersistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy
	// PV.Name of the appropriate PersistentVolume. Used to generate cloud
	// volume name.
	PVName string
	// PVC is reference to the claim that lead to provisioning of a new PV.
	// Provisioners *must* create a PV that would be matched by this PVC,
	// i.e. with required capacity, accessMode, labels matching PVC.Selector and
	// so on.
	PVC *v1.PersistentVolumeClaim
	// Unique name of Kubernetes cluster.
	ClusterName string
	// Tags to attach to the real volume in the cloud provider - e.g. AWS EBS
	CloudTags *map[string]string
	// Volume provisioning parameters from StorageClass
	Parameters map[string]string
}

// VolumePlugin is an interface to volume plugins that can be used on a
// kubernetes node (e.g. by kubelet) to instantiate and manage volumes.
type VolumePlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init(host VolumeHost) error

	// Name returns the plugin's name.  Plugins should use namespaced names
	// such as "example.com/volume".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	GetPluginName() string

	// GetVolumeName returns the name/ID to uniquely identifying the actual
	// backing device, directory, path, etc. referenced by the specified volume
	// spec.
	// For Attachable volumes, this value must be able to be passed back to
	// volume Detach methods to identify the device to act on.
	// If the plugin does not support the given spec, this returns an error.
	GetVolumeName(spec *Spec) (string, error)

	// CanSupport tests whether the plugin supports a given volume
	// specification from the API.  The spec pointer should be considered
	// const.
	CanSupport(spec *Spec) bool

	// RequiresRemount returns true if this plugin requires mount calls to be
	// reexecuted. Atomically updating volumes, like Downward API, depend on
	// this to update the contents of the volume.
	RequiresRemount() bool

	// NewMounter creates a new volume.Mounter from an API specification.
	// Ownership of the spec pointer in *not* transferred.
	// - spec: The v1.Volume spec
	// - pod: The enclosing pod
	NewMounter(spec *Spec, podRef *v1.Pod, opts VolumeOptions) (Mounter, error)

	// NewUnmounter creates a new volume.Unmounter from recoverable state.
	// - name: The volume name, as per the v1.Volume spec.
	// - podUID: The UID of the enclosing pod
	NewUnmounter(name string, podUID types.UID) (Unmounter, error)

	// ConstructVolumeSpec constructs a volume spec based on the given volume name
	// and mountPath. The spec may have incomplete information due to limited
	// information from input. This function is used by volume manager to reconstruct
	// volume spec by reading the volume directories from disk
	ConstructVolumeSpec(volumeName, mountPath string) (*Spec, error)
}

// PersistentVolumePlugin is an extended interface of VolumePlugin and is used
// by volumes that want to provide long term persistence of data
type PersistentVolumePlugin interface {
	VolumePlugin
	// GetAccessModes describes the ways a given volume can be accessed/mounted.
	GetAccessModes() []v1.PersistentVolumeAccessMode
}

// RecyclableVolumePlugin is an extended interface of VolumePlugin and is used
// by persistent volumes that want to be recycled before being made available
// again to new claims
type RecyclableVolumePlugin interface {
	VolumePlugin
	// NewRecycler creates a new volume.Recycler which knows how to reclaim this
	// resource after the volume's release from a PersistentVolumeClaim. The
	// recycler will use the provided recorder to write any events that might be
	// interesting to user. It's expected that caller will pass these events to
	// the PV being recycled.
	NewRecycler(pvName string, spec *Spec, eventRecorder RecycleEventRecorder) (Recycler, error)
}

// DeletableVolumePlugin is an extended interface of VolumePlugin and is used
// by persistent volumes that want to be deleted from the cluster after their
// release from a PersistentVolumeClaim.
type DeletableVolumePlugin interface {
	VolumePlugin
	// NewDeleter creates a new volume.Deleter which knows how to delete this
	// resource in accordance with the underlying storage provider after the
	// volume's release from a claim
	NewDeleter(spec *Spec) (Deleter, error)
}

const (
	// Name of a volume in external cloud that is being provisioned and thus
	// should be ignored by rest of Kubernetes.
	ProvisionedVolumeName = "placeholder-for-provisioning"
)

// ProvisionableVolumePlugin is an extended interface of VolumePlugin and is
// used to create volumes for the cluster.
type ProvisionableVolumePlugin interface {
	VolumePlugin
	// NewProvisioner creates a new volume.Provisioner which knows how to
	// create PersistentVolumes in accordance with the plugin's underlying
	// storage provider
	NewProvisioner(options VolumeOptions) (Provisioner, error)
}

// AttachableVolumePlugin is an extended interface of VolumePlugin and is used for volumes that require attachment
// to a node before mounting.
type AttachableVolumePlugin interface {
	VolumePlugin
	NewAttacher() (Attacher, error)
	NewDetacher() (Detacher, error)
	GetDeviceMountRefs(deviceMountPath string) ([]string, error)
}

// VolumeHost is an interface that plugins can use to access the kubelet.
type VolumeHost interface {
	// GetPluginDir returns the absolute path to a directory under which
	// a given plugin may store data.  This directory might not actually
	// exist on disk yet.  For plugin data that is per-pod, see
	// GetPodPluginDir().
	GetPluginDir(pluginName string) string

	// GetPodVolumeDir returns the absolute path a directory which
	// represents the named volume under the named plugin for the given
	// pod.  If the specified pod does not exist, the result of this call
	// might not exist.
	GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string

	// GetPodPluginDir returns the absolute path to a directory under which
	// a given plugin may store data for a given pod.  If the specified pod
	// does not exist, the result of this call might not exist.  This
	// directory might not actually exist on disk yet.
	GetPodPluginDir(podUID types.UID, pluginName string) string

	// GetKubeClient returns a client interface
	GetKubeClient() clientset.Interface

	// NewWrapperMounter finds an appropriate plugin with which to handle
	// the provided spec.  This is used to implement volume plugins which
	// "wrap" other plugins.  For example, the "secret" volume is
	// implemented in terms of the "emptyDir" volume.
	NewWrapperMounter(volName string, spec Spec, pod *v1.Pod, opts VolumeOptions) (Mounter, error)

	// NewWrapperUnmounter finds an appropriate plugin with which to handle
	// the provided spec.  See comments on NewWrapperMounter for more
	// context.
	NewWrapperUnmounter(volName string, spec Spec, podUID types.UID) (Unmounter, error)

	// Get cloud provider from kubelet.
	GetCloudProvider() cloudprovider.Interface

	// Get mounter interface.
	GetMounter() mount.Interface

	// Get writer interface for writing data to disk.
	GetWriter() io.Writer

	// Returns the hostname of the host kubelet is running on
	GetHostName() string

	// Returns host IP or nil in the case of error.
	GetHostIP() (net.IP, error)

	// Returns node allocatable
	GetNodeAllocatable() (v1.ResourceList, error)
}

// VolumePluginMgr tracks registered plugins.
type VolumePluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]VolumePlugin
}

// Spec is an internal representation of a volume.  All API volume types translate to Spec.
type Spec struct {
	Volume           *v1.Volume
	PersistentVolume *v1.PersistentVolume
	ReadOnly         bool
}

// Name returns the name of either Volume or PersistentVolume, one of which must not be nil.
func (spec *Spec) Name() string {
	switch {
	case spec.Volume != nil:
		return spec.Volume.Name
	case spec.PersistentVolume != nil:
		return spec.PersistentVolume.Name
	default:
		return ""
	}
}

// VolumeConfig is how volume plugins receive configuration.  An instance
// specific to the plugin will be passed to the plugin's
// ProbeVolumePlugins(config) func.  Reasonable defaults will be provided by
// the binary hosting the plugins while allowing override of those default
// values.  Those config values are then set to an instance of VolumeConfig
// and passed to the plugin.
//
// Values in VolumeConfig are intended to be relevant to several plugins, but
// not necessarily all plugins.  The preference is to leverage strong typing
// in this struct.  All config items must have a descriptive but non-specific
// name (i.e, RecyclerMinimumTimeout is OK but RecyclerMinimumTimeoutForNFS is
// !OK).  An instance of config will be given directly to the plugin, so
// config names specific to plugins are unneeded and wrongly expose plugins in
// this VolumeConfig struct.
//
// OtherAttributes is a map of string values intended for one-off
// configuration of a plugin or config that is only relevant to a single
// plugin.  All values are passed by string and require interpretation by the
// plugin. Passing config as strings is the least desirable option but can be
// used for truly one-off configuration. The binary should still use strong
// typing for this value when binding CLI values before they are passed as
// strings in OtherAttributes.
type VolumeConfig struct {
	// RecyclerPodTemplate is pod template that understands how to scrub clean
	// a persistent volume after its release. The template is used by plugins
	// which override specific properties of the pod in accordance with that
	// plugin. See NewPersistentVolumeRecyclerPodTemplate for the properties
	// that are expected to be overridden.
	RecyclerPodTemplate *v1.Pod

	// RecyclerMinimumTimeout is the minimum amount of time in seconds for the
	// recycler pod's ActiveDeadlineSeconds attribute. Added to the minimum
	// timeout is the increment per Gi of capacity.
	RecyclerMinimumTimeout int

	// RecyclerTimeoutIncrement is the number of seconds added to the recycler
	// pod's ActiveDeadlineSeconds for each Gi of capacity in the persistent
	// volume. Example: 5Gi volume x 30s increment = 150s + 30s minimum = 180s
	// ActiveDeadlineSeconds for recycler pod
	RecyclerTimeoutIncrement int

	// PVName is name of the PersistentVolume instance that is being recycled.
	// It is used to generate unique recycler pod name.
	PVName string

	// OtherAttributes stores config as strings.  These strings are opaque to
	// the system and only understood by the binary hosting the plugin and the
	// plugin itself.
	OtherAttributes map[string]string

	// ProvisioningEnabled configures whether provisioning of this plugin is
	// enabled or not. Currently used only in host_path plugin.
	ProvisioningEnabled bool
}

// NewSpecFromVolume creates an Spec from an v1.Volume
func NewSpecFromVolume(vs *v1.Volume) *Spec {
	return &Spec{
		Volume: vs,
	}
}

// NewSpecFromPersistentVolume creates an Spec from an v1.PersistentVolume
func NewSpecFromPersistentVolume(pv *v1.PersistentVolume, readOnly bool) *Spec {
	return &Spec{
		PersistentVolume: pv,
		ReadOnly:         readOnly,
	}
}

// InitPlugins initializes each plugin.  All plugins must have unique names.
// This must be called exactly once before any New* methods are called on any
// plugins.
func (pm *VolumePluginMgr) InitPlugins(plugins []VolumePlugin, host VolumeHost) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.plugins == nil {
		pm.plugins = map[string]VolumePlugin{}
	}

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.GetPluginName()
		if errs := validation.IsQualifiedName(name); len(errs) != 0 {
			allErrs = append(allErrs, fmt.Errorf("volume plugin has invalid name: %q: %s", name, strings.Join(errs, ";")))
			continue
		}

		if _, found := pm.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("volume plugin %q was registered more than once", name))
			continue
		}
		err := plugin.Init(host)
		if err != nil {
			glog.Errorf("Failed to load volume plugin %s, error: %s", plugin, err.Error())
			allErrs = append(allErrs, err)
			continue
		}
		pm.plugins[name] = plugin
		glog.V(1).Infof("Loaded volume plugin %q", name)
	}
	return utilerrors.NewAggregate(allErrs)
}

// FindPluginBySpec looks for a plugin that can support a given volume
// specification.  If no plugins can support or more than one plugin can
// support it, return error.
func (pm *VolumePluginMgr) FindPluginBySpec(spec *Spec) (VolumePlugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	matches := []string{}
	for k, v := range pm.plugins {
		if v.CanSupport(spec) {
			matches = append(matches, k)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matches, ","))
	}
	return pm.plugins[matches[0]], nil
}

// FindPluginByName fetches a plugin by name or by legacy name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindPluginByName(name string) (VolumePlugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	// Once we can get rid of legacy names we can reduce this to a map lookup.
	matches := []string{}
	for k, v := range pm.plugins {
		if v.GetPluginName() == name {
			matches = append(matches, k)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matches, ","))
	}
	return pm.plugins[matches[0]], nil
}

// FindPersistentPluginBySpec looks for a persistent volume plugin that can
// support a given volume specification.  If no plugin is found, return an
// error
func (pm *VolumePluginMgr) FindPersistentPluginBySpec(spec *Spec) (PersistentVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("Could not find volume plugin for spec: %#v", spec)
	}
	if persistentVolumePlugin, ok := volumePlugin.(PersistentVolumePlugin); ok {
		return persistentVolumePlugin, nil
	}
	return nil, fmt.Errorf("no persistent volume plugin matched")
}

// FindPersistentPluginByName fetches a persistent volume plugin by name.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindPersistentPluginByName(name string) (PersistentVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if persistentVolumePlugin, ok := volumePlugin.(PersistentVolumePlugin); ok {
		return persistentVolumePlugin, nil
	}
	return nil, fmt.Errorf("no persistent volume plugin matched")
}

// FindRecyclablePluginByName fetches a persistent volume plugin by name.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindRecyclablePluginBySpec(spec *Spec) (RecyclableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if recyclableVolumePlugin, ok := volumePlugin.(RecyclableVolumePlugin); ok {
		return recyclableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no recyclable volume plugin matched")
}

// FindProvisionablePluginByName fetches  a persistent volume plugin by name.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindProvisionablePluginByName(name string) (ProvisionableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if provisionableVolumePlugin, ok := volumePlugin.(ProvisionableVolumePlugin); ok {
		return provisionableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no provisionable volume plugin matched")
}

// FindDeletablePluginBySppec fetches a persistent volume plugin by spec.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindDeletablePluginBySpec(spec *Spec) (DeletableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if deletableVolumePlugin, ok := volumePlugin.(DeletableVolumePlugin); ok {
		return deletableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no deletable volume plugin matched")
}

// FindDeletablePluginByName fetches a persistent volume plugin by name.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindDeletablePluginByName(name string) (DeletableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if deletableVolumePlugin, ok := volumePlugin.(DeletableVolumePlugin); ok {
		return deletableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no deletable volume plugin matched")
}

// FindCreatablePluginBySpec fetches a persistent volume plugin by name.  If
// no plugin is found, returns error.
func (pm *VolumePluginMgr) FindCreatablePluginBySpec(spec *Spec) (ProvisionableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if provisionableVolumePlugin, ok := volumePlugin.(ProvisionableVolumePlugin); ok {
		return provisionableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no creatable volume plugin matched")
}

// FindAttachablePluginBySpec fetches a persistent volume plugin by name.
// Unlike the other "FindPlugin" methods, this does not return error if no
// plugin is found.  All volumes require a mounter and unmounter, but not
// every volume will have an attacher/detacher.
func (pm *VolumePluginMgr) FindAttachablePluginBySpec(spec *Spec) (AttachableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if attachableVolumePlugin, ok := volumePlugin.(AttachableVolumePlugin); ok {
		return attachableVolumePlugin, nil
	}
	return nil, nil
}

// FindAttachablePluginByName fetches an attachable volume plugin by name.
// Unlike the other "FindPlugin" methods, this does not return error if no
// plugin is found.  All volumes require a mounter and unmounter, but not
// every volume will have an attacher/detacher.
func (pm *VolumePluginMgr) FindAttachablePluginByName(name string) (AttachableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if attachablePlugin, ok := volumePlugin.(AttachableVolumePlugin); ok {
		return attachablePlugin, nil
	}
	return nil, nil
}

// NewPersistentVolumeRecyclerPodTemplate creates a template for a recycler
// pod.  By default, a recycler pod simply runs "rm -rf" on a volume and tests
// for emptiness.  Most attributes of the template will be correct for most
// plugin implementations.  The following attributes can be overridden per
// plugin via configuration:
//
// 1.  pod.Spec.Volumes[0].VolumeSource must be overridden.  Recycler
//     implementations without a valid VolumeSource will fail.
// 2.  pod.GenerateName helps distinguish recycler pods by name.  Recommended.
//     Default is "pv-recycler-".
// 3.  pod.Spec.ActiveDeadlineSeconds gives the recycler pod a maximum timeout
//     before failing.  Recommended.  Default is 60 seconds.
//
// See HostPath and NFS for working recycler examples
func NewPersistentVolumeRecyclerPodTemplate() *v1.Pod {
	timeout := int64(60)
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			GenerateName: "pv-recycler-",
			Namespace:    v1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			ActiveDeadlineSeconds: &timeout,
			RestartPolicy:         v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "vol",
					// IMPORTANT!  All plugins using this template MUST
					// override pod.Spec.Volumes[0].VolumeSource Recycler
					// implementations without a valid VolumeSource will fail.
					VolumeSource: v1.VolumeSource{},
				},
			},
			Containers: []v1.Container{
				{
					Name:    "pv-recycler",
					Image:   "gcr.io/google_containers/busybox",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "test -e /scrub && rm -rf /scrub/..?* /scrub/.[!.]* /scrub/*  && test -z \"$(ls -A /scrub)\" || exit 1"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "vol",
							MountPath: "/scrub",
						},
					},
				},
			},
		},
	}
	return pod
}
