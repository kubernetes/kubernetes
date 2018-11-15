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

	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	csiclientset "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"
)

type ProbeOperation uint32
type ProbeEvent struct {
	Plugin     VolumePlugin // VolumePlugin that was added/updated/removed. if ProbeEvent.Op is 'ProbeRemove', Plugin should be nil
	PluginName string
	Op         ProbeOperation // The operation to the plugin
}

const (
	// Common parameter which can be specified in StorageClass to specify the desired FSType
	// Provisioners SHOULD implement support for this if they are block device based
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Default value depends on the provisioner
	VolumeParameterFSType = "fstype"

	ProbeAddOrUpdate ProbeOperation = 1 << iota
	ProbeRemove
)

// VolumeOptions contains option information about a volume.
type VolumeOptions struct {
	// The attributes below are required by volume.Provisioner
	// TODO: refactor all of this out of volumes when an admin can configure
	// many kinds of provisioners.

	// Reclamation policy for a persistent volume
	PersistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy
	// Mount options for a persistent volume
	MountOptions []string
	// Suggested PV.Name of the PersistentVolume to provision.
	// This is a generated name guaranteed to be unique in Kubernetes cluster.
	// If you choose not to use it as volume name, ensure uniqueness by either
	// combining it with your value or create unique values of your own.
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

type DynamicPluginProber interface {
	Init() error

	// If an error occurs, events are undefined.
	Probe() (events []ProbeEvent, err error)
}

// VolumePlugin is an interface to volume plugins that can be used on a
// kubernetes node (e.g. by kubelet) to instantiate and manage volumes.
type VolumePlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init(host VolumeHost) error

	// Name returns the plugin's name.  Plugins must use namespaced names
	// such as "example.com/volume" and contain exactly one '/' character.
	// The "kubernetes.io" namespace is reserved for plugins which are
	// bundled with kubernetes.
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

	// SupportsMountOption returns true if volume plugins supports Mount options
	// Specifying mount options in a volume plugin that doesn't support
	// user specified mount options will result in error creating persistent volumes
	SupportsMountOption() bool

	// SupportsBulkVolumeVerification checks if volume plugin type is capable
	// of enabling bulk polling of all nodes. This can speed up verification of
	// attached volumes by quite a bit, but underlying pluging must support it.
	SupportsBulkVolumeVerification() bool
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

	// Recycle knows how to reclaim this
	// resource after the volume's release from a PersistentVolumeClaim.
	// Recycle will use the provided recorder to write any events that might be
	// interesting to user. It's expected that caller will pass these events to
	// the PV being recycled.
	Recycle(pvName string, spec *Spec, eventRecorder recyclerclient.RecycleEventRecorder) error
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
	DeviceMountableVolumePlugin
	NewAttacher() (Attacher, error)
	NewDetacher() (Detacher, error)
}

// DeviceMountableVolumePlugin is an extended interface of VolumePlugin and is used
// for volumes that requires mount device to a node before binding to volume to pod.
type DeviceMountableVolumePlugin interface {
	VolumePlugin
	NewDeviceMounter() (DeviceMounter, error)
	NewDeviceUnmounter() (DeviceUnmounter, error)
	GetDeviceMountRefs(deviceMountPath string) ([]string, error)
}

// ExpandableVolumePlugin is an extended interface of VolumePlugin and is used for volumes that can be
// expanded
type ExpandableVolumePlugin interface {
	VolumePlugin
	ExpandVolumeDevice(spec *Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error)
	RequiresFSResize() bool
}

// FSResizableVolumePlugin is an extension of ExpandableVolumePlugin and is used for volumes (flex)
// that require extra steps on nodes for expansion to complete
type FSResizableVolumePlugin interface {
	ExpandableVolumePlugin
	ExpandFS(spec *Spec, devicePath, deviceMountPath string, newSize, oldSize resource.Quantity) error
}

// VolumePluginWithAttachLimits is an extended interface of VolumePlugin that restricts number of
// volumes that can be attached to a node.
type VolumePluginWithAttachLimits interface {
	VolumePlugin
	// Return maximum number of volumes that can be attached to a node for this plugin.
	// The key must be same as string returned by VolumeLimitKey function. The returned
	// map may look like:
	//     - { "storage-limits-aws-ebs": 39 }
	//     - { "storage-limits-gce-pd": 10 }
	// A volume plugin may return error from this function - if it can not be used on a given node or not
	// applicable in given environment (where environment could be cloudprovider or any other dependency)
	// For example - calling this function for EBS volume plugin on a GCE node should
	// result in error.
	// The returned values are stored in node allocatable property and will be used
	// by scheduler to determine how many pods with volumes can be scheduled on given node.
	GetVolumeLimits() (map[string]int64, error)
	// Return volume limit key string to be used in node capacity constraints
	// The key must start with prefix storage-limits-. For example:
	//    - storage-limits-aws-ebs
	//    - storage-limits-csi-cinder
	// The key should respect character limit of ResourceName type
	// This function may be called by kubelet or scheduler to identify node allocatable property
	// which stores volumes limits.
	VolumeLimitKey(spec *Spec) string
}

// BlockVolumePlugin is an extend interface of VolumePlugin and is used for block volumes support.
type BlockVolumePlugin interface {
	VolumePlugin
	// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
	// Ownership of the spec pointer in *not* transferred.
	// - spec: The v1.Volume spec
	// - pod: The enclosing pod
	NewBlockVolumeMapper(spec *Spec, podRef *v1.Pod, opts VolumeOptions) (BlockVolumeMapper, error)
	// NewBlockVolumeUnmapper creates a new volume.BlockVolumeUnmapper from recoverable state.
	// - name: The volume name, as per the v1.Volume spec.
	// - podUID: The UID of the enclosing pod
	NewBlockVolumeUnmapper(name string, podUID types.UID) (BlockVolumeUnmapper, error)
	// ConstructBlockVolumeSpec constructs a volume spec based on the given
	// podUID, volume name and a pod device map path.
	// The spec may have incomplete information due to limited information
	// from input. This function is used by volume manager to reconstruct
	// volume spec by reading the volume directories from disk.
	ConstructBlockVolumeSpec(podUID types.UID, volumeName, mountPath string) (*Spec, error)
}

// VolumeHost is an interface that plugins can use to access the kubelet.
type VolumeHost interface {
	// GetPluginDir returns the absolute path to a directory under which
	// a given plugin may store data.  This directory might not actually
	// exist on disk yet.  For plugin data that is per-pod, see
	// GetPodPluginDir().
	GetPluginDir(pluginName string) string

	// GetVolumeDevicePluginDir returns the absolute path to a directory
	// under which a given plugin may store data.
	// ex. plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/
	GetVolumeDevicePluginDir(pluginName string) string

	// GetPodsDir returns the absolute path to a directory where all the pods
	// information is stored
	GetPodsDir() string

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

	// GetPodVolumeDeviceDir returns the absolute path a directory which
	// represents the named plugin for the given pod.
	// If the specified pod does not exist, the result of this call
	// might not exist.
	// ex. pods/{podUid}/{DefaultKubeletVolumeDevicesDirName}/{escapeQualifiedPluginName}/
	GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string

	// GetKubeClient returns a client interface
	GetKubeClient() clientset.Interface

	// GetCSIClient returns a client interface to csi.storage.k8s.io
	GetCSIClient() csiclientset.Interface

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
	GetMounter(pluginName string) mount.Interface

	// Returns the hostname of the host kubelet is running on
	GetHostName() string

	// Returns host IP or nil in the case of error.
	GetHostIP() (net.IP, error)

	// Returns node allocatable.
	GetNodeAllocatable() (v1.ResourceList, error)

	// Returns a function that returns a secret.
	GetSecretFunc() func(namespace, name string) (*v1.Secret, error)

	// Returns a function that returns a configmap.
	GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error)

	GetServiceAccountTokenFunc() func(namespace, name string, tr *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error)

	DeleteServiceAccountTokenFunc() func(podUID types.UID)

	// Returns an interface that should be used to execute any utilities in volume plugins
	GetExec(pluginName string) mount.Exec

	// Returns the labels on the node
	GetNodeLabels() (map[string]string, error)

	// Returns the name of the node
	GetNodeName() types.NodeName

	// Returns the event recorder of kubelet.
	GetEventRecorder() record.EventRecorder
}

// VolumePluginMgr tracks registered plugins.
type VolumePluginMgr struct {
	mutex         sync.Mutex
	plugins       map[string]VolumePlugin
	prober        DynamicPluginProber
	probedPlugins map[string]VolumePlugin
	Host          VolumeHost
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

// IsKubeletExpandable returns true for volume types that can be expanded only by the node
// and not the controller. Currently Flex volume is the only one in this category since
// it is typically not installed on the controller
func (spec *Spec) IsKubeletExpandable() bool {
	switch {
	case spec.Volume != nil:
		return spec.Volume.FlexVolume != nil
	case spec.PersistentVolume != nil:
		return spec.PersistentVolume.Spec.FlexVolume != nil
	default:
		return false

	}
}

// KubeletExpandablePluginName creates and returns a name for the plugin
// this is used in context on the controller where the plugin lookup fails
// as volume expansion on controller isn't supported, but a plugin name is
// required
func (spec *Spec) KubeletExpandablePluginName() string {
	switch {
	case spec.Volume != nil && spec.Volume.FlexVolume != nil:
		return spec.Volume.FlexVolume.Driver
	case spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FlexVolume != nil:
		return spec.PersistentVolume.Spec.FlexVolume.Driver
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
func (pm *VolumePluginMgr) InitPlugins(plugins []VolumePlugin, prober DynamicPluginProber, host VolumeHost) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	pm.Host = host

	if prober == nil {
		// Use a dummy prober to prevent nil deference.
		pm.prober = &dummyPluginProber{}
	} else {
		pm.prober = prober
	}
	if err := pm.prober.Init(); err != nil {
		// Prober init failure should not affect the initialization of other plugins.
		klog.Errorf("Error initializing dynamic plugin prober: %s", err)
		pm.prober = &dummyPluginProber{}
	}

	if pm.plugins == nil {
		pm.plugins = map[string]VolumePlugin{}
	}
	if pm.probedPlugins == nil {
		pm.probedPlugins = map[string]VolumePlugin{}
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
			klog.Errorf("Failed to load volume plugin %s, error: %s", name, err.Error())
			allErrs = append(allErrs, err)
			continue
		}
		pm.plugins[name] = plugin
		klog.V(1).Infof("Loaded volume plugin %q", name)
	}
	return utilerrors.NewAggregate(allErrs)
}

func (pm *VolumePluginMgr) initProbedPlugin(probedPlugin VolumePlugin) error {
	name := probedPlugin.GetPluginName()
	if errs := validation.IsQualifiedName(name); len(errs) != 0 {
		return fmt.Errorf("volume plugin has invalid name: %q: %s", name, strings.Join(errs, ";"))
	}

	err := probedPlugin.Init(pm.Host)
	if err != nil {
		return fmt.Errorf("Failed to load volume plugin %s, error: %s", name, err.Error())
	}

	klog.V(1).Infof("Loaded volume plugin %q", name)
	return nil
}

// FindPluginBySpec looks for a plugin that can support a given volume
// specification.  If no plugins can support or more than one plugin can
// support it, return error.
func (pm *VolumePluginMgr) FindPluginBySpec(spec *Spec) (VolumePlugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if spec == nil {
		return nil, fmt.Errorf("Could not find plugin because volume spec is nil")
	}

	matchedPluginNames := []string{}
	matches := []VolumePlugin{}
	for k, v := range pm.plugins {
		if v.CanSupport(spec) {
			matchedPluginNames = append(matchedPluginNames, k)
			matches = append(matches, v)
		}
	}

	pm.refreshProbedPlugins()
	for _, plugin := range pm.probedPlugins {
		if plugin.CanSupport(spec) {
			matchedPluginNames = append(matchedPluginNames, plugin.GetPluginName())
			matches = append(matches, plugin)
		}
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matchedPluginNames, ","))
	}
	return matches[0], nil
}

// FindPluginByName fetches a plugin by name or by legacy name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindPluginByName(name string) (VolumePlugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	// Once we can get rid of legacy names we can reduce this to a map lookup.
	matchedPluginNames := []string{}
	matches := []VolumePlugin{}
	for k, v := range pm.plugins {
		if v.GetPluginName() == name {
			matchedPluginNames = append(matchedPluginNames, k)
			matches = append(matches, v)
		}
	}

	pm.refreshProbedPlugins()
	for _, plugin := range pm.probedPlugins {
		if plugin.GetPluginName() == name {
			matchedPluginNames = append(matchedPluginNames, plugin.GetPluginName())
			matches = append(matches, plugin)
		}
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matchedPluginNames, ","))
	}
	return matches[0], nil
}

// Check if probedPlugin cache update is required.
// If it is, initialize all probed plugins and replace the cache with them.
func (pm *VolumePluginMgr) refreshProbedPlugins() {
	events, err := pm.prober.Probe()
	if err != nil {
		klog.Errorf("Error dynamically probing plugins: %s", err)
		return // Use cached plugins upon failure.
	}

	for _, event := range events {
		if event.Op == ProbeAddOrUpdate {
			if err := pm.initProbedPlugin(event.Plugin); err != nil {
				klog.Errorf("Error initializing dynamically probed plugin %s; error: %s",
					event.Plugin.GetPluginName(), err)
				continue
			}
			pm.probedPlugins[event.Plugin.GetPluginName()] = event.Plugin
		} else if event.Op == ProbeRemove {
			// Plugin is not available on ProbeRemove event, only PluginName
			delete(pm.probedPlugins, event.PluginName)
		} else {
			klog.Errorf("Unknown Operation on PluginName: %s.",
				event.Plugin.GetPluginName())
		}
	}
}

// ListVolumePluginWithLimits returns plugins that have volume limits on nodes
func (pm *VolumePluginMgr) ListVolumePluginWithLimits() []VolumePluginWithAttachLimits {
	matchedPlugins := []VolumePluginWithAttachLimits{}
	for _, v := range pm.plugins {
		if plugin, ok := v.(VolumePluginWithAttachLimits); ok {
			matchedPlugins = append(matchedPlugins, plugin)
		}
	}
	return matchedPlugins
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

// FindVolumePluginWithLimitsBySpec returns volume plugin that has a limit on how many
// of them can be attached to a node
func (pm *VolumePluginMgr) FindVolumePluginWithLimitsBySpec(spec *Spec) (VolumePluginWithAttachLimits, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("Could not find volume plugin for spec : %#v", spec)
	}

	if limitedPlugin, ok := volumePlugin.(VolumePluginWithAttachLimits); ok {
		return limitedPlugin, nil
	}
	return nil, fmt.Errorf("no plugin with limits found")
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

// FindAttachablePluginBySpec fetches a persistent volume plugin by spec.
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

// FindDeviceMountablePluginBySpec fetches a persistent volume plugin by spec.
func (pm *VolumePluginMgr) FindDeviceMountablePluginBySpec(spec *Spec) (DeviceMountableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if deviceMountableVolumePlugin, ok := volumePlugin.(DeviceMountableVolumePlugin); ok {
		return deviceMountableVolumePlugin, nil
	}
	return nil, nil
}

// FindDeviceMountablePluginByName fetches a devicemountable volume plugin by name.
func (pm *VolumePluginMgr) FindDeviceMountablePluginByName(name string) (DeviceMountableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if deviceMountableVolumePlugin, ok := volumePlugin.(DeviceMountableVolumePlugin); ok {
		return deviceMountableVolumePlugin, nil
	}
	return nil, nil
}

// FindExpandablePluginBySpec fetches a persistent volume plugin by spec.
func (pm *VolumePluginMgr) FindExpandablePluginBySpec(spec *Spec) (ExpandableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		if spec.IsKubeletExpandable() {
			// for kubelet expandable volumes, return a noop plugin that
			// returns success for expand on the controller
			klog.Warningf("FindExpandablePluginBySpec(%s) -> returning noopExpandableVolumePluginInstance", spec.Name())
			return &noopExpandableVolumePluginInstance{spec}, nil
		}
		klog.Warningf("FindExpandablePluginBySpec(%s) -> err:%v", spec.Name(), err)
		return nil, err
	}

	if expandableVolumePlugin, ok := volumePlugin.(ExpandableVolumePlugin); ok {
		return expandableVolumePlugin, nil
	}
	return nil, nil
}

// FindExpandablePluginBySpec fetches a persistent volume plugin by name.
func (pm *VolumePluginMgr) FindExpandablePluginByName(name string) (ExpandableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}

	if expandableVolumePlugin, ok := volumePlugin.(ExpandableVolumePlugin); ok {
		return expandableVolumePlugin, nil
	}
	return nil, nil
}

// FindMapperPluginBySpec fetches a block volume plugin by spec.
func (pm *VolumePluginMgr) FindMapperPluginBySpec(spec *Spec) (BlockVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}

	if blockVolumePlugin, ok := volumePlugin.(BlockVolumePlugin); ok {
		return blockVolumePlugin, nil
	}
	return nil, nil
}

// FindMapperPluginByName fetches a block volume plugin by name.
func (pm *VolumePluginMgr) FindMapperPluginByName(name string) (BlockVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}

	if blockVolumePlugin, ok := volumePlugin.(BlockVolumePlugin); ok {
		return blockVolumePlugin, nil
	}
	return nil, nil
}

// FindFSResizablePluginBySpec fetches a persistent volume plugin by spec
func (pm *VolumePluginMgr) FindFSResizablePluginBySpec(spec *Spec) (FSResizableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if fsResizablePlugin, ok := volumePlugin.(FSResizableVolumePlugin); ok {
		return fsResizablePlugin, nil
	}
	return nil, nil
}

// FindFSResizablePluginByName fetches a persistent volume plugin by name
func (pm *VolumePluginMgr) FindFSResizablePluginByName(name string) (FSResizableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}

	if fsResizablePlugin, ok := volumePlugin.(FSResizableVolumePlugin); ok {
		return fsResizablePlugin, nil
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
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pv-recycler-",
			Namespace:    metav1.NamespaceDefault,
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
					Image:   "busybox:1.27",
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

// Check validity of recycle pod template
// List of checks:
// - at least one volume is defined in the recycle pod template
// If successful, returns nil
// if unsuccessful, returns an error.
func ValidateRecyclerPodTemplate(pod *v1.Pod) error {
	if len(pod.Spec.Volumes) < 1 {
		return fmt.Errorf("does not contain any volume(s)")
	}
	return nil
}

type dummyPluginProber struct{}

func (*dummyPluginProber) Init() error                  { return nil }
func (*dummyPluginProber) Probe() ([]ProbeEvent, error) { return nil, nil }
