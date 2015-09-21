/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/validation"
)

// VolumeOptions contains option information about a volume.
type VolumeOptions struct {
	// The rootcontext to use when performing mounts for a volume.
	// This is a temporary measure in order to set the rootContext of tmpfs mounts correctly.
	// it will be replaced and expanded on by future SecurityContext work.
	RootContext string

	// The attributes below are required by volume.Creater
	// perhaps CreaterVolumeOptions struct?

	// CapacityMB is the size in MB of a volume.
	CapacityMB int
	// AccessModes of a volume
	AccessModes []api.PersistentVolumeAccessMode
	// Reclamation policy for a persistent volume
	PersistentVolumeReclaimPolicy api.PersistentVolumeReclaimPolicy
}

// VolumePlugin is an interface to volume plugins that can be used on a
// kubernetes node (e.g. by kubelet) to instantiate and manage volumes.
type VolumePlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init(host VolumeHost)

	// Name returns the plugin's name.  Plugins should use namespaced names
	// such as "example.com/volume".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	Name() string

	// CanSupport tests whether the plugin supports a given volume
	// specification from the API.  The spec pointer should be considered
	// const.
	CanSupport(spec *Spec) bool

	// NewBuilder creates a new volume.Builder from an API specification.
	// Ownership of the spec pointer in *not* transferred.
	// - spec: The api.Volume spec
	// - pod: The enclosing pod
	NewBuilder(spec *Spec, podRef *api.Pod, opts VolumeOptions) (Builder, error)

	// NewCleaner creates a new volume.Cleaner from recoverable state.
	// - name: The volume name, as per the api.Volume spec.
	// - podUID: The UID of the enclosing pod
	NewCleaner(name string, podUID types.UID) (Cleaner, error)
}

// PersistentVolumePlugin is an extended interface of VolumePlugin and is used
// by volumes that want to provide long term persistence of data
type PersistentVolumePlugin interface {
	VolumePlugin
	// GetAccessModes describes the ways a given volume can be accessed/mounted.
	GetAccessModes() []api.PersistentVolumeAccessMode
}

// RecyclableVolumePlugin is an extended interface of VolumePlugin and is used
// by persistent volumes that want to be recycled before being made available again to new claims
type RecyclableVolumePlugin interface {
	VolumePlugin
	// NewRecycler creates a new volume.Recycler which knows how to reclaim this resource
	// after the volume's release from a PersistentVolumeClaim
	NewRecycler(spec *Spec) (Recycler, error)
}

// DeletableVolumePlugin is an extended interface of VolumePlugin and is used by persistent volumes that want
// to be deleted from the cluster after their release from a PersistentVolumeClaim.
type DeletableVolumePlugin interface {
	VolumePlugin
	// NewDeleter creates a new volume.Deleter which knows how to delete this resource
	// in accordance with the underlying storage provider after the volume's release from a claim
	NewDeleter(spec *Spec) (Deleter, error)
}

// CreatableVolumePlugin is an extended interface of VolumePlugin and is used to create volumes for the cluster.
type CreatableVolumePlugin interface {
	VolumePlugin
	// NewCreater creates a new volume.Creater which knows how to create PersistentVolumes in accordance with
	// the plugin's underlying storage provider
	NewCreater(options VolumeOptions) (Creater, error)
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
	GetKubeClient() client.Interface

	// NewWrapperBuilder finds an appropriate plugin with which to handle
	// the provided spec.  This is used to implement volume plugins which
	// "wrap" other plugins.  For example, the "secret" volume is
	// implemented in terms of the "emptyDir" volume.
	NewWrapperBuilder(spec *Spec, pod *api.Pod, opts VolumeOptions) (Builder, error)

	// NewWrapperCleaner finds an appropriate plugin with which to handle
	// the provided spec.  See comments on NewWrapperBuilder for more
	// context.
	NewWrapperCleaner(spec *Spec, podUID types.UID) (Cleaner, error)

	// Get cloud provider from kubelet.
	GetCloudProvider() cloudprovider.Interface

	// Get mounter interface.
	GetMounter() mount.Interface

	// Get writer interface for writing data to disk.
	GetWriter() io.Writer
}

// VolumePluginMgr tracks registered plugins.
type VolumePluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]VolumePlugin
}

// Spec is an internal representation of a volume.  All API volume types translate to Spec.
type Spec struct {
	Volume           *api.Volume
	PersistentVolume *api.PersistentVolume
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

// VolumeConfig is how volume plugins receive configuration.  An instance specific to the plugin will be passed to
// the plugin's ProbeVolumePlugins(config) func.  Reasonable defaults will be provided by the binary hosting
// the plugins while allowing override of those default values.  Those config values are then set to an instance of
// VolumeConfig and passed to the plugin.
//
// Values in VolumeConfig are intended to be relevant to several plugins, but not necessarily all plugins.  The
// preference is to leverage strong typing in this struct.  All config items must have a descriptive but non-specific
// name (i.e, RecyclerMinimumTimeout is OK but RecyclerMinimumTimeoutForNFS is !OK).  An instance of config will be
// given directly to the plugin, so config names specific to plugins are unneeded and wrongly expose plugins
// in this VolumeConfig struct.
//
// OtherAttributes is a map of string values intended for one-off configuration of a plugin or config that is only
// relevant to a single plugin.  All values are passed by string and require interpretation by the plugin.
// Passing config as strings is the least desirable option but can be used for truly one-off configuration.
// The binary should still use strong typing for this value when binding CLI values before they are passed as strings
// in OtherAttributes.
type VolumeConfig struct {
	// RecyclerPodTemplate is pod template that understands how to scrub clean a persistent volume after its release.
	// The template is used by plugins which override specific properties of the pod in accordance with that plugin.
	// See NewPersistentVolumeRecyclerPodTemplate for the properties that are expected to be overridden.
	RecyclerPodTemplate *api.Pod

	// RecyclerMinimumTimeout is the minimum amount of time in seconds for the recycler pod's ActiveDeadlineSeconds attribute.
	// Added to the minimum timeout is the increment per Gi of capacity.
	RecyclerMinimumTimeout int

	// RecyclerTimeoutIncrement is the number of seconds added to the recycler pod's ActiveDeadlineSeconds for each
	// Gi of capacity in the persistent volume.
	// Example: 5Gi volume x 30s increment = 150s + 30s minimum = 180s ActiveDeadlineSeconds for recycler pod
	RecyclerTimeoutIncrement int

	// OtherAttributes stores config as strings.  These strings are opaque to the system and only understood by the binary
	// hosting the plugin and the plugin itself.
	OtherAttributes map[string]string
}

// NewSpecFromVolume creates an Spec from an api.Volume
func NewSpecFromVolume(vs *api.Volume) *Spec {
	return &Spec{
		Volume: vs,
	}
}

// NewSpecFromPersistentVolume creates an Spec from an api.PersistentVolume
func NewSpecFromPersistentVolume(pv *api.PersistentVolume, readOnly bool) *Spec {
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
		name := plugin.Name()
		if !validation.IsQualifiedName(name) {
			allErrs = append(allErrs, fmt.Errorf("volume plugin has invalid name: %#v", plugin))
			continue
		}

		if _, found := pm.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("volume plugin %q was registered more than once", name))
			continue
		}
		plugin.Init(host)
		pm.plugins[name] = plugin
		glog.V(1).Infof("Loaded volume plugin %q", name)
	}
	return errors.NewAggregate(allErrs)
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
		if v.Name() == name {
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

// FindPersistentPluginBySpec looks for a persistent volume plugin that can support a given volume
// specification.  If no plugin is found, return an error
func (pm *VolumePluginMgr) FindPersistentPluginBySpec(spec *Spec) (PersistentVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("Could not find volume plugin for spec: %+v", spec)
	}
	if persistentVolumePlugin, ok := volumePlugin.(PersistentVolumePlugin); ok {
		return persistentVolumePlugin, nil
	}
	return nil, fmt.Errorf("no persistent volume plugin matched")
}

// FindPersistentPluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
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

// FindRecyclablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
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

// FindDeletablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
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

// FindCreatablePluginBySpec fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindCreatablePluginBySpec(spec *Spec) (CreatableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if creatableVolumePlugin, ok := volumePlugin.(CreatableVolumePlugin); ok {
		return creatableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no creatable volume plugin matched")
}

// NewPersistentVolumeRecyclerPodTemplate creates a template for a recycler pod.  By default, a recycler pod simply runs
// "rm -rf" on a volume and tests for emptiness.  Most attributes of the template will be correct for most
// plugin implementations.  The following attributes can be overridden per plugin via configuration:
//
// 1.  pod.Spec.Volumes[0].VolumeSource must be overridden.  Recycler implementations without a valid VolumeSource will fail.
// 2.  pod.GenerateName helps distinguish recycler pods by name.  Recommended. Default is "pv-recycler-".
// 3.  pod.Spec.ActiveDeadlineSeconds gives the recycler pod a maximum timeout before failing.  Recommended.  Default is 60 seconds.
//
// See HostPath and NFS for working recycler examples
func NewPersistentVolumeRecyclerPodTemplate() *api.Pod {
	timeout := int64(60)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-recycler-",
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			ActiveDeadlineSeconds: &timeout,
			RestartPolicy:         api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: "vol",
					// IMPORTANT!  All plugins using this template MUST override pod.Spec.Volumes[0].VolumeSource
					// Recycler implementations without a valid VolumeSource will fail.
					VolumeSource: api.VolumeSource{},
				},
			},
			Containers: []api.Container{
				{
					Name:    "pv-recycler",
					Image:   "gcr.io/google_containers/busybox",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "test -e /scrub && echo $(date) > /scrub/trash.txt && rm -rf /scrub/* /scrub/.* && test -z \"$(ls -A /scrub)\" || exit 1"},
					VolumeMounts: []api.VolumeMount{
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
