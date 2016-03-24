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

package host_path

import (
	"fmt"
	"os"
	"regexp"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
// The volumeConfig arg provides the ability to configure volume behavior.  It is implemented as a pointer to allow nils.
// The hostPathPlugin is used to store the volumeConfig and give it, when needed, to the func that creates HostPath Recyclers.
// Tests that exercise recycling should not use this func but instead use ProbeRecyclablePlugins() to override default behavior.
func ProbeVolumePlugins(volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&hostPathPlugin{
			host:               nil,
			newRecyclerFunc:    newRecycler,
			newDeleterFunc:     newDeleter,
			newProvisionerFunc: newProvisioner,
			config:             volumeConfig,
		},
	}
}

func ProbeRecyclableVolumePlugins(recyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error), volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&hostPathPlugin{
			host:               nil,
			newRecyclerFunc:    recyclerFunc,
			newProvisionerFunc: newProvisioner,
			config:             volumeConfig,
		},
	}
}

type hostPathPlugin struct {
	host volume.VolumeHost
	// decouple creating Recyclers/Deleters/Provisioners by deferring to a function.  Allows for easier testing.
	newRecyclerFunc    func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	newDeleterFunc     func(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error)
	newProvisionerFunc func(options volume.VolumeOptions, host volume.VolumeHost) (volume.Provisioner, error)
	config             volume.VolumeConfig
}

var _ volume.VolumePlugin = &hostPathPlugin{}
var _ volume.PersistentVolumePlugin = &hostPathPlugin{}
var _ volume.RecyclableVolumePlugin = &hostPathPlugin{}
var _ volume.DeletableVolumePlugin = &hostPathPlugin{}
var _ volume.ProvisionableVolumePlugin = &hostPathPlugin{}

const (
	hostPathPluginName = "kubernetes.io/host-path"
)

func (plugin *hostPathPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *hostPathPlugin) Name() string {
	return hostPathPluginName
}

func (plugin *hostPathPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath != nil) ||
		(spec.Volume != nil && spec.Volume.HostPath != nil)
}

func (plugin *hostPathPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *hostPathPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	if spec.Volume != nil && spec.Volume.HostPath != nil {
		path := spec.Volume.HostPath.Path
		return &hostPathBuilder{
			hostPath: &hostPath{path: path},
			readOnly: false,
		}, nil
	} else {
		path := spec.PersistentVolume.Spec.HostPath.Path
		return &hostPathBuilder{
			hostPath: &hostPath{path: path},
			readOnly: spec.ReadOnly,
		}, nil
	}
}

func (plugin *hostPathPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &hostPathCleaner{&hostPath{
		path: "",
	}}, nil
}

func (plugin *hostPathPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.host, plugin.config)
}

func (plugin *hostPathPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterFunc(spec, plugin.host)
}

func (plugin *hostPathPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerFunc(options, plugin.host)
}

func newRecycler(spec *volume.Spec, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	path := spec.PersistentVolume.Spec.HostPath.Path
	return &hostPathRecycler{
		name:    spec.Name(),
		path:    path,
		host:    host,
		config:  config,
		timeout: volume.CalculateTimeoutForVolume(config.RecyclerMinimumTimeout, config.RecyclerTimeoutIncrement, spec.PersistentVolume),
	}, nil
}

func newDeleter(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	path := spec.PersistentVolume.Spec.HostPath.Path
	return &hostPathDeleter{name: spec.Name(), path: path, host: host}, nil
}

func newProvisioner(options volume.VolumeOptions, host volume.VolumeHost) (volume.Provisioner, error) {
	return &hostPathProvisioner{options: options, host: host}, nil
}

// HostPath volumes represent a bare host file or directory mount.
// The direct at the specified path will be directly exposed to the container.
type hostPath struct {
	path string
	volume.MetricsNil
}

func (hp *hostPath) GetPath() string {
	return hp.path
}

type hostPathBuilder struct {
	*hostPath
	readOnly bool
}

var _ volume.Builder = &hostPathBuilder{}

func (b *hostPathBuilder) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// SetUp does nothing.
func (b *hostPathBuilder) SetUp(fsGroup *int64) error {
	return nil
}

// SetUpAt does not make sense for host paths - probably programmer error.
func (b *hostPathBuilder) SetUpAt(dir string, fsGroup *int64) error {
	return fmt.Errorf("SetUpAt() does not make sense for host paths")
}

func (b *hostPathBuilder) GetPath() string {
	return b.path
}

type hostPathCleaner struct {
	*hostPath
}

var _ volume.Cleaner = &hostPathCleaner{}

// TearDown does nothing.
func (c *hostPathCleaner) TearDown() error {
	return nil
}

// TearDownAt does not make sense for host paths - probably programmer error.
func (c *hostPathCleaner) TearDownAt(dir string) error {
	return fmt.Errorf("TearDownAt() does not make sense for host paths")
}

// hostPathRecycler implements a Recycler for the HostPath plugin
// This implementation is meant for testing only and only works in a single node cluster
type hostPathRecycler struct {
	name    string
	path    string
	host    volume.VolumeHost
	config  volume.VolumeConfig
	timeout int64
	volume.MetricsNil
}

func (r *hostPathRecycler) GetPath() string {
	return r.path
}

// Recycle recycles/scrubs clean a HostPath volume.
// Recycle blocks until the pod has completed or any error occurs.
// HostPath recycling only works in single node clusters and is meant for testing purposes only.
func (r *hostPathRecycler) Recycle() error {
	pod := r.config.RecyclerPodTemplate
	// overrides
	pod.Spec.ActiveDeadlineSeconds = &r.timeout
	pod.GenerateName = "pv-recycler-hostpath-"
	pod.Spec.Volumes[0].VolumeSource = api.VolumeSource{
		HostPath: &api.HostPathVolumeSource{
			Path: r.path,
		},
	}
	return volume.RecycleVolumeByWatchingPodUntilCompletion(pod, r.host.GetKubeClient())
}

// hostPathProvisioner implements a Provisioner for the HostPath plugin
// This implementation is meant for testing only and only works in a single node cluster.
type hostPathProvisioner struct {
	host    volume.VolumeHost
	options volume.VolumeOptions
}

// Create for hostPath simply creates a local /tmp/hostpath_pv/%s directory as a new PersistentVolume.
// This Provisioner is meant for development and testing only and WILL NOT WORK in a multi-node cluster.
func (r *hostPathProvisioner) Provision(pv *api.PersistentVolume) error {
	if pv.Spec.HostPath == nil {
		return fmt.Errorf("pv.Spec.HostPath cannot be nil")
	}
	return os.MkdirAll(pv.Spec.HostPath.Path, 0750)
}

func (r *hostPathProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	fullpath := fmt.Sprintf("/tmp/hostpath_pv/%s", util.NewUUID())
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-hostpath-",
			Annotations: map[string]string{
				"kubernetes.io/createdby": "hostpath-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: r.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   r.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): r.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fullpath,
				},
			},
		},
	}, nil
}

// hostPathDeleter deletes a hostPath PV from the cluster.
// This deleter only works on a single host cluster and is for testing purposes only.
type hostPathDeleter struct {
	name string
	path string
	host volume.VolumeHost
	volume.MetricsNil
}

func (r *hostPathDeleter) GetPath() string {
	return r.path
}

// Delete for hostPath removes the local directory so long as it is beneath /tmp/*.
// THIS IS FOR TESTING AND LOCAL DEVELOPMENT ONLY!  This message should scare you away from using
// this deleter for anything other than development and testing.
func (r *hostPathDeleter) Delete() error {
	regexp := regexp.MustCompile("/tmp/.+")
	if !regexp.MatchString(r.GetPath()) {
		return fmt.Errorf("host_path deleter only supports /tmp/.+ but received provided %s", r.GetPath())
	}
	return os.RemoveAll(r.GetPath())
}
