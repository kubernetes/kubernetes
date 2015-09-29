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
	"k8s.io/kubernetes/pkg/api/resource"
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
			host:            nil,
			newRecyclerFunc: newRecycler,
			newDeleterFunc:  newDeleter,
			newCreaterFunc:  newCreater,
			config:          volumeConfig,
		},
	}
}

func ProbeRecyclableVolumePlugins(recyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error), volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&hostPathPlugin{
			host:            nil,
			newRecyclerFunc: recyclerFunc,
			newCreaterFunc:  newCreater,
			config:          volumeConfig,
		},
	}
}

type hostPathPlugin struct {
	host volume.VolumeHost
	// decouple creating Recyclers/Deleters/Creaters by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	newDeleterFunc  func(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error)
	newCreaterFunc  func(options volume.VolumeOptions, host volume.VolumeHost) (volume.Creater, error)
	config          volume.VolumeConfig
}

var _ volume.VolumePlugin = &hostPathPlugin{}
var _ volume.PersistentVolumePlugin = &hostPathPlugin{}
var _ volume.RecyclableVolumePlugin = &hostPathPlugin{}
var _ volume.DeletableVolumePlugin = &hostPathPlugin{}
var _ volume.CreatableVolumePlugin = &hostPathPlugin{}

const (
	hostPathPluginName = "kubernetes.io/host-path"
)

func (plugin *hostPathPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
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
		return &hostPathBuilder{
			hostPath: &hostPath{path: spec.Volume.HostPath.Path},
			readOnly: false,
		}, nil
	} else {
		return &hostPathBuilder{
			hostPath: &hostPath{path: spec.PersistentVolume.Spec.HostPath.Path},
			readOnly: spec.ReadOnly,
		}, nil
	}
}

func (plugin *hostPathPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &hostPathCleaner{&hostPath{""}}, nil
}

func (plugin *hostPathPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.host, plugin.config)
}

func (plugin *hostPathPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterFunc(spec, plugin.host)
}

func (plugin *hostPathPlugin) NewCreater(options volume.VolumeOptions) (volume.Creater, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newCreaterFunc(options, plugin.host)
}

func newRecycler(spec *volume.Spec, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	return &hostPathRecycler{
		name:    spec.Name(),
		path:    spec.PersistentVolume.Spec.HostPath.Path,
		host:    host,
		config:  config,
		timeout: volume.CalculateTimeoutForVolume(config.RecyclerMinimumTimeout, config.RecyclerTimeoutIncrement, spec.PersistentVolume),
	}, nil
}

func newDeleter(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	return &hostPathDeleter{spec.Name(), spec.PersistentVolume.Spec.HostPath.Path, host}, nil
}

func newCreater(options volume.VolumeOptions, host volume.VolumeHost) (volume.Creater, error) {
	return &hostPathCreater{options: options, host: host}, nil
}

// HostPath volumes represent a bare host file or directory mount.
// The direct at the specified path will be directly exposed to the container.
type hostPath struct {
	path string
}

func (hp *hostPath) GetPath() string {
	return hp.path
}

type hostPathBuilder struct {
	*hostPath
	readOnly bool
}

var _ volume.Builder = &hostPathBuilder{}

// SetUp does nothing.
func (b *hostPathBuilder) SetUp() error {
	return nil
}

// SetUpAt does not make sense for host paths - probably programmer error.
func (b *hostPathBuilder) SetUpAt(dir string) error {
	return fmt.Errorf("SetUpAt() does not make sense for host paths")
}

func (b *hostPathBuilder) IsReadOnly() bool {
	return b.readOnly
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

// hostPathRecycler implements a dynamic provisioning Recycler for the HostPath plugin
// This implementation is meant for testing only and only works in a single node cluster
type hostPathRecycler struct {
	name    string
	path    string
	host    volume.VolumeHost
	config  volume.VolumeConfig
	timeout int64
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

// hostPathCreater implements a dynamic provisioning Creater for the HostPath plugin
// This implementation is meant for testing only and only works in a single node cluster.
type hostPathCreater struct {
	host    volume.VolumeHost
	options volume.VolumeOptions
}

// Create for hostPath simply creates a local /tmp/hostpath_pv/%s directory as a new PersistentVolume.
// This Creater is meant for development and testing only and WILL NOT WORK in a multi-node cluster.
func (r *hostPathCreater) Create() (*api.PersistentVolume, error) {
	fullpath := fmt.Sprintf("/tmp/hostpath_pv/%s", util.NewUUID())
	err := os.MkdirAll(fullpath, 0750)
	if err != nil {
		return nil, err
	}

	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-hostpath-",
			Labels: map[string]string{
				"createdby": "hostpath dynamic provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: r.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   r.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dMi", r.options.CapacityMB)),
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
