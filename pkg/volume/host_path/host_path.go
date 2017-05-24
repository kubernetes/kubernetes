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

package host_path

import (
	"fmt"
	"os"
	"regexp"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
// The volumeConfig arg provides the ability to configure volume behavior.  It is implemented as a pointer to allow nils.
// The hostPathPlugin is used to store the volumeConfig and give it, when needed, to the func that creates HostPath Recyclers.
// Tests that exercise recycling should not use this func but instead use ProbeRecyclablePlugins() to override default behavior.
func ProbeVolumePlugins(volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&hostPathPlugin{
			host:   nil,
			config: volumeConfig,
		},
	}
}

type hostPathPlugin struct {
	host   volume.VolumeHost
	config volume.VolumeConfig
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

func (plugin *hostPathPlugin) GetPluginName() string {
	return hostPathPluginName
}

func (plugin *hostPathPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.Path, nil
}

func (plugin *hostPathPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath != nil) ||
		(spec.Volume != nil && spec.Volume.HostPath != nil)
}

func (plugin *hostPathPlugin) RequiresRemount() bool {
	return false
}

func (plugin *hostPathPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *hostPathPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	hostPathVolumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	return &hostPathMounter{
		hostPath: &hostPath{path: hostPathVolumeSource.Path},
		readOnly: readOnly,
	}, nil
}

func (plugin *hostPathPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &hostPathUnmounter{&hostPath{
		path: "",
	}}, nil
}

func (plugin *hostPathPlugin) NewRecycler(pvName string, spec *volume.Spec, eventRecorder volume.RecycleEventRecorder) (volume.Recycler, error) {
	return newRecycler(pvName, spec, eventRecorder, plugin.host, plugin.config)
}

func (plugin *hostPathPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return newDeleter(spec, plugin.host)
}

func (plugin *hostPathPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if !plugin.config.ProvisioningEnabled {
		return nil, fmt.Errorf("Provisioning in volume plugin %q is disabled", plugin.GetPluginName())
	}
	return newProvisioner(options, plugin.host, plugin)
}

func (plugin *hostPathPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	hostPathVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{
				Path: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(hostPathVolume), nil
}

func newRecycler(pvName string, spec *volume.Spec, eventRecorder volume.RecycleEventRecorder, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	path := spec.PersistentVolume.Spec.HostPath.Path
	return &hostPathRecycler{
		name:          spec.Name(),
		path:          path,
		host:          host,
		config:        config,
		timeout:       volume.CalculateTimeoutForVolume(config.RecyclerMinimumTimeout, config.RecyclerTimeoutIncrement, spec.PersistentVolume),
		pvName:        pvName,
		eventRecorder: eventRecorder,
	}, nil
}

func newDeleter(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.HostPath is nil")
	}
	path := spec.PersistentVolume.Spec.HostPath.Path
	return &hostPathDeleter{name: spec.Name(), path: path, host: host}, nil
}

func newProvisioner(options volume.VolumeOptions, host volume.VolumeHost, plugin *hostPathPlugin) (volume.Provisioner, error) {
	return &hostPathProvisioner{options: options, host: host, plugin: plugin}, nil
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

type hostPathMounter struct {
	*hostPath
	readOnly bool
}

var _ volume.Mounter = &hostPathMounter{}

func (b *hostPathMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *hostPathMounter) CanMount() error {
	return nil
}

// SetUp does nothing.
func (b *hostPathMounter) SetUp(fsGroup *int64) error {
	return nil
}

// SetUpAt does not make sense for host paths - probably programmer error.
func (b *hostPathMounter) SetUpAt(dir string, fsGroup *int64) error {
	return fmt.Errorf("SetUpAt() does not make sense for host paths")
}

func (b *hostPathMounter) GetPath() string {
	return b.path
}

type hostPathUnmounter struct {
	*hostPath
}

var _ volume.Unmounter = &hostPathUnmounter{}

// TearDown does nothing.
func (c *hostPathUnmounter) TearDown() error {
	return nil
}

// TearDownAt does not make sense for host paths - probably programmer error.
func (c *hostPathUnmounter) TearDownAt(dir string) error {
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
	pvName        string
	eventRecorder volume.RecycleEventRecorder
}

func (r *hostPathRecycler) GetPath() string {
	return r.path
}

// Recycle recycles/scrubs clean a HostPath volume.
// Recycle blocks until the pod has completed or any error occurs.
// HostPath recycling only works in single node clusters and is meant for testing purposes only.
func (r *hostPathRecycler) Recycle() error {
	templateClone, err := conversion.NewCloner().DeepCopy(r.config.RecyclerPodTemplate)
	if err != nil {
		return err
	}
	pod := templateClone.(*api.Pod)
	// overrides
	pod.Spec.ActiveDeadlineSeconds = &r.timeout
	pod.Spec.Volumes[0].VolumeSource = api.VolumeSource{
		HostPath: &api.HostPathVolumeSource{
			Path: r.path,
		},
	}
	return volume.RecycleVolumeByWatchingPodUntilCompletion(r.pvName, pod, r.host.GetKubeClient(), r.eventRecorder)
}

// hostPathProvisioner implements a Provisioner for the HostPath plugin
// This implementation is meant for testing only and only works in a single node cluster.
type hostPathProvisioner struct {
	host    volume.VolumeHost
	options volume.VolumeOptions
	plugin  *hostPathPlugin
}

// Create for hostPath simply creates a local /tmp/hostpath_pv/%s directory as a new PersistentVolume.
// This Provisioner is meant for development and testing only and WILL NOT WORK in a multi-node cluster.
func (r *hostPathProvisioner) Provision() (*api.PersistentVolume, error) {
	fullpath := fmt.Sprintf("/tmp/hostpath_pv/%s", uuid.NewUUID())

	capacity := r.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: r.options.PVName,
			Annotations: map[string]string{
				"kubernetes.io/createdby": "hostpath-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: r.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   r.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fullpath,
				},
			},
		},
	}
	if len(r.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = r.plugin.GetAccessModes()
	}

	return pv, os.MkdirAll(pv.Spec.HostPath.Path, 0750)
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

func getVolumeSource(
	spec *volume.Spec) (*api.HostPathVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.HostPath != nil {
		return spec.Volume.HostPath, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.HostPath != nil {
		return spec.PersistentVolume.Spec.HostPath, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an HostPath volume type")
}
