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

package local

import (
	"fmt"
	"os"
	"regexp"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
// The volumeConfig arg provides the ability to configure volume behavior.  It is implemented as a pointer to allow nils.
// The localPlugin is used to store the volumeConfig and give it, when needed, to the func that Recycles.
// Tests that exercise recycling should not use this func but instead use ProbeRecyclablePlugins() to override default behavior.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&localPlugin{
			host:   nil,
			config: volume.VolumeConfig{},
		},
	}
}

type localPlugin struct {
	host   volume.VolumeHost
	config volume.VolumeConfig
}

var _ volume.VolumePlugin = &localPlugin{}
var _ volume.PersistentVolumePlugin = &localPlugin{}
var _ volume.DeletableVolumePlugin = &localPlugin{}
var _ volume.RecyclableVolumePlugin = nil
var _ volume.ProvisionableVolumePlugin = nil

const (
	localPluginName = "kubernetes.io/local"
)

func (plugin *localPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *localPlugin) GetPluginName() string {
	return localPluginName
}

func (plugin *localPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.Path, nil
}

func (plugin *localPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LocalStorage != nil)
}

func (plugin *localPlugin) RequiresRemount() bool {
	return false
}

func (plugin *localPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *localPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *localPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

func (plugin *localPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	localVolumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	return &localMounter{
		local:    &local{path: localVolumeSource.Path},
		readOnly: readOnly,
	}, nil
}

func (plugin *localPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &localUnmounter{&local{
		path: "",
	}}, nil
}

func (plugin *localPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return newDeleter(spec, plugin.host)
}

func (plugin *localPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	localVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			LocalStorage: &v1.LocalStorageVolumeSource{
				Path:     volumeName,
				NodeName: "",
			},
		},
	}
	return volume.NewSpecFromVolume(localVolume), nil
}

func newDeleter(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LocalStorage == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Local is nil")
	}
	path := spec.PersistentVolume.Spec.LocalStorage.Path
	return &localDeleter{name: spec.Name(), path: path, host: host}, nil
}

// Local volumes represent a bare host file or directory mount.
// The direct at the specified path will be directly exposed to the container.
type local struct {
	path string
	volume.MetricsNil
}

func (hp *local) GetPath() string {
	return hp.path
}

type localMounter struct {
	*local
	readOnly bool
}

var _ volume.Mounter = &localMounter{}

func (b *localMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *localMounter) CanMount() error {
	return nil
}

// SetUp does nothing.
func (b *localMounter) SetUp(fsGroup *int64) error {
	return nil
}

// SetUpAt does not make sense for host paths - probably programmer error.
func (b *localMounter) SetUpAt(dir string, fsGroup *int64) error {
	return fmt.Errorf("SetUpAt() does not make sense for host paths")
}

func (b *localMounter) GetPath() string {
	return b.path
}

type localUnmounter struct {
	*local
}

var _ volume.Unmounter = &localUnmounter{}

// TearDown does nothing.
func (c *localUnmounter) TearDown() error {
	return nil
}

// TearDownAt does not make sense for host paths - probably programmer error.
func (c *localUnmounter) TearDownAt(dir string) error {
	return fmt.Errorf("TearDownAt() does not make sense for host paths")
}

// localDeleter deletes a local PV from the cluster.
// This deleter only works on a single host cluster and is for testing purposes only.
type localDeleter struct {
	name string
	path string
	host volume.VolumeHost
	volume.MetricsNil
}

func (r *localDeleter) GetPath() string {
	return r.path
}

// Delete for local removes the local directory so long as it is beneath /tmp/*.
// THIS IS FOR TESTING AND LOCAL DEVELOPMENT ONLY!  This message should scare you away from using
// this deleter for anything other than development and testing.
func (r *localDeleter) Delete() error {
	regexp := regexp.MustCompile("/tmp/.+")
	if !regexp.MatchString(r.GetPath()) {
		return fmt.Errorf("host_path deleter only supports /tmp/.+ but received provided %s", r.GetPath())
	}
	return os.RemoveAll(r.GetPath())
}

func getVolumeSource(
	spec *volume.Spec) (*v1.LocalStorageVolumeSource, bool, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LocalStorage != nil {
		return spec.PersistentVolume.Spec.LocalStorage, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an Local volume type")
}
