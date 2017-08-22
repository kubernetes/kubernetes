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

package scaleio

import (
	"errors"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	sioName           = "scaleio"
	sioPluginName     = "kubernetes.io/scaleio"
	sioConfigFileName = "sioconf.dat"
)

type sioPlugin struct {
	host      volume.VolumeHost
	mounter   mount.Interface
	volumeMtx keymutex.KeyMutex
}

func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &sioPlugin{
		host: nil,
	}
	return []volume.VolumePlugin{p}
}

// *******************
// VolumePlugin Impl
// *******************
var _ volume.VolumePlugin = &sioPlugin{}

func (p *sioPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	p.mounter = host.GetMounter()
	p.volumeMtx = keymutex.NewKeyMutex()
	return nil
}

func (p *sioPlugin) GetPluginName() string {
	return sioPluginName
}

func (p *sioPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	source, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		return "", err
	}

	return source.VolumeName, nil
}

func (p *sioPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.ScaleIO != nil) ||
		(spec.Volume != nil && spec.Volume.ScaleIO != nil)
}

func (p *sioPlugin) RequiresRemount() bool {
	return false
}

func (p *sioPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {
	sioSource, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		glog.Error(log("failed to extract ScaleIOVolumeSource from spec: %v", err))
		return nil, err
	}

	return &sioVolume{
		pod:         pod,
		spec:        spec,
		source:      sioSource,
		namespace:   pod.Namespace,
		volSpecName: spec.Name(),
		volName:     sioSource.VolumeName,
		podUID:      pod.UID,
		readOnly:    sioSource.ReadOnly,
		fsType:      sioSource.FSType,
		plugin:      p,
	}, nil
}

// NewUnmounter creates a representation of the volume to unmount
// The specName param can be used to carry the namespace value (if needed) using format:
// specName = [<namespace>nsSep]<somevalue> where the specname is pre-pended with the namespace
func (p *sioPlugin) NewUnmounter(specName string, podUID types.UID) (volume.Unmounter, error) {
	glog.V(4).Info(log("Unmounter for %s", specName))

	return &sioVolume{
		podUID:      podUID,
		volSpecName: specName,
		plugin:      p,
	}, nil
}

func (p *sioPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	sioVol := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(sioVol), nil
}

// SupportsMountOption returns true if volume plugins supports Mount options
// Specifying mount options in a volume plugin that doesn't support
// user specified mount options will result in error creating persistent volumes
func (p *sioPlugin) SupportsMountOption() bool {
	return false
}

// SupportsBulkVolumeVerification checks if volume plugin type is capable
// of enabling bulk polling of all nodes. This can speed up verification of
// attached volumes by quite a bit, but underlying pluging must support it.
func (p *sioPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

//******************************
// PersistentVolumePlugin Impl
// *****************************
var _ volume.PersistentVolumePlugin = &sioPlugin{}

func (p *sioPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

// ***************************
// DeletableVolumePlugin Impl
//****************************
var _ volume.DeletableVolumePlugin = &sioPlugin{}

func (p *sioPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	sioSource, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		glog.Error(log("deleter failed to extract source from spec: %v", err))
		return nil, err
	}

	namespace := spec.PersistentVolume.Spec.ClaimRef.Namespace

	return &sioVolume{
		spec:        spec,
		source:      sioSource,
		namespace:   namespace,
		volSpecName: spec.Name(),
		volName:     sioSource.VolumeName,
		plugin:      p,
		readOnly:    sioSource.ReadOnly,
	}, nil
}

// *********************************
// ProvisionableVolumePlugin Impl
// *********************************
var _ volume.ProvisionableVolumePlugin = &sioPlugin{}

func (p *sioPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	glog.V(4).Info(log("creating Provisioner"))

	configData := options.Parameters
	if configData == nil {
		glog.Error(log("provisioner missing parameters, unable to continue"))
		return nil, errors.New("option parameters missing")
	}

	namespace := options.PVC.Namespace

	return &sioVolume{
		configData:  configData,
		plugin:      p,
		options:     options,
		namespace:   namespace,
		volSpecName: options.PVName,
	}, nil
}
