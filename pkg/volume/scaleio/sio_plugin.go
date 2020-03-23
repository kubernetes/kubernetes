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

	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/utils/keymutex"
)

const (
	sioPluginName     = "kubernetes.io/scaleio"
	sioConfigFileName = "sioconf.dat"
)

type sioPlugin struct {
	host      volume.VolumeHost
	volumeMtx keymutex.KeyMutex
}

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
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
	p.volumeMtx = keymutex.NewHashed(0)
	return nil
}

func (p *sioPlugin) GetPluginName() string {
	return sioPluginName
}

func (p *sioPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	attribs, err := getVolumeSourceAttribs(spec)
	if err != nil {
		return "", err
	}
	return attribs.volName, nil
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

	// extract source info from either ScaleIOVolumeSource or ScaleIOPersistentVolumeSource type
	attribs, err := getVolumeSourceAttribs(spec)
	if err != nil {
		return nil, errors.New(log("mounter failed to extract volume attributes from spec: %v", err))
	}

	secretName, secretNS, err := getSecretAndNamespaceFromSpec(spec, pod)
	if err != nil {
		return nil, errors.New(log("failed to get secret name or secretNamespace: %v", err))
	}

	return &sioVolume{
		pod:             pod,
		spec:            spec,
		secretName:      secretName,
		secretNamespace: secretNS,
		volSpecName:     spec.Name(),
		volName:         attribs.volName,
		podUID:          pod.UID,
		readOnly:        attribs.readOnly,
		fsType:          attribs.fsType,
		plugin:          p,
	}, nil
}

// NewUnmounter creates a representation of the volume to unmount
func (p *sioPlugin) NewUnmounter(specName string, podUID types.UID) (volume.Unmounter, error) {
	klog.V(4).Info(log("Unmounter for %s", specName))

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
	attribs, err := getVolumeSourceAttribs(spec)
	if err != nil {
		klog.Error(log("deleter failed to extract volume attributes from spec: %v", err))
		return nil, err
	}

	secretName, secretNS, err := getSecretAndNamespaceFromSpec(spec, nil)
	if err != nil {
		return nil, errors.New(log("failed to get secret name or secretNamespace: %v", err))
	}

	return &sioVolume{
		spec:            spec,
		secretName:      secretName,
		secretNamespace: secretNS,
		volSpecName:     spec.Name(),
		volName:         attribs.volName,
		plugin:          p,
		readOnly:        attribs.readOnly,
	}, nil
}

// *********************************
// ProvisionableVolumePlugin Impl
// *********************************
var _ volume.ProvisionableVolumePlugin = &sioPlugin{}

func (p *sioPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	klog.V(4).Info(log("creating Provisioner"))

	configData := options.Parameters
	if configData == nil {
		klog.Error(log("provisioner missing parameters, unable to continue"))
		return nil, errors.New("option parameters missing")
	}

	// Supports ref of name of secret a couple of ways:
	// options.Parameters["secretRef"] for backward compat, or
	// options.Parameters["secretName"]
	secretName := configData[confKey.secretName]
	if secretName == "" {
		secretName = configData["secretName"]
		configData[confKey.secretName] = secretName
	}

	secretNS := configData[confKey.secretNamespace]
	if secretNS == "" {
		secretNS = options.PVC.Namespace
	}

	return &sioVolume{
		configData:      configData,
		plugin:          p,
		options:         options,
		secretName:      secretName,
		secretNamespace: secretNS,
		volSpecName:     options.PVName,
	}, nil
}
