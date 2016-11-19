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

package scaleio

import (
	"errors"

	"github.com/golang/glog"
	api "k8s.io/kubernetes/pkg/api/v1"
	meta "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	sioName             = "scaleio"
	sioPluginName       = "kubernetes.io/scaleio"
	sioDefaultNamespace = "default"
)

type sioPlugin struct {
	host    volume.VolumeHost
	mounter mount.Interface
}

func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &sioPlugin{
		host: nil,
	}
	return []volume.VolumePlugin{p}
	return nil
}

// *******************
// VolumePlugin Impl
// *******************
var _ volume.VolumePlugin = &sioPlugin{}

func (p *sioPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	p.mounter = host.GetMounter()
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
		glog.Errorf("sio: failed to extract ScaleIOVolumeSource from spec: %v", err)
		return nil, err
	}
	glog.V(4).Infof("sio: creating new Mounter for volume %v", sioSource.VolumeName)

	configData, err := setupConfigData(p, sioSource)
	if err != nil {
		glog.Errorf("sio: failed to setup configuration data: %v", err)
		return nil, err
	}

	mgr, err := newSioMgr(configData)
	if err != nil {
		return nil, err
	}

	return &sioVolume{
		pod:        pod,
		spec:       spec,
		source:     sioSource,
		volName:    sioSource.VolumeName,
		podUID:     pod.UID,
		readOnly:   sioSource.ReadOnly,
		plugin:     p,
		configData: configData,
		sioMgr:     mgr,
	}, nil
}

func (p *sioPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	glog.V(4).Infof("sio: creating new UnMounter for volume %v", volName)

	kubeClient := p.host.GetKubeClient()
	confMap, err := kubeClient.Core().ConfigMaps(sioDefaultNamespace).Get(volName, meta.GetOptions{})
	if err != nil {
		glog.Errorf("sio: unable to create Unmounter, failed to get ConfigMap %s: %v", volName, err)
		return nil, err
	}
	configs := confMap.Data
	assertConfigDefaults(configs)
	if err := validateConfigs(configs); err != nil {
		return nil, err
	}
	mgr, err := newSioMgr(configs)
	if err != nil {
		return nil, err
	}

	return &sioVolume{
		sioMgr:  mgr,
		podUID:  podUID,
		volName: volName,
		plugin:  p,
	}, nil
}

func (p *sioPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glog.V(4).Infof("sio: ConstructVolumeSpec(volume=%v, mountPath=%v)", volumeName, mountPath)
	sioVolumeSpec := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{
				VolumeName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(sioVolumeSpec), nil
}

//******************************
// PersistentVolumePlugin Impl
// *****************************
var _ volume.PersistentVolumePlugin = &sioPlugin{}

func (p *sioPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

// ***************************
// DeletableVolumePlugin Impl
//****************************
var _ volume.DeletableVolumePlugin = &sioPlugin{}

func (p *sioPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	sioSource, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		return nil, err
	}

	kubeClient := p.host.GetKubeClient()
	volName := sioSource.VolumeName
	confMap, err := kubeClient.Core().ConfigMaps(sioDefaultNamespace).Get(volName, meta.GetOptions{})
	if err != nil {
		glog.Errorf("sio: unable to create Deleter, failed to get ConfigMap %s: %v", volName, err)
		return nil, err
	}
	configs := confMap.Data
	assertConfigDefaults(configs)
	if validateConfigs(configs); err != nil {
		return nil, err
	}

	mgr, err := newSioMgr(configs)
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("sio: creating new Deleter for volume %v", volName)

	return &sioVolume{
		sioMgr:   mgr,
		spec:     spec,
		source:   sioSource,
		volName:  sioSource.VolumeName,
		plugin:   p,
		readOnly: sioSource.ReadOnly,
	}, nil
}

// *********************************
// ProvisionableVolumePlugin Impl
// *********************************
var _ volume.ProvisionableVolumePlugin = &sioPlugin{}

func (p *sioPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	glog.V(4).Info("sio: creating Provisioner")

	configData := options.Parameters
	if configData == nil {
		return nil, errors.New("missing options")
	}
	assertConfigDefaults(configData)

	secret, err := getSecret(p, configData[confKey.secretRef])
	if err != nil {
		glog.Errorf("sio: failed to get secret: %v", err)
		return nil, err
	}
	mapSecret(configData, secret)

	// ensure required configs are given
	validateConfigs(configData)

	// persist/update configMap
	err = saveConfigMap(p, configData)
	if err != nil {
		glog.Errorf("sio: unable to save/update configMap: %v", err)
		return nil, err
	}
	mgr, err := newSioMgr(configData)
	if err != nil {
		return nil, err
	}

	return &sioVolume{
		sioMgr:  mgr,
		plugin:  p,
		options: options,
	}, nil
}
