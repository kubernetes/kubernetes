/*
Copyright 2016 The Kubernetes Authors.

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

package libstorage

import (
	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

const (
	lsPluginName = "kubernetes.io/libstorage"
)

type lsPlugin struct {
	host volume.VolumeHost
	lsMgr
}

func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &lsPlugin{
		host: nil,
	}
	return []volume.VolumePlugin{p}
}

// *******************
// VolumePlugin Impl
// *******************
var _ volume.VolumePlugin = &lsPlugin{}

func (p *lsPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	return nil
}

func (p *lsPlugin) GetPluginName() string {
	return lsPluginName
}

func (p *lsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	source, err := getLibStorageSource(spec)
	if err != nil {
		return "", err
	}
	return source.VolumeName, nil
}

func (p *lsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LibStorage != nil) ||
		(spec.Volume != nil && spec.Volume.LibStorage != nil)
}

func (p *lsPlugin) RequiresRemount() bool {
	return false
}

func (p *lsPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {

	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("libStorage: creating new Mounter for volume %v", lsSource.VolumeName)

	p.resetMgr(lsSource.Host, lsSource.Service)

	return &lsVolume{
		podUID:   pod.UID,
		volName:  lsSource.VolumeName,
		fsType:   lsSource.FSType,
		mounter:  p.host.GetMounter(),
		plugin:   p,
		readOnly: spec.ReadOnly,
	}, nil
}

func (p *lsPlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	glog.V(4).Infof("libStorage: creating new UnMounter for volume %v\n", name)
	return &lsVolume{
		podUID:  podUID,
		volName: name,
		plugin:  p,
		mounter: p.host.GetMounter(),
	}, nil
}

func (p *lsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glog.V(4).Infof("libStorage: ConstructVolumeSpec(volume=%v, mountPath=%v)", volumeName, mountPath)
	lsVolumeSpec := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			LibStorage: &api.LibStorageVolumeSource{
				VolumeName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(lsVolumeSpec), nil
}

//******************************
// PersistentVolumePlugin Impl
// *****************************
var _ volume.PersistentVolumePlugin = &lsPlugin{}

func (p *lsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

//******************************
// AttachableVolumePlugin Impl
//******************************
//var _ volume.AttachableVolumePlugin = &lsPlugin{}

//func (p *lsPlugin) NewAttacher() (volume.Attacher, error) {
//	return &lsVolume{
//		mounter: p.host.GetMounter(),
//		plugin:  p,
//	}, nil
//}

//func (p *lsPlugin) NewDetacher() (volume.Detacher, error) {
//	return &lsVolume{
//		mounter: p.host.GetMounter(),
//		plugin:  p,
//	}, nil
//}

//func (p *lsPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
//	mounter := p.host.GetMounter()
//	return mount.GetMountRefs(mounter, deviceMountPath)
//}

// ***************************
// DeletableVolumePlugin Impl
//****************************
var _ volume.DeletableVolumePlugin = &lsPlugin{}

func (p *lsPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("libStorage: creating new Deleter for volume %v\n", lsSource.VolumeName)

	p.resetMgr(lsSource.Host, lsSource.Service)

	return &lsVolume{
		volName:  lsSource.VolumeName,
		fsType:   lsSource.FSType,
		mounter:  p.host.GetMounter(),
		plugin:   p,
		readOnly: spec.ReadOnly,
	}, nil
}

// *********************************
// ProvisionableVolumePlugin Impl
// *********************************
var _ volume.ProvisionableVolumePlugin = &lsPlugin{}

func (p *lsPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	glog.V(4).Info("libStorage: creating Provisioner")
	// extract libstorage configs
	volName := options.Parameters["volumeName"]
	lsHost := options.Parameters["host"]
	lsServ := options.Parameters["service"]
	fsType := options.Parameters["fsType"]

	p.resetMgr(lsHost, lsServ)

	return &lsVolume{
		volName: volName,
		fsType:  fsType,
		plugin:  p,
		options: options,
	}, nil
}

func (p *lsPlugin) resetMgr(host, service string) {
	if p.lsMgr == nil {
		mgr := newLibStorageMgr(host, service)
		mgr.lsPluginDir = p.lsPluginDir()
		p.lsMgr = mgr
	}
}

func (p *lsPlugin) lsPluginDir() string {
	return p.host.GetPluginDir(lsPluginName)
}
