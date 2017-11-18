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

package csi

import (
	"errors"
	"fmt"
	"path"
	"time"

	"github.com/golang/glog"
	api "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	csipb "k8s.io/kubernetes/pkg/volume/csi/proto/csi"
)

const (
	csiName       = "csi"
	csiPluginName = "kubernetes.io/csi"
	//csiAddrTemplate = "/var/lib/kubelet/plugins/csi/sockets/%v/kubeletproxy.sock"
	csiAddrTemplate = "/tmp/kubeletproxy.sock"
	csiTimeout      = 15 * time.Second
)

var (
	// csiVersion supported csi version
	csiVersion = &csipb.Version{Major: 0, Minor: 1, Patch: 0}
)

type csiPlugin struct {
	host volume.VolumeHost
}

// ProbeVolumePlugins returns implemented plugins
func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &csiPlugin{
		host: nil,
	}
	return []volume.VolumePlugin{p}
}

// volume.VolumePlugin methods
var _ volume.VolumePlugin = &csiPlugin{}

func (p *csiPlugin) Init(host volume.VolumeHost) error {
	glog.Info(log("plugin initializing..."))
	p.host = host
	return nil
}

func (p *csiPlugin) GetPluginName() string {
	return csiPluginName
}

func (p *csiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	csi, err := getCSISourceFromSpec(spec)
	if err != nil {
		return "", err
	}
	return csi.VolumeHandle, nil
}

func (p *csiPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil
}

func (p *csiPlugin) RequiresRemount() bool {
	return false
}

func (p *csiPlugin) NewMounter(
	spec *volume.Spec,
	pod *api.Pod,
	_ volume.VolumeOptions) (volume.Mounter, error) {
	pvSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		return nil, err
	}
	//addr := fmt.Sprintf(csiAddrTemplate, pvSource.Driver)
	glog.V(4).Infof(log("setting up mounter for [volume=%v,driver=%v]", pvSource.VolumeHandle, pvSource.Driver))
	addr := csiAddrTemplate
	client := newCsiDriverClient("unix", addr)
	mounter := &csiMountMgr{
		plugin:     p,
		spec:       spec,
		pod:        pod,
		podUID:     pod.UID,
		driverName: pvSource.Driver,
		volumeID:   pvSource.VolumeHandle,
		csiClient:  client,
	}
	return mounter, nil
}

func (p *csiPlugin) NewUnmounter(specName string, podUID types.UID) (volume.Unmounter, error) {
	glog.V(4).Infof(log("setting up unmounter for [name=%v, podUID=%v]", specName, podUID))
	addr := csiAddrTemplate
	client := newCsiDriverClient("unix", addr)
	unmounter := &csiMountMgr{
		plugin:    p,
		podUID:    podUID,
		csiClient: client,
	}
	return unmounter, nil
}

func (p *csiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glog.V(4).Infof(log("constructing volume spec [vol=%v, path=%v]", volumeName, mountPath))

	// extract driverName/volumeId from end of mountPath
	dir, volID := path.Split(mountPath)
	driverName := path.Base(dir)

	pv := &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: volumeName,
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volID,
				},
			},
		},
	}
	return volume.NewSpecFromPersistentVolume(pv, false), nil
}

func (p *csiPlugin) SupportsMountOption() bool {
	return false
}

func (p *csiPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

// volume.AttachableVolumePlugin methods
var _ volume.AttachableVolumePlugin = &csiPlugin{}

func (p *csiPlugin) NewAttacher() (volume.Attacher, error) {
	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("unable to get kubernetes client from host"))
		return nil, errors.New("unable to get Kubernetes client")
	}

	return &csiAttacher{
		plugin:        p,
		k8s:           k8s,
		waitSleepTime: 1 * time.Second,
	}, nil
}

func (p *csiPlugin) NewDetacher() (volume.Detacher, error) {
	return nil, errors.New("unimplemented")
}

func (p *csiPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	m := p.host.GetMounter(p.GetPluginName())
	return mount.GetMountRefs(m, deviceMountPath)
}

func getCSISourceFromSpec(spec *volume.Spec) (*api.CSIPersistentVolumeSource, error) {
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.CSI != nil {
		return spec.PersistentVolume.Spec.CSI, nil
	}

	return nil, fmt.Errorf("CSIPersistentVolumeSource not defined in spec")
}

// log prepends log string with `kubernetes.io/csi`
func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("%s: %s", csiPluginName, msg), parts...)
}
