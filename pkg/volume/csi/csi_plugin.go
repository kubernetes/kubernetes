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

package csi

import (
	"errors"
	"fmt"
	"path"
	"regexp"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/golang/glog"
	api "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	csiName       = "csi"
	csiPluginName = "kubernetes.io/csi"

	// TODO (vladimirvivien) implement a more dynamic way to discover
	// the unix domain socket path for each installed csi driver.
	// TODO (vladimirvivien) would be nice to name socket with a .sock extension
	// for consistency.
	csiAddrTemplate = "/var/lib/kubelet/plugins/%v"
	csiTimeout      = 15 * time.Second
	volNameSep      = "^"
)

var (
	// csiVersion supported csi version
	csiVersion     = &csipb.Version{Major: 0, Minor: 1, Patch: 0}
	driverNameRexp = regexp.MustCompile(`^[A-Za-z]+(\.?-?_?[A-Za-z0-9-])+$`)
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

// GetvolumeName returns a concatenated string of CSIVolumeSource.Driver<volNameSe>CSIVolumeSource.VolumeHandle
// That string value is used in Detach() to extract driver name and volumeName.
func (p *csiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	csi, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("plugin.GetVolumeName failed to extract volume source from spec: %v", err))
		return "", err
	}

	//TODO (vladimirvivien) this validation should be done at the API validation check
	if !isDriverNameValid(csi.Driver) {
		glog.Error(log("plugin.GetVolumeName failed to create volume name: invalid csi driver name %s", csi.Driver))
		return "", errors.New("invalid csi driver name")
	}

	// return driverName<separator>volumeHandle
	return fmt.Sprintf("%s%s%s", csi.Driver, volNameSep, csi.VolumeHandle), nil
}

func (p *csiPlugin) CanSupport(spec *volume.Spec) bool {
	// TODO (vladimirvivien) CanSupport should also take into account
	// the availability/registration of specified Driver in the volume source
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

	// TODO (vladimirvivien) consider moving this check in API validation
	// check Driver name to conform to CSI spec
	if !isDriverNameValid(pvSource.Driver) {
		glog.Error(log("driver name does not conform to CSI spec: %s", pvSource.Driver))
		return nil, errors.New("driver name is invalid")
	}

	// before it is used in any paths such as socket etc
	addr := fmt.Sprintf(csiAddrTemplate, pvSource.Driver)
	glog.V(4).Infof(log("setting up mounter for [volume=%v,driver=%v]", pvSource.VolumeHandle, pvSource.Driver))
	client := newCsiDriverClient("unix", addr)

	k8s := p.host.GetKubeClient()
	if k8s == nil {
		glog.Error(log("failed to get a kubernetes client"))
		return nil, errors.New("failed to get a Kubernetes client")
	}

	mounter := &csiMountMgr{
		plugin:     p,
		k8s:        k8s,
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
	unmounter := &csiMountMgr{
		plugin: p,
		podUID: podUID,
	}
	return unmounter, nil
}

func (p *csiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glog.V(4).Infof(log("constructing volume spec [pv.Name=%v, path=%v]", volumeName, mountPath))

	// extract driverName/volumeId from end of mountPath
	dir, volID := path.Split(mountPath)
	volID = kstrings.UnescapeQualifiedNameForDisk(volID)
	driverName := path.Base(dir)

	// TODO (vladimirvivien) consider moving this check in API validation
	if !isDriverNameValid(driverName) {
		glog.Error(log("failed while reconstructing volume spec csi: driver name extracted from path is invalid: [path=%s; driverName=%s]", mountPath, driverName))
		return nil, errors.New("invalid csi driver name from path")
	}

	glog.V(4).Info(log("plugin.ConstructVolumeSpec extracted [volumeID=%s; driverName=%s]", volID, driverName))

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
	// TODO (vladimirvivien) use CSI VolumeCapability.MountVolume.mount_flags
	// to probe for the result for this method:w
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

// isDriverNameValid validates the driverName using CSI spec
func isDriverNameValid(name string) bool {
	if len(name) == 0 || len(name) > 63 {
		return false
	}
	return driverNameRexp.MatchString(name)
}
