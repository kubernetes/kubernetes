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

package portworx

import (
	"github.com/golang/glog"
	osdapi "github.com/libopenstorage/openstorage/api"
	osdclient "github.com/libopenstorage/openstorage/api/client"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	osdspec "github.com/libopenstorage/openstorage/api/spec"
	osdvolume "github.com/libopenstorage/openstorage/volume"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	osdMgmtPort      = "9001"
	osdDriverVersion = "v1"
	pxdDriverName    = "pxd"
	pwxSockName      = "pwx"
)

type PortworxVolumeUtil struct {
	portworxClient *osdclient.Client
}

// CreateVolume creates a Portworx volume.
func (util *PortworxVolumeUtil) CreateVolume(p *portworxVolumeProvisioner) (string, int, map[string]string, error) {
	hostName := p.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return "", 0, nil, err
	}

	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// Portworx Volumes are specified in GB
	requestGB := int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024))

	var labels map[string]string
	if p.options.CloudTags == nil {
		labels = make(map[string]string)
	} else {
		labels = *p.options.CloudTags
	}

	specHandler := osdspec.NewSpecHandler()
	spec, err := specHandler.SpecFromOpts(p.options.Parameters)
	if err != nil {
		return "", 0, nil, err
	}
	spec.Size = uint64(requestGB * 1024 * 1024 * 1024)
	source := osdapi.Source{}
	locator := osdapi.VolumeLocator{
		Name:         p.options.PVName,
		VolumeLabels: labels,
	}
	volumeID, err := client.Create(&locator, &source, spec)
	if err != nil {
		glog.V(2).Infof("Error creating Portworx Volume : %v", err)
	}
	return volumeID, requestGB, nil, err
}

// DeleteVolume deletes a Portworx volume
func (util *PortworxVolumeUtil) DeleteVolume(d *portworxVolumeDeleter) error {
	hostName := d.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return err
	}

	err = client.Delete(d.volumeID)
	if err != nil {
		glog.V(2).Infof("Error deleting Portworx Volume (%v): %v", d.volName, err)
		return err
	}
	return nil
}

// AttachVolume attaches a Portworx Volume
func (util *PortworxVolumeUtil) AttachVolume(m *portworxVolumeMounter) (string, error) {
	hostName := m.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return "", err
	}

	devicePath, err := client.Attach(m.volName)
	if err != nil {
		glog.V(2).Infof("Error attaching Portworx Volume (%v): %v", m.volName, err)
		return "", err
	}
	return devicePath, nil
}

// DetachVolume detaches a Portworx Volume
func (util *PortworxVolumeUtil) DetachVolume(u *portworxVolumeUnmounter) error {
	hostName := u.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return err
	}

	err = client.Detach(u.volName)
	if err != nil {
		glog.V(2).Infof("Error detaching Portworx Volume (%v): %v", u.volName, err)
		return err
	}
	return nil
}

// MountVolume mounts a Portworx Volume on the specified mountPath
func (util *PortworxVolumeUtil) MountVolume(m *portworxVolumeMounter, mountPath string) error {
	hostName := m.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return err
	}

	err = client.Mount(m.volName, mountPath)
	if err != nil {
		glog.V(2).Infof("Error mounting Portworx Volume (%v) on Path (%v): %v", m.volName, mountPath, err)
		return err
	}
	return nil
}

// UnmountVolume unmounts a Portworx Volume
func (util *PortworxVolumeUtil) UnmountVolume(u *portworxVolumeUnmounter, mountPath string) error {
	hostName := u.plugin.host.GetHostName()
	client, err := util.osdClient(hostName)
	if err != nil {
		return err
	}

	err = client.Unmount(u.volName, mountPath)
	if err != nil {
		glog.V(2).Infof("Error unmounting Portworx Volume (%v) on Path (%v): %v", u.volName, mountPath, err)
		return err
	}
	return nil
}

func (util *PortworxVolumeUtil) osdClient(hostName string) (osdvolume.VolumeDriver, error) {
	if util.portworxClient == nil {
		driverClient, err := volumeclient.NewDriverClient("", pxdDriverName, osdDriverVersion)
		if err != nil {
			return nil, err
		}
		util.portworxClient = driverClient
	}

	return volumeclient.VolumeDriver(util.portworxClient), nil
}
