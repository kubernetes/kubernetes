/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package gce_pd

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
)

func ProbeProvisionableVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		// SSD is first = default on GCE
		&provisionableGcePersistentDiskPlugin{
			host:     nil,
			name:     "kubernetes.io/gce-pd-ssd",
			diskType: "pd-ssd",
		},
		&provisionableGcePersistentDiskPlugin{
			host:     nil,
			name:     "kubernetes.io/gce-pd-standard",
			diskType: "pd-standard",
		},
	}
}

type provisionableGcePersistentDiskPlugin struct {
	host volume.VolumeHost
	// Name of the provisioner
	name string
	// GCE PD disk type
	diskType string
}

var _ volume.VolumePlugin = &provisionableGcePersistentDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &provisionableGcePersistentDiskPlugin{}

func (plugin *provisionableGcePersistentDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *provisionableGcePersistentDiskPlugin) Name() string {
	return plugin.name
}

func (plugin *provisionableGcePersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.GCEPersistentDisk != nil) ||
		(spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil)
}

func (plugin *provisionableGcePersistentDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = accessModes
	}
	return plugin.newProvisionerInternal(options, &GCEDiskUtil{})
}

func (plugin *provisionableGcePersistentDiskPlugin) newProvisionerInternal(options volume.VolumeOptions, manager pdManager) (volume.Provisioner, error) {
	return &gcePersistentDiskProvisioner{
		manager: manager,
		plugin:  plugin,
		options: options,
	}, nil
}

type gcePersistentDiskProvisioner struct {
	manager pdManager
	plugin  *provisionableGcePersistentDiskPlugin
	options volume.VolumeOptions
}

var _ volume.Provisioner = &gcePersistentDiskProvisioner{}

func (c *gcePersistentDiskProvisioner) Provision(pv *api.PersistentVolume) error {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		return err
	}
	pv.Spec.PersistentVolumeSource.GCEPersistentDisk.PDName = volumeID
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return nil
}

func (c *gcePersistentDiskProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	// Provide dummy api.PersistentVolume.Spec, it will be filled in
	// gcePersistentDiskProvisioner.Provision()
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-gce-",
			Labels:       map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "gce-pd-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): c.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
					PDName:    "dummy",
					FSType:    "ext4",
					Partition: 0,
					ReadOnly:  false,
				},
			},
		},
	}, nil
}
