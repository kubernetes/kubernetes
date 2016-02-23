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

package cinder

import (
	"errors"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/volume"
)

func ProbeProvisionableVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&provisionableCinderPlugin{
			host:       nil,
			name:       "kubernetes.io/cinder-provisioning-default",
			volumeType: "",
		},
		// TODO: add more provisioners based on configMap
	}
}

type provisionableCinderPlugin struct {
	host volume.VolumeHost
	// Name of the provisioner
	name string
	// Cinder volume type
	volumeType string
}

var _ volume.VolumePlugin = &provisionableCinderPlugin{}
var _ volume.ProvisionableVolumePlugin = &provisionableCinderPlugin{}

func (plugin *provisionableCinderPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *provisionableCinderPlugin) Name() string {
	return plugin.name
}

func (plugin *provisionableCinderPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder != nil) ||
		(spec.Volume != nil && spec.Volume.Cinder != nil)
}

func (plugin *provisionableCinderPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = accessModes
	}
	return plugin.newProvisionerInternal(options, &CinderDiskUtil{})
}

func (plugin *provisionableCinderPlugin) newProvisionerInternal(options volume.VolumeOptions, manager cdManager) (volume.Provisioner, error) {
	return &cinderVolumeProvisioner{
		manager: manager,
		plugin:  plugin,
		options: options,
	}, nil
}

func (plugin *provisionableCinderPlugin) getCloudProvider() (*openstack.OpenStack, error) {
	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	os := cloud.(*openstack.OpenStack)
	if os == nil {
		return nil, errors.New("Invalid cloud provider: expected OpenStack")
	}
	return os, nil
}

type cinderVolumeProvisioner struct {
	manager cdManager
	plugin  *provisionableCinderPlugin
	options volume.VolumeOptions
}

var _ volume.Provisioner = &cinderVolumeProvisioner{}

func (c *cinderVolumeProvisioner) Provision(pv *api.PersistentVolume) error {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		return err
	}
	pv.Spec.PersistentVolumeSource.Cinder.VolumeID = volumeID
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return nil
}

func (c *cinderVolumeProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	// Provide dummy api.PersistentVolume.Spec, it will be filled in
	// cinderVolumeProvisioner.Provision()
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-cinder-",
			Labels:       map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "cinder-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): c.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				Cinder: &api.CinderVolumeSource{
					VolumeID: "dummy",
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}, nil

}
