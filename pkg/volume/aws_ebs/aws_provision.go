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

package aws_ebs

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
)

func ProbeProvisionableVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		// SSD is first = default on AWS
		&provisionableAwsPersistentDiskPlugin{
			host:       nil,
			name:       "kubernetes.io/aws-ebs-ssd",
			volumeType: "gp2",
		},
		&provisionableAwsPersistentDiskPlugin{
			host:       nil,
			name:       "kubernetes.io/aws-ebs-standard",
			volumeType: "standard",
		},
		// TODO: add IOPS provisioners based on configMap
	}
}

type provisionableAwsPersistentDiskPlugin struct {
	host volume.VolumeHost
	// Name of the provisioner
	name string
	// AWS EBS volume type
	volumeType string
}

var _ volume.VolumePlugin = &provisionableAwsPersistentDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &provisionableAwsPersistentDiskPlugin{}

func (plugin *provisionableAwsPersistentDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *provisionableAwsPersistentDiskPlugin) Name() string {
	return plugin.name
}

func (plugin *provisionableAwsPersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AWSElasticBlockStore != nil) ||
		(spec.Volume != nil && spec.Volume.AWSElasticBlockStore != nil)
}

func (plugin *provisionableAwsPersistentDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = accessModes
	}
	return plugin.newProvisionerInternal(options, &AWSDiskUtil{})
}

func (plugin *provisionableAwsPersistentDiskPlugin) newProvisionerInternal(options volume.VolumeOptions, manager ebsManager) (volume.Provisioner, error) {
	return &awsElasticBlockStoreProvisioner{
		manager: manager,
		plugin:  plugin,
		options: options,
	}, nil
}

type awsElasticBlockStoreProvisioner struct {
	manager ebsManager
	plugin  *provisionableAwsPersistentDiskPlugin
	options volume.VolumeOptions
}

var _ volume.Provisioner = &awsElasticBlockStoreProvisioner{}

func (c *awsElasticBlockStoreProvisioner) Provision(pv *api.PersistentVolume) error {
	volumeID, sizeGB, err := c.manager.CreateVolume(c)
	if err != nil {
		return err
	}
	pv.Spec.PersistentVolumeSource.AWSElasticBlockStore.VolumeID = volumeID
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return nil
}

func (c *awsElasticBlockStoreProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	// Provide dummy api.PersistentVolume.Spec, it will be filled in
	// awsElasticBlockStoreProvisioner.Provision()
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-aws-",
			Labels:       map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "aws-ebs-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): c.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
					VolumeID:  "dummy",
					FSType:    "ext4",
					Partition: 0,
					ReadOnly:  false,
				},
			},
		},
	}, nil
}
