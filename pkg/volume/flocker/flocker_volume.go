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

package flocker

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
)

type volumeManager interface {
	// Creates a volume
	CreateVolume(provisioner *flockerVolumeProvisioner) (datasetUUID string, volumeSizeGB int, labels map[string]string, err error)
	// Deletes a volume
	DeleteVolume(deleter *flockerVolumeDeleter) error
}

type flockerVolumeDeleter struct {
	*flockerVolume
}

var _ volume.Deleter = &flockerVolumeDeleter{}

func (b *flockerVolumeDeleter) GetPath() string {
	return getPath(b.podUID, b.volName, b.plugin.host)
}

func (d *flockerVolumeDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type flockerVolumeProvisioner struct {
	*flockerVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &flockerVolumeProvisioner{}

func (c *flockerVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {

	if len(c.options.Parameters) > 0 {
		return nil, fmt.Errorf("Provisioning failed: Specified at least one unsupported parameter")
	}

	if c.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("Provisioning failed: Specified unsupported selector")
	}

	datasetUUID, sizeGB, labels, err := c.manager.CreateVolume(c)
	if err != nil {
		return nil, err
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name:   c.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "flocker-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Flocker: &v1.FlockerVolumeSource{
					DatasetUUID: datasetUUID,
				},
			},
		},
	}
	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
		}
	}

	return pv, nil
}
