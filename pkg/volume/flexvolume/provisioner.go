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

package flexvolume

import (
	"encoding/json"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
)

type flexVolumeProvisioner struct {
	spec    *volume.Spec
	plugin  *flexVolumePlugin
	options volume.VolumeOptions
}

var _ volume.Provisioner = &flexVolumeProvisioner{}

func (p *flexVolumeProvisioner) constructPersistentVolume(name string, sizeMB uint32, options, labels map[string]string) *api.PersistentVolume {

	// Add volume name option to to persistent volume.
	options[kVolumeName] = name
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   p.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": p.plugin.driverName + "-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: p.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   p.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dMi", sizeMB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				FlexVolume: &api.FlexVolumeSource{
					Driver:   p.plugin.driverName,
					Options:  options,
					ReadOnly: false,
				},
			},
		},
	}

	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
		}
	}
	return pv
}

func (d *flexVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	call := d.plugin.NewDriverCall(createCmd)

	// Generate unique volume name.
	name := volume.GenerateVolumeName(d.options.ClusterName, d.options.PVName, 63)

	// Add name.
	call.Append(name)

	// Add size option.
	requestBytes := d.options.Capacity.Value()
	requestMB := volume.RoundUpSize(requestBytes, 1024*1024)
	if d.options.Parameters == nil {
		d.options.Parameters = make(map[string]string)
	}
	d.options.Parameters[optionsVolumeSizeMB] = string(requestMB)

	call.AppendSpec(d.spec, d.plugin.host, d.options.Parameters)

	// Add cloud tags.
	jsonBytes, err := json.Marshal(*d.options.CloudTags)
	if err != nil {
		return nil,fmt.Errorf("Failed to marshal spec, error: %s", err.Error())
	}
	call.Append(string(jsonBytes))

	// TODO: implement c.options.ProvisionerSelector parsing
	if d.options.Selector != nil {
		return nil, fmt.Errorf("claim.Spec.Selector is not supported for %s", d.plugin.driverName)
	}

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*provisionerDefaults)(d).Provision()
	} else if err != nil {
		return nil, err
	}

	if status.Volume == nil {
		return nil, fmt.Errorf("%s: Failed to create volume: %s", d.plugin.driverName, d.spec.Name)
	}

	pv := d.constructPersistentVolume(status.Volume.ID, status.Volume.SizeMB, status.Volume.Labels, status.Volume.AttachAndMountOptions)

	return pv, nil
}
