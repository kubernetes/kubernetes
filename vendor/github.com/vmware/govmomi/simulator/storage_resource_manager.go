/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"strconv"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type StorageResourceManager struct {
	mo.StorageResourceManager
}

func NewStorageResourceManager(ref types.ManagedObjectReference) object.Reference {
	m := &StorageResourceManager{}
	m.Self = ref

	return m
}

func (m *StorageResourceManager) ConfigureStorageDrsForPodTask(req *types.ConfigureStorageDrsForPod_Task) soap.HasFault {
	task := CreateTask(m, "configureStorageDrsForPod", func(*Task) (types.AnyType, types.BaseMethodFault) {
		cluster := Map.Get(req.Pod).(*StoragePod)

		if s := req.Spec.PodConfigSpec; s != nil {
			config := &cluster.PodStorageDrsEntry.StorageDrsConfig.PodConfig

			if s.Enabled != nil {
				config.Enabled = *s.Enabled
			}
			if s.DefaultVmBehavior != "" {
				config.DefaultVmBehavior = s.DefaultVmBehavior
			}
		}

		return nil, nil
	})

	return &methods.ConfigureStorageDrsForPod_TaskBody{
		Res: &types.ConfigureStorageDrsForPod_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *StorageResourceManager) pod(ref *types.ManagedObjectReference) *StoragePod {
	if ref == nil {
		return nil
	}
	cluster := Map.Get(*ref).(*StoragePod)
	config := &cluster.PodStorageDrsEntry.StorageDrsConfig.PodConfig

	if !config.Enabled {
		return nil
	}

	if len(cluster.ChildEntity) == 0 {
		return nil
	}

	return cluster
}

func (m *StorageResourceManager) RecommendDatastores(req *types.RecommendDatastores) soap.HasFault {
	spec := req.StorageSpec.PodSelectionSpec
	body := new(methods.RecommendDatastoresBody)
	res := new(types.RecommendDatastoresResponse)
	key := 0
	invalid := func(prop string) soap.HasFault {
		body.Fault_ = Fault("", &types.InvalidArgument{
			InvalidProperty: prop,
		})
		return body
	}
	add := func(cluster *StoragePod, ds types.ManagedObjectReference) {
		key++
		res.Returnval.Recommendations = append(res.Returnval.Recommendations, types.ClusterRecommendation{
			Key:            strconv.Itoa(key),
			Type:           "V1",
			Time:           time.Now(),
			Rating:         1,
			Reason:         "storagePlacement",
			ReasonText:     "Satisfy storage initial placement requests",
			WarningText:    "",
			WarningDetails: (*types.LocalizableMessage)(nil),
			Prerequisite:   nil,
			Action: []types.BaseClusterAction{
				&types.StoragePlacementAction{
					ClusterAction: types.ClusterAction{
						Type:   "StoragePlacementV1",
						Target: (*types.ManagedObjectReference)(nil),
					},
					Vm: (*types.ManagedObjectReference)(nil),
					RelocateSpec: types.VirtualMachineRelocateSpec{
						Service:      (*types.ServiceLocator)(nil),
						Folder:       (*types.ManagedObjectReference)(nil),
						Datastore:    &ds,
						DiskMoveType: "moveAllDiskBackingsAndAllowSharing",
						Pool:         (*types.ManagedObjectReference)(nil),
						Host:         (*types.ManagedObjectReference)(nil),
						Disk:         nil,
						Transform:    "",
						DeviceChange: nil,
						Profile:      nil,
					},
					Destination:       ds,
					SpaceUtilBefore:   5.00297212600708,
					SpaceDemandBefore: 5.00297212600708,
					SpaceUtilAfter:    5.16835880279541,
					SpaceDemandAfter:  5.894514083862305,
					IoLatencyBefore:   0,
				},
			},
			Target: &cluster.Self,
		})
	}

	var devices object.VirtualDeviceList

	switch types.StoragePlacementSpecPlacementType(req.StorageSpec.Type) {
	case types.StoragePlacementSpecPlacementTypeCreate:
		if req.StorageSpec.ResourcePool == nil {
			return invalid("resourcePool")
		}
		if req.StorageSpec.ConfigSpec == nil {
			return invalid("configSpec")
		}
		for _, d := range req.StorageSpec.ConfigSpec.DeviceChange {
			devices = append(devices, d.GetVirtualDeviceConfigSpec().Device)
		}
		cluster := m.pod(spec.StoragePod)
		if cluster == nil {
			if f := req.StorageSpec.ConfigSpec.Files; f == nil || f.VmPathName == "" {
				return invalid("configSpec.files")
			}
		}
	case types.StoragePlacementSpecPlacementTypeClone:
		if req.StorageSpec.Folder == nil {
			return invalid("folder")
		}
		if req.StorageSpec.Vm == nil {
			return invalid("vm")
		}
		if req.StorageSpec.CloneName == "" {
			return invalid("cloneName")
		}
		if req.StorageSpec.CloneSpec == nil {
			return invalid("cloneSpec")
		}
	}

	for _, placement := range spec.InitialVmConfig {
		cluster := m.pod(&placement.StoragePod)
		if cluster == nil {
			return invalid("podSelectionSpec.storagePod")
		}

		for _, disk := range placement.Disk {
			if devices.FindByKey(disk.DiskId) == nil {
				return invalid("podSelectionSpec.initialVmConfig.disk.fileBacking")
			}
		}

		for _, ds := range cluster.ChildEntity {
			add(cluster, ds)
		}
	}

	body.Res = res
	return body
}
