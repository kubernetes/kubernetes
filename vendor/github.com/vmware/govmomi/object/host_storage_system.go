/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"
	"errors"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type HostStorageSystem struct {
	Common
}

func NewHostStorageSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostStorageSystem {
	return &HostStorageSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostStorageSystem) RetrieveDiskPartitionInfo(ctx context.Context, devicePath string) (*types.HostDiskPartitionInfo, error) {
	req := types.RetrieveDiskPartitionInfo{
		This:       s.Reference(),
		DevicePath: []string{devicePath},
	}

	res, err := methods.RetrieveDiskPartitionInfo(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil || len(res.Returnval) == 0 {
		return nil, errors.New("no partition info")
	}

	return &res.Returnval[0], nil
}

func (s HostStorageSystem) ComputeDiskPartitionInfo(ctx context.Context, devicePath string, layout types.HostDiskPartitionLayout) (*types.HostDiskPartitionInfo, error) {
	req := types.ComputeDiskPartitionInfo{
		This:       s.Reference(),
		DevicePath: devicePath,
		Layout:     layout,
	}

	res, err := methods.ComputeDiskPartitionInfo(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (s HostStorageSystem) UpdateDiskPartitionInfo(ctx context.Context, devicePath string, spec types.HostDiskPartitionSpec) error {
	req := types.UpdateDiskPartitions{
		This:       s.Reference(),
		DevicePath: devicePath,
		Spec:       spec,
	}

	_, err := methods.UpdateDiskPartitions(ctx, s.c, &req)
	return err
}

func (s HostStorageSystem) RescanAllHba(ctx context.Context) error {
	req := types.RescanAllHba{
		This: s.Reference(),
	}

	_, err := methods.RescanAllHba(ctx, s.c, &req)
	return err
}

func (s HostStorageSystem) Refresh(ctx context.Context) error {
	req := types.RefreshStorageSystem{
		This: s.Reference(),
	}

	_, err := methods.RefreshStorageSystem(ctx, s.c, &req)
	return err
}

func (s HostStorageSystem) RescanVmfs(ctx context.Context) error {
	req := types.RescanVmfs{
		This: s.Reference(),
	}

	_, err := methods.RescanVmfs(ctx, s.c, &req)
	return err
}

func (s HostStorageSystem) MarkAsSsd(ctx context.Context, uuid string) (*Task, error) {
	req := types.MarkAsSsd_Task{
		This:         s.Reference(),
		ScsiDiskUuid: uuid,
	}

	res, err := methods.MarkAsSsd_Task(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(s.c, res.Returnval), nil
}

func (s HostStorageSystem) MarkAsNonSsd(ctx context.Context, uuid string) (*Task, error) {
	req := types.MarkAsNonSsd_Task{
		This:         s.Reference(),
		ScsiDiskUuid: uuid,
	}

	res, err := methods.MarkAsNonSsd_Task(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(s.c, res.Returnval), nil
}

func (s HostStorageSystem) MarkAsLocal(ctx context.Context, uuid string) (*Task, error) {
	req := types.MarkAsLocal_Task{
		This:         s.Reference(),
		ScsiDiskUuid: uuid,
	}

	res, err := methods.MarkAsLocal_Task(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(s.c, res.Returnval), nil
}

func (s HostStorageSystem) MarkAsNonLocal(ctx context.Context, uuid string) (*Task, error) {
	req := types.MarkAsNonLocal_Task{
		This:         s.Reference(),
		ScsiDiskUuid: uuid,
	}

	res, err := methods.MarkAsNonLocal_Task(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(s.c, res.Returnval), nil
}

func (s HostStorageSystem) AttachScsiLun(ctx context.Context, uuid string) error {
	req := types.AttachScsiLun{
		This:    s.Reference(),
		LunUuid: uuid,
	}

	_, err := methods.AttachScsiLun(ctx, s.c, &req)

	return err
}

func (s HostStorageSystem) QueryUnresolvedVmfsVolumes(ctx context.Context) ([]types.HostUnresolvedVmfsVolume, error) {
	req := &types.QueryUnresolvedVmfsVolume{
		This: s.Reference(),
	}

	res, err := methods.QueryUnresolvedVmfsVolume(ctx, s.Client(), req)
	if err != nil {
		return nil, err
	}
	return res.Returnval, nil
}

func (s HostStorageSystem) UnmountVmfsVolume(ctx context.Context, vmfsUuid string) error {
	req := &types.UnmountVmfsVolume{
		This:     s.Reference(),
		VmfsUuid: vmfsUuid,
	}

	_, err := methods.UnmountVmfsVolume(ctx, s.Client(), req)
	if err != nil {
		return err
	}

	return nil
}
