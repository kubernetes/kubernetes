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
	"errors"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
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
