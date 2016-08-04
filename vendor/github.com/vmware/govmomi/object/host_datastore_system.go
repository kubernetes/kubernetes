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
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type HostDatastoreSystem struct {
	Common
}

func NewHostDatastoreSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostDatastoreSystem {
	return &HostDatastoreSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostDatastoreSystem) CreateLocalDatastore(ctx context.Context, name string, path string) (*Datastore, error) {
	req := types.CreateLocalDatastore{
		This: s.Reference(),
		Name: name,
		Path: path,
	}

	res, err := methods.CreateLocalDatastore(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewDatastore(s.Client(), res.Returnval), nil
}

func (s HostDatastoreSystem) CreateNasDatastore(ctx context.Context, spec types.HostNasVolumeSpec) (*Datastore, error) {
	req := types.CreateNasDatastore{
		This: s.Reference(),
		Spec: spec,
	}

	res, err := methods.CreateNasDatastore(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewDatastore(s.Client(), res.Returnval), nil
}

func (s HostDatastoreSystem) CreateVmfsDatastore(ctx context.Context, spec types.VmfsDatastoreCreateSpec) (*Datastore, error) {
	req := types.CreateVmfsDatastore{
		This: s.Reference(),
		Spec: spec,
	}

	res, err := methods.CreateVmfsDatastore(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewDatastore(s.Client(), res.Returnval), nil
}

func (s HostDatastoreSystem) Remove(ctx context.Context, ds *Datastore) error {
	req := types.RemoveDatastore{
		This:      s.Reference(),
		Datastore: ds.Reference(),
	}

	_, err := methods.RemoveDatastore(ctx, s.Client(), &req)
	if err != nil {
		return err
	}

	return nil
}

func (s HostDatastoreSystem) QueryAvailableDisksForVmfs(ctx context.Context) ([]types.HostScsiDisk, error) {
	req := types.QueryAvailableDisksForVmfs{
		This: s.Reference(),
	}

	res, err := methods.QueryAvailableDisksForVmfs(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (s HostDatastoreSystem) QueryVmfsDatastoreCreateOptions(ctx context.Context, devicePath string) ([]types.VmfsDatastoreOption, error) {
	req := types.QueryVmfsDatastoreCreateOptions{
		This:       s.Reference(),
		DevicePath: devicePath,
	}

	res, err := methods.QueryVmfsDatastoreCreateOptions(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}
