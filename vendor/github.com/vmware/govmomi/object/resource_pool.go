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

	"github.com/vmware/govmomi/nfc"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type ResourcePool struct {
	Common
}

func NewResourcePool(c *vim25.Client, ref types.ManagedObjectReference) *ResourcePool {
	return &ResourcePool{
		Common: NewCommon(c, ref),
	}
}

// Owner returns the ResourcePool owner as a ClusterComputeResource or ComputeResource.
func (p ResourcePool) Owner(ctx context.Context) (Reference, error) {
	var pool mo.ResourcePool

	err := p.Properties(ctx, p.Reference(), []string{"owner"}, &pool)
	if err != nil {
		return nil, err
	}

	return NewReference(p.Client(), pool.Owner), nil
}

func (p ResourcePool) ImportVApp(ctx context.Context, spec types.BaseImportSpec, folder *Folder, host *HostSystem) (*nfc.Lease, error) {
	req := types.ImportVApp{
		This: p.Reference(),
		Spec: spec,
	}

	if folder != nil {
		ref := folder.Reference()
		req.Folder = &ref
	}

	if host != nil {
		ref := host.Reference()
		req.Host = &ref
	}

	res, err := methods.ImportVApp(ctx, p.c, &req)
	if err != nil {
		return nil, err
	}

	return nfc.NewLease(p.c, res.Returnval), nil
}

func (p ResourcePool) Create(ctx context.Context, name string, spec types.ResourceConfigSpec) (*ResourcePool, error) {
	req := types.CreateResourcePool{
		This: p.Reference(),
		Name: name,
		Spec: spec,
	}

	res, err := methods.CreateResourcePool(ctx, p.c, &req)
	if err != nil {
		return nil, err
	}

	return NewResourcePool(p.c, res.Returnval), nil
}

func (p ResourcePool) CreateVApp(ctx context.Context, name string, resSpec types.ResourceConfigSpec, configSpec types.VAppConfigSpec, folder *Folder) (*VirtualApp, error) {
	req := types.CreateVApp{
		This:       p.Reference(),
		Name:       name,
		ResSpec:    resSpec,
		ConfigSpec: configSpec,
	}

	if folder != nil {
		ref := folder.Reference()
		req.VmFolder = &ref
	}

	res, err := methods.CreateVApp(ctx, p.c, &req)
	if err != nil {
		return nil, err
	}

	return NewVirtualApp(p.c, res.Returnval), nil
}

func (p ResourcePool) UpdateConfig(ctx context.Context, name string, config *types.ResourceConfigSpec) error {
	req := types.UpdateConfig{
		This:   p.Reference(),
		Name:   name,
		Config: config,
	}

	if config != nil && config.Entity == nil {
		ref := p.Reference()

		// Create copy of config so changes won't leak back to the caller
		newConfig := *config
		newConfig.Entity = &ref
		req.Config = &newConfig
	}

	_, err := methods.UpdateConfig(ctx, p.c, &req)
	return err
}

func (p ResourcePool) DestroyChildren(ctx context.Context) error {
	req := types.DestroyChildren{
		This: p.Reference(),
	}

	_, err := methods.DestroyChildren(ctx, p.c, &req)
	return err
}

func (p ResourcePool) Destroy(ctx context.Context) (*Task, error) {
	req := types.Destroy_Task{
		This: p.Reference(),
	}

	res, err := methods.Destroy_Task(ctx, p.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(p.c, res.Returnval), nil
}
