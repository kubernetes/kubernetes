/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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

package ovf

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type Manager struct {
	types.ManagedObjectReference

	c *vim25.Client
}

func NewManager(c *vim25.Client) *Manager {
	return &Manager{*c.ServiceContent.OvfManager, c}
}

// CreateDescriptor wraps methods.CreateDescriptor
func (m *Manager) CreateDescriptor(ctx context.Context, obj mo.Reference, cdp types.OvfCreateDescriptorParams) (*types.OvfCreateDescriptorResult, error) {
	req := types.CreateDescriptor{
		This: m.Reference(),
		Obj:  obj.Reference(),
		Cdp:  cdp,
	}

	res, err := methods.CreateDescriptor(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// CreateImportSpec wraps methods.CreateImportSpec
func (m *Manager) CreateImportSpec(ctx context.Context, ovfDescriptor string, resourcePool mo.Reference, datastore mo.Reference, cisp types.OvfCreateImportSpecParams) (*types.OvfCreateImportSpecResult, error) {
	req := types.CreateImportSpec{
		This:          m.Reference(),
		OvfDescriptor: ovfDescriptor,
		ResourcePool:  resourcePool.Reference(),
		Datastore:     datastore.Reference(),
		Cisp:          cisp,
	}

	res, err := methods.CreateImportSpec(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// ParseDescriptor wraps methods.ParseDescriptor
func (m *Manager) ParseDescriptor(ctx context.Context, ovfDescriptor string, pdp types.OvfParseDescriptorParams) (*types.OvfParseDescriptorResult, error) {
	req := types.ParseDescriptor{
		This:          m.Reference(),
		OvfDescriptor: ovfDescriptor,
		Pdp:           pdp,
	}

	res, err := methods.ParseDescriptor(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// ValidateHost wraps methods.ValidateHost
func (m *Manager) ValidateHost(ctx context.Context, ovfDescriptor string, host mo.Reference, vhp types.OvfValidateHostParams) (*types.OvfValidateHostResult, error) {
	req := types.ValidateHost{
		This:          m.Reference(),
		OvfDescriptor: ovfDescriptor,
		Host:          host.Reference(),
		Vhp:           vhp,
	}

	res, err := methods.ValidateHost(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}
