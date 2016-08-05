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

type OvfManager struct {
	Common
}

func NewOvfManager(c *vim25.Client) *OvfManager {
	o := OvfManager{
		Common: NewCommon(c, *c.ServiceContent.OvfManager),
	}

	return &o
}

// CreateDescriptor wraps methods.CreateDescriptor
func (o OvfManager) CreateDescriptor(ctx context.Context, obj Reference, cdp types.OvfCreateDescriptorParams) (*types.OvfCreateDescriptorResult, error) {
	req := types.CreateDescriptor{
		This: o.Reference(),
		Obj:  obj.Reference(),
		Cdp:  cdp,
	}

	res, err := methods.CreateDescriptor(ctx, o.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// CreateImportSpec wraps methods.CreateImportSpec
func (o OvfManager) CreateImportSpec(ctx context.Context, ovfDescriptor string, resourcePool Reference, datastore Reference, cisp types.OvfCreateImportSpecParams) (*types.OvfCreateImportSpecResult, error) {
	req := types.CreateImportSpec{
		This:          o.Reference(),
		OvfDescriptor: ovfDescriptor,
		ResourcePool:  resourcePool.Reference(),
		Datastore:     datastore.Reference(),
		Cisp:          cisp,
	}

	res, err := methods.CreateImportSpec(ctx, o.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// ParseDescriptor wraps methods.ParseDescriptor
func (o OvfManager) ParseDescriptor(ctx context.Context, ovfDescriptor string, pdp types.OvfParseDescriptorParams) (*types.OvfParseDescriptorResult, error) {
	req := types.ParseDescriptor{
		This:          o.Reference(),
		OvfDescriptor: ovfDescriptor,
		Pdp:           pdp,
	}

	res, err := methods.ParseDescriptor(ctx, o.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// ValidateHost wraps methods.ValidateHost
func (o OvfManager) ValidateHost(ctx context.Context, ovfDescriptor string, host Reference, vhp types.OvfValidateHostParams) (*types.OvfValidateHostResult, error) {
	req := types.ValidateHost{
		This:          o.Reference(),
		OvfDescriptor: ovfDescriptor,
		Host:          host.Reference(),
		Vhp:           vhp,
	}

	res, err := methods.ValidateHost(ctx, o.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}
