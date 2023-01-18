/*
Copyright (c) 2021 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type TenantManager struct {
	Common
}

func NewTenantManager(c *vim25.Client) *TenantManager {
	t := TenantManager{
		Common: NewCommon(c, *c.ServiceContent.TenantManager),
	}

	return &t
}

func (t TenantManager) MarkServiceProviderEntities(ctx context.Context, entities []types.ManagedObjectReference) error {
	req := types.MarkServiceProviderEntities{
		This:   t.Reference(),
		Entity: entities,
	}

	_, err := methods.MarkServiceProviderEntities(ctx, t.Client(), &req)
	if err != nil {
		return err
	}

	return nil
}

func (t TenantManager) UnmarkServiceProviderEntities(ctx context.Context, entities []types.ManagedObjectReference) error {
	req := types.UnmarkServiceProviderEntities{
		This:   t.Reference(),
		Entity: entities,
	}

	_, err := methods.UnmarkServiceProviderEntities(ctx, t.Client(), &req)
	if err != nil {
		return err
	}

	return nil
}

func (t TenantManager) RetrieveServiceProviderEntities(ctx context.Context) ([]types.ManagedObjectReference, error) {
	req := types.RetrieveServiceProviderEntities{
		This: t.Reference(),
	}

	res, err := methods.RetrieveServiceProviderEntities(ctx, t.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}
