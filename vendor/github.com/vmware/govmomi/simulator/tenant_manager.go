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

package simulator

import (
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type TenantManager struct {
	mo.TenantTenantManager

	spEntities map[types.ManagedObjectReference]bool
}

func (t *TenantManager) init(r *Registry) {
	t.spEntities = make(map[types.ManagedObjectReference]bool)
}

func (t *TenantManager) markEntities(entities []types.ManagedObjectReference) {
	for _, e := range entities {
		t.spEntities[e] = true
	}
}

func (t *TenantManager) unmarkEntities(entities []types.ManagedObjectReference) {
	for _, e := range entities {
		_, ok := t.spEntities[e]
		if ok {
			delete(t.spEntities, e)
		}
	}
}

func (t *TenantManager) getEntities() []types.ManagedObjectReference {
	entities := []types.ManagedObjectReference{}
	for e := range t.spEntities {
		entities = append(entities, e)
	}
	return entities
}

func (t *TenantManager) MarkServiceProviderEntities(req *types.MarkServiceProviderEntities) soap.HasFault {
	body := new(methods.MarkServiceProviderEntitiesBody)
	t.markEntities(req.Entity)
	body.Res = &types.MarkServiceProviderEntitiesResponse{}
	return body
}

func (t *TenantManager) UnmarkServiceProviderEntities(req *types.UnmarkServiceProviderEntities) soap.HasFault {
	body := new(methods.UnmarkServiceProviderEntitiesBody)
	t.unmarkEntities(req.Entity)
	body.Res = &types.UnmarkServiceProviderEntitiesResponse{}
	return body
}

func (t *TenantManager) RetrieveServiceProviderEntities(req *types.RetrieveServiceProviderEntities) soap.HasFault {
	body := new(methods.RetrieveServiceProviderEntitiesBody)
	body.Res = &types.RetrieveServiceProviderEntitiesResponse{
		Returnval: t.getEntities(),
	}
	return body
}
