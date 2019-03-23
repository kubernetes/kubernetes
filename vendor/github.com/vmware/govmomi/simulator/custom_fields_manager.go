/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type CustomFieldsManager struct {
	mo.CustomFieldsManager

	nextKey int32
}

func NewCustomFieldsManager(ref types.ManagedObjectReference) object.Reference {
	m := &CustomFieldsManager{}
	m.Self = ref
	return m
}

func (c *CustomFieldsManager) find(key int32) (int, *types.CustomFieldDef) {
	for i, field := range c.Field {
		if field.Key == key {
			return i, &c.Field[i]
		}
	}

	return -1, nil
}

func (c *CustomFieldsManager) AddCustomFieldDef(req *types.AddCustomFieldDef) soap.HasFault {
	body := &methods.AddCustomFieldDefBody{}

	def := types.CustomFieldDef{
		Key:                     c.nextKey,
		Name:                    req.Name,
		ManagedObjectType:       req.MoType,
		Type:                    req.MoType,
		FieldDefPrivileges:      req.FieldDefPolicy,
		FieldInstancePrivileges: req.FieldPolicy,
	}

	c.Field = append(c.Field, def)
	c.nextKey++

	body.Res = &types.AddCustomFieldDefResponse{
		Returnval: def,
	}
	return body
}

func (c *CustomFieldsManager) RemoveCustomFieldDef(req *types.RemoveCustomFieldDef) soap.HasFault {
	body := &methods.RemoveCustomFieldDefBody{}

	i, field := c.find(req.Key)
	if field == nil {
		body.Fault_ = Fault("", &types.NotFound{})
		return body
	}

	c.Field = append(c.Field[:i], c.Field[i+1:]...)

	body.Res = &types.RemoveCustomFieldDefResponse{}
	return body
}

func (c *CustomFieldsManager) RenameCustomFieldDef(req *types.RenameCustomFieldDef) soap.HasFault {
	body := &methods.RenameCustomFieldDefBody{}

	_, field := c.find(req.Key)
	if field == nil {
		body.Fault_ = Fault("", &types.NotFound{})
		return body
	}

	field.Name = req.Name

	body.Res = &types.RenameCustomFieldDefResponse{}
	return body
}

func (c *CustomFieldsManager) SetField(req *types.SetField) soap.HasFault {
	body := &methods.SetFieldBody{}

	entity := Map.Get(req.Entity).(mo.Entity).Entity()
	Map.WithLock(entity, func() {
		entity.CustomValue = append(entity.CustomValue, &types.CustomFieldStringValue{
			CustomFieldValue: types.CustomFieldValue{Key: req.Key},
			Value:            req.Value,
		})
	})

	body.Res = &types.SetFieldResponse{}
	return body
}
