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

// Iterates through all entities of passed field type;
// Removes found field from their custom field properties.
func entitiesFieldRemove(field types.CustomFieldDef) {
	entities := Map.All(field.ManagedObjectType)
	for _, e := range entities {
		entity := e.Entity()
		Map.WithLock(entity, func() {
			aFields := entity.AvailableField
			for i, aField := range aFields {
				if aField.Key == field.Key {
					entity.AvailableField = append(aFields[:i], aFields[i+1:]...)
					break
				}
			}

			values := e.Entity().Value
			for i, value := range values {
				if value.(*types.CustomFieldStringValue).Key == field.Key {
					entity.Value = append(values[:i], values[i+1:]...)
					break
				}
			}

			cValues := e.Entity().CustomValue
			for i, cValue := range cValues {
				if cValue.(*types.CustomFieldStringValue).Key == field.Key {
					entity.CustomValue = append(cValues[:i], cValues[i+1:]...)
					break
				}
			}
		})
	}
}

// Iterates through all entities of passed field type;
// Renames found field in entity's AvailableField property.
func entitiesFieldRename(field types.CustomFieldDef) {
	entities := Map.All(field.ManagedObjectType)
	for _, e := range entities {
		entity := e.Entity()
		Map.WithLock(entity, func() {
			aFields := entity.AvailableField
			for i, aField := range aFields {
				if aField.Key == field.Key {
					aFields[i].Name = field.Name
					break
				}
			}
		})
	}
}

func (c *CustomFieldsManager) findByNameType(name, moType string) (int, *types.CustomFieldDef) {
	for i, field := range c.Field {
		if (field.ManagedObjectType == "" || field.ManagedObjectType == moType || moType == "") &&
			field.Name == name {
			return i, &c.Field[i]
		}
	}

	return -1, nil
}

func (c *CustomFieldsManager) findByKey(key int32) (int, *types.CustomFieldDef) {
	for i, field := range c.Field {
		if field.Key == key {
			return i, &c.Field[i]
		}
	}

	return -1, nil
}

func (c *CustomFieldsManager) AddCustomFieldDef(req *types.AddCustomFieldDef) soap.HasFault {
	body := &methods.AddCustomFieldDefBody{}

	_, field := c.findByNameType(req.Name, req.MoType)
	if field != nil {
		body.Fault_ = Fault("", &types.DuplicateName{
			Name:   req.Name,
			Object: c.Reference(),
		})
		return body
	}

	def := types.CustomFieldDef{
		Key:                     c.nextKey,
		Name:                    req.Name,
		ManagedObjectType:       req.MoType,
		Type:                    req.MoType,
		FieldDefPrivileges:      req.FieldDefPolicy,
		FieldInstancePrivileges: req.FieldPolicy,
	}

	entities := Map.All(req.MoType)
	for _, e := range entities {
		entity := e.Entity()
		Map.WithLock(entity, func() {
			entity.AvailableField = append(entity.AvailableField, def)
		})
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

	i, field := c.findByKey(req.Key)
	if field == nil {
		body.Fault_ = Fault("", &types.NotFound{})
		return body
	}

	entitiesFieldRemove(*field)

	c.Field = append(c.Field[:i], c.Field[i+1:]...)

	body.Res = &types.RemoveCustomFieldDefResponse{}
	return body
}

func (c *CustomFieldsManager) RenameCustomFieldDef(req *types.RenameCustomFieldDef) soap.HasFault {
	body := &methods.RenameCustomFieldDefBody{}

	_, field := c.findByKey(req.Key)
	if field == nil {
		body.Fault_ = Fault("", &types.NotFound{})
		return body
	}

	field.Name = req.Name

	entitiesFieldRename(*field)

	body.Res = &types.RenameCustomFieldDefResponse{}
	return body
}

func (c *CustomFieldsManager) SetField(ctx *Context, req *types.SetField) soap.HasFault {
	body := &methods.SetFieldBody{}

	_, field := c.findByKey(req.Key)
	if field == nil {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "key"})
		return body
	}

	newValue := &types.CustomFieldStringValue{
		CustomFieldValue: types.CustomFieldValue{Key: req.Key},
		Value:            req.Value,
	}

	entity := Map.Get(req.Entity).(mo.Entity).Entity()
	ctx.WithLock(entity, func() {
		entity.CustomValue = append(entity.CustomValue, newValue)
		entity.Value = append(entity.Value, newValue)
	})

	body.Res = &types.SetFieldResponse{}
	return body
}
