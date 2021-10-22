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
	"context"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

func TestObjectCustomFields(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	defer m.Remove()
	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	cfm, err := object.GetCustomFieldsManager(c.Client)
	if err != nil {
		t.Fatal(err)
	}

	vm := Map.Any("VirtualMachine").(*VirtualMachine)
	vmm := object.NewVirtualMachine(c.Client, vm.Reference())

	fieldName := "testField"
	fieldValue := "12345"

	// Test that field is not created
	err = vmm.SetCustomValue(ctx, fieldName, fieldValue)
	if err == nil {
		t.Fatalf("expected error")
	}

	if len(vm.AvailableField) != 0 {
		t.Fatalf("vm.AvailableField length expected 0, got %d", len(vm.AvailableField))
	}

	// Create field
	field, err := cfm.Add(ctx, fieldName, vm.Reference().Type, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(vm.AvailableField) != 1 {
		t.Fatalf("len(vm.AvailableField) expected 1, got %d", len(vm.AvailableField))
	}

	if vm.AvailableField[0].Key != field.Key {
		t.Fatalf("vm.AvailableField[0].Key expected %d, got %d", field.Key, vm.AvailableField[0].Key)
	}
	if vm.AvailableField[0].Name != field.Name {
		t.Fatalf("vm.AvailableField[0].Name expected %s, got %s", field.Name, vm.AvailableField[0].Name)
	}
	if vm.AvailableField[0].Type != field.Type {
		t.Fatalf("vm.AvailableField[0].Type expected %s, got %s", field.Type, vm.AvailableField[0].Type)
	}
	if vm.AvailableField[0].ManagedObjectType != field.ManagedObjectType {
		t.Fatalf("vm.AvailableField[0].ManagedObjectType expected %s, got %s",
			field.ManagedObjectType, vm.AvailableField[0].ManagedObjectType)
	}
	if vm.AvailableField[0].FieldDefPrivileges != field.FieldDefPrivileges {
		t.Fatalf("vm.AvailableField[0].FieldDefPrivileges expected %s, got %s",
			field.FieldDefPrivileges, vm.AvailableField[0].FieldDefPrivileges)
	}
	if vm.AvailableField[0].FieldInstancePrivileges != field.FieldInstancePrivileges {
		t.Fatalf("vm.AvailableField[0].FieldInstancePrivileges expected %s, got %s",
			field.FieldInstancePrivileges, vm.AvailableField[0].FieldInstancePrivileges)
	}

	// Test that field with duplicate name can't be created
	_, err = cfm.Add(ctx, fieldName, vm.Reference().Type, nil, nil)
	if err == nil {
		t.Fatalf("expected error")
	}

	// Create second field
	_, err = cfm.Add(ctx, fieldName+"2", vm.Reference().Type, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(vm.AvailableField) != 2 {
		t.Fatalf("len(vm.AvailableField) expected 2, got %d", len(vm.AvailableField))
	}

	// Set field
	err = vmm.SetCustomValue(ctx, fieldName, fieldValue)
	if err != nil {
		t.Fatal(err)
	}

	if len(vm.CustomValue) != 1 {
		t.Fatalf("len(vm.CustomValue) expected 1, got %d", len(vm.CustomValue))
	}

	if len(vm.Value) != 1 {
		t.Fatalf("len(vm.Value) expected 1, got %d", len(vm.Value))
	}

	if vm.CustomValue[0].(*types.CustomFieldStringValue).Key != field.Key {
		t.Fatalf("vm.CustomValue[0].Key expected %d, got %d",
			field.Key, vm.CustomValue[0].(*types.CustomFieldStringValue).Key)
	}
	if vm.CustomValue[0].(*types.CustomFieldStringValue).Value != fieldValue {
		t.Fatalf("vm.CustomValue[0].Value expected %s, got %s",
			fieldValue, vm.CustomValue[0].(*types.CustomFieldStringValue).Value)
	}

	if vm.Value[0].(*types.CustomFieldStringValue).Key != field.Key {
		t.Fatalf("vm.Value[0].Key expected %d, got %d",
			field.Key, vm.Value[0].(*types.CustomFieldStringValue).Key)
	}
	if vm.Value[0].(*types.CustomFieldStringValue).Value != fieldValue {
		t.Fatalf("vm.Value[0].Value expected %s, got %s",
			fieldValue, vm.Value[0].(*types.CustomFieldStringValue).Value)
	}

	// Rename field
	newName := field.Name + "_renamed"
	err = cfm.Rename(ctx, field.Key, newName)
	if err != nil {
		t.Fatal(err)
	}

	if vm.AvailableField[0].Name != newName {
		t.Fatalf("vm.AvailableField[0].Name expected %s, got %s", newName, vm.AvailableField[0].Name)
	}

	// Remove field
	err = cfm.Remove(ctx, field.Key)
	if err != nil {
		t.Fatal(err)
	}

	if len(vm.AvailableField) != 1 {
		t.Fatalf("len(vm.AvailableField) expected 1, got %d", len(vm.AvailableField))
	}

	if len(vm.CustomValue) != 0 {
		t.Fatalf("len(vm.CustomValue) expected 0, got %d", len(vm.CustomValue))
	}

	if len(vm.Value) != 0 {
		t.Fatalf("len(vm.Value) expected 0, got %d", len(vm.Value))
	}

	// Test that remaining field key is different from removed field
	if vm.AvailableField[0].Key == field.Key {
		t.Fatalf("vm.AvailableField[0].Key expected to not be equal %d", field.Key)
	}

	// Remove remaining field
	err = cfm.Remove(ctx, vm.AvailableField[0].Key)
	if err != nil {
		t.Fatal(err)
	}

	if len(vm.AvailableField) != 0 {
		t.Fatalf("len(vm.AvailableField) expected 0, got %d", len(vm.AvailableField))
	}
}
