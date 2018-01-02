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

package guest

import (
	"context"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type OperationsManager struct {
	c  *vim25.Client
	vm types.ManagedObjectReference
}

func NewOperationsManager(c *vim25.Client, vm types.ManagedObjectReference) *OperationsManager {
	return &OperationsManager{c, vm}
}

func (m OperationsManager) retrieveOne(ctx context.Context, p string, dst *mo.GuestOperationsManager) error {
	pc := property.DefaultCollector(m.c)
	return pc.RetrieveOne(ctx, *m.c.ServiceContent.GuestOperationsManager, []string{p}, dst)
}

func (m OperationsManager) AuthManager(ctx context.Context) (*AuthManager, error) {
	var g mo.GuestOperationsManager

	err := m.retrieveOne(ctx, "authManager", &g)
	if err != nil {
		return nil, err
	}

	return &AuthManager{*g.AuthManager, m.vm, m.c}, nil
}

func (m OperationsManager) FileManager(ctx context.Context) (*FileManager, error) {
	var g mo.GuestOperationsManager

	err := m.retrieveOne(ctx, "fileManager", &g)
	if err != nil {
		return nil, err
	}

	return &FileManager{*g.FileManager, m.vm, m.c}, nil
}

func (m OperationsManager) ProcessManager(ctx context.Context) (*ProcessManager, error) {
	var g mo.GuestOperationsManager

	err := m.retrieveOne(ctx, "processManager", &g)
	if err != nil {
		return nil, err
	}

	return &ProcessManager{*g.ProcessManager, m.vm, m.c}, nil
}
