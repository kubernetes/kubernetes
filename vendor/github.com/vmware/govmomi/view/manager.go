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

package view

import (
	"context"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type Manager struct {
	object.Common
}

func NewManager(c *vim25.Client) *Manager {
	m := Manager{
		object.NewCommon(c, *c.ServiceContent.ViewManager),
	}

	return &m
}

func (m Manager) CreateListView(ctx context.Context, objects []types.ManagedObjectReference) (*ListView, error) {
	req := types.CreateListView{
		This: m.Common.Reference(),
		Obj:  objects,
	}

	res, err := methods.CreateListView(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewListView(m.Client(), res.Returnval), nil
}

func (m Manager) CreateContainerView(ctx context.Context, container types.ManagedObjectReference, managedObjectTypes []string, recursive bool) (*ContainerView, error) {

	req := types.CreateContainerView{
		This:      m.Common.Reference(),
		Container: container,
		Recursive: recursive,
		Type:      managedObjectTypes,
	}

	res, err := methods.CreateContainerView(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewContainerView(m.Client(), res.Returnval), nil
}
