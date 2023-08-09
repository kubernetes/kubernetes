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

package view

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type ListView struct {
	ManagedObjectView
}

func NewListView(c *vim25.Client, ref types.ManagedObjectReference) *ListView {
	return &ListView{
		ManagedObjectView: *NewManagedObjectView(c, ref),
	}
}

func (v ListView) Add(ctx context.Context, refs []types.ManagedObjectReference) error {
	req := types.ModifyListView{
		This: v.Reference(),
		Add:  refs,
	}
	_, err := methods.ModifyListView(ctx, v.Client(), &req)
	return err
}

func (v ListView) Remove(ctx context.Context, refs []types.ManagedObjectReference) error {
	req := types.ModifyListView{
		This:   v.Reference(),
		Remove: refs,
	}
	_, err := methods.ModifyListView(ctx, v.Client(), &req)
	return err
}

func (v ListView) Reset(ctx context.Context, refs []types.ManagedObjectReference) error {
	req := types.ResetListView{
		This: v.Reference(),
		Obj:  refs,
	}
	_, err := methods.ResetListView(ctx, v.Client(), &req)
	return err
}
