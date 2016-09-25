/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

type OptionManager struct {
	Common
}

func NewOptionManager(c *vim25.Client, ref types.ManagedObjectReference) *OptionManager {
	return &OptionManager{
		Common: NewCommon(c, ref),
	}
}

func (m OptionManager) Query(ctx context.Context, name string) ([]types.BaseOptionValue, error) {
	req := types.QueryOptions{
		This: m.Reference(),
		Name: name,
	}

	res, err := methods.QueryOptions(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m OptionManager) Update(ctx context.Context, value []types.BaseOptionValue) error {
	req := types.UpdateOptions{
		This:         m.Reference(),
		ChangedValue: value,
	}

	_, err := methods.UpdateOptions(ctx, m.Client(), &req)
	return err
}
