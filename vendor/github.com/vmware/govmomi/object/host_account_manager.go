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

type HostAccountManager struct {
	Common
}

func NewHostAccountManager(c *vim25.Client, ref types.ManagedObjectReference) *HostAccountManager {
	return &HostAccountManager{
		Common: NewCommon(c, ref),
	}
}

func (m HostAccountManager) Create(ctx context.Context, user *types.HostAccountSpec) error {
	req := types.CreateUser{
		This: m.Reference(),
		User: user,
	}

	_, err := methods.CreateUser(ctx, m.Client(), &req)
	return err
}

func (m HostAccountManager) Update(ctx context.Context, user *types.HostAccountSpec) error {
	req := types.UpdateUser{
		This: m.Reference(),
		User: user,
	}

	_, err := methods.UpdateUser(ctx, m.Client(), &req)
	return err
}

func (m HostAccountManager) Remove(ctx context.Context, userName string) error {
	req := types.RemoveUser{
		This:     m.Reference(),
		UserName: userName,
	}

	_, err := methods.RemoveUser(ctx, m.Client(), &req)
	return err
}
