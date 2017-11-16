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
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type HostServiceSystem struct {
	Common
}

func NewHostServiceSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostServiceSystem {
	return &HostServiceSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostServiceSystem) Service(ctx context.Context) ([]types.HostService, error) {
	var ss mo.HostServiceSystem

	err := s.Properties(ctx, s.Reference(), []string{"serviceInfo.service"}, &ss)
	if err != nil {
		return nil, err
	}

	return ss.ServiceInfo.Service, nil
}

func (s HostServiceSystem) Start(ctx context.Context, id string) error {
	req := types.StartService{
		This: s.Reference(),
		Id:   id,
	}

	_, err := methods.StartService(ctx, s.Client(), &req)
	return err
}

func (s HostServiceSystem) Stop(ctx context.Context, id string) error {
	req := types.StopService{
		This: s.Reference(),
		Id:   id,
	}

	_, err := methods.StopService(ctx, s.Client(), &req)
	return err
}

func (s HostServiceSystem) Restart(ctx context.Context, id string) error {
	req := types.RestartService{
		This: s.Reference(),
		Id:   id,
	}

	_, err := methods.RestartService(ctx, s.Client(), &req)
	return err
}

func (s HostServiceSystem) UpdatePolicy(ctx context.Context, id string, policy string) error {
	req := types.UpdateServicePolicy{
		This:   s.Reference(),
		Id:     id,
		Policy: policy,
	}

	_, err := methods.UpdateServicePolicy(ctx, s.Client(), &req)
	return err
}
