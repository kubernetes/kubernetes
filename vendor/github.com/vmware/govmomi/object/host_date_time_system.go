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
	"time"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type HostDateTimeSystem struct {
	Common
}

func NewHostDateTimeSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostDateTimeSystem {
	return &HostDateTimeSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostDateTimeSystem) UpdateConfig(ctx context.Context, config types.HostDateTimeConfig) error {
	req := types.UpdateDateTimeConfig{
		This:   s.Reference(),
		Config: config,
	}

	_, err := methods.UpdateDateTimeConfig(ctx, s.c, &req)
	return err
}

func (s HostDateTimeSystem) Update(ctx context.Context, date time.Time) error {
	req := types.UpdateDateTime{
		This:     s.Reference(),
		DateTime: date,
	}

	_, err := methods.UpdateDateTime(ctx, s.c, &req)
	return err
}

func (s HostDateTimeSystem) Query(ctx context.Context) (*time.Time, error) {
	req := types.QueryDateTime{
		This: s.Reference(),
	}

	res, err := methods.QueryDateTime(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}
