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

package object

import (
	"errors"
	"fmt"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type HttpNfcLease struct {
	Common
}

func NewHttpNfcLease(c *vim25.Client, ref types.ManagedObjectReference) *HttpNfcLease {
	return &HttpNfcLease{
		Common: NewCommon(c, ref),
	}
}

// HttpNfcLeaseAbort wraps methods.HttpNfcLeaseAbort
func (o HttpNfcLease) HttpNfcLeaseAbort(ctx context.Context, fault *types.LocalizedMethodFault) error {
	req := types.HttpNfcLeaseAbort{
		This:  o.Reference(),
		Fault: fault,
	}

	_, err := methods.HttpNfcLeaseAbort(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// HttpNfcLeaseComplete wraps methods.HttpNfcLeaseComplete
func (o HttpNfcLease) HttpNfcLeaseComplete(ctx context.Context) error {
	req := types.HttpNfcLeaseComplete{
		This: o.Reference(),
	}

	_, err := methods.HttpNfcLeaseComplete(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// HttpNfcLeaseGetManifest wraps methods.HttpNfcLeaseGetManifest
func (o HttpNfcLease) HttpNfcLeaseGetManifest(ctx context.Context) error {
	req := types.HttpNfcLeaseGetManifest{
		This: o.Reference(),
	}

	_, err := methods.HttpNfcLeaseGetManifest(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// HttpNfcLeaseProgress wraps methods.HttpNfcLeaseProgress
func (o HttpNfcLease) HttpNfcLeaseProgress(ctx context.Context, percent int32) error {
	req := types.HttpNfcLeaseProgress{
		This:    o.Reference(),
		Percent: percent,
	}

	_, err := methods.HttpNfcLeaseProgress(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

func (o HttpNfcLease) Wait(ctx context.Context) (*types.HttpNfcLeaseInfo, error) {
	var lease mo.HttpNfcLease

	pc := property.DefaultCollector(o.c)
	err := property.Wait(ctx, pc, o.Reference(), []string{"state", "info", "error"}, func(pc []types.PropertyChange) bool {
		done := false

		for _, c := range pc {
			if c.Val == nil {
				continue
			}

			switch c.Name {
			case "error":
				val := c.Val.(types.LocalizedMethodFault)
				lease.Error = &val
				done = true
			case "info":
				val := c.Val.(types.HttpNfcLeaseInfo)
				lease.Info = &val
			case "state":
				lease.State = c.Val.(types.HttpNfcLeaseState)
				if lease.State != types.HttpNfcLeaseStateInitializing {
					done = true
				}
			}
		}

		return done
	})

	if err != nil {
		return nil, err
	}

	if lease.State == types.HttpNfcLeaseStateReady {
		return lease.Info, nil
	}

	if lease.Error != nil {
		return nil, errors.New(lease.Error.LocalizedMessage)
	}

	return nil, fmt.Errorf("unexpected nfc lease state: %s", lease.State)
}
