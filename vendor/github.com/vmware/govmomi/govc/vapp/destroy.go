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

package vapp

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type destroy struct {
	*flags.DatacenterFlag
}

func init() {
	cli.Register("vapp.destroy", &destroy{})
}

func (cmd *destroy) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
}

func (cmd *destroy) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *destroy) Usage() string {
	return "VAPP..."
}

func (cmd *destroy) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, arg := range f.Args() {
		vapps, err := finder.VirtualAppList(ctx, arg)
		if err != nil {
			if _, ok := err.(*find.NotFoundError); ok {
				// Ignore if vapp cannot be found
				continue
			}

			return err
		}

		for _, vapp := range vapps {
			powerOff := func() error {
				task, err := vapp.PowerOff(ctx, false)
				if err != nil {
					return err
				}
				err = task.Wait(ctx)
				if err != nil {
					// it's safe to ignore if the vapp is already powered off
					if f, ok := err.(types.HasFault); ok {
						switch f.Fault().(type) {
						case *types.InvalidPowerState:
							return nil
						}
					}
					return err
				}
				return nil
			}
			if err := powerOff(); err != nil {
				return err
			}

			destroy := func() error {
				task, err := vapp.Destroy(ctx)
				if err != nil {
					return err
				}
				err = task.Wait(ctx)
				if err != nil {
					return err
				}
				return nil
			}
			if err := destroy(); err != nil {
				return err
			}
		}
	}

	return nil
}
