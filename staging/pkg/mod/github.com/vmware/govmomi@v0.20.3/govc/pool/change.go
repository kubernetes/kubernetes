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

package pool

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.DatacenterFlag
	*ResourceConfigSpecFlag

	name string
}

func init() {
	cli.Register("pool.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
	cmd.ResourceConfigSpecFlag = &ResourceConfigSpecFlag{
		ResourceConfigSpec: types.ResourceConfigSpec{
			CpuAllocation: types.ResourceAllocationInfo{
				Shares: new(types.SharesInfo),
			},
			MemoryAllocation: types.ResourceAllocationInfo{
				Shares: new(types.SharesInfo),
			},
		},
	}
	cmd.ResourceConfigSpecFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "name", "", "Resource pool name")
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourceConfigSpecFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Usage() string {
	return "POOL..."
}

func (cmd *change) Description() string {
	return "Change the configuration of one or more resource POOLs.\n" + poolNameHelp
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, ra := range []*types.ResourceAllocationInfo{&cmd.CpuAllocation, &cmd.MemoryAllocation} {
		if ra.Shares.Level == "" {
			ra.Shares = nil
		}
	}

	for _, arg := range f.Args() {
		pools, err := finder.ResourcePoolListAll(ctx, arg)
		if err != nil {
			return err
		}

		for _, pool := range pools {
			err := pool.UpdateConfig(ctx, cmd.name, &cmd.ResourceConfigSpec)
			if err != nil {
				return err
			}
		}
	}

	return nil
}
