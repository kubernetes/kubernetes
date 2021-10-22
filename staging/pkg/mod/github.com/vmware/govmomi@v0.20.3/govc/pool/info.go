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
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.DatacenterFlag
	*flags.OutputFlag

	pools bool
	apps  bool
}

func init() {
	cli.Register("pool.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.pools, "p", true, "List resource pools")
	f.BoolVar(&cmd.apps, "a", false, "List virtual app resource pools")
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Usage() string {
	return "POOL..."
}

func (cmd *info) Description() string {
	return "Retrieve information about one or more resource POOLs.\n" + poolNameHelp
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	var res infoResult
	var props []string

	if cmd.OutputFlag.All() {
		props = nil
	} else {
		props = []string{
			"name",
			"config.cpuAllocation",
			"config.memoryAllocation",
			"runtime.cpu",
			"runtime.memory",
		}
	}

	var vapps []*object.VirtualApp

	for _, arg := range f.Args() {
		if cmd.pools {
			objects, err := finder.ResourcePoolList(ctx, arg)
			if err != nil {
				if _, ok := err.(*find.NotFoundError); !ok {
					return err
				}
			}
			res.objects = append(res.objects, objects...)
		}

		if cmd.apps {
			apps, err := finder.VirtualAppList(ctx, arg)
			if err != nil {
				if _, ok := err.(*find.NotFoundError); !ok {
					return err
				}
			}
			vapps = append(vapps, apps...)
		}
	}

	if len(res.objects) != 0 {
		refs := make([]types.ManagedObjectReference, 0, len(res.objects))
		for _, o := range res.objects {
			refs = append(refs, o.Reference())
		}

		pc := property.DefaultCollector(c)
		err = pc.Retrieve(ctx, refs, props, &res.ResourcePools)
		if err != nil {
			return err
		}
	}

	if len(vapps) != 0 {
		var apps []mo.VirtualApp
		refs := make([]types.ManagedObjectReference, 0, len(vapps))
		for _, o := range vapps {
			refs = append(refs, o.Reference())
			p := object.NewResourcePool(c, o.Reference())
			p.InventoryPath = o.InventoryPath
			res.objects = append(res.objects, p)
		}

		pc := property.DefaultCollector(c)
		err = pc.Retrieve(ctx, refs, props, &apps)
		if err != nil {
			return err
		}

		for _, app := range apps {
			res.ResourcePools = append(res.ResourcePools, app.ResourcePool)
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	ResourcePools []mo.ResourcePool
	objects       []*object.ResourcePool
}

func (r *infoResult) Write(w io.Writer) error {
	// Maintain order via r.objects as Property collector does not always return results in order.
	objects := make(map[types.ManagedObjectReference]mo.ResourcePool, len(r.ResourcePools))
	for _, o := range r.ResourcePools {
		objects[o.Reference()] = o
	}

	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, o := range r.objects {
		pool := objects[o.Reference()]
		fmt.Fprintf(tw, "Name:\t%s\n", pool.Name)
		fmt.Fprintf(tw, "  Path:\t%s\n", o.InventoryPath)

		writeInfo(tw, "CPU", "MHz", &pool.Runtime.Cpu, pool.Config.CpuAllocation)
		pool.Runtime.Memory.MaxUsage >>= 20
		pool.Runtime.Memory.OverallUsage >>= 20
		writeInfo(tw, "Mem", "MB", &pool.Runtime.Memory, pool.Config.MemoryAllocation)
	}

	return tw.Flush()
}

func writeInfo(w io.Writer, name string, units string, ru *types.ResourcePoolResourceUsage, ra types.ResourceAllocationInfo) {
	usage := 100.0 * float64(ru.OverallUsage) / float64(ru.MaxUsage)
	shares := ""
	limit := "unlimited"

	if ra.Shares.Level == types.SharesLevelCustom {
		shares = fmt.Sprintf(" (%d)", ra.Shares.Shares)
	}

	if ra.Limit != nil {
		limit = fmt.Sprintf("%d%s", *ra.Limit, units)
	}

	fmt.Fprintf(w, "  %s Usage:\t%d%s (%0.1f%%)\n", name, ru.OverallUsage, units, usage)
	fmt.Fprintf(w, "  %s Shares:\t%s%s\n", name, ra.Shares.Level, shares)
	fmt.Fprintf(w, "  %s Reservation:\t%d%s (expandable=%v)\n", name, *ra.Reservation, units, *ra.ExpandableReservation)
	fmt.Fprintf(w, "  %s Limit:\t%s\n", name, limit)
}
