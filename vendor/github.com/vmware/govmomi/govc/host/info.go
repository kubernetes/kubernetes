/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package host

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	var res infoResult
	var props []string

	if cmd.OutputFlag.JSON {
		props = nil // Load everything
	} else {
		props = []string{"summary"} // Load summary
	}

	// We could do without the -host flag, leaving it for compat
	host, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	// Default only if there is a single host
	if host == nil && f.NArg() == 0 {
		host, err = cmd.HostSystem()
		if err != nil {
			return err
		}
	}

	if host != nil {
		res.objects = append(res.objects, host)
	} else {
		res.objects, err = cmd.HostSystems(f.Args())
		if err != nil {
			return err
		}
	}

	if len(res.objects) != 0 {
		refs := make([]types.ManagedObjectReference, 0, len(res.objects))
		for _, o := range res.objects {
			refs = append(refs, o.Reference())
		}

		pc := property.DefaultCollector(c)
		err = pc.Retrieve(ctx, refs, props, &res.HostSystems)
		if err != nil {
			return err
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	HostSystems []mo.HostSystem
	objects     []*object.HostSystem
}

func (r *infoResult) Write(w io.Writer) error {
	// Maintain order via r.objects as Property collector does not always return results in order.
	objects := make(map[types.ManagedObjectReference]mo.HostSystem, len(r.HostSystems))
	for _, o := range r.HostSystems {
		objects[o.Reference()] = o
	}

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, o := range r.objects {
		host := objects[o.Reference()]
		s := host.Summary
		h := s.Hardware
		z := s.QuickStats
		ncpu := int32(h.NumCpuPkgs * h.NumCpuCores)
		cpuUsage := 100 * float64(z.OverallCpuUsage) / float64(ncpu*h.CpuMhz)
		memUsage := 100 * float64(z.OverallMemoryUsage<<20) / float64(h.MemorySize)

		fmt.Fprintf(tw, "Name:\t%s\n", s.Config.Name)
		fmt.Fprintf(tw, "  Path:\t%s\n", o.InventoryPath)
		fmt.Fprintf(tw, "  Manufacturer:\t%s\n", h.Vendor)
		fmt.Fprintf(tw, "  Logical CPUs:\t%d CPUs @ %dMHz\n", ncpu, h.CpuMhz)
		fmt.Fprintf(tw, "  Processor type:\t%s\n", h.CpuModel)
		fmt.Fprintf(tw, "  CPU usage:\t%d MHz (%.1f%%)\n", z.OverallCpuUsage, cpuUsage)
		fmt.Fprintf(tw, "  Memory:\t%dMB\n", h.MemorySize/(1024*1024))
		fmt.Fprintf(tw, "  Memory usage:\t%d MB (%.1f%%)\n", z.OverallMemoryUsage, memUsage)
		fmt.Fprintf(tw, "  Boot time:\t%s\n", s.Runtime.BootTime)
	}

	return tw.Flush()
}
