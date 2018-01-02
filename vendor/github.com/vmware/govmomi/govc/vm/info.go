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

package vm

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"context"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"

	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.SearchFlag

	WaitForIP       bool
	General         bool
	ExtraConfig     bool
	Resources       bool
	ToolsConfigInfo bool
}

func init() {
	cli.Register("vm.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.WaitForIP, "waitip", false, "Wait for VM to acquire IP address")
	f.BoolVar(&cmd.General, "g", true, "Show general summary")
	f.BoolVar(&cmd.ExtraConfig, "e", false, "Show ExtraConfig")
	f.BoolVar(&cmd.Resources, "r", false, "Show resource summary")
	f.BoolVar(&cmd.ToolsConfigInfo, "t", false, "Show ToolsConfigInfo")
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		if _, ok := err.(*find.NotFoundError); ok {
			// Continue with empty VM slice
		} else {
			return err
		}
	}

	refs := make([]types.ManagedObjectReference, 0, len(vms))
	for _, vm := range vms {
		refs = append(refs, vm.Reference())
	}

	var res infoResult
	var props []string

	if cmd.OutputFlag.JSON {
		props = nil // Load everything
	} else {
		props = []string{"summary"} // Load summary
		if cmd.General {
			props = append(props, "guest.ipAddress")
		}
		if cmd.ExtraConfig {
			props = append(props, "config.extraConfig")
		}
		if cmd.Resources {
			props = append(props, "datastore", "network")
		}
		if cmd.ToolsConfigInfo {
			props = append(props, "config.tools")
		}
	}

	pc := property.DefaultCollector(c)
	if len(refs) != 0 {
		err = pc.Retrieve(ctx, refs, props, &res.VirtualMachines)
		if err != nil {
			return err
		}
	}

	if cmd.WaitForIP {
		for i, vm := range res.VirtualMachines {
			if vm.Guest == nil || vm.Guest.IpAddress == "" {
				_, err = vms[i].WaitForIP(ctx)
				if err != nil {
					return err
				}
				// Reload virtual machine object
				err = pc.RetrieveOne(ctx, vms[i].Reference(), props, &res.VirtualMachines[i])
				if err != nil {
					return err
				}
			}
		}
	}

	if !cmd.OutputFlag.JSON {
		res.objects = vms
		res.cmd = cmd
		if err = res.collectReferences(pc, ctx); err != nil {
			return err
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	VirtualMachines []mo.VirtualMachine
	objects         []*object.VirtualMachine
	entities        map[types.ManagedObjectReference]string
	cmd             *info
}

// collectReferences builds a unique set of MORs to the set of VirtualMachines,
// so we can collect properties in a single call for each reference type {host,datastore,network}.
func (r *infoResult) collectReferences(pc *property.Collector, ctx context.Context) error {
	r.entities = make(map[types.ManagedObjectReference]string) // MOR -> Name map

	var host []mo.HostSystem
	var network []mo.Network
	var opaque []mo.OpaqueNetwork
	var dvp []mo.DistributedVirtualPortgroup
	var datastore []mo.Datastore
	// Table to drive inflating refs to their mo.* counterparts (dest)
	// and save() the Name to r.entities w/o using reflection here.
	// Note that we cannot use a []mo.ManagedEntity here, since mo.Network has its own 'Name' field,
	// the mo.Network.ManagedEntity.Name field will not be set.
	vrefs := map[string]*struct {
		dest interface{}
		refs []types.ManagedObjectReference
		save func()
	}{
		"HostSystem": {
			&host, nil, func() {
				for _, e := range host {
					r.entities[e.Reference()] = e.Name
				}
			},
		},
		"Network": {
			&network, nil, func() {
				for _, e := range network {
					r.entities[e.Reference()] = e.Name
				}
			},
		},
		"OpaqueNetwork": {
			&opaque, nil, func() {
				for _, e := range opaque {
					r.entities[e.Reference()] = e.Name
				}
			},
		},
		"DistributedVirtualPortgroup": {
			&dvp, nil, func() {
				for _, e := range dvp {
					r.entities[e.Reference()] = e.Name
				}
			},
		},
		"Datastore": {
			&datastore, nil, func() {
				for _, e := range datastore {
					r.entities[e.Reference()] = e.Name
				}
			},
		},
	}

	xrefs := make(map[types.ManagedObjectReference]bool)
	// Add MOR to vrefs[kind].refs avoiding any duplicates.
	addRef := func(refs ...types.ManagedObjectReference) {
		for _, ref := range refs {
			if _, exists := xrefs[ref]; exists {
				return
			}
			xrefs[ref] = true
			vref := vrefs[ref.Type]
			vref.refs = append(vref.refs, ref)
		}
	}

	for _, vm := range r.VirtualMachines {
		if r.cmd.General {
			if ref := vm.Summary.Runtime.Host; ref != nil {
				addRef(*ref)
			}
		}

		if r.cmd.Resources {
			addRef(vm.Datastore...)
			addRef(vm.Network...)
		}
	}

	for _, vref := range vrefs {
		if vref.refs == nil {
			continue
		}
		err := pc.Retrieve(ctx, vref.refs, []string{"name"}, vref.dest)
		if err != nil {
			return err
		}
		vref.save()
	}

	return nil
}

func (r *infoResult) entityNames(refs []types.ManagedObjectReference) string {
	var names []string
	for _, ref := range refs {
		names = append(names, r.entities[ref])
	}
	return strings.Join(names, ", ")
}

func (r *infoResult) Write(w io.Writer) error {
	// Maintain order via r.objects as Property collector does not always return results in order.
	objects := make(map[types.ManagedObjectReference]mo.VirtualMachine, len(r.VirtualMachines))
	for _, o := range r.VirtualMachines {
		objects[o.Reference()] = o
	}

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, o := range r.objects {
		vm := objects[o.Reference()]
		s := vm.Summary

		fmt.Fprintf(tw, "Name:\t%s\n", s.Config.Name)

		if r.cmd.General {
			hostName := "<unavailable>"

			if href := vm.Summary.Runtime.Host; href != nil {
				if name, ok := r.entities[*href]; ok {
					hostName = name
				}
			}

			fmt.Fprintf(tw, "  Path:\t%s\n", o.InventoryPath)
			fmt.Fprintf(tw, "  UUID:\t%s\n", s.Config.Uuid)
			fmt.Fprintf(tw, "  Guest name:\t%s\n", s.Config.GuestFullName)
			fmt.Fprintf(tw, "  Memory:\t%dMB\n", s.Config.MemorySizeMB)
			fmt.Fprintf(tw, "  CPU:\t%d vCPU(s)\n", s.Config.NumCpu)
			fmt.Fprintf(tw, "  Power state:\t%s\n", s.Runtime.PowerState)
			fmt.Fprintf(tw, "  Boot time:\t%s\n", s.Runtime.BootTime)
			fmt.Fprintf(tw, "  IP address:\t%s\n", s.Guest.IpAddress)
			fmt.Fprintf(tw, "  Host:\t%s\n", hostName)
		}

		if r.cmd.Resources {
			if s.Storage == nil {
				s.Storage = new(types.VirtualMachineStorageSummary)
			}
			fmt.Fprintf(tw, "  CPU usage:\t%dMHz\n", s.QuickStats.OverallCpuUsage)
			fmt.Fprintf(tw, "  Host memory usage:\t%dMB\n", s.QuickStats.HostMemoryUsage)
			fmt.Fprintf(tw, "  Guest memory usage:\t%dMB\n", s.QuickStats.GuestMemoryUsage)
			fmt.Fprintf(tw, "  Storage uncommitted:\t%s\n", units.ByteSize(s.Storage.Uncommitted))
			fmt.Fprintf(tw, "  Storage committed:\t%s\n", units.ByteSize(s.Storage.Committed))
			fmt.Fprintf(tw, "  Storage unshared:\t%s\n", units.ByteSize(s.Storage.Unshared))
			fmt.Fprintf(tw, "  Storage:\t%s\n", r.entityNames(vm.Datastore))
			fmt.Fprintf(tw, "  Network:\t%s\n", r.entityNames(vm.Network))
		}

		if r.cmd.ExtraConfig {
			fmt.Fprintf(tw, "  ExtraConfig:\n")
			for _, v := range vm.Config.ExtraConfig {
				fmt.Fprintf(tw, "    %s:\t%s\n", v.GetOptionValue().Key, v.GetOptionValue().Value)
			}
		}

		if r.cmd.ToolsConfigInfo {
			t := vm.Config.Tools
			fmt.Fprintf(tw, "  ToolsConfigInfo:\n")
			fmt.Fprintf(tw, "    ToolsVersion:\t%d\n", t.ToolsVersion)
			fmt.Fprintf(tw, "    AfterPowerOn:\t%s\n", flags.NewOptionalBool(&t.AfterPowerOn).String())
			fmt.Fprintf(tw, "    AfterResume:\t%s\n", flags.NewOptionalBool(&t.AfterResume).String())
			fmt.Fprintf(tw, "    BeforeGuestStandby:\t%s\n", flags.NewOptionalBool(&t.BeforeGuestStandby).String())
			fmt.Fprintf(tw, "    BeforeGuestShutdown:\t%s\n", flags.NewOptionalBool(&t.BeforeGuestShutdown).String())
			fmt.Fprintf(tw, "    BeforeGuestReboot:\t%s\n", flags.NewOptionalBool(&t.BeforeGuestReboot).String())
			fmt.Fprintf(tw, "    ToolsUpgradePolicy:\t%s\n", t.ToolsUpgradePolicy)
			fmt.Fprintf(tw, "    PendingCustomization:\t%s\n", t.PendingCustomization)
			fmt.Fprintf(tw, "    SyncTimeWithHost:\t%s\n", flags.NewOptionalBool(&t.SyncTimeWithHost).String())
		}
	}

	return tw.Flush()
}
