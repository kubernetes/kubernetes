/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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

package option

import (
	"context"
	"flag"
	"io"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClusterFlag
	*flags.HostSystemFlag
	*flags.VirtualMachineFlag
}

func init() {
	cli.Register("vm.option.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	cmd.ClusterFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClusterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.VirtualMachineFlag.Process(ctx)
}

func (cmd *info) Usage() string {
	return "[GUEST_ID]..."
}

func (cmd *info) Description() string {
	return `VM config options for CLUSTER.

The config option data contains information about the execution environment for a VM
in the given CLUSTER, and optionally for a specific HOST.

This command only supports '-json' or '-dump' output, defaulting to the latter.

Examples:
  govc vm.option.info -cluster C0
  govc vm.option.info -cluster C0 ubuntu64Guest
  govc vm.option.info -cluster C0 -json | jq .GuestOSDescriptor[].Id
  govc vm.option.info -host my_hostname
  govc vm.option.info -vm my_vm`
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	vmf := cmd.VirtualMachineFlag

	if vmf.JSON == false && vmf.Dump == false {
		vmf.Dump = true // Default to -dump as there is no plain-text format atm
	}

	c, err := vmf.Client()
	if err != nil {
		return err
	}

	var ref types.ManagedObjectReference

	host, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return err
	}
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}
	if vm == nil {
		if host == nil {
			finder, ferr := cmd.ClusterFlag.Finder()
			if ferr != nil {
				return ferr
			}

			cr, ferr := finder.ComputeResourceOrDefault(ctx, cmd.ClusterFlag.Name)
			if ferr != nil {
				return ferr
			}
			ref = cr.Reference()
		} else {
			var h mo.HostSystem
			err = host.Properties(ctx, host.Reference(), []string{"parent"}, &h)
			if err != nil {
				return err
			}
			ref = *h.Parent
		}
	} else {
		ref = vm.Reference()
	}

	var content []types.ObjectContent

	err = property.DefaultCollector(c).RetrieveOne(ctx, ref, []string{"environmentBrowser"}, &content)
	if err != nil {
		return err
	}

	req := types.QueryConfigOptionEx{
		This: content[0].PropSet[0].Val.(types.ManagedObjectReference),
		Spec: &types.EnvironmentBrowserConfigOptionQuerySpec{
			GuestId: f.Args(),
		},
	}

	if host != nil {
		req.Spec.Host = types.NewReference(host.Reference())
	}

	opt, err := methods.QueryConfigOptionEx(ctx, c, &req)
	if err != nil {
		return err
	}

	return vmf.WriteResult(&infoResult{opt.Returnval})
}

type infoResult struct {
	*types.VirtualMachineConfigOption
}

func (r *infoResult) Write(w io.Writer) error {
	return flag.ErrHelp
}

func (r *infoResult) Dump() interface{} {
	return r.VirtualMachineConfigOption
}
