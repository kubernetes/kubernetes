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
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type markasvm struct {
	*flags.SearchFlag
	*flags.ResourcePoolFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("vm.markasvm", &markasvm{})
}

func (cmd *markasvm) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)
	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *markasvm) Process(ctx context.Context) error {
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *markasvm) Usage() string {
	return "VM..."
}

func (cmd *markasvm) Description() string {
	return `Mark VM template as a virtual machine.

Examples:
  govc vm.markasvm $name -host host1
  govc vm.markasvm $name -pool cluster1/Resources`
}

func (cmd *markasvm) Run(ctx context.Context, f *flag.FlagSet) error {
	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	pool, err := cmd.ResourcePoolIfSpecified()
	if err != nil {
		return err
	}

	host, err := cmd.HostSystemFlag.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	if pool == nil {
		if host == nil {
			return flag.ErrHelp
		}

		pool, err = host.ResourcePool(ctx)
		if err != nil {
			return err
		}
	}

	for _, vm := range vms {
		err := vm.MarkAsVirtualMachine(ctx, *pool, host)
		if err != nil {
			return err
		}
	}

	return nil
}
