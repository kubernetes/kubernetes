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

package network

import (
	"context"
	"errors"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type add struct {
	*flags.VirtualMachineFlag
	*flags.NetworkFlag
}

func init() {
	cli.Register("vm.network.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
	cmd.NetworkFlag, ctx = flags.NewNetworkFlag(ctx)
	cmd.NetworkFlag.Register(ctx, f)
}

func (cmd *add) Description() string {
	return `Add network adapter to VM.

Examples:
  govc vm.network.add -vm $vm -net "VM Network" -net.adapter e1000e
  govc device.info -vm $vm ethernet-*`
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.NetworkFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachineFlag.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return errors.New("please specify a vm")
	}

	// Set network if specified as extra argument.
	if f.NArg() > 0 {
		_ = cmd.NetworkFlag.Set(f.Arg(0))
	}

	net, err := cmd.NetworkFlag.Device()
	if err != nil {
		return err
	}

	return vm.AddDevice(ctx, net)
}
