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
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.VirtualMachineFlag
	*flags.NetworkFlag
}

func init() {
	cli.Register("vm.network.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
	cmd.NetworkFlag, ctx = flags.NewNetworkFlag(ctx)
	cmd.NetworkFlag.Register(ctx, f)
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.NetworkFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Usage() string {
	return "DEVICE"
}

func (cmd *change) Description() string {
	return `Change network DEVICE configuration.

Examples:
  govc vm.network.change -vm $vm -net PG2 ethernet-0
  govc vm.network.change -vm $vm -net.address 00:00:0f:2e:5d:69 ethernet-0
  govc device.info -vm $vm ethernet-*`
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachineFlag.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return errors.New("please specify a vm")
	}

	name := f.Arg(0)

	if name == "" {
		return errors.New("please specify a device name")
	}

	// Set network if specified as extra argument.
	if f.NArg() > 1 {
		_ = cmd.NetworkFlag.Set(f.Arg(1))
	}

	devices, err := vm.Device(ctx)
	if err != nil {
		return err
	}

	net := devices.Find(name)

	if net == nil {
		return fmt.Errorf("device '%s' not found", name)
	}

	dev, err := cmd.NetworkFlag.Device()
	if err != nil {
		return err
	}

	current := net.(types.BaseVirtualEthernetCard).GetVirtualEthernetCard()
	changed := dev.(types.BaseVirtualEthernetCard).GetVirtualEthernetCard()

	current.Backing = changed.Backing

	if changed.MacAddress != "" {
		current.MacAddress = changed.MacAddress
	}

	if changed.AddressType != "" {
		current.AddressType = changed.AddressType
	}

	return vm.EditDevice(ctx, net)
}
