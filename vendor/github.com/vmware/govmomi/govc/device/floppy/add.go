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

package floppy

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type add struct {
	*flags.VirtualMachineFlag
}

func init() {
	cli.Register("device.floppy.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
}

func (cmd *add) Description() string {
	return `Add floppy device to VM.

Examples:
  govc device.floppy.add -vm $vm
  govc device.info floppy-*`
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	devices, err := vm.Device(ctx)
	if err != nil {
		return err
	}

	d, err := devices.CreateFloppy()
	if err != nil {
		return err
	}

	err = vm.AddDevice(ctx, d)
	if err != nil {
		return err
	}

	// output name of device we just created
	devices, err = vm.Device(ctx)
	if err != nil {
		return err
	}

	devices = devices.SelectByType(d)

	name := devices.Name(devices[len(devices)-1])

	fmt.Println(name)

	return nil
}
