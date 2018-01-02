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

package cdrom

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type add struct {
	*flags.VirtualMachineFlag

	controller string
}

func init() {
	cli.Register("device.cdrom.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.StringVar(&cmd.controller, "controller", "", "IDE controller name")
}

func (cmd *add) Description() string {
	return `Add CD-ROM device to VM.

Examples:
  govc device.cdrom.add -vm $vm
  govc device.ls -vm $vm | grep ide-
  govc device.cdrom.add -vm $vm -controller ide-200
  govc device.info cdrom-*`
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

	c, err := devices.FindIDEController(cmd.controller)
	if err != nil {
		return err
	}

	d, err := devices.CreateCdrom(c)
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
