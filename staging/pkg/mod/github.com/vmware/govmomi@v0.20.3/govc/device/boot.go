/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

package device

import (
	"context"
	"flag"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type boot struct {
	*flags.VirtualMachineFlag

	order string
	types.VirtualMachineBootOptions
}

func init() {
	cli.Register("device.boot", &boot{})
}

func (cmd *boot) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.Int64Var(&cmd.BootDelay, "delay", 0, "Delay in ms before starting the boot sequence")
	f.StringVar(&cmd.order, "order", "", "Boot device order [-,floppy,cdrom,ethernet,disk]")
	f.Int64Var(&cmd.BootRetryDelay, "retry-delay", 0, "Delay in ms before a boot retry")

	cmd.BootRetryEnabled = types.NewBool(false)
	f.BoolVar(cmd.BootRetryEnabled, "retry", false, "If true, retry boot after retry-delay")

	cmd.EnterBIOSSetup = types.NewBool(false)
	f.BoolVar(cmd.EnterBIOSSetup, "setup", false, "If true, enter BIOS setup on next boot")
}

func (cmd *boot) Description() string {
	return `Configure VM boot settings.

Examples:
  govc device.boot -vm $vm -delay 1000 -order floppy,cdrom,ethernet,disk
  govc device.boot -vm $vm -order - # reset boot order`
}

func (cmd *boot) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *boot) Run(ctx context.Context, f *flag.FlagSet) error {
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

	if cmd.order != "" {
		o := strings.Split(cmd.order, ",")
		cmd.BootOrder = devices.BootOrder(o)
	}

	return vm.SetBootOptions(ctx, &cmd.VirtualMachineBootOptions)
}
