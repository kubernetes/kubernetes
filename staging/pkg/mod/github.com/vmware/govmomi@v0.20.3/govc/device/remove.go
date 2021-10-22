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
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type remove struct {
	*flags.VirtualMachineFlag
	keepFiles bool
}

func init() {
	cli.Register("device.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
	f.BoolVar(&cmd.keepFiles, "keep", false, "Keep files in datastore")
}

func (cmd *remove) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *remove) Usage() string {
	return "DEVICE..."
}

func (cmd *remove) Description() string {
	return `Remove DEVICE from VM.

Examples:
  govc device.remove -vm $name cdrom-3000
  govc device.remove -vm $name disk-1000
  govc device.remove -vm $name -keep disk-*`
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
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

	for _, name := range f.Args() {
		device := match(name, devices)
		if len(device) == 0 {
			return fmt.Errorf("device '%s' not found", name)
		}

		if err = vm.RemoveDevice(ctx, cmd.keepFiles, device...); err != nil {
			return err
		}
	}

	return nil
}
