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

type disconnect struct {
	*flags.VirtualMachineFlag
}

func init() {
	cli.Register("device.disconnect", &disconnect{})
}

func (cmd *disconnect) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
}

func (cmd *disconnect) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *disconnect) Usage() string {
	return "DEVICE..."
}

func (cmd *disconnect) Description() string {
	return `Disconnect DEVICE on VM.

Examples:
  govc device.disconnect -vm $name cdrom-3000`
}

func (cmd *disconnect) Run(ctx context.Context, f *flag.FlagSet) error {
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
		device := devices.Find(name)
		if device == nil {
			return fmt.Errorf("device '%s' not found", name)
		}

		if err = devices.Disconnect(device); err != nil {
			return err
		}

		if err = vm.EditDevice(ctx, device); err != nil {
			return err
		}
	}

	return nil
}
