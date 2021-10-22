/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package serial

import (
	"context"
	"flag"
	"fmt"
	"path"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/mo"
)

type connect struct {
	*flags.VirtualMachineFlag

	proxy  string
	device string
	client bool
}

func init() {
	cli.Register("device.serial.connect", &connect{})
}

func (cmd *connect) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.StringVar(&cmd.proxy, "vspc-proxy", "", "vSPC proxy URI")
	f.StringVar(&cmd.device, "device", "", "serial port device name")
	f.BoolVar(&cmd.client, "client", false, "Use client direction")
}

func (cmd *connect) Usage() string {
	return "URI"
}

func (cmd *connect) Description() string {
	return `Connect service URI to serial port.

If "-" is given as URI, connects file backed device with file name of
device name + .log suffix in the VM Config.Files.LogDirectory.

Defaults to the first serial port if no DEVICE is given.

Examples:
  govc device.ls | grep serialport-
  govc device.serial.connect -vm $vm -device serialport-8000 telnet://:33233
  govc device.info -vm $vm serialport-*
  govc device.serial.connect -vm $vm "[datastore1] $vm/console.log"
  govc device.serial.connect -vm $vm -
  govc datastore.tail -f $vm/serialport-8000.log`
}

func (cmd *connect) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *connect) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

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

	d, err := devices.FindSerialPort(cmd.device)
	if err != nil {
		return err
	}

	uri := f.Arg(0)

	if uri == "-" {
		var mvm mo.VirtualMachine
		err = vm.Properties(ctx, vm.Reference(), []string{"config.files.logDirectory"}, &mvm)
		if err != nil {
			return err
		}

		uri = path.Join(mvm.Config.Files.LogDirectory, fmt.Sprintf("%s.log", devices.Name(d)))
	}

	return vm.EditDevice(ctx, devices.ConnectSerialPort(d, uri, cmd.client, cmd.proxy))
}
