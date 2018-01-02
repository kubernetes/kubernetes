/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package usb

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type add struct {
	*flags.VirtualMachineFlag

	controller  string
	autoConnect bool
	ehciEnabled bool
}

func init() {
	cli.Register("device.usb.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	ctypes := []string{"usb", "xhci"}
	f.StringVar(&cmd.controller, "type", ctypes[0],
		fmt.Sprintf("USB controller type (%s)", strings.Join(ctypes, "|")))

	f.BoolVar(&cmd.autoConnect, "auto", true, "Enable ability to hot plug devices")
	f.BoolVar(&cmd.ehciEnabled, "ehci", true, "Enable enhanced host controller interface (USB 2.0)")
}

func (cmd *add) Description() string {
	return `Add USB device to VM.

Examples:
  govc device.usb.add -vm $vm
  govc device.usb.add -type xhci -vm $vm
  govc device.info usb*`
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

	var d types.BaseVirtualDevice

	switch cmd.controller {
	case "usb":
		c := new(types.VirtualUSBController)
		c.AutoConnectDevices = &cmd.autoConnect
		c.EhciEnabled = &cmd.ehciEnabled
		d = c
	case "xhci":
		c := new(types.VirtualUSBXHCIController)
		c.AutoConnectDevices = &cmd.autoConnect
		d = c
	default:
		return flag.ErrHelp
	}

	err = vm.AddDevice(ctx, d)
	if err != nil {
		return err
	}

	// output name of device we just created
	devices, err := vm.Device(ctx)
	if err != nil {
		return err
	}

	devices = devices.SelectByType(d)

	name := devices.Name(devices[len(devices)-1])

	fmt.Println(name)

	return nil
}
