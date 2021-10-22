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
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type ls struct {
	*flags.VirtualMachineFlag

	boot bool
}

func init() {
	cli.Register("device.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.BoolVar(&cmd.boot, "boot", false, "List devices configured in the VM's boot options")
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Description() string {
	return `List devices for VM.

Examples:
  govc device.ls -vm $name
  govc device.ls -vm $name disk-*
  govc device.ls -vm $name -json | jq '.Devices[].Name'`
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
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

	if f.NArg() != 0 {
		var matches object.VirtualDeviceList
		for _, name := range f.Args() {
			device := match(name, devices)
			if len(device) == 0 {
				return fmt.Errorf("device '%s' not found", name)
			}
			matches = append(matches, device...)
		}
		devices = matches
	}

	if cmd.boot {
		options, err := vm.BootOptions(ctx)
		if err != nil {
			return err
		}

		devices = devices.SelectBootOrder(options.BootOrder)
	}

	res := lsResult{toLsList(devices), devices}
	return cmd.WriteResult(&res)
}

type lsDevice struct {
	Name    string
	Type    string
	Summary string
}

func toLsList(devices object.VirtualDeviceList) []lsDevice {
	var res []lsDevice

	for _, device := range devices {
		res = append(res, lsDevice{
			Name:    devices.Name(device),
			Type:    devices.TypeName(device),
			Summary: device.GetVirtualDevice().DeviceInfo.GetDescription().Summary,
		})
	}

	return res
}

type lsResult struct {
	Devices []lsDevice
	list    object.VirtualDeviceList
}

func (r *lsResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 3, 0, 2, ' ', 0)

	for _, device := range r.list {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", r.list.Name(device), r.list.TypeName(device),
			device.GetVirtualDevice().DeviceInfo.GetDescription().Summary)
	}

	return tw.Flush()
}
