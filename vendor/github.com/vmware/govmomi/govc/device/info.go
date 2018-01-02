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

package device

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.VirtualMachineFlag
	*flags.OutputFlag
	*flags.NetworkFlag
}

func init() {
	cli.Register("device.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.NetworkFlag, ctx = flags.NewNetworkFlag(ctx)
	cmd.NetworkFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.NetworkFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Usage() string {
	return "[DEVICE]..."
}

func (cmd *info) match(p string, devices object.VirtualDeviceList) object.VirtualDeviceList {
	var matches object.VirtualDeviceList
	match := func(name string) bool {
		matched, _ := path.Match(p, name)
		return matched
	}

	for _, device := range devices {
		name := devices.Name(device)
		eq := name == p
		if eq || match(name) {
			matches = append(matches, device)
		}
		if eq {
			break
		}
	}

	return matches
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
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

	res := infoResult{
		list: devices,
	}

	if cmd.NetworkFlag.IsSet() {
		net, err := cmd.Network()
		if err != nil {
			return err
		}

		backing, err := net.EthernetCardBackingInfo(ctx)
		if err != nil {
			return err
		}

		devices = devices.SelectByBackingInfo(backing)
	}

	if f.NArg() == 0 {
		res.Devices = devices
	} else {
		for _, name := range f.Args() {
			matches := cmd.match(name, devices)
			if len(matches) == 0 {
				return fmt.Errorf("device '%s' not found", name)
			}

			res.Devices = append(res.Devices, matches...)
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	Devices object.VirtualDeviceList
	// need the full list of devices to lookup attached devices and controllers
	list object.VirtualDeviceList
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, device := range r.Devices {
		d := device.GetVirtualDevice()
		info := d.DeviceInfo.GetDescription()

		fmt.Fprintf(tw, "Name:\t%s\n", r.Devices.Name(device))
		fmt.Fprintf(tw, "  Type:\t%s\n", r.Devices.TypeName(device))
		fmt.Fprintf(tw, "  Label:\t%s\n", info.Label)
		fmt.Fprintf(tw, "  Summary:\t%s\n", info.Summary)
		fmt.Fprintf(tw, "  Key:\t%d\n", d.Key)

		if c, ok := device.(types.BaseVirtualController); ok {
			var attached []string
			for _, key := range c.GetVirtualController().Device {
				attached = append(attached, r.Devices.Name(r.list.FindByKey(key)))
			}
			fmt.Fprintf(tw, "  Devices:\t%s\n", strings.Join(attached, ", "))
		} else {
			if c := r.list.FindByKey(d.ControllerKey); c != nil {
				fmt.Fprintf(tw, "  Controller:\t%s\n", r.Devices.Name(c))
				if d.UnitNumber != nil {
					fmt.Fprintf(tw, "  Unit number:\t%d\n", *d.UnitNumber)
				} else {
					fmt.Fprintf(tw, "  Unit number:\t<nil>\n")
				}
			}
		}

		if ca := d.Connectable; ca != nil {
			fmt.Fprintf(tw, "  Connected:\t%t\n", ca.Connected)
			fmt.Fprintf(tw, "  Start connected:\t%t\n", ca.StartConnected)
			fmt.Fprintf(tw, "  Guest control:\t%t\n", ca.AllowGuestControl)
			fmt.Fprintf(tw, "  Status:\t%s\n", ca.Status)
		}

		switch md := device.(type) {
		case types.BaseVirtualEthernetCard:
			fmt.Fprintf(tw, "  MAC Address:\t%s\n", md.GetVirtualEthernetCard().MacAddress)
			fmt.Fprintf(tw, "  Address type:\t%s\n", md.GetVirtualEthernetCard().AddressType)
		case *types.VirtualDisk:
			if b, ok := md.Backing.(types.BaseVirtualDeviceFileBackingInfo); ok {
				fmt.Fprintf(tw, "  File:\t%s\n", b.GetVirtualDeviceFileBackingInfo().FileName)
			}
			if b, ok := md.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok && b.Parent != nil {
				fmt.Fprintf(tw, "  Parent:\t%s\n", b.Parent.GetVirtualDeviceFileBackingInfo().FileName)
			}
		case *types.VirtualSerialPort:
			if b, ok := md.Backing.(*types.VirtualSerialPortURIBackingInfo); ok {
				fmt.Fprintf(tw, "  Direction:\t%s\n", b.Direction)
				fmt.Fprintf(tw, "  Service URI:\t%s\n", b.ServiceURI)
				fmt.Fprintf(tw, "  Proxy URI:\t%s\n", b.ProxyURI)
			}
		}
	}

	return tw.Flush()
}
