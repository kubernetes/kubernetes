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

package vm

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/govc/host/esxcli"
	"github.com/vmware/govmomi/object"
)

type ip struct {
	*flags.OutputFlag
	*flags.SearchFlag

	esx  bool
	all  bool
	v4   bool
	wait time.Duration
	nic  string
}

func init() {
	cli.Register("vm.ip", &ip{})
}

func (cmd *ip) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.esx, "esxcli", false, "Use esxcli instead of guest tools")
	f.BoolVar(&cmd.all, "a", false, "Wait for an IP address on all NICs")
	f.StringVar(&cmd.nic, "n", "", "Wait for IP address on NIC, specified by device name or MAC")
	f.BoolVar(&cmd.v4, "v4", false, "Only report IPv4 addresses")
	f.DurationVar(&cmd.wait, "wait", time.Hour, "Wait time for the VM obtain an IP address")
}

func (cmd *ip) Usage() string {
	return "VM..."
}

func (cmd *ip) Description() string {
	return `List IPs for VM.

By default the vm.ip command depends on vmware-tools to report the 'guest.ipAddress' field and will
wait until it has done so.  This value can also be obtained using:

  govc vm.info -json $vm | jq -r .VirtualMachines[].Guest.IpAddress

When given the '-a' flag, only IP addresses for which there is a corresponding virtual nic are listed.
If there are multiple nics, the listed addresses will be comma delimited.  The '-a' flag depends on
vmware-tools to report the 'guest.net' field and will wait until it has done so for all nics.
Note that this list includes IPv6 addresses if any, use '-v4' to filter them out.  IP addresses reported
by tools for which there is no virtual nic are not included, for example that of the 'docker0' interface.

These values can also be obtained using:

  govc vm.info -json $vm | jq -r .VirtualMachines[].Guest.Net[].IpConfig.IpAddress[].IpAddress

When given the '-n' flag, filters '-a' behavior to the nic specified by MAC address or device name.

The 'esxcli' flag does not require vmware-tools to be installed, but does require the ESX host to
have the /Net/GuestIPHack setting enabled.

The 'wait' flag default to 1hr (original default was infinite).  If a VM does not obtain an IP within
the wait time, the command will still exit with status 0.

Examples:
  govc vm.ip $vm
  govc vm.ip -wait 5m $vm
  govc vm.ip -a -v4 $vm
  govc vm.ip -n 00:0c:29:57:7b:c3 $vm
  govc vm.ip -n ethernet-0 $vm
  govc host.esxcli system settings advanced set -o /Net/GuestIPHack -i 1
  govc vm.ip -esxcli $vm`
}

func (cmd *ip) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ip) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	var get func(*object.VirtualMachine, context.Context) (string, error)

	if cmd.esx {
		get = func(vm *object.VirtualMachine, deadline context.Context) (string, error) {
			guest := esxcli.NewGuestInfo(c)

			ticker := time.NewTicker(time.Millisecond * 500)
			defer ticker.Stop()

			for {
				select {
				case <-ticker.C:
					ip, err := guest.IpAddress(vm)
					if err != nil {
						return "", err
					}

					if ip != "0.0.0.0" {
						return ip, nil
					}
				case <-deadline.Done():
					return "", nil
				}
			}
		}
	} else {
		var hwaddr []string
		if cmd.nic != "" {
			hwaddr = strings.Split(cmd.nic, ",")
		}

		get = func(vm *object.VirtualMachine, deadline context.Context) (string, error) {
			if cmd.all || hwaddr != nil {
				macs, err := vm.WaitForNetIP(deadline, cmd.v4, hwaddr...)
				if err != nil {
					return "", err
				}

				var ips []string
				for _, addrs := range macs {
					for _, ip := range addrs {
						ips = append(ips, ip)
					}
				}
				return strings.Join(ips, ","), nil
			}
			return vm.WaitForIP(deadline)
		}
	}

	for _, vm := range vms {
		deadline, cancel := context.WithDeadline(ctx, time.Now().Add(cmd.wait))

		ip, err := get(vm, deadline)
		if err != nil {
			if deadline.Err() != context.DeadlineExceeded {
				cancel()
				return err
			}
		}

		cancel()

		if ip == "" {
			continue
		}

		// TODO(PN): Display inventory path to VM
		fmt.Fprintf(cmd, "%s\n", ip)
	}

	return nil
}
