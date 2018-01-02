/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package vnic

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.vnic.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	ns, err := cmd.HostNetworkSystem()
	if err != nil {
		return err
	}

	var mns mo.HostNetworkSystem

	m, err := host.ConfigManager().VirtualNicManager(ctx)
	if err != nil {
		return err
	}

	info, err := m.Info(ctx)
	if err != nil {
		return err
	}

	err = ns.Properties(ctx, ns.Reference(), []string{"networkInfo"}, &mns)
	if err != nil {
		return err
	}

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	type dnet struct {
		dvp mo.DistributedVirtualPortgroup
		dvs mo.VmwareDistributedVirtualSwitch
	}

	dnets := make(map[string]*dnet)

	for _, nic := range mns.NetworkInfo.Vnic {
		fmt.Fprintf(tw, "Device:\t%s\n", nic.Device)

		if dvp := nic.Spec.DistributedVirtualPort; dvp != nil {
			dn, ok := dnets[dvp.PortgroupKey]

			if !ok {
				dn = new(dnet)
				o := object.NewDistributedVirtualPortgroup(host.Client(), types.ManagedObjectReference{
					Type:  "DistributedVirtualPortgroup",
					Value: dvp.PortgroupKey,
				})

				err = o.Properties(ctx, o.Reference(), []string{"name", "config.distributedVirtualSwitch"}, &dn.dvp)
				if err != nil {
					return err
				}

				err = o.Properties(ctx, *dn.dvp.Config.DistributedVirtualSwitch, []string{"name"}, &dn.dvs)
				if err != nil {
					return err
				}

				dnets[dvp.PortgroupKey] = dn
			}

			fmt.Fprintf(tw, "Network label:\t%s\n", dn.dvp.Name)
			fmt.Fprintf(tw, "Switch:\t%s\n", dn.dvs.Name)
		} else {
			fmt.Fprintf(tw, "Network label:\t%s\n", nic.Portgroup)
			for _, pg := range mns.NetworkInfo.Portgroup {
				if pg.Spec.Name == nic.Portgroup {
					fmt.Fprintf(tw, "Switch:\t%s\n", pg.Spec.VswitchName)
					break
				}
			}
		}

		fmt.Fprintf(tw, "IP address:\t%s\n", nic.Spec.Ip.IpAddress)
		fmt.Fprintf(tw, "TCP/IP stack:\t%s\n", nic.Spec.NetStackInstanceKey)

		var services []string
		for _, nc := range info.NetConfig {
			for _, dev := range nc.SelectedVnic {
				key := nc.NicType + "." + nic.Key
				if dev == key {
					services = append(services, nc.NicType)
				}
			}

		}
		fmt.Fprintf(tw, "Enabled services:\t%s\n", strings.Join(services, ", "))
	}

	return tw.Flush()
}
