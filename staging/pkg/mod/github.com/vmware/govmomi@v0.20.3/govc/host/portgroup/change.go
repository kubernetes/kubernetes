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

package portgroup

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.ClientFlag
	*flags.HostSystemFlag

	types.HostPortGroupSpec
	types.HostNetworkSecurityPolicy
}

func init() {
	cli.Register("host.portgroup.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.VlanId = -1
	f.Var(flags.NewInt32(&cmd.VlanId), "vlan-id", "VLAN ID")
	f.StringVar(&cmd.Name, "name", "", "Portgroup name")
	f.StringVar(&cmd.VswitchName, "vswitch-name", "", "vSwitch name")

	f.Var(flags.NewOptionalBool(&cmd.AllowPromiscuous), "allow-promiscuous", "Allow promiscuous mode")
	f.Var(flags.NewOptionalBool(&cmd.ForgedTransmits), "forged-transmits", "Allow forged transmits")
	f.Var(flags.NewOptionalBool(&cmd.MacChanges), "mac-changes", "Allow MAC changes")
}

func (cmd *change) Description() string {
	return `Change configuration of HOST portgroup NAME.

Examples:
  govc host.portgroup.change -allow-promiscuous -forged-transmits -mac-changes "VM Network"
  govc host.portgroup.change -vswitch-name vSwitch1 "Management Network"`
}

func (cmd *change) Usage() string {
	return "NAME"
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	pg, err := networkInfoPortgroup(ctx, cmd.ClientFlag, cmd.HostSystemFlag)
	if err != nil {
		return err
	}

	ns, err := cmd.HostNetworkSystem()
	if err != nil {
		return err
	}

	name := f.Arg(0)
	var current *types.HostPortGroupSpec

	for _, g := range pg {
		if g.Spec.Name == name {
			current = &g.Spec
			break
		}
	}

	if current != nil {
		if cmd.Name == "" {
			cmd.Name = current.Name
		}
		if cmd.VswitchName == "" {
			cmd.VswitchName = current.VswitchName
		}
		if cmd.VlanId < 0 {
			cmd.VlanId = current.VlanId
		}
	}

	cmd.HostPortGroupSpec.Policy.Security = &cmd.HostNetworkSecurityPolicy

	return ns.UpdatePortGroup(ctx, name, cmd.HostPortGroupSpec)
}
