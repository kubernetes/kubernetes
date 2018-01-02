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

package portgroup

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type add struct {
	*flags.HostSystemFlag

	spec types.HostPortGroupSpec
}

func init() {
	cli.Register("host.portgroup.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.StringVar(&cmd.spec.VswitchName, "vswitch", "", "vSwitch Name")
	f.Var(flags.NewInt32(&cmd.spec.VlanId), "vlan", "VLAN ID")
}

func (cmd *add) Description() string {
	return `Add portgroup to HOST.

Examples:
  govc host.portgroup.add -vswitch vSwitch0 -vlan 3201 bridge`
}

func (cmd *add) Usage() string {
	return "NAME"
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	ns, err := cmd.HostNetworkSystem()
	if err != nil {
		return err
	}

	cmd.spec.Name = f.Arg(0)

	return ns.AddPortGroup(ctx, cmd.spec)
}
