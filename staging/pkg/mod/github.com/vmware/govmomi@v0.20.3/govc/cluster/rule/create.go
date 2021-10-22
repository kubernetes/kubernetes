/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package rule

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type create struct {
	*SpecFlag
	*InfoFlag

	vmhost       bool
	affinity     bool
	antiaffinity bool
	depends      bool
}

func init() {
	cli.Register("cluster.rule.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SpecFlag = new(SpecFlag)
	cmd.SpecFlag.Register(ctx, f)

	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)

	f.BoolVar(&cmd.vmhost, "vm-host", false, "Virtual Machines to Hosts")
	f.BoolVar(&cmd.affinity, "affinity", false, "Keep Virtual Machines Together")
	f.BoolVar(&cmd.antiaffinity, "anti-affinity", false, "Separate Virtual Machines")
	f.BoolVar(&cmd.depends, "depends", false, "Virtual Machines to Virtual Machines")
}

func (cmd *create) Process(ctx context.Context) error {
	if cmd.name == "" {
		return flag.ErrHelp
	}
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *create) Usage() string {
	return "NAME..."
}

func (cmd *create) Description() string {
	return `Create cluster rule.

Rules are not enabled by default, use the 'enable' flag to enable upon creation or cluster.rule.change after creation.

One of '-affinity', '-anti-affinity', '-depends' or '-vm-host' must be provided to specify the rule type.

With '-affinity' or '-anti-affinity', at least 2 vm NAME arguments must be specified.

With '-depends', vm group NAME and vm group dependency NAME arguments must be specified.

With '-vm-host', use the '-vm-group' flag combined with the '-host-affine-group' and/or '-host-anti-affine-group' flags.

Examples:
  govc cluster.rule.create -name pod1 -enable -affinity vm_a vm_b vm_c
  govc cluster.rule.create -name pod2 -enable -anti-affinity vm_d vm_e vm_f
  govc cluster.rule.create -name pod3 -enable -mandatory -vm-host -vm-group my_vms -host-affine-group my_hosts
  govc cluster.rule.create -name pod4 -depends vm_group_app vm_group_db`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	update := types.ArrayUpdateSpec{Operation: types.ArrayUpdateOperationAdd}
	var rule types.BaseClusterRuleInfo
	var err error

	switch {
	case cmd.vmhost:
		rule = &cmd.ClusterVmHostRuleInfo
	case cmd.affinity:
		rule = &cmd.ClusterAffinityRuleSpec
		if len(args) < 2 {
			return flag.ErrHelp // can't create this rule without 2 or more hosts
		}
		cmd.ClusterAffinityRuleSpec.Vm, err = cmd.ObjectList(ctx, "VirtualMachine", args)
		if err != nil {
			return err
		}
	case cmd.antiaffinity:
		rule = &cmd.ClusterAntiAffinityRuleSpec
		if len(args) < 2 {
			return flag.ErrHelp // can't create this rule without 2 or more hosts
		}
		cmd.ClusterAntiAffinityRuleSpec.Vm, err = cmd.ObjectList(ctx, "VirtualMachine", args)
		if err != nil {
			return err
		}
	case cmd.depends:
		if len(args) != 2 {
			return flag.ErrHelp
		}
		rule = &types.ClusterDependencyRuleInfo{
			VmGroup:          args[0],
			DependsOnVmGroup: args[1],
		}
	default:
		return flag.ErrHelp
	}

	if cmd.Enabled == nil {
		// ClusterDependencyRuleInfo throws InvalidArgument if Enabled == nil
		cmd.Enabled = types.NewBool(false)
	}

	info := rule.GetClusterRuleInfo()
	info.Name = cmd.name
	info.Enabled = cmd.Enabled
	info.Mandatory = cmd.Mandatory
	info.UserCreated = types.NewBool(true)

	return cmd.Apply(ctx, update, rule)
}
