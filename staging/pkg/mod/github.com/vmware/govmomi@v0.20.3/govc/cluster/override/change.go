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

package override

import (
	"context"
	"flag"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/cluster"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.ClusterFlag
	*flags.VirtualMachineFlag

	drs types.ClusterDrsVmConfigInfo
	das types.ClusterDasVmConfigInfo
}

func init() {
	cli.Register("cluster.override.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	cmd.ClusterFlag.Register(ctx, f)
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	// DRS
	f.Var(flags.NewOptionalBool(&cmd.drs.Enabled), "drs-enabled", "Enable DRS")

	f.StringVar((*string)(&cmd.drs.Behavior), "drs-mode", "", cluster.DrsBehaviorUsage())

	// HA
	rp := []string{
		string(types.DasVmPriorityDisabled),
		string(types.DasVmPriorityLow),
		string(types.DasVmPriorityMedium),
		string(types.DasVmPriorityHigh),
	}
	cmd.das.DasSettings = new(types.ClusterDasVmSettings)

	f.StringVar((*string)(&cmd.das.DasSettings.RestartPriority), "ha-restart-priority", "", "HA restart priority: "+strings.Join(rp, ", "))
}

func (cmd *change) Description() string {
	return `Change cluster VM overrides.

Examples:
  govc cluster.override.change -cluster cluster_1 -vm vm_1 -ha-restart-priority high
  govc cluster.override.change -cluster cluster_1 -vm vm_2 -drs-enabled=false
  govc cluster.override.change -cluster cluster_1 -vm vm_3 -drs-enabled -drs-mode fullyAutomated`
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.ClusterFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.VirtualMachineFlag.Process(ctx)
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	cluster, err := cmd.Cluster()
	if err != nil {
		return err
	}

	config, err := cluster.Configuration(ctx)
	if err != nil {
		return err
	}

	spec := &types.ClusterConfigSpecEx{}
	cmd.drs.Key = vm.Reference()
	cmd.das.Key = vm.Reference()

	if cmd.drs.Behavior != "" || cmd.drs.Enabled != nil {
		op := types.ArrayUpdateOperationAdd
		for _, c := range config.DrsVmConfig {
			if c.Key == cmd.drs.Key {
				op = types.ArrayUpdateOperationEdit
				break
			}
		}

		spec.DrsVmConfigSpec = []types.ClusterDrsVmConfigSpec{
			{
				ArrayUpdateSpec: types.ArrayUpdateSpec{
					Operation: op,
				},
				Info: &cmd.drs,
			},
		}
	}

	if cmd.das.DasSettings.RestartPriority != "" {
		op := types.ArrayUpdateOperationAdd
		for _, c := range config.DasVmConfig {
			if c.Key == cmd.das.Key {
				op = types.ArrayUpdateOperationEdit
				break
			}
		}

		spec.DasVmConfigSpec = []types.ClusterDasVmConfigSpec{
			{
				ArrayUpdateSpec: types.ArrayUpdateSpec{
					Operation: op,
				},
				Info: &cmd.das,
			},
		}
	}

	if spec.DrsVmConfigSpec == nil && spec.DasVmConfigSpec == nil {
		return flag.ErrHelp
	}

	return cmd.Reconfigure(ctx, spec)
}
