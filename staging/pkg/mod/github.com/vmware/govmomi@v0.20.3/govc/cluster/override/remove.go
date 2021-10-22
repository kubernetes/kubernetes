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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type remove struct {
	*flags.ClusterFlag
	*flags.VirtualMachineFlag
}

func init() {
	cli.Register("cluster.override.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	cmd.ClusterFlag.Register(ctx, f)
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
}

func (cmd *remove) Description() string {
	return `Remove cluster VM overrides.

Examples:
  govc cluster.override.remove -cluster cluster_1 -vm vm_1`
}

func (cmd *remove) Process(ctx context.Context) error {
	if err := cmd.ClusterFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.VirtualMachineFlag.Process(ctx)
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
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
	ref := vm.Reference()

	for _, c := range config.DrsVmConfig {
		if c.Key == ref {
			spec.DrsVmConfigSpec = []types.ClusterDrsVmConfigSpec{
				{
					ArrayUpdateSpec: types.ArrayUpdateSpec{
						Operation: types.ArrayUpdateOperationRemove,
						RemoveKey: ref,
					},
				},
			}
			break
		}
	}

	for _, c := range config.DasVmConfig {
		if c.Key == ref {
			spec.DasVmConfigSpec = []types.ClusterDasVmConfigSpec{
				{
					ArrayUpdateSpec: types.ArrayUpdateSpec{
						Operation: types.ArrayUpdateOperationRemove,
						RemoveKey: ref,
					},
				},
			}
			break
		}
	}

	return cmd.Reconfigure(ctx, spec)
}
