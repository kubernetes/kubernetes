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

package cluster

import (
	"context"
	"flag"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.DatacenterFlag

	types.ClusterConfigSpecEx
}

func init() {
	cli.Register("cluster.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.DrsConfig = new(types.ClusterDrsConfigInfo)
	cmd.DasConfig = new(types.ClusterDasConfigInfo)
	cmd.VsanConfig = new(types.VsanClusterConfigInfo)
	cmd.VsanConfig.DefaultConfig = new(types.VsanClusterConfigInfoHostDefaultInfo)

	// DRS
	f.Var(flags.NewOptionalBool(&cmd.DrsConfig.Enabled), "drs-enabled", "Enable DRS")

	drsModes := []string{
		string(types.DrsBehaviorManual),
		string(types.DrsBehaviorPartiallyAutomated),
		string(types.DrsBehaviorFullyAutomated),
	}
	f.StringVar((*string)(&cmd.DrsConfig.DefaultVmBehavior), "drs-mode", "",
		"DRS behavior for virtual machines: "+strings.Join(drsModes, ", "))

	// HA
	f.Var(flags.NewOptionalBool(&cmd.DasConfig.Enabled), "ha-enabled", "Enable HA")

	// vSAN
	f.Var(flags.NewOptionalBool(&cmd.VsanConfig.Enabled), "vsan-enabled", "Enable vSAN")
	f.Var(flags.NewOptionalBool(&cmd.VsanConfig.DefaultConfig.AutoClaimStorage), "vsan-autoclaim", "Autoclaim storage on cluster hosts")
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Usage() string {
	return "CLUSTER..."
}

func (cmd *change) Description() string {
	return `Change configuration of the given clusters.

Examples:
  govc cluster.change -drs-enabled -vsan-enabled -vsan-autoclaim ClusterA
  govc cluster.change -drs-enabled=false ClusterB`
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, path := range f.Args() {
		clusters, err := finder.ClusterComputeResourceList(ctx, path)
		if err != nil {
			return err
		}

		for _, cluster := range clusters {
			task, err := cluster.Reconfigure(ctx, &cmd.ClusterConfigSpecEx, true)
			if err != nil {
				return err
			}

			_, err = task.WaitForResult(ctx, nil)
			if err != nil {
				return err
			}
		}
	}

	return nil
}
