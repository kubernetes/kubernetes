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

type change struct {
	*SpecFlag
	*InfoFlag
}

func init() {
	cli.Register("cluster.rule.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SpecFlag = new(SpecFlag)
	cmd.SpecFlag.Register(ctx, f)

	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)
}

func (cmd *change) Process(ctx context.Context) error {
	if cmd.name == "" {
		return flag.ErrHelp
	}
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *change) Usage() string {
	return `NAME...`
}

func (cmd *change) Description() string {
	return `Change cluster rule.

Examples:
  govc cluster.rule.change -cluster my_cluster -name my_rule -enable=false`
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	update := types.ArrayUpdateSpec{Operation: types.ArrayUpdateOperationEdit}
	rule, err := cmd.Rule(ctx)
	if err != nil {
		return err
	}

	var vms *[]types.ManagedObjectReference

	switch r := rule.info.(type) {
	case *types.ClusterAffinityRuleSpec:
		vms = &r.Vm
	case *types.ClusterAntiAffinityRuleSpec:
		vms = &r.Vm
	}

	if vms != nil && f.NArg() != 0 {
		refs, err := cmd.ObjectList(ctx, rule.kind, f.Args())
		if err != nil {
			return err
		}

		*vms = refs
	}

	info := rule.info.GetClusterRuleInfo()
	info.Name = cmd.name
	info.Enabled = cmd.Enabled
	info.Mandatory = cmd.Mandatory

	return cmd.Apply(ctx, update, rule.info)
}
