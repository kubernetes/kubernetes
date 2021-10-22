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

type remove struct {
	*InfoFlag
}

func init() {
	cli.Register("cluster.rule.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)
}

func (cmd *remove) Process(ctx context.Context) error {
	if cmd.name == "" {
		return flag.ErrHelp
	}
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *remove) Description() string {
	return `Remove cluster rule.

Examples:
  govc cluster.group.remove -cluster my_cluster -name my_rule`
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
	rule, err := cmd.Rule(ctx)
	if err != nil {
		return err
	}

	update := types.ArrayUpdateSpec{
		Operation: types.ArrayUpdateOperationRemove,
		RemoveKey: rule.info.GetClusterRuleInfo().Key,
	}

	return cmd.Apply(ctx, update, nil)
}
