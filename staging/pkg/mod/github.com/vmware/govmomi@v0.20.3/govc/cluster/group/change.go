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

package group

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*InfoFlag
}

func init() {
	cli.Register("cluster.group.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
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
	return `Set cluster group members.

Examples:
  govc cluster.group.change -name my_group vm_a vm_b vm_c # set
  govc cluster.group.change -name my_group vm_a vm_b vm_c $(govc cluster.group.ls -name my_group) vm_d # add
  govc cluster.group.ls -name my_group | grep -v vm_b | xargs govc cluster.group.change -name my_group vm_a vm_b vm_c # remove`
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	update := types.ArrayUpdateSpec{Operation: types.ArrayUpdateOperationEdit}
	group, err := cmd.Group(ctx)
	if err != nil {
		return err
	}

	refs, err := cmd.ObjectList(ctx, group.kind, f.Args())
	if err != nil {
		return err
	}

	*group.refs = refs

	return cmd.Apply(ctx, update, group.info)
}
