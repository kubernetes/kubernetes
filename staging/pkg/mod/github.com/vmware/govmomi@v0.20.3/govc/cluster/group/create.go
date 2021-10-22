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

type create struct {
	*InfoFlag

	vm   bool
	host bool
}

func init() {
	cli.Register("cluster.group.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)

	f.BoolVar(&cmd.vm, "vm", false, "Create cluster VM group")
	f.BoolVar(&cmd.host, "host", false, "Create cluster Host group")
}

func (cmd *create) Process(ctx context.Context) error {
	if cmd.name == "" {
		return flag.ErrHelp
	}
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *create) Description() string {
	return `Create cluster group.

One of '-vm' or '-host' must be provided to specify the group type.

Examples:
  govc cluster.group.create -name my_vm_group -vm vm_a vm_b vm_c
  govc cluster.group.create -name my_host_group -host host_a host_b host_c`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	update := types.ArrayUpdateSpec{Operation: types.ArrayUpdateOperationAdd}
	var info types.BaseClusterGroupInfo
	var err error

	switch {
	case cmd.vm:
		info = new(types.ClusterVmGroup)
	case cmd.host:
		info = new(types.ClusterHostGroup)
	default:
		return flag.ErrHelp
	}

	info.GetClusterGroupInfo().Name = cmd.name

	group := newGroupInfo(info)
	*group.refs, err = cmd.ObjectList(ctx, group.kind, f.Args())
	if err != nil {
		return err
	}

	return cmd.Apply(ctx, update, info)
}
