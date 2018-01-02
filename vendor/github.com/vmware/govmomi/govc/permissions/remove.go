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

package permissions

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type remove struct {
	*PermissionFlag

	types.Permission

	role string
}

func init() {
	cli.Register("permissions.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)

	f.StringVar(&cmd.Principal, "principal", "", "User or group for which the permission is defined")
	f.BoolVar(&cmd.Group, "group", false, "True, if principal refers to a group name; false, for a user name")
}

func (cmd *remove) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *remove) Usage() string {
	return "[PATH]..."
}

func (cmd *remove) Description() string {
	return `Removes a permission rule from managed entities.

Examples:
  govc permissions.remove -principal root
  govc permissions.remove -principal $USER@vsphere.local -role Admin /dc1/host/cluster1`
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
	refs, err := cmd.ManagedObjects(ctx, f.Args())
	if err != nil {
		return err
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	for _, ref := range refs {
		err = m.RemoveEntityPermission(ctx, ref, cmd.Principal, cmd.Group)
		if err != nil {
			return err
		}
	}

	return nil
}
