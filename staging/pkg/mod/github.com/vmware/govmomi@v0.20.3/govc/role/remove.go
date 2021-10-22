/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package role

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/permissions"
)

type remove struct {
	*permissions.PermissionFlag

	force bool
}

func init() {
	cli.Register("role.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = permissions.NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)

	f.BoolVar(&cmd.force, "force", false, "Force removal if role is in use")
}

func (cmd *remove) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *remove) Usage() string {
	return "NAME"
}

func (cmd *remove) Description() string {
	return `Remove authorization role.

Examples:
  govc role.remove MyRole
  govc role.remove MyRole -force`
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	role, err := cmd.Role(f.Arg(0))
	if err != nil {
		return err
	}

	return m.RemoveRole(ctx, role.RoleId, !cmd.force)
}
