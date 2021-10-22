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

type usage struct {
	*permissions.PermissionFlag
}

func init() {
	cli.Register("role.usage", &usage{})
}

func (cmd *usage) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = permissions.NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)
}

func (cmd *usage) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *usage) Usage() string {
	return "NAME..."
}

func (cmd *usage) Description() string {
	return `List usage for role NAME.

Examples:
  govc role.usage
  govc role.usage Admin`
}

func (cmd *usage) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	if f.NArg() == 0 {
		cmd.List.Permissions, err = m.RetrieveAllPermissions(ctx)
		if err != nil {
			return err
		}
	} else {
		for _, name := range f.Args() {
			role, err := cmd.Role(name)
			if err != nil {
				return err
			}

			perms, err := m.RetrieveRolePermissions(ctx, role.RoleId)
			if err != nil {
				return err
			}

			cmd.List.Add(perms)
		}
	}

	return cmd.WriteResult(&cmd.List)
}
