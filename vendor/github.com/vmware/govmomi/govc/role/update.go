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

type update struct {
	*permissions.PermissionFlag

	name   string
	remove bool
	add    bool
}

func init() {
	cli.Register("role.update", &update{})
}

func (cmd *update) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = permissions.NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "name", "", "Change role name")
	f.BoolVar(&cmd.remove, "r", false, "Remove given PRIVILEGE(s)")
	f.BoolVar(&cmd.add, "a", false, "Remove given PRIVILEGE(s)")
}

func (cmd *update) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *update) Usage() string {
	return "NAME [PRIVILEGE]..."
}

func (cmd *update) Description() string {
	return `Update authorization role.

Set, Add or Remove role PRIVILEGE(s).

Examples:
  govc role.update MyRole $(govc role.ls Admin | grep VirtualMachine.)
  govc role.update -r MyRole $(govc role.ls Admin | grep VirtualMachine.GuestOperations.)
  govc role.update -a MyRole $(govc role.ls Admin | grep Datastore.)
  govc role.update -name RockNRole MyRole`
}

func (cmd *update) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
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

	ids := role.Privilege
	args := f.Args()[1:]

	if cmd.add {
		ids = append(ids, args...)
	} else if cmd.remove {
		ids = nil
		rm := make(map[string]bool, len(args))
		for _, arg := range args {
			rm[arg] = true
		}

		for _, id := range role.Privilege {
			if !rm[id] {
				ids = append(ids, id)
			}
		}
	} else if len(args) != 0 {
		ids = args
	}

	if cmd.name == "" {
		cmd.name = role.Name
	}

	return m.UpdateRole(ctx, role.RoleId, cmd.name, ids)
}
