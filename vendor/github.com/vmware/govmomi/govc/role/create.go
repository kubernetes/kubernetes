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

type create struct {
	*permissions.PermissionFlag
}

func init() {
	cli.Register("role.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = permissions.NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Usage() string {
	return "NAME [PRIVILEGE]..."
}

func (cmd *create) Description() string {
	return `Create authorization role.

Optionally populate the role with the given PRIVILEGE(s).

Examples:
  govc role.create MyRole
  govc role.create NoDC $(govc role.ls Admin | grep -v Datacenter.)`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	_, err = m.AddRole(ctx, f.Arg(0), f.Args()[1:])
	return err
}
