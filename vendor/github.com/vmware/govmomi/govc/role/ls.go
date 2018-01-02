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
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/permissions"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type ls struct {
	*permissions.PermissionFlag
}

func init() {
	cli.Register("role.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = permissions.NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "[NAME]"
}

func (cmd *ls) Description() string {
	return `List authorization roles.

If NAME is provided, list privileges for the role.

Examples:
  govc role.ls
  govc role.ls Admin`
}

type lsRoleList object.AuthorizationRoleList

func (rl lsRoleList) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, role := range rl {
		fmt.Fprintf(tw, "%s\t%s\n", role.Name, role.Info.GetDescription().Summary)
	}

	return tw.Flush()
}

type lsRole types.AuthorizationRole

func (r lsRole) Write(w io.Writer) error {
	for _, p := range r.Privilege {
		fmt.Println(p)
	}
	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() > 1 {
		return flag.ErrHelp
	}

	_, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	if f.NArg() == 1 {
		role, err := cmd.Role(f.Arg(0))
		if err != nil {
			return err
		}

		return cmd.WriteResult(lsRole(*role))
	}

	return cmd.WriteResult(lsRoleList(cmd.Roles))
}
