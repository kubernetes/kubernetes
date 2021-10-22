/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package user

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/ssoadmin"
	"github.com/vmware/govmomi/ssoadmin/types"
)

type id struct {
	*flags.ClientFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("sso.user.id", &id{})
}

func (cmd *id) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *id) Usage() string {
	return "NAME"
}

func (cmd *id) Description() string {
	return `Print SSO user and group IDs.

Examples:
  govc sso.user.id
  govc sso.user.id Administrator
  govc sso.user.id -json Administrator`
}

func (cmd *id) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.OutputFlag.Process(ctx)
}

type userID struct {
	User  *types.AdminUser
	Group []types.PrincipalId
}

func (r *userID) Write(w io.Writer) error {
	var groups []string
	for _, g := range r.Group {
		groups = append(groups, g.Name)
	}
	fmt.Fprintf(w, "%s=%s@%s groups=%s\n", r.User.Kind, r.User.Id.Name, r.User.Id.Domain, strings.Join(groups, ","))
	return nil
}

func (r *userID) Dump() interface{} {
	return struct {
		User  *types.AdminUser
		Group []types.PrincipalId
	}{r.User, r.Group}
}

func (cmd *id) Run(ctx context.Context, f *flag.FlagSet) error {
	arg := f.Arg(0)
	if arg == "" {
		arg = cmd.Userinfo().Username()
	}

	return withClient(ctx, cmd.ClientFlag, func(c *ssoadmin.Client) error {
		var err error
		var u userID

		if u.User, err = c.FindUser(ctx, arg); err != nil {
			return err
		}
		if u.User == nil {
			return fmt.Errorf("%q: no such user", arg)
		}

		if u.Group, err = c.FindParentGroups(ctx, u.User.Id); err != nil {
			return err
		}

		return cmd.WriteResult(&u)
	})
}
