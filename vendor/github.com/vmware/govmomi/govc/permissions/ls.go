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
)

type ls struct {
	*PermissionFlag

	inherited bool
}

func init() {
	cli.Register("permissions.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PermissionFlag, ctx = NewPermissionFlag(ctx)
	cmd.PermissionFlag.Register(ctx, f)

	f.BoolVar(&cmd.inherited, "a", true, "Include inherited permissions defined by parent entities")
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.PermissionFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "[PATH]..."
}

func (cmd *ls) Description() string {
	return `List the permissions defined on or effective on managed entities.

Examples:
  govc permissions.ls
  govc permissions.ls /dc1/host/cluster1`
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	refs, err := cmd.ManagedObjects(ctx, f.Args())
	if err != nil {
		return err
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	for _, ref := range refs {
		perms, err := m.RetrieveEntityPermissions(ctx, ref, cmd.inherited)
		if err != nil {
			return err
		}

		cmd.List.Add(perms)
	}

	return cmd.WriteResult(&cmd.List)
}
