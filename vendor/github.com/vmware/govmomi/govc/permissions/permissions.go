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
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type List struct {
	Roles object.AuthorizationRoleList `json:",omitempty"`

	Permissions []types.Permission `json:",omitempty"`

	f *PermissionFlag
}

type PermissionFlag struct {
	*flags.DatacenterFlag
	*flags.OutputFlag

	asRef bool

	m *object.AuthorizationManager

	List
}

func NewPermissionFlag(ctx context.Context) (*PermissionFlag, context.Context) {
	f := &PermissionFlag{}
	f.List.f = f
	f.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	f.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	return f, ctx
}

func (f *PermissionFlag) Register(ctx context.Context, fs *flag.FlagSet) {
	f.DatacenterFlag.Register(ctx, fs)
	f.OutputFlag.Register(ctx, fs)

	fs.BoolVar(&f.asRef, "i", false, "Use moref instead of inventory path")
}

func (f *PermissionFlag) Process(ctx context.Context) error {
	if err := f.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := f.OutputFlag.Process(ctx); err != nil {
		return err
	}

	return nil
}

func (f *PermissionFlag) Manager(ctx context.Context) (*object.AuthorizationManager, error) {
	if f.m != nil {
		return f.m, nil
	}

	c, err := f.Client()
	if err != nil {
		return nil, err
	}

	f.m = object.NewAuthorizationManager(c)
	f.Roles, err = f.m.RoleList(ctx)

	return f.m, err
}

func (f *PermissionFlag) Role(name string) (*types.AuthorizationRole, error) {
	role := f.Roles.ByName(name)
	if role == nil {
		return nil, fmt.Errorf("role %q not found", name)
	}
	return role, nil
}

func (f *PermissionFlag) ManagedObjects(ctx context.Context, args []string) ([]types.ManagedObjectReference, error) {
	if !f.asRef {
		return f.DatacenterFlag.ManagedObjects(ctx, args)
	}

	var refs []types.ManagedObjectReference

	for _, arg := range args {
		var ref types.ManagedObjectReference
		if ref.FromString(arg) {
			refs = append(refs, ref)
		} else {
			return nil, fmt.Errorf("invalid moref: %s", arg)
		}
	}

	return refs, nil
}

func (l *List) Write(w io.Writer) error {
	ctx := context.Background()
	finder, err := l.f.Finder()
	if err != nil {
		return err
	}

	refs := make(map[types.ManagedObjectReference]string)

	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", "Role", "Entity", "Principal", "Propagate")

	for _, perm := range l.Permissions {
		propagate := "No"
		if perm.Propagate {
			propagate = "Yes"
		}

		name := l.Roles.ById(perm.RoleId).Name

		p := "-"
		if perm.Entity != nil {
			if l.f.asRef {
				p = perm.Entity.String()
			} else {
				// convert moref to inventory path
				if p = refs[*perm.Entity]; p == "" {
					e, err := finder.Element(ctx, *perm.Entity)
					if err == nil {
						p = e.Path
					}

					refs[*perm.Entity] = p
				}
			}
		}

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", name, p, perm.Principal, propagate)
	}

	return tw.Flush()
}

func (l *List) Add(perms []types.Permission) {
	l.Permissions = append(l.Permissions, perms...)
}
