/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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

package pool

import (
	"context"
	"flag"
	"fmt"
	"path"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type create struct {
	*flags.DatacenterFlag
	*ResourceConfigSpecFlag
}

func init() {
	cli.Register("pool.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.ResourceConfigSpecFlag = NewResourceConfigSpecFlag()
	cmd.ResourceConfigSpecFlag.Register(ctx, f)
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourceConfigSpecFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Usage() string {
	return "POOL..."
}

func (cmd *create) Description() string {
	return "Create one or more resource POOLs.\n" + poolCreateHelp
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, arg := range f.Args() {
		dir := path.Dir(arg)
		base := path.Base(arg)
		parents, err := finder.ResourcePoolList(ctx, dir)
		if err != nil {
			if _, ok := err.(*find.NotFoundError); ok {
				return fmt.Errorf("cannot create resource pool '%s': parent not found", base)
			}
			return err
		}

		for _, parent := range parents {
			_, err = parent.Create(ctx, base, cmd.ResourceConfigSpec)
			if err != nil {
				return err
			}
		}
	}

	return nil
}
