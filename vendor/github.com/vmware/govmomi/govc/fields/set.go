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

package fields

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type set struct {
	*flags.DatacenterFlag
}

func init() {
	cli.Register("fields.set", &set{})
}

func (cmd *set) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
}

func (cmd *set) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *set) Usage() string {
	return "KEY VALUE PATH..."
}

func (cmd *set) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() < 3 {
		return flag.ErrHelp
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := object.GetCustomFieldsManager(c)
	if err != nil {
		return err
	}

	args := f.Args()

	key, err := m.FindKey(ctx, args[0])
	if err != nil {
		return err
	}

	val := args[1]

	objs, err := cmd.ManagedObjects(ctx, args[2:])
	if err != nil {
		return err
	}

	for _, ref := range objs {
		err := m.Set(ctx, ref, key, val)
		if err != nil {
			return err
		}
	}

	return nil
}
