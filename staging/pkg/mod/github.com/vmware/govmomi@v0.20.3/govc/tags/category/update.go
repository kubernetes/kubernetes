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

package category

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
)

type update struct {
	*flags.ClientFlag
	cat   tags.Category
	multi *bool
}

func init() {
	cli.Register("tags.category.update", &update{})
}

func (cmd *update) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	f.StringVar(&cmd.cat.Name, "n", "", "Name of category")
	f.StringVar(&cmd.cat.Description, "d", "", "Description")
	f.Var((*kinds)(&cmd.cat.AssociableTypes), "t", "Object types")
	f.Var(flags.NewOptionalBool(&cmd.multi), "m", "Allow multiple tags per object")
}

func (cmd *update) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *update) Usage() string {
	return "NAME"
}

func (cmd *update) Description() string {
	return `Update category.

The '-t' flag can only be used to add new object types.  Removing category types is not supported by vCenter.

Examples:
  govc tags.category.update -n k8s-vcp-region -d "Kubernetes VCP region" k8s-region
  govc tags.category.update -t ClusterComputeResource k8s-zone`
}

func (cmd *update) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	arg := f.Arg(0)

	if cmd.multi != nil {
		cmd.cat.Cardinality = cardinality(*cmd.multi)
	}

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		m := tags.NewManager(c)
		cat, err := m.GetCategory(ctx, arg)
		if err != nil {
			return err
		}
		cat.Patch(&cmd.cat)

		return m.UpdateCategory(ctx, cat)
	})
}
