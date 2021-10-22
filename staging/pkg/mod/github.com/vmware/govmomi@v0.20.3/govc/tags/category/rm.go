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
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
)

type rm struct {
	*flags.ClientFlag
	force bool
}

func init() {
	cli.Register("tags.category.rm", &rm{})
}

func (cmd *rm) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	f.BoolVar(&cmd.force, "f", false, "Delete tag regardless of attached objects")
}

func (cmd *rm) Usage() string {
	return "NAME"
}

func (cmd *rm) Description() string {
	return `Delete category NAME.

Fails if category is used by any tag, unless the '-f' flag is provided.

Examples:
  govc tags.category.rm k8s-region
  govc tags.category.rm -f k8s-zone`
}

func (cmd *rm) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}
	categoryID := f.Arg(0)

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		m := tags.NewManager(c)
		cat, err := m.GetCategory(ctx, categoryID)
		if err != nil {
			return err
		}
		if cmd.force == false {
			ctags, err := m.ListTagsForCategory(ctx, cat.ID)
			if err != nil {
				return err
			}
			if len(ctags) > 0 {
				return fmt.Errorf("category %s used by %d tags", categoryID, len(ctags))
			}
		}
		return m.DeleteCategory(ctx, cat)
	})
}
