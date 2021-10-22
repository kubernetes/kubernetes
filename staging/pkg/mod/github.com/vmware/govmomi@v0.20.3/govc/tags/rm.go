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

package tags

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

	cat   string
	force bool
}

func init() {
	cli.Register("tags.rm", &rm{})
}

func (cmd *rm) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.cat, "c", "", "Tag category")
	f.BoolVar(&cmd.force, "f", false, "Delete tag regardless of attached objects")
}

func (cmd *rm) Usage() string {
	return "NAME"
}

func (cmd *rm) Description() string {
	return `Delete tag NAME.

Fails if tag is attached to any object, unless the '-f' flag is provided.

Examples:
  govc tags.rm k8s-zone-us-ca1
  govc tags.rm -f -c k8s-zone us-ca2`
}

func (cmd *rm) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	tagID := f.Arg(0)

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		m := tags.NewManager(c)
		if cmd.force == false {
			objs, err := m.ListAttachedObjects(ctx, tagID)
			if err != nil {
				return err
			}
			if len(objs) > 0 {
				return fmt.Errorf("tag %s has %d attached objects", tagID, len(objs))
			}
		}
		tag, err := m.GetTagForCategory(ctx, tagID, cmd.cat)
		if err != nil {
			return err
		}
		return m.DeleteTag(ctx, tag)
	})
}
