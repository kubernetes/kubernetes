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
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
	c string
	C bool
}

func init() {
	cli.Register("tags.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	cmd.OutputFlag.Register(ctx, f)
	f.StringVar(&cmd.c, "c", "", "Category name")
	f.BoolVar(&cmd.C, "C", true, "Display category name instead of ID")
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.OutputFlag.Process(ctx)
}

func (cmd *info) Usage() string {
	return "NAME"
}

func (cmd *info) Description() string {
	return `Display tags info.

If NAME is provided, display info for only that tag.  Otherwise display info for all tags.

Examples:
  govc tags.info
  govc tags.info k8s-zone-us-ca1
  govc tags.info -c k8s-zone`
}

type infoResult []tags.Tag

func (t infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, item := range t {
		fmt.Fprintf(tw, "Name:\t%s\n", item.Name)
		fmt.Fprintf(tw, "  ID:\t%s\n", item.ID)
		fmt.Fprintf(tw, "  Description:\t%s\n", item.Description)
		fmt.Fprintf(tw, "  Category:\t%s\n", item.CategoryID)
		fmt.Fprintf(tw, "  UsedBy: %s\n", item.UsedBy)
	}

	return tw.Flush()
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		m := tags.NewManager(c)
		var res lsResult
		var err error

		if cmd.c == "" {
			res, err = m.GetTags(ctx)
		} else {
			res, err = m.GetTagsForCategory(ctx, cmd.c)
		}
		if err != nil {
			return err
		}

		if f.NArg() == 1 {
			arg := f.Arg(0)
			src := res
			res = nil
			for i := range src {
				if src[i].Name == arg || src[i].ID == arg {
					res = append(res, src[i])
					break
				}
			}
			if len(res) == 0 {
				return fmt.Errorf("tag %q not found", arg)
			}
		}

		if cmd.C {
			categories, err := m.GetCategories(ctx)
			if err != nil {
				return err
			}
			m := make(map[string]tags.Category)
			for _, category := range categories {
				m[category.ID] = category
			}
			for i := range res {
				res[i].CategoryID = m[res[i].CategoryID].Name
			}
		}

		return cmd.WriteResult(infoResult(res))
	})
}
