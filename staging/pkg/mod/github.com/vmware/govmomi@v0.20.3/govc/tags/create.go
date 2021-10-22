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

type create struct {
	*flags.ClientFlag
	tag tags.Tag
}

func init() {
	cli.Register("tags.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	f.StringVar(&cmd.tag.CategoryID, "c", "", "Category name")
	f.StringVar(&cmd.tag.Description, "d", "", "Description of tag")
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Usage() string {
	return "NAME"
}

func (cmd *create) Description() string {
	return `Create tag.

The '-c' option to specify a tag category is required.
This command will output the ID of the new tag.

Examples:
  govc tags.create -d "Kubernetes Zone US CA1" -c k8s-zone k8s-zone-us-ca1`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 || cmd.tag.CategoryID == "" {
		return flag.ErrHelp
	}

	cmd.tag.Name = f.Arg(0)

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		id, err := tags.NewManager(c).CreateTag(ctx, &cmd.tag)
		if err != nil {
			return err
		}

		fmt.Println(id)
		return nil
	})
}
