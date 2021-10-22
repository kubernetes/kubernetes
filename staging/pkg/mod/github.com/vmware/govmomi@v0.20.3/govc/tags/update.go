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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
)

type update struct {
	*flags.ClientFlag

	tag tags.Tag
	cat string
}

func init() {
	cli.Register("tags.update", &update{})
}

func (cmd *update) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.tag.Name, "n", "", "Name of tag")
	f.StringVar(&cmd.tag.Description, "d", "", "Description of tag")
	f.StringVar(&cmd.cat, "c", "", "Tag category")
}

func (cmd *update) Usage() string {
	return "NAME"
}

func (cmd *update) Description() string {
	return `Update tag.

Examples:
  govc tags.update -d "K8s zone US-CA1" k8s-zone-us-ca1
  govc tags.update -d "K8s zone US-CA1" -c k8s-zone us-ca1`
}

func (cmd *update) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}
	arg := f.Arg(0)

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		m := tags.NewManager(c)
		tag, err := m.GetTagForCategory(ctx, arg, cmd.cat)
		if err != nil {
			return err
		}
		tag.Patch(&cmd.tag)
		return m.UpdateTag(ctx, tag)
	})
}
