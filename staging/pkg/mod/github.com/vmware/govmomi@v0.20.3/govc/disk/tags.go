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

package disk

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vslm"
)

type tags struct {
	*flags.ClientFlag
	types.VslmTagEntry
	attach bool
}

func init() {
	cli.Register("disk.tags.attach", &tags{attach: true})
	cli.Register("disk.tags.detach", &tags{attach: false})
}

func (cmd *tags) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.ParentCategoryName, "c", "", "Tag category")
}

func (cmd *tags) Usage() string {
	return "NAME ID"
}

func (cmd *tags) name() string {
	if cmd.attach {
		return "attach"
	}
	return "detach"
}

func (cmd *tags) Description() string {
	if cmd.attach {
		return `Attach tag NAME to disk ID.

Examples:
  govc disk.tags.attach -c k8s-region k8s-region-us $id`
	}

	return `Detach tag NAME from disk ID.

Examples:
  govc disk.tags.detach -c k8s-region k8s-region-us $id`
}

func (cmd *tags) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m := vslm.NewObjectManager(c)
	cmd.TagName = f.Arg(0)
	method := m.DetachTag
	if cmd.attach {
		method = m.AttachTag
	}
	return method(ctx, f.Arg(1), cmd.VslmTagEntry)
}
