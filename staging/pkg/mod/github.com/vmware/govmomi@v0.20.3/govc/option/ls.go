/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package option

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type List struct {
	*flags.ClientFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("option.ls", &List{})
}

func (cmd *List) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *List) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *List) Usage() string {
	return "[NAME]"
}

var ListDescription = `List option with the given NAME.

If NAME ends with a dot, all options for that subtree are listed.`

func (cmd *List) Description() string {
	return ListDescription + `

Examples:
  govc option.ls
  govc option.ls config.vpxd.sso.
  govc option.ls config.vpxd.sso.sts.uri`
}

func (cmd *List) Query(ctx context.Context, f *flag.FlagSet, m *object.OptionManager) error {
	var err error
	var opts []types.BaseOptionValue

	if f.NArg() > 1 {
		return flag.ErrHelp
	}

	if f.NArg() == 1 {
		opts, err = m.Query(ctx, f.Arg(0))
	} else {
		var om mo.OptionManager
		err = m.Properties(ctx, m.Reference(), []string{"setting"}, &om)
		opts = om.Setting
	}

	if err != nil {
		return err
	}

	return cmd.WriteResult(optionResult(opts))
}

func (cmd *List) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m := object.NewOptionManager(c, *c.ServiceContent.Setting)

	return cmd.Query(ctx, f, m)
}

type optionResult []types.BaseOptionValue

func (r optionResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)
	for _, opt := range r {
		o := opt.GetOptionValue()
		fmt.Fprintf(tw, "%s:\t%v\n", o.Key, o.Value)
	}
	return tw.Flush()
}
