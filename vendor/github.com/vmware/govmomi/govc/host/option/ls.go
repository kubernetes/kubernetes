/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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
	"github.com/vmware/govmomi/vim25/types"
)

type ls struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.option.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "NAME"
}

func (cmd *ls) Description() string {
	return `List option with the given NAME.

If NAME ends with a dot, all options for that subtree are listed.`
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	m, err := host.ConfigManager().OptionManager(ctx)
	if err != nil {
		return err
	}

	opts, err := m.Query(ctx, f.Arg(0))
	if err != nil {
		return err
	}

	return cmd.WriteResult(optionResult(opts))
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
