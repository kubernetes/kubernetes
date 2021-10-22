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

package extension

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
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("extension.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Usage() string {
	return "[KEY]..."
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := object.GetExtensionManager(c)
	if err != nil {
		return err
	}

	list, err := m.List(ctx)
	if err != nil {
		return err
	}

	var res infoResult

	if f.NArg() == 0 {
		res.Extensions = list
	} else {
		exts := make(map[string]types.Extension)
		for _, e := range list {
			exts[e.Key] = e
		}

		for _, key := range f.Args() {
			if e, ok := exts[key]; ok {
				res.Extensions = append(res.Extensions, e)
			} else {
				return fmt.Errorf("extension %s not found", key)
			}
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	Extensions []types.Extension
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, e := range r.Extensions {
		fmt.Fprintf(tw, "Name:\t%s\n", e.Key)
		fmt.Fprintf(tw, "  Version:\t%s\n", e.Version)
		fmt.Fprintf(tw, "  Description:\t%s\n", e.Description.GetDescription().Summary)
		fmt.Fprintf(tw, "  Company:\t%s\n", e.Company)
		fmt.Fprintf(tw, "  Last heartbeat time:\t%s\n", e.LastHeartbeatTime)
		fmt.Fprintf(tw, "  Subject name:\t%s\n", e.SubjectName)
		fmt.Fprintf(tw, "  Type:\t%s\n", e.Type)
	}

	return tw.Flush()
}
