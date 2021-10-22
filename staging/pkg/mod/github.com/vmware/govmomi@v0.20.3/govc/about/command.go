/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package about

import (
	"context"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type about struct {
	*flags.ClientFlag
	*flags.OutputFlag

	Long bool
}

func init() {
	cli.Register("about", &about{})
}

func (cmd *about) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.Long, "l", false, "Include service content")
}

func (cmd *about) Description() string {
	return `Display About info for HOST.

System information including the name, type, version, and build number.

Examples:
  govc about
  govc about -json | jq -r .About.ProductLineId`
}

func (cmd *about) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *about) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	res := infoResult{
		a: &c.ServiceContent.About,
	}

	if cmd.Long {
		res.Content = &c.ServiceContent
	} else {
		res.About = res.a
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	Content *types.ServiceContent `json:",omitempty"`
	About   *types.AboutInfo      `json:",omitempty"`
	a       *types.AboutInfo
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "Name:\t%s\n", r.a.Name)
	fmt.Fprintf(tw, "Vendor:\t%s\n", r.a.Vendor)
	fmt.Fprintf(tw, "Version:\t%s\n", r.a.Version)
	fmt.Fprintf(tw, "Build:\t%s\n", r.a.Build)
	fmt.Fprintf(tw, "OS type:\t%s\n", r.a.OsType)
	fmt.Fprintf(tw, "API type:\t%s\n", r.a.ApiType)
	fmt.Fprintf(tw, "API version:\t%s\n", r.a.ApiVersion)
	fmt.Fprintf(tw, "Product ID:\t%s\n", r.a.ProductLineId)
	fmt.Fprintf(tw, "UUID:\t%s\n", r.a.InstanceUuid)
	return tw.Flush()
}

func (r *infoResult) Dump() interface{} {
	if r.Content != nil {
		return r.Content
	}
	return r.About
}
