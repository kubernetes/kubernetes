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
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type about struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("about", &about{})
}

func (cmd *about) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *about) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *about) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}
	a := c.ServiceContent.About

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "Name:\t%s\n", a.Name)
	fmt.Fprintf(tw, "Vendor:\t%s\n", a.Vendor)
	fmt.Fprintf(tw, "Version:\t%s\n", a.Version)
	fmt.Fprintf(tw, "Build:\t%s\n", a.Build)
	fmt.Fprintf(tw, "OS type:\t%s\n", a.OsType)
	fmt.Fprintf(tw, "API type:\t%s\n", a.ApiType)
	fmt.Fprintf(tw, "API version:\t%s\n", a.ApiVersion)
	fmt.Fprintf(tw, "Product ID:\t%s\n", a.ProductLineId)
	fmt.Fprintf(tw, "UUID:\t%s\n", a.InstanceUuid)
	return tw.Flush()
}
