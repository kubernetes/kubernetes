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

package service

import (
	"context"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/lookup"
	"github.com/vmware/govmomi/lookup/types"
)

type ls struct {
	*flags.ClientFlag
	*flags.OutputFlag

	long bool
	url  bool

	types.LookupServiceRegistrationFilter
}

func init() {
	cli.Register("sso.service.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.long, "l", false, "Long listing format")
	f.BoolVar(&cmd.url, "U", false, "List endpoint URL(s) only")

	cmd.LookupServiceRegistrationFilter.EndpointType = new(types.LookupServiceRegistrationEndpointType)
	cmd.LookupServiceRegistrationFilter.ServiceType = new(types.LookupServiceRegistrationServiceType)
	f.StringVar(&cmd.SiteId, "s", "", "Site ID")
	f.StringVar(&cmd.NodeId, "n", "", "Node ID")
	f.StringVar(&cmd.ServiceType.Type, "t", "", "Service type")
	f.StringVar(&cmd.ServiceType.Product, "p", "", "Service product")
	f.StringVar(&cmd.EndpointType.Type, "T", "", "Endpoint type")
	f.StringVar(&cmd.EndpointType.Protocol, "P", "", "Endpoint protocol")
}

func (cmd *ls) Description() string {
	return `List platform services.

Examples:
  govc sso.service.ls
  govc sso.service.ls -t vcenterserver -P vmomi
  govc sso.service.ls -t sso:sts
  govc sso.service.ls -t sso:sts -U
  govc sso.service.ls -t sso:sts -json | jq -r .[].ServiceEndpoints[].Url`
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.OutputFlag.Process(ctx)
}

type infoResult []types.LookupServiceRegistrationInfo

func (r infoResult) Dump() interface{} {
	return []types.LookupServiceRegistrationInfo(r)
}

func (r infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, info := range r {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", info.ServiceType.Product, info.ServiceType.Type, info.ServiceId)
	}

	return tw.Flush()
}

type infoResultLong []types.LookupServiceRegistrationInfo

func (r infoResultLong) Dump() interface{} {
	return []types.LookupServiceRegistrationInfo(r)
}

func (r infoResultLong) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, info := range r {
		for _, s := range info.ServiceEndpoints {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n",
				info.ServiceType.Product, info.ServiceType.Type, info.ServiceId,
				s.EndpointType.Protocol, s.EndpointType.Type, s.Url)
		}
	}

	return tw.Flush()
}

type infoResultURL []types.LookupServiceRegistrationInfo

func (r infoResultURL) Dump() interface{} {
	return []types.LookupServiceRegistrationInfo(r)
}

func (r infoResultURL) Write(w io.Writer) error {
	for _, info := range r {
		for _, s := range info.ServiceEndpoints {
			fmt.Fprintln(w, s.Url)
		}
	}

	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	vc, err := cmd.Client()
	if err != nil {
		return err
	}

	c, err := lookup.NewClient(ctx, vc)
	if err != nil {
		return err
	}

	info, err := c.List(ctx, &cmd.LookupServiceRegistrationFilter)
	if err != nil {
		return err
	}

	switch {
	case cmd.long:
		return cmd.WriteResult(infoResultLong(info))
	case cmd.url:
		return cmd.WriteResult(infoResultURL(info))
	default:
		return cmd.WriteResult(infoResult(info))
	}
}
