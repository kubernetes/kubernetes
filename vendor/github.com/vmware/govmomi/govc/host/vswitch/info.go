/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package vswitch

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.vswitch.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
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

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}

	ns, err := cmd.HostNetworkSystem()
	if err != nil {
		return err
	}

	var mns mo.HostNetworkSystem

	pc := property.DefaultCollector(client)
	err = pc.RetrieveOne(ctx, ns.Reference(), []string{"networkInfo.vswitch"}, &mns)
	if err != nil {
		return err
	}

	r := &infoResult{mns.NetworkInfo.Vswitch}

	return cmd.WriteResult(r)
}

type infoResult struct {
	Vswitch []types.HostVirtualSwitch
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for i, s := range r.Vswitch {
		if i > 0 {
			fmt.Fprintln(tw)
		}
		fmt.Fprintf(tw, "Name:\t%s\n", s.Name)
		fmt.Fprintf(tw, "Portgroup:\t%s\n", keys("key-vim.host.PortGroup-", s.Portgroup))
		fmt.Fprintf(tw, "Pnic:\t%s\n", keys("key-vim.host.PhysicalNic-", s.Pnic))
		fmt.Fprintf(tw, "MTU:\t%d\n", s.Mtu)
		fmt.Fprintf(tw, "Ports:\t%d\n", s.NumPorts)
		fmt.Fprintf(tw, "Ports Available:\t%d\n", s.NumPortsAvailable)
		HostNetworkPolicy(tw, s.Spec.Policy)
	}

	return tw.Flush()
}

func keys(key string, vals []string) string {
	for i, val := range vals {
		vals[i] = strings.TrimPrefix(val, key)
	}
	return strings.Join(vals, ", ")
}

func enabled(b *bool) string {
	if b != nil && *b {
		return "Yes"
	}
	return "No"
}

func HostNetworkPolicy(w io.Writer, p *types.HostNetworkPolicy) {
	if p == nil || p.Security == nil {
		return // e.g. Workstation
	}
	fmt.Fprintf(w, "Allow promiscuous mode:\t%s\n", enabled(p.Security.AllowPromiscuous))
	fmt.Fprintf(w, "Allow forged transmits:\t%s\n", enabled(p.Security.ForgedTransmits))
	fmt.Fprintf(w, "Allow MAC changes:\t%s\n", enabled(p.Security.MacChanges))
}
