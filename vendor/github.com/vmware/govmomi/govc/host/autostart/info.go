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

package autostart

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
)

type info struct {
	cli.Command

	*AutostartFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("host.autostart.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.AutostartFlag, ctx = newAutostartFlag(ctx)
	cmd.AutostartFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.AutostartFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Usage() string {
	return ""
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}

	mhas, err := cmd.HostAutoStartManager()
	if err != nil {
		return err
	}

	return cmd.WriteResult(&infoResult{client, mhas})
}

type infoResult struct {
	client *vim25.Client
	mhas   *mo.HostAutoStartManager
}

func (r *infoResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.mhas)
}

// vmPaths resolves the paths for the VMs in the result.
func (r *infoResult) vmPaths() (map[string]string, error) {
	ctx := context.TODO()
	paths := make(map[string]string)
	for _, info := range r.mhas.Config.PowerInfo {
		mes, err := mo.Ancestors(ctx, r.client, r.client.ServiceContent.PropertyCollector, info.Key)
		if err != nil {
			return nil, err
		}

		path := ""
		for _, me := range mes {
			// Skip root entity in building inventory path.
			if me.Parent == nil {
				continue
			}
			path += "/" + me.Name
		}

		paths[info.Key.Value] = path
	}

	return paths, nil
}

func (r *infoResult) Write(w io.Writer) error {
	paths, err := r.vmPaths()
	if err != nil {
		return err
	}

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "VM")
	fmt.Fprintf(tw, "\tStartAction")
	fmt.Fprintf(tw, "\tStartDelay")
	fmt.Fprintf(tw, "\tStartOrder")
	fmt.Fprintf(tw, "\tStopAction")
	fmt.Fprintf(tw, "\tStopDelay")
	fmt.Fprintf(tw, "\tWaitForHeartbeat")
	fmt.Fprintf(tw, "\n")

	for _, info := range r.mhas.Config.PowerInfo {
		fmt.Fprintf(tw, "%s", paths[info.Key.Value])
		fmt.Fprintf(tw, "\t%s", info.StartAction)
		fmt.Fprintf(tw, "\t%d", info.StartDelay)
		fmt.Fprintf(tw, "\t%d", info.StartOrder)
		fmt.Fprintf(tw, "\t%s", info.StopAction)
		fmt.Fprintf(tw, "\t%d", info.StopDelay)
		fmt.Fprintf(tw, "\t%s", info.WaitForHeartbeat)
		fmt.Fprintf(tw, "\n")
	}

	_ = tw.Flush()
	return nil
}
