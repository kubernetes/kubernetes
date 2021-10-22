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

package override

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClusterFlag
}

func init() {
	cli.Register("cluster.override.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	cmd.ClusterFlag.Register(ctx, f)
}

func (cmd *info) Description() string {
	return `Cluster VM overrides info.

Examples:
  govc cluster.override.info
  govc cluster.override.info -json`
}

func (cmd *info) Process(ctx context.Context) error {
	return cmd.ClusterFlag.Process(ctx)
}

type Override struct {
	id   types.ManagedObjectReference
	Name string
	Host string                        `json:",omitempty"`
	DRS  *types.ClusterDrsVmConfigInfo `json:",omitempty"`
	DAS  *types.ClusterDasVmConfigInfo `json:",omitempty"`
}

type infoResult struct {
	Overrides map[string]*Override
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, entry := range r.Overrides {
		behavior := fmt.Sprintf("Default (%s)", types.DrsBehaviorFullyAutomated)
		if entry.DRS != nil {
			if *entry.DRS.Enabled {
				behavior = string(entry.DRS.Behavior)
			}
		}

		priority := fmt.Sprintf("Default (%s)", types.DasVmPriorityMedium)
		if entry.DAS != nil {
			priority = entry.DAS.DasSettings.RestartPriority
		}

		fmt.Fprintf(tw, "Name:\t%s\n", entry.Name)
		fmt.Fprintf(tw, "  DRS Automation Level:\t%s\n", strings.Title(behavior))
		fmt.Fprintf(tw, "  HA Restart Priority:\t%s\n", strings.Title(priority))
		fmt.Fprintf(tw, "  Host:\t%s\n", entry.Host)
	}

	return tw.Flush()
}

func (r *infoResult) entry(id types.ManagedObjectReference) *Override {
	key := id.String()
	vm, ok := r.Overrides[key]
	if !ok {
		r.Overrides[key] = &Override{id: id}
		vm = r.Overrides[key]
	}
	return vm
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	cluster, err := cmd.Cluster()
	if err != nil {
		return err
	}

	config, err := cluster.Configuration(ctx)
	if err != nil {
		return err
	}

	res := &infoResult{
		Overrides: make(map[string]*Override),
	}

	for i := range config.DasVmConfig {
		vm := res.entry(config.DasVmConfig[i].Key)

		vm.DAS = &config.DasVmConfig[i]
	}

	for i := range config.DrsVmConfig {
		vm := res.entry(config.DrsVmConfig[i].Key)

		vm.DRS = &config.DrsVmConfig[i]
	}

	for _, o := range res.Overrides {
		// TODO: can optimize to reduce round trips
		vm := object.NewVirtualMachine(cluster.Client(), o.id)
		o.Name, _ = vm.ObjectName(ctx)
		if h, herr := vm.HostSystem(ctx); herr == nil {
			o.Host, _ = h.ObjectName(ctx)
		}
	}

	return cmd.WriteResult(res)
}
