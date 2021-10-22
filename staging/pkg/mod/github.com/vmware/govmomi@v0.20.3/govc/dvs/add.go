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

package dvs

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type add struct {
	*flags.HostSystemFlag

	path string
	pnic string
}

func init() {
	cli.Register("dvs.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.StringVar(&cmd.path, "dvs", "", "DVS path")
	f.StringVar(&cmd.pnic, "pnic", "vmnic0", "Name of the host physical NIC")
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *add) Usage() string {
	return "HOST..."
}

func (cmd *add) Description() string {
	return `Add hosts to DVS.

Examples:
  govc dvs.add -dvs dvsName -pnic vmnic1 hostA hostB hostC`
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	net, err := finder.Network(ctx, cmd.path)
	if err != nil {
		return err
	}

	dvs, ok := net.(*object.DistributedVirtualSwitch)
	if !ok {
		return fmt.Errorf("%s (%T) is not of type %T", cmd.path, net, dvs)
	}

	var s mo.DistributedVirtualSwitch
	err = dvs.Properties(ctx, dvs.Reference(), []string{"config"}, &s)
	if err != nil {
		return err
	}

	backing := new(types.DistributedVirtualSwitchHostMemberPnicBacking)

	for _, vmnic := range strings.Split(cmd.pnic, ",") {
		backing.PnicSpec = append(backing.PnicSpec, types.DistributedVirtualSwitchHostMemberPnicSpec{
			PnicDevice: strings.TrimSpace(vmnic),
		})
	}

	config := &types.DVSConfigSpec{
		ConfigVersion: s.Config.GetDVSConfigInfo().ConfigVersion,
	}

	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	existing := make(map[string]bool)
	// TODO: host.pnic.info command
	for _, member := range s.Config.GetDVSConfigInfo().Host {
		existing[member.Config.Host.Value] = true
	}

	for _, host := range hosts {
		ref := host.Reference()
		if existing[ref.Value] {
			fmt.Fprintf(os.Stderr, "%s is already a member of %s\n", host.InventoryPath, dvs.InventoryPath)
			continue
		}

		config.Host = append(config.Host, types.DistributedVirtualSwitchHostMemberConfigSpec{
			Operation: "add",
			Host:      ref,
			Backing:   backing,
		})
	}

	if len(config.Host) == 0 {
		return nil
	}

	task, err := dvs.Reconfigure(ctx, config)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("adding %d hosts to dvs %s... ", len(config.Host), dvs.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}
