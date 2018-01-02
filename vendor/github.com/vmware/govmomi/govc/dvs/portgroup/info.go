/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package portgroup

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.DatacenterFlag

	pg         string
	active     bool
	connected  bool
	inside     bool
	uplinkPort bool
	vlanID     int
	count      uint
}

func init() {
	cli.Register("dvs.portgroup.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.StringVar(&cmd.pg, "pg", "", "Distributed Virtual Portgroup")
	f.BoolVar(&cmd.active, "active", false, "Filter by port active or inactive status")
	f.BoolVar(&cmd.connected, "connected", false, "Filter by port connected or disconnected status")
	f.BoolVar(&cmd.inside, "inside", true, "Filter by port inside or outside status")
	f.BoolVar(&cmd.uplinkPort, "uplinkPort", false, "Filter for uplink ports")
	f.IntVar(&cmd.vlanID, "vlan", 0, "Filter by VLAN ID (0 = unfiltered)")
	f.UintVar(&cmd.count, "count", 0, "Number of matches to return (0 = unlimited)")
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}

	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	client, err := cmd.Client()
	if err != nil {
		return err
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	// Retrieve DVS reference
	net, err := finder.Network(ctx, f.Arg(0))
	if err != nil {
		return err
	}

	// Convert to DVS object type
	dvs, ok := net.(*object.DistributedVirtualSwitch)
	if !ok {
		return fmt.Errorf("%s (%s) is not a DVS", f.Arg(0), net.Reference().Type)
	}

	// Set base search criteria
	criteria := types.DistributedVirtualSwitchPortCriteria{
		Connected:  types.NewBool(cmd.connected),
		Active:     types.NewBool(cmd.active),
		UplinkPort: types.NewBool(cmd.uplinkPort),
		Inside:     types.NewBool(cmd.inside),
	}

	// If a distributed virtual portgroup path is set, then add its portgroup key to the base criteria
	if len(cmd.pg) > 0 {
		// Retrieve distributed virtual portgroup reference
		net, err = finder.Network(ctx, cmd.pg)
		if err != nil {
			return err
		}

		// Convert distributed virtual portgroup object type
		dvpg, ok := net.(*object.DistributedVirtualPortgroup)
		if !ok {
			return fmt.Errorf("%s (%s) is not a DVPG", cmd.pg, net.Reference().Type)
		}

		// Obtain portgroup key property
		var dvp mo.DistributedVirtualPortgroup
		if err := dvpg.Properties(ctx, dvpg.Reference(), []string{"key"}, &dvp); err != nil {
			return err
		}

		// Add portgroup key to port search criteria
		criteria.PortgroupKey = []string{dvp.Key}
	}

	// Prepare request
	req := types.FetchDVPorts{
		This:     dvs.Reference(),
		Criteria: &criteria,
	}

	// Fetch ports
	res, err := methods.FetchDVPorts(ctx, client, &req)
	if err != nil {
		return err
	}

	var returnedPorts uint

	// Iterate over returned ports
	for _, port := range res.Returnval {
		portConfigSetting := port.Config.Setting.(*types.VMwareDVSPortSetting)
		portVlan := portConfigSetting.Vlan.(*types.VmwareDistributedVirtualSwitchVlanIdSpec)
		portVlanID := portVlan.VlanId

		// Show port info if: VLAN ID is not defined, or VLAN ID matches requested VLAN
		if cmd.vlanID == 0 || portVlanID == int32(cmd.vlanID) {
			returnedPorts++

			fmt.Printf("PortgroupKey: %s\n", port.PortgroupKey)
			fmt.Printf("DvsUuid:      %s\n", port.DvsUuid)
			fmt.Printf("VlanId:       %d\n", portVlanID)
			fmt.Printf("PortKey:      %s\n\n", port.Key)

			// If we are limiting the count and have reached the count, then stop returning output
			if cmd.count > 0 && returnedPorts == cmd.count {
				break
			}
		}
	}

	return nil
}
