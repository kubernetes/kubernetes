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
	"io"
	"reflect"
	"sort"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
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
	count      int
	dvsRules   bool
}

var protocols = map[int32]string{
	1:  "icmp",
	2:  "igmp",
	6:  "tcp",
	17: "udp",
	58: "ipv6-icmp",
}

type trafficRule struct {
	Description        string
	Direction          string
	Action             string
	Protocol           string
	SourceAddress      string
	SourceIpPort       string
	DestinationAddress string
	DestinationIpPort  string
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
	f.IntVar(&cmd.count, "count", 0, "Number of matches to return (0 = unlimited)")
	f.BoolVar(&cmd.dvsRules, "r", false, "Show DVS rules")
}

func (cmd *info) Usage() string {
	return "DVS"
}

func (cmd *info) Description() string {
	return `Portgroup info for DVS.

Examples:
  govc dvs.portgroup.info DSwitch
  govc dvs.portgroup.info -pg InternalNetwork DSwitch
  govc find / -type DistributedVirtualSwitch | xargs -n1 govc dvs.portgroup.info`
}

type infoResult struct {
	Port []types.DistributedVirtualPort
	cmd  *info
}

func printPort(port types.BaseDvsIpPort) string {
	if port != nil {
		switch port.(type) {
		case *types.DvsSingleIpPort:
			return fmt.Sprintf("%d", port.(*types.DvsSingleIpPort).PortNumber)
		case *types.DvsIpPortRange:
			return fmt.Sprintf("%d-%d", port.(*types.DvsIpPortRange).StartPortNumber, port.(*types.DvsIpPortRange).EndPortNumber)
		}
	}
	return "Any"
}

func printAddress(address types.BaseIpAddress) string {
	if address != nil {
		switch (address).(type) {
		case *types.SingleIp:
			return fmt.Sprintf("%s", address.(*types.SingleIp).Address)
		case *types.IpRange:
			return fmt.Sprintf("%s/%d", address.(*types.IpRange).AddressPrefix, address.(*types.IpRange).PrefixLength)
		}
	}
	return "Any"
}

func printAction(action types.BaseDvsNetworkRuleAction) string {
	if action != nil {
		switch (action).(type) {
		case *types.DvsAcceptNetworkRuleAction:
			return fmt.Sprintf("Accept")
		case *types.DvsDropNetworkRuleAction:
			return fmt.Sprintf("Drop")
		}
	}
	return "n/a"
}

func printTable(trafficRuleSet map[int]map[int]trafficRule, portID int) {
	if len(trafficRuleSet[portID]) == 0 {
		return
	}

	keys := []int{}
	for k := range trafficRuleSet[portID] {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	tabWidthInt := 22
	tabWidth := fmt.Sprintf("%d", tabWidthInt)
	headLen := 9*(tabWidthInt+2) - 1
	fmt.Printf("+" + strings.Repeat("-", headLen) + "+\n")
	format := "| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s| %-" + tabWidth +
		"s|\n"
	fmt.Printf(format,
		"Sequence",
		"Description",
		"Direction",
		"Action",
		"Protocol",
		"SourceAddress",
		"SourceIpPort",
		"DestinationAddress",
		"DestinationIpPort")
	fmt.Printf("+" + strings.Repeat("-", headLen) + "+\n")
	for _, id := range keys {
		fmt.Printf(format,
			fmt.Sprintf("%d", id),
			trafficRuleSet[portID][id].Description,
			trafficRuleSet[portID][id].Direction,
			trafficRuleSet[portID][id].Action,
			trafficRuleSet[portID][id].Protocol,
			trafficRuleSet[portID][id].SourceAddress,
			trafficRuleSet[portID][id].SourceIpPort,
			trafficRuleSet[portID][id].DestinationAddress,
			trafficRuleSet[portID][id].DestinationIpPort)
	}
	fmt.Printf("+" + strings.Repeat("-", headLen) + "+\n")
}

func (r *infoResult) Write(w io.Writer) error {
	trafficRuleSet := make(map[int]map[int]trafficRule)
	for portID, port := range r.Port {
		var vlanID int32
		setting := port.Config.Setting.(*types.VMwareDVSPortSetting)

		switch vlan := setting.Vlan.(type) {
		case *types.VmwareDistributedVirtualSwitchVlanIdSpec:
			vlanID = vlan.VlanId
		case *types.VmwareDistributedVirtualSwitchTrunkVlanSpec:
		case *types.VmwareDistributedVirtualSwitchPvlanSpec:
			vlanID = vlan.PvlanId
		}

		// Show port info if: VLAN ID is not defined, or VLAN ID matches requested VLAN
		if r.cmd.vlanID == 0 || vlanID == int32(r.cmd.vlanID) {
			fmt.Printf("PortgroupKey: %s\n", port.PortgroupKey)
			fmt.Printf("DvsUuid:      %s\n", port.DvsUuid)
			fmt.Printf("VlanId:       %d\n", vlanID)
			fmt.Printf("PortKey:      %s\n\n", port.Key)

			trafficRuleSet[portID] = make(map[int]trafficRule)

			if r.cmd.dvsRules && setting.FilterPolicy != nil &&
				setting.FilterPolicy.FilterConfig != nil &&
				len(setting.FilterPolicy.FilterConfig) > 0 {

				rules := setting.FilterPolicy.FilterConfig[0].GetDvsTrafficFilterConfig()
				if rules != nil && rules.TrafficRuleset != nil && rules.TrafficRuleset.Rules != nil {
					for _, rule := range rules.TrafficRuleset.Rules {
						for _, q := range rule.Qualifier {
							var protocol string
							if val, ok := protocols[q.GetDvsIpNetworkRuleQualifier().Protocol.Value]; ok {
								protocol = val
							} else {
								protocol = fmt.Sprintf("%d", q.GetDvsIpNetworkRuleQualifier().Protocol.Value)
							}

							trafficRuleSet[portID][int(rule.Sequence)] = trafficRule{
								Description:        rule.Description,
								Direction:          rule.Direction,
								Action:             printAction(rule.Action),
								Protocol:           protocol,
								SourceAddress:      printAddress(q.GetDvsIpNetworkRuleQualifier().SourceAddress),
								SourceIpPort:       printPort(q.GetDvsIpNetworkRuleQualifier().SourceIpPort),
								DestinationAddress: printAddress(q.GetDvsIpNetworkRuleQualifier().DestinationAddress),
								DestinationIpPort:  printPort(q.GetDvsIpNetworkRuleQualifier().DestinationIpPort),
							}
						}
					}
				}
			}
		}
	}

	if r.cmd.dvsRules && len(r.Port) > 0 {
		eq := 0
		for i, _ := range r.Port {
			if i > 0 {
				reflect.DeepEqual(trafficRuleSet[i-1], trafficRuleSet[i])
				if reflect.DeepEqual(trafficRuleSet[i-1], trafficRuleSet[i]) {
					eq++
				} else {
					fmt.Printf("%s and %s port rules are unequal\n", r.Port[i-1].Key, r.Port[i].Key)
					break
				}
			}
		}

		if eq == len(trafficRuleSet)-1 {
			printTable(trafficRuleSet, 0)
		} else {
			for portID, port := range r.Port {
				fmt.Printf("\nPortKey:      %s\n", port.Key)
				printTable(trafficRuleSet, portID)
			}
		}
	}

	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
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

	res, err := dvs.FetchDVPorts(ctx, &criteria)
	if err != nil {
		return err
	}

	// Truncate output to -count if specified
	if cmd.count > 0 && cmd.count < len(res) {
		res = res[:cmd.count]
	}

	info := infoResult{
		cmd:  cmd,
		Port: res,
	}

	return cmd.WriteResult(&info)
}
