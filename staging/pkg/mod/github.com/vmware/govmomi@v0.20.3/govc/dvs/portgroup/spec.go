/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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
	"strings"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type DVPortgroupConfigSpec struct {
	types.DVPortgroupConfigSpec
}

func (spec *DVPortgroupConfigSpec) Register(ctx context.Context, f *flag.FlagSet) {
	ptypes := []string{
		string(types.DistributedVirtualPortgroupPortgroupTypeEarlyBinding),
		string(types.DistributedVirtualPortgroupPortgroupTypeLateBinding),
		string(types.DistributedVirtualPortgroupPortgroupTypeEphemeral),
	}

	f.StringVar(&spec.Type, "type", ptypes[0],
		fmt.Sprintf("Portgroup type (%s)", strings.Join(ptypes, "|")))

	f.Var(flags.NewInt32(&spec.NumPorts), "nports", "Number of ports")

	config := new(types.VMwareDVSPortSetting)
	vlan := new(types.VmwareDistributedVirtualSwitchVlanIdSpec)
	spec.DefaultPortConfig = config
	config.Vlan = vlan

	f.Var(flags.NewInt32(&vlan.VlanId), "vlan", "VLAN ID")
}

func (spec *DVPortgroupConfigSpec) Spec() types.DVPortgroupConfigSpec {
	return spec.DVPortgroupConfigSpec
}
