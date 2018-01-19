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

package simulator

import (
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HostFirewallSystem struct {
	mo.HostFirewallSystem
}

func NewHostFirewallSystem(_ *mo.HostSystem) *HostFirewallSystem {
	info := esx.HostFirewallInfo

	return &HostFirewallSystem{
		HostFirewallSystem: mo.HostFirewallSystem{
			FirewallInfo: &info,
		},
	}
}

func DisableRuleset(info *types.HostFirewallInfo, id string) bool {
	for i := range info.Ruleset {
		if info.Ruleset[i].Key == id {
			info.Ruleset[i].Enabled = false
			return true
		}
	}

	return false
}

func (s *HostFirewallSystem) DisableRuleset(req *types.DisableRuleset) soap.HasFault {
	body := &methods.DisableRulesetBody{}

	if DisableRuleset(s.HostFirewallSystem.FirewallInfo, req.Id) {
		body.Res = new(types.DisableRulesetResponse)
		return body
	}

	body.Fault_ = Fault("", &types.NotFound{})

	return body
}

func EnableRuleset(info *types.HostFirewallInfo, id string) bool {
	for i := range info.Ruleset {
		if info.Ruleset[i].Key == id {
			info.Ruleset[i].Enabled = true
			return true
		}
	}

	return false
}

func (s *HostFirewallSystem) EnableRuleset(req *types.EnableRuleset) soap.HasFault {
	body := &methods.EnableRulesetBody{}

	if EnableRuleset(s.HostFirewallSystem.FirewallInfo, req.Id) {
		body.Res = new(types.EnableRulesetResponse)
		return body
	}

	body.Fault_ = Fault("", &types.NotFound{})

	return body
}
