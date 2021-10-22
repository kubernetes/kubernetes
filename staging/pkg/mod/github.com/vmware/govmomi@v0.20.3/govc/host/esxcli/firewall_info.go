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

package esxcli

import "github.com/vmware/govmomi/object"

type FirewallInfo struct {
	Loaded        bool
	Enabled       bool
	DefaultAction string
}

// GetFirewallInfo via 'esxcli network firewall get'
// The HostFirewallSystem type does not expose this data.
// This helper can be useful in particular to determine if the firewall is enabled or disabled.
func GetFirewallInfo(s *object.HostSystem) (*FirewallInfo, error) {
	x, err := NewExecutor(s.Client(), s)
	if err != nil {
		return nil, err
	}

	res, err := x.Run([]string{"network", "firewall", "get"})
	if err != nil {
		return nil, err
	}

	info := &FirewallInfo{
		Loaded:        res.Values[0]["Loaded"][0] == "true",
		Enabled:       res.Values[0]["Enabled"][0] == "true",
		DefaultAction: res.Values[0]["DefaultAction"][0],
	}

	return info, nil
}
