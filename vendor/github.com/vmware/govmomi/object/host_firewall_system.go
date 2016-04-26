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

package object

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type HostFirewallSystem struct {
	Common
}

func NewHostFirewallSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostFirewallSystem {
	return &HostFirewallSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostFirewallSystem) DisableRuleset(ctx context.Context, id string) error {
	req := types.DisableRuleset{
		This: s.Reference(),
		Id:   id,
	}

	_, err := methods.DisableRuleset(ctx, s.c, &req)
	return err
}

func (s HostFirewallSystem) EnableRuleset(ctx context.Context, id string) error {
	req := types.EnableRuleset{
		This: s.Reference(),
		Id:   id,
	}

	_, err := methods.EnableRuleset(ctx, s.c, &req)
	return err
}

func (s HostFirewallSystem) Refresh(ctx context.Context) error {
	req := types.RefreshFirewall{
		This: s.Reference(),
	}

	_, err := methods.RefreshFirewall(ctx, s.c, &req)
	return err
}

func (s HostFirewallSystem) Info(ctx context.Context) (*types.HostFirewallInfo, error) {
	var fs mo.HostFirewallSystem

	err := s.Properties(ctx, s.Reference(), []string{"firewallInfo"}, &fs)
	if err != nil {
		return nil, err
	}

	return fs.FirewallInfo, nil
}

// HostFirewallRulesetList provides helpers for a slice of types.HostFirewallRuleset
type HostFirewallRulesetList []types.HostFirewallRuleset

// ByRule returns a HostFirewallRulesetList where Direction, PortType and Protocol are equal and Port is within range
func (l HostFirewallRulesetList) ByRule(rule types.HostFirewallRule) HostFirewallRulesetList {
	var matches HostFirewallRulesetList

	for _, rs := range l {
		for _, r := range rs.Rule {
			if r.PortType != rule.PortType ||
				r.Protocol != rule.Protocol ||
				r.Direction != rule.Direction {
				continue
			}

			if r.EndPort == 0 && rule.Port == r.Port ||
				rule.Port >= r.Port && rule.Port <= r.EndPort {
				matches = append(matches, rs)
				break
			}
		}
	}

	return matches
}

// EnabledByRule returns a HostFirewallRulesetList with Match(rule) applied and filtered via Enabled()
// if enabled param is true, otherwise filtered via Disabled().
// An error is returned if the resulting list is empty.
func (l HostFirewallRulesetList) EnabledByRule(rule types.HostFirewallRule, enabled bool) (HostFirewallRulesetList, error) {
	var matched, skipped HostFirewallRulesetList
	var matchedKind, skippedKind string

	l = l.ByRule(rule)

	if enabled {
		matched = l.Enabled()
		matchedKind = "enabled"

		skipped = l.Disabled()
		skippedKind = "disabled"
	} else {
		matched = l.Disabled()
		matchedKind = "disabled"

		skipped = l.Enabled()
		skippedKind = "enabled"
	}

	if len(matched) == 0 {
		msg := fmt.Sprintf("%d %s firewall rulesets match %s %s %s %d, %d %s rulesets match",
			len(matched), matchedKind,
			rule.Direction, rule.Protocol, rule.PortType, rule.Port,
			len(skipped), skippedKind)

		if len(skipped) != 0 {
			msg += fmt.Sprintf(": %s", strings.Join(skipped.Keys(), ", "))
		}

		return nil, errors.New(msg)
	}

	return matched, nil
}

// Enabled returns a HostFirewallRulesetList with enabled rules
func (l HostFirewallRulesetList) Enabled() HostFirewallRulesetList {
	var matches HostFirewallRulesetList

	for _, rs := range l {
		if rs.Enabled {
			matches = append(matches, rs)
		}
	}

	return matches
}

// Disabled returns a HostFirewallRulesetList with disabled rules
func (l HostFirewallRulesetList) Disabled() HostFirewallRulesetList {
	var matches HostFirewallRulesetList

	for _, rs := range l {
		if !rs.Enabled {
			matches = append(matches, rs)
		}
	}

	return matches
}

// Keys returns the HostFirewallRuleset.Key for each ruleset in the list
func (l HostFirewallRulesetList) Keys() []string {
	var keys []string

	for _, rs := range l {
		keys = append(keys, rs.Key)
	}

	return keys
}
