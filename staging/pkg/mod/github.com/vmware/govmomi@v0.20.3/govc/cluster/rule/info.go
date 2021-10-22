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

package rule

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
)

type info struct {
	*InfoFlag
}

func init() {
	cli.Register("cluster.rule.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *info) Description() string {
	return `Provides detailed infos about cluster rules, their types and rule members.

Examples:
  govc cluster.rule.info -cluster my_cluster
  govc cluster.rule.info -cluster my_cluster -name my_rule`
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	var res ruleResult

	rules, err := cmd.Rules(ctx)
	if err != nil {
		return err
	}

	for _, rule := range rules {
		ruleName := rule.GetClusterRuleInfo().Name
		ruleInfo := GetExtendedClusterRuleInfo(rule)
		if cmd.name == "" || cmd.name == ruleName {
			res = append(res, fmt.Sprintf("Rule: %s", ruleName))
			res = append(res, fmt.Sprintf("  Type: %s", ruleInfo.ruleType))
			switch ruleInfo.ruleType {
			case "ClusterAffinityRuleSpec", "ClusterAntiAffinityRuleSpec":
				names, err := cmd.Names(ctx, *ruleInfo.refs)
				if err != nil {
					cmd.WriteResult(res)
					return err
				}

				for _, ref := range *ruleInfo.refs {
					res = append(res, fmt.Sprintf("  VM: %s", names[ref]))
				}
			case "ClusterVmHostRuleInfo":
				res = append(res, fmt.Sprintf("  vmGroupName: %s", ruleInfo.vmGroupName))
				res = append(res, fmt.Sprintf("  affineHostGroupName %s", ruleInfo.affineHostGroupName))
				res = append(res, fmt.Sprintf("  antiAffineHostGroupName %s", ruleInfo.antiAffineHostGroupName))
			case "ClusterDependencyRuleInfo":
				res = append(res, fmt.Sprintf("  VmGroup %s", ruleInfo.VmGroup))
				res = append(res, fmt.Sprintf("  DependsOnVmGroup %s", ruleInfo.DependsOnVmGroup))
			default:
				res = append(res, "unknown rule type, no further rule details known")
			}
		}

	}

	return cmd.WriteResult(res)
}
