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
	"io"

	"github.com/vmware/govmomi/govc/cli"
)

type ls struct {
	*InfoFlag
}

func init() {
	cli.Register("cluster.rule.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)
}

func (cmd *ls) Process(ctx context.Context) error {
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *ls) Description() string {
	return `List cluster rules and rule members.

Examples:
  govc cluster.rule.ls -cluster my_cluster
  govc cluster.rule.ls -cluster my_cluster -name my_rule
  govc cluster.rule.ls -cluster my_cluster -l
  govc cluster.rule.ls -cluster my_cluster -name my_rule -l`
}

type ruleResult []string

func (r ruleResult) Write(w io.Writer) error {
	for i := range r {
		fmt.Fprintln(w, r[i])
	}

	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	var res ruleResult

	if cmd.name == "" {
		rules, err := cmd.Rules(ctx)
		if err != nil {
			return err
		}

		for _, g := range rules {
			ruleName := g.GetClusterRuleInfo().Name
			if cmd.Long {
				ruleTypeInfo := GetExtendedClusterRuleInfo(g).ruleType
				res = append(res, fmt.Sprintf("%s (%s)", ruleName, ruleTypeInfo))
			} else {
				res = append(res, fmt.Sprintf("%s", ruleName))
			}
		}
	} else {
		rule, err := cmd.Rule(ctx)
		if err != nil {
			return err
		}

		//res = append(res, rule.ruleType+":")
		switch rule.ruleType {
		case "ClusterAffinityRuleSpec", "ClusterAntiAffinityRuleSpec":
			names, err := cmd.Names(ctx, *rule.refs)
			if err != nil {
				cmd.WriteResult(res)
				return err
			}

			for _, ref := range *rule.refs {
				res = extendedAppend(res, cmd.Long, names[ref], "VM")
			}
		case "ClusterVmHostRuleInfo":
			res = extendedAppend(res, cmd.Long, rule.vmGroupName, "vmGroupName")
			res = extendedAppend(res, cmd.Long, rule.affineHostGroupName, "affineHostGroupName")
			res = extendedAppend(res, cmd.Long, rule.antiAffineHostGroupName, "antiAffineHostGroupName")
		case "ClusterDependencyRuleInfo":
			res = extendedAppend(res, cmd.Long, rule.VmGroup, "VmGroup")
			res = extendedAppend(res, cmd.Long, rule.DependsOnVmGroup, "DependsOnVmGroup")
		default:
			res = append(res, "unknown rule type, no further rule details known")
		}

	}

	return cmd.WriteResult(res)
}

func extendedAppend(res ruleResult, Long bool, entryValue string, entryType string) ruleResult {
	var newres ruleResult
	if Long {
		newres = append(res, fmt.Sprintf("%s (%s)", entryValue, entryType))
	} else {
		newres = append(res, entryValue)
	}
	return newres
}
