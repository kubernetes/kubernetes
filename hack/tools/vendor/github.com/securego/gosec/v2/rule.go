// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gosec

import (
	"go/ast"
	"reflect"
)

// The Rule interface used by all rules supported by gosec.
type Rule interface {
	ID() string
	Match(ast.Node, *Context) (*Issue, error)
}

// RuleBuilder is used to register a rule definition with the analyzer
type RuleBuilder func(id string, c Config) (Rule, []ast.Node)

// A RuleSet contains a mapping of lists of rules to the type of AST node they
// should be run on and a mapping of rule ID's to whether the rule are
// suppressed.
// The analyzer will only invoke rules contained in the list associated with the
// type of AST node it is currently visiting.
type RuleSet struct {
	Rules             map[reflect.Type][]Rule
	RuleSuppressedMap map[string]bool
}

// NewRuleSet constructs a new RuleSet
func NewRuleSet() RuleSet {
	return RuleSet{make(map[reflect.Type][]Rule), make(map[string]bool)}
}

// Register adds a trigger for the supplied rule for the the
// specified ast nodes.
func (r RuleSet) Register(rule Rule, isSuppressed bool, nodes ...ast.Node) {
	for _, n := range nodes {
		t := reflect.TypeOf(n)
		if rules, ok := r.Rules[t]; ok {
			r.Rules[t] = append(rules, rule)
		} else {
			r.Rules[t] = []Rule{rule}
		}
	}
	r.RuleSuppressedMap[rule.ID()] = isSuppressed
}

// RegisteredFor will return all rules that are registered for a
// specified ast node.
func (r RuleSet) RegisteredFor(n ast.Node) []Rule {
	if rules, found := r.Rules[reflect.TypeOf(n)]; found {
		return rules
	}
	return []Rule{}
}

// IsRuleSuppressed will return whether the rule is suppressed.
func (r RuleSet) IsRuleSuppressed(ruleID string) bool {
	return r.RuleSuppressedMap[ruleID]
}
