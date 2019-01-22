/*
Copyright 2017 The Kubernetes Authors.

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

package rules

import (
	"testing"

	adreg "k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

type ruleTest struct {
	rule    adreg.RuleWithOperations
	match   []admission.Attributes
	noMatch []admission.Attributes
}
type tests map[string]ruleTest

func a(group, version, resource, subresource, name string, operation admission.Operation) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, nil,
		schema.GroupVersionKind{Group: group, Version: version, Kind: "k" + resource},
		"ns", name,
		schema.GroupVersionResource{Group: group, Version: version, Resource: resource}, subresource,
		operation,
		false,
		nil,
	)
}

func attrList(a ...admission.Attributes) []admission.Attributes {
	return a
}

func TestGroup(t *testing.T) {
	table := tests{
		"wildcard": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIGroups: []string{"*"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
			),
		},
		"exact": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIGroups: []string{"g1", "g2"},
				},
			},
			match: attrList(
				a("g1", "v", "r", "", "name", admission.Create),
				a("g2", "v2", "r3", "", "name", admission.Create),
			),
			noMatch: attrList(
				a("g3", "v", "r", "", "name", admission.Create),
				a("g4", "v", "r", "", "name", admission.Create),
			),
		},
	}

	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.rule, m}
			if !r.group() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.rule, m}
			if r.group() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestVersion(t *testing.T) {
	table := tests{
		"wildcard": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIVersions: []string{"*"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
			),
		},
		"exact": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIVersions: []string{"v1", "v2"},
				},
			},
			match: attrList(
				a("g1", "v1", "r", "", "name", admission.Create),
				a("g2", "v2", "r", "", "name", admission.Create),
			),
			noMatch: attrList(
				a("g1", "v3", "r", "", "name", admission.Create),
				a("g2", "v4", "r", "", "name", admission.Create),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.rule, m}
			if !r.version() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.rule, m}
			if r.version() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestOperation(t *testing.T) {
	table := tests{
		"wildcard": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.OperationAll}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "", "name", admission.Update),
				a("g", "v", "r", "", "name", admission.Delete),
				a("g", "v", "r", "", "name", admission.Connect),
			),
		},
		"create": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Create}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Update),
				a("g", "v", "r", "", "name", admission.Delete),
				a("g", "v", "r", "", "name", admission.Connect),
			),
		},
		"update": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Update}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Update),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "", "name", admission.Delete),
				a("g", "v", "r", "", "name", admission.Connect),
			),
		},
		"delete": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Delete}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Delete),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "", "name", admission.Update),
				a("g", "v", "r", "", "name", admission.Connect),
			),
		},
		"connect": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Connect}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Connect),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "", "name", admission.Update),
				a("g", "v", "r", "", "name", admission.Delete),
			),
		},
		"multiple": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Update, adreg.Delete}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Update),
				a("g", "v", "r", "", "name", admission.Delete),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "", "name", admission.Connect),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.rule, m}
			if !r.operation() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.rule, m}
			if r.operation() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestResource(t *testing.T) {
	table := tests{
		"no subresources": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("2", "v", "r2", "", "name", admission.Create),
			),
			noMatch: attrList(
				a("g", "v", "r", "exec", "name", admission.Create),
				a("2", "v", "r2", "proxy", "name", admission.Create),
			),
		},
		"r & subresources": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"r/*"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "exec", "name", admission.Create),
			),
			noMatch: attrList(
				a("2", "v", "r2", "", "name", admission.Create),
				a("2", "v", "r2", "proxy", "name", admission.Create),
			),
		},
		"r & subresources or r2": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"r/*", "r2"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("g", "v", "r", "exec", "name", admission.Create),
				a("2", "v", "r2", "", "name", admission.Create),
			),
			noMatch: attrList(
				a("2", "v", "r2", "proxy", "name", admission.Create),
			),
		},
		"proxy or exec": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*/proxy", "*/exec"},
				},
			},
			match: attrList(
				a("g", "v", "r", "exec", "name", admission.Create),
				a("2", "v", "r2", "proxy", "name", admission.Create),
				a("2", "v", "r3", "proxy", "name", admission.Create),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create),
				a("2", "v", "r2", "", "name", admission.Create),
				a("2", "v", "r4", "scale", "name", admission.Create),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.rule, m}
			if !r.resource() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.rule, m}
			if r.resource() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}
