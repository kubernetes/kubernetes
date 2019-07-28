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
	"fmt"
	"testing"

	adreg "k8s.io/api/admissionregistration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
)

type ruleTest struct {
	rule    adreg.RuleWithOperations
	match   []admission.Attributes
	noMatch []admission.Attributes
}
type tests map[string]ruleTest

func a(group, version, resource, subresource, name string, operation admission.Operation, operationOptions runtime.Object) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, nil,
		schema.GroupVersionKind{Group: group, Version: version, Kind: "k" + resource},
		"ns", name,
		schema.GroupVersionResource{Group: group, Version: version, Resource: resource}, subresource,
		operation,
		operationOptions,
		false,
		nil,
	)
}

func namespacedAttributes(group, version, resource, subresource, name string, operation admission.Operation, operationOptions runtime.Object) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, nil,
		schema.GroupVersionKind{Group: group, Version: version, Kind: "k" + resource},
		"ns", name,
		schema.GroupVersionResource{Group: group, Version: version, Resource: resource}, subresource,
		operation,
		operationOptions,
		false,
		nil,
	)
}

func clusterScopedAttributes(group, version, resource, subresource, name string, operation admission.Operation, operationOptions runtime.Object) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, nil,
		schema.GroupVersionKind{Group: group, Version: version, Kind: "k" + resource},
		"", name,
		schema.GroupVersionResource{Group: group, Version: version, Resource: resource}, subresource,
		operation,
		operationOptions,
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
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"exact": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIGroups: []string{"g1", "g2"},
				},
			},
			match: attrList(
				a("g1", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g2", "v2", "r3", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("g3", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g4", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
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
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"exact": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					APIVersions: []string{"v1", "v2"},
				},
			},
			match: attrList(
				a("g1", "v1", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g2", "v2", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("g1", "v3", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g2", "v4", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
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
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
				a("g", "v", "r", "", "name", admission.Connect, nil),
			),
		},
		"create": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Create}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
				a("g", "v", "r", "", "name", admission.Connect, nil),
			),
		},
		"update": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Update}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
				a("g", "v", "r", "", "name", admission.Connect, nil),
			),
		},
		"delete": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Delete}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
				a("g", "v", "r", "", "name", admission.Connect, nil),
			),
		},
		"connect": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Connect}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Connect, nil),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
			),
		},
		"multiple": {
			rule: adreg.RuleWithOperations{Operations: []adreg.OperationType{adreg.Update, adreg.Delete}},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Update, &metav1.UpdateOptions{}),
				a("g", "v", "r", "", "name", admission.Delete, &metav1.DeleteOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "", "name", admission.Connect, nil),
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
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "proxy", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"r & subresources": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"r/*"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("2", "v", "r2", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "proxy", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"r & subresources or r2": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"r/*", "r2"},
				},
			},
			match: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("2", "v", "r2", "proxy", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"proxy or exec": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*/proxy", "*/exec"},
				},
			},
			match: attrList(
				a("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "proxy", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r3", "proxy", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				a("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r2", "", "name", admission.Create, &metav1.CreateOptions{}),
				a("2", "v", "r4", "scale", "name", admission.Create, &metav1.CreateOptions{}),
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

func TestScope(t *testing.T) {
	cluster := adreg.ClusterScope
	namespace := adreg.NamespacedScope
	allscopes := adreg.AllScopes
	table := tests{
		"cluster scope": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*"},
					Scope:     &cluster,
				},
			},
			match: attrList(
				clusterScopedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				namespacedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"namespace scope": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*"},
					Scope:     &namespace,
				},
			},
			match: attrList(
				namespacedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(
				clusterScopedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
			),
		},
		"all scopes": {
			rule: adreg.RuleWithOperations{
				Rule: adreg.Rule{
					Resources: []string{"*"},
					Scope:     &allscopes,
				},
			},
			match: attrList(
				namespacedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("g", "v", "r", "", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("g", "v", "r", "exec", "name", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				clusterScopedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "", "ns", admission.Create, &metav1.CreateOptions{}),
				namespacedAttributes("", "v1", "namespaces", "finalize", "ns", admission.Create, &metav1.CreateOptions{}),
			),
			noMatch: attrList(),
		},
	}
	keys := sets.NewString()
	for name := range table {
		keys.Insert(name)
	}
	for _, name := range keys.List() {
		tt := table[name]
		for i, m := range tt.match {
			t.Run(fmt.Sprintf("%s_match_%d", name, i), func(t *testing.T) {
				r := Matcher{tt.rule, m}
				if !r.scope() {
					t.Errorf("%v: expected match %#v", name, m)
				}
			})
		}
		for i, m := range tt.noMatch {
			t.Run(fmt.Sprintf("%s_nomatch_%d", name, i), func(t *testing.T) {
				r := Matcher{tt.rule, m}
				if r.scope() {
					t.Errorf("%v: expected no match %#v", name, m)
				}
			})
		}
	}
}
