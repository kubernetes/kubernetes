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

package exclusionrules

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

type exclusionRuleTest struct {
	exclusionRule webhookadmission.ExclusionRule
	match         []admission.Attributes
	noMatch       []admission.Attributes
}
type tests map[string]exclusionRuleTest

func attrList(a ...admission.Attributes) []admission.Attributes {
	return a
}

func a(group, version, kind, namespace, name string) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, nil,
		schema.GroupVersionKind{Group: group, Version: version, Kind: kind},
		namespace, name,
		schema.GroupVersionResource{Group: group, Version: version, Resource: ""}, "",
		"",
		nil,
		false,
		nil,
	)
}

func TestGroup(t *testing.T) {
	table := tests{
		"wildcard": {
			exclusionRule: webhookadmission.ExclusionRule{
				APIGroups: []string{"*"},
			},
			match: attrList(
				a("g", "v", "r", "", "name"),
			),
		},
		"exact": {
			exclusionRule: webhookadmission.ExclusionRule{
				APIGroups: []string{"g1", "g2"},
			},
			match: attrList(
				a("g1", "v", "r", "", "name"),
				a("g2", "v2", "r3", "", "name"),
			),
			noMatch: attrList(
				a("g3", "v", "r", "", "name"),
				a("g4", "v", "r", "", "name"),
			),
		},
	}

	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.exclusionRule, m}
			if !r.group() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.exclusionRule, m}
			if r.group() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestVersion(t *testing.T) {
	table := tests{
		"wildcard": {
			exclusionRule: webhookadmission.ExclusionRule{
				APIVersions: []string{"*"},
			},
			match: attrList(
				a("g", "v", "r", "", "name"),
			),
		},
		"exact": {
			exclusionRule: webhookadmission.ExclusionRule{
				APIVersions: []string{"v1", "v2"},
			},
			match: attrList(
				a("g1", "v1", "r", "", "name"),
				a("g2", "v2", "r", "", "name"),
			),
			noMatch: attrList(
				a("g1", "v3", "r", "", "name"),
				a("g2", "v4", "r", "", "name"),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.exclusionRule, m}
			if !r.version() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.exclusionRule, m}
			if r.version() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestKind(t *testing.T) {
	table := tests{
		"exact": {
			exclusionRule: webhookadmission.ExclusionRule{
				Kind: "Lease",
			},
			match: attrList(
				a("g1", "v1", "Lease", "", "name"),
				a("g2", "v2", "Lease", "", "name"),
			),
			noMatch: attrList(
				a("g1", "v3", "Deployment", "", "name"),
				a("g2", "v4", "Pod", "", "name"),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.exclusionRule, m}
			if !r.kind() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.exclusionRule, m}
			if r.kind() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestName(t *testing.T) {
	table := tests{
		"exact": {
			exclusionRule: webhookadmission.ExclusionRule{
				Name: "kube-scheduler",
			},
			match: attrList(
				a("g1", "v1", "Lease", "", "kube-scheduler"),
				a("g2", "v2", "Lease", "", "kube-scheduler"),
			),
			noMatch: attrList(
				a("g1", "v3", "Deployment", "", "something"),
				a("g2", "v4", "Pod", "", "else"),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.exclusionRule, m}
			if !r.name() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.exclusionRule, m}
			if r.name() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}

func TestNamespace(t *testing.T) {
	table := tests{
		"exact": {
			exclusionRule: webhookadmission.ExclusionRule{
				Namespace: "kube-system",
			},
			match: attrList(
				a("g1", "v1", "Lease", "kube-system", ""),
				a("g2", "v2", "Endpoint", "kube-system", ""),
			),
			noMatch: attrList(
				a("g1", "v3", "Deployment", "something", "something"),
				a("g2", "v4", "Pod", "else", "else"),
			),
		},
	}
	for name, tt := range table {
		for _, m := range tt.match {
			r := Matcher{tt.exclusionRule, m}
			if !r.namespace() {
				t.Errorf("%v: expected match %#v", name, m)
			}
		}
		for _, m := range tt.noMatch {
			r := Matcher{tt.exclusionRule, m}
			if r.namespace() {
				t.Errorf("%v: expected no match %#v", name, m)
			}
		}
	}
}
