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

package webhook

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/pkg/api"
)

type ruleTest struct {
	test             string
	rule             Rule
	attrAPIGroup     string
	attrNamespace    string
	attrOperation    admission.Operation
	attrResource     string
	attrResourceName string
	attrSubResource  string
	shouldMatch      bool
}

func TestAPIGroupMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "apiGroups empty match",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "apiGroup match",
			rule: Rule{
				APIGroups: []string{"my-group"},
			},
			attrAPIGroup: "my-group",
			shouldMatch:  true,
		},
		ruleTest{
			test: "apiGroup mismatch",
			rule: Rule{
				APIGroups: []string{"my-group"},
			},
			attrAPIGroup: "your-group",
			shouldMatch:  false,
		},
	}

	runTests(t, "apiGroups", tests)
}

func TestMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "empty rule matches",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "all properties match",
			rule: Rule{
				APIGroups:     []string{"my-group"},
				Namespaces:    []string{"my-ns"},
				Operations:    []admission.Operation{admission.Create},
				ResourceNames: []string{"my-name"},
				Resources:     []string{"pods/status"},
			},
			shouldMatch:      true,
			attrAPIGroup:     "my-group",
			attrNamespace:    "my-ns",
			attrOperation:    admission.Create,
			attrResource:     "pods",
			attrResourceName: "my-name",
			attrSubResource:  "status",
		},
		ruleTest{
			test: "no properties match",
			rule: Rule{
				APIGroups:     []string{"my-group"},
				Namespaces:    []string{"my-ns"},
				Operations:    []admission.Operation{admission.Create},
				ResourceNames: []string{"my-name"},
				Resources:     []string{"pods/status"},
			},
			shouldMatch:      false,
			attrAPIGroup:     "your-group",
			attrNamespace:    "your-ns",
			attrOperation:    admission.Delete,
			attrResource:     "secrets",
			attrResourceName: "your-name",
		},
	}

	runTests(t, "", tests)
}

func TestNamespaceMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "namespaces empty match",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "namespace match",
			rule: Rule{
				Namespaces: []string{"my-ns"},
			},
			attrNamespace: "my-ns",
			shouldMatch:   true,
		},
		ruleTest{
			test: "namespace mismatch",
			rule: Rule{
				Namespaces: []string{"my-ns"},
			},
			attrNamespace: "your-ns",
			shouldMatch:   false,
		},
	}

	runTests(t, "namespaces", tests)
}

func TestOperationMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "operations empty match",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "operation match",
			rule: Rule{
				Operations: []admission.Operation{admission.Create},
			},
			attrOperation: admission.Create,
			shouldMatch:   true,
		},
		ruleTest{
			test: "operation mismatch",
			rule: Rule{
				Operations: []admission.Operation{admission.Create},
			},
			attrOperation: admission.Delete,
			shouldMatch:   false,
		},
	}

	runTests(t, "operations", tests)
}

func TestResourceMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "resources empty match",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "resource match",
			rule: Rule{
				Resources: []string{"pods"},
			},
			attrResource: "pods",
			shouldMatch:  true,
		},
		ruleTest{
			test: "resource mismatch",
			rule: Rule{
				Resources: []string{"pods"},
			},
			attrResource: "secrets",
			shouldMatch:  false,
		},
		ruleTest{
			test: "resource with subresource match",
			rule: Rule{
				Resources: []string{"pods/status"},
			},
			attrResource:    "pods",
			attrSubResource: "status",
			shouldMatch:     true,
		},
		ruleTest{
			test: "resource with subresource mismatch",
			rule: Rule{
				Resources: []string{"pods"},
			},
			attrResource:    "pods",
			attrSubResource: "status",
			shouldMatch:     false,
		},
	}

	runTests(t, "resources", tests)
}

func TestResourceNameMatches(t *testing.T) {
	tests := []ruleTest{
		ruleTest{
			test:        "resourceNames empty match",
			rule:        Rule{},
			shouldMatch: true,
		},
		ruleTest{
			test: "resourceName match",
			rule: Rule{
				ResourceNames: []string{"my-name"},
			},
			attrResourceName: "my-name",
			shouldMatch:      true,
		},
		ruleTest{
			test: "resourceName mismatch",
			rule: Rule{
				ResourceNames: []string{"my-name"},
			},
			attrResourceName: "your-name",
			shouldMatch:      false,
		},
	}

	runTests(t, "resourceNames", tests)
}

func runTests(t *testing.T, prop string, tests []ruleTest) {
	for _, tt := range tests {
		if tt.attrResource == "" {
			tt.attrResource = "pods"
		}

		res := api.Resource(tt.attrResource).WithVersion("version")

		if tt.attrAPIGroup != "" {
			res.Group = tt.attrAPIGroup
		}

		attr := admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), tt.attrNamespace, tt.attrResourceName, res, tt.attrSubResource, tt.attrOperation, nil)
		var attrVal string
		var ruleVal []string
		var matches bool

		switch prop {
		case "":
			matches = Matches(tt.rule, attr)
		case "apiGroups":
			attrVal = tt.attrAPIGroup
			matches = APIGroupMatches(tt.rule, attr)
			ruleVal = tt.rule.APIGroups
		case "namespaces":
			attrVal = tt.attrNamespace
			matches = NamespaceMatches(tt.rule, attr)
			ruleVal = tt.rule.Namespaces
		case "operations":
			attrVal = string(tt.attrOperation)
			matches = OperationMatches(tt.rule, attr)
			ruleVal = make([]string, len(tt.rule.Operations))

			for _, rOp := range tt.rule.Operations {
				ruleVal = append(ruleVal, string(rOp))
			}
		case "resources":
			attrVal = tt.attrResource
			matches = ResourceMatches(tt.rule, attr)
			ruleVal = tt.rule.Resources
		case "resourceNames":
			attrVal = tt.attrResourceName
			matches = ResourceNamesMatches(tt.rule, attr)
			ruleVal = tt.rule.ResourceNames
		default:
			t.Errorf("Unexpected test property: %s", prop)
		}

		if matches && !tt.shouldMatch {
			if prop == "" {
				testError(t, tt.test, "Expected match")
			} else {
				testError(t, tt.test, fmt.Sprintf("Expected %s rule property not to match %s against one of the following: %s", prop, attrVal, strings.Join(ruleVal, ", ")))
			}
		} else if !matches && tt.shouldMatch {
			if prop == "" {
				testError(t, tt.test, "Unexpected match")
			} else {
				testError(t, tt.test, fmt.Sprintf("Expected %s rule property to match %s against one of the following: %s", prop, attrVal, strings.Join(ruleVal, ", ")))
			}
		}
	}
}

func testError(t *testing.T, name, msg string) {
	t.Errorf("test failed (%s): %s", name, msg)
}
