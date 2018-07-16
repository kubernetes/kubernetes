/*
Copyright 2018 The Kubernetes Authors.

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

package generators

import (
	"fmt"
	"io"

	"k8s.io/kube-openapi/pkg/generators/rules"

	"github.com/golang/glog"
	"k8s.io/gengo/types"
)

// apiLinter is the framework hosting mutliple API rules and recording API rule
// violations
type apiLinter struct {
	// API rules that implement APIRule interface and output API rule violations
	rules      []APIRule
	violations []apiViolation
}

// newAPILinter creates an apiLinter object with API rules in package rules. Please
// add APIRule here when new API rule is implemented.
func newAPILinter() *apiLinter {
	return &apiLinter{
		rules: []APIRule{
			&rules.NamesMatch{},
		},
	}
}

// apiViolation uniquely identifies single API rule violation
type apiViolation struct {
	// Name of rule from APIRule.Name()
	rule string

	packageName string
	typeName    string

	// Optional: name of field that violates API rule. Empty fieldName implies that
	// the entire type violates the rule.
	field string
}

// APIRule is the interface for validating API rule on Go types
type APIRule interface {
	// Validate evaluates API rule on type t and returns a list of field names in
	// the type that violate the rule. Empty field name [""] implies the entire
	// type violates the rule.
	Validate(t *types.Type) ([]string, error)

	// Name returns the name of APIRule
	Name() string
}

// validate runs all API rules on type t and records any API rule violation
func (l *apiLinter) validate(t *types.Type) error {
	for _, r := range l.rules {
		glog.V(5).Infof("validating API rule %v for type %v", r.Name(), t)
		fields, err := r.Validate(t)
		if err != nil {
			return err
		}
		for _, field := range fields {
			l.violations = append(l.violations, apiViolation{
				rule:        r.Name(),
				packageName: t.Name.Package,
				typeName:    t.Name.Name,
				field:       field,
			})
		}
	}
	return nil
}

// report prints any API rule violation to writer w and returns error if violation exists
func (l *apiLinter) report(w io.Writer) error {
	for _, v := range l.violations {
		fmt.Fprintf(w, "API rule violation: %s,%s,%s,%s\n", v.rule, v.packageName, v.typeName, v.field)
	}
	if len(l.violations) > 0 {
		return fmt.Errorf("API rule violations exist")
	}
	return nil
}
