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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"sort"

	"k8s.io/kube-openapi/pkg/generators/rules"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/types"
	"k8s.io/klog/v2"
)

const apiViolationFileType = "api-violation"

type apiViolationFile struct {
	// Since our file actually is unrelated to the package structure, use a
	// path that hasn't been mangled by the framework.
	unmangledPath string
}

func (a apiViolationFile) AssembleFile(f *generator.File, path string) error {
	path = a.unmangledPath
	klog.V(2).Infof("Assembling file %q", path)
	if path == "-" {
		_, err := io.Copy(os.Stdout, &f.Body)
		return err
	}

	output, err := os.Create(path)
	if err != nil {
		return err
	}
	defer output.Close()
	_, err = io.Copy(output, &f.Body)
	return err
}

func (a apiViolationFile) VerifyFile(f *generator.File, path string) error {
	if path == "-" {
		// Nothing to verify against.
		return nil
	}
	path = a.unmangledPath

	formatted := f.Body.Bytes()
	existing, err := ioutil.ReadFile(path)
	if err != nil {
		return fmt.Errorf("unable to read file %q for comparison: %v", path, err)
	}
	if bytes.Compare(formatted, existing) == 0 {
		return nil
	}

	// Be nice and find the first place where they differ
	// (Copied from gengo's default file type)
	i := 0
	for i < len(formatted) && i < len(existing) && formatted[i] == existing[i] {
		i++
	}
	eDiff, fDiff := existing[i:], formatted[i:]
	if len(eDiff) > 100 {
		eDiff = eDiff[:100]
	}
	if len(fDiff) > 100 {
		fDiff = fDiff[:100]
	}
	return fmt.Errorf("output for %q differs; first existing/expected diff: \n  %q\n  %q", path, string(eDiff), string(fDiff))
}

func newAPIViolationGen() *apiViolationGen {
	return &apiViolationGen{
		linter: newAPILinter(),
	}
}

type apiViolationGen struct {
	generator.DefaultGen

	linter *apiLinter
}

func (v *apiViolationGen) FileType() string { return apiViolationFileType }
func (v *apiViolationGen) Filename() string {
	return "this file is ignored by the file assembler"
}

func (v *apiViolationGen) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	klog.V(5).Infof("validating API rules for type %v", t)
	if err := v.linter.validate(t); err != nil {
		return err
	}
	return nil
}

// Finalize prints the API rule violations to report file (if specified from
// arguments) or stdout (default)
func (v *apiViolationGen) Finalize(c *generator.Context, w io.Writer) error {
	// NOTE: we don't return error here because we assume that the report file will
	// get evaluated afterwards to determine if error should be raised. For example,
	// you can have make rules that compare the report file with existing known
	// violations (whitelist) and determine no error if no change is detected.
	v.linter.report(w)
	return nil
}

// apiLinter is the framework hosting multiple API rules and recording API rule
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
			&rules.OmitEmptyMatchCase{},
			&rules.ListTypeMissing{},
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

// apiViolations implements sort.Interface for []apiViolation based on the fields: rule,
// packageName, typeName and field.
type apiViolations []apiViolation

func (a apiViolations) Len() int      { return len(a) }
func (a apiViolations) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a apiViolations) Less(i, j int) bool {
	if a[i].rule != a[j].rule {
		return a[i].rule < a[j].rule
	}
	if a[i].packageName != a[j].packageName {
		return a[i].packageName < a[j].packageName
	}
	if a[i].typeName != a[j].typeName {
		return a[i].typeName < a[j].typeName
	}
	return a[i].field < a[j].field
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
		klog.V(5).Infof("validating API rule %v for type %v", r.Name(), t)
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
	sort.Sort(apiViolations(l.violations))
	for _, v := range l.violations {
		fmt.Fprintf(w, "API rule violation: %s,%s,%s,%s\n", v.rule, v.packageName, v.typeName, v.field)
	}
	if len(l.violations) > 0 {
		return fmt.Errorf("API rule violations exist")
	}
	return nil
}
