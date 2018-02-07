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

// API linter follows the same rules as openapi-gen, and validates API conventions on every type or package
// that requires API definition generation. The following rules are the same from openapi-gen:
//
// - To generate definition for a specific type or package add "+k8s:openapi-gen=true" tag to the type/package comment lines.
// - To exclude a type or a member from a tagged package/type, add "+k8s:openapi-gen=false" tag to the comment lines.
//
// This directory contains the gengo framework for API linter `api-linter.go` and
// other implementations of API convention validators. Each validator should
// implement the interface `linters.APIValidator`, and be passed to
//
//     func newAPILinter(sanitizedName, filename string, apiValidators []APIValidator) generator.Generator
//
// as part of argument `apiValidators`.

package linters

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/kube-openapi/pkg/util"

	"github.com/golang/glog"
)

const (
	// This is the comment tag that carries parameters for OpenAPI generation
	tagName     = "k8s:openapi-gen"
	tagOptional = "optional"

	// Known values for the tag
	tagValueTrue  = "true"
	tagValueFalse = "false"
)

// CustomArgs is used by the gengo framework to pass args specific to this linter.
type CustomArgs struct {
	// WhitelistFilename is the whitelist csv filename.
	WhitelistFilename string
}

func getAPITagValue(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[tagName]
}

func hasAPITagValue(comments []string, value string) bool {
	tagValues := getAPITagValue(comments)
	for _, val := range tagValues {
		if val == value {
			return true
		}
	}
	return false
}

// hasOptionalTag returns true if the member has +optional in its comments or
// omitempty in its json tags.
func hasOptionalTag(m *types.Member) bool {
	hasOptionalCommentTag := types.ExtractCommentTags(
		"+", m.CommentLines)[tagOptional] != nil
	hasOptionalJSONTag := strings.Contains(
		reflect.StructTag(m.Tags).Get("json"), "omitempty")
	return hasOptionalCommentTag || hasOptionalJSONTag
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer("", nil),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "raw"
}

// Packages returns a API linter that validates API conventions and generates a report file.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	return generator.Packages{
		&generator.DefaultPackage{
			PackagePath: "k8s.io/kubernetes/staging/src/k8s.io/code-generator/cmd/api-linter/report",
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				filename := ""
				if customArgs, ok := arguments.CustomArgs.(*CustomArgs); ok {
					filename = customArgs.WhitelistFilename
				}
				apiValidators := []APIValidator{
					goJSONNameMatchAPIConvention{},
				}
				return []generator.Generator{newAPILinter(arguments.OutputFileBaseName, filename, apiValidators)}
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				// There is a conflict between this codegen and codecgen, we should avoid types generated for codecgen
				if strings.HasPrefix(t.Name.Name, "codecSelfer") {
					return false
				}
				pkg := context.Universe.Package(t.Name.Package)
				if hasAPITagValue(pkg.Comments, tagValueTrue) {
					return !hasAPITagValue(t.CommentLines, tagValueFalse)
				}
				return hasAPITagValue(t.CommentLines, tagValueTrue)
			},
		},
	}
}

func (l *apiLinter) Filter(c *generator.Context, t *types.Type) bool {
	// There is a conflict between this codegen and codecgen, we should avoid types generated for codecgen
	return !strings.HasPrefix(t.Name.Name, "codecSelfer")
}

func (l *apiLinter) Imports(c *generator.Context) []string {
	return []string{}
}

// apiLinter validates API spec against API conventions.
type apiLinter struct {
	generator.DefaultGen
	validators        []APIValidator
	violations        []apiViolation
	whitelist         map[apiViolation]bool
	whitelistFilename string
}

// apiViolation uniquely identifies one API violation. All three fields (linterName, typeName and violationID) should not contain ','
// to work properly with whitelist parsing (whitelist entries in the format of: linterName,typeName,violationID).
type apiViolation struct {
	// Validator's name from APIValidator.Name()
	linterName string
	// Type being validated
	typeName string
	// Returned from APIValidator.Validate(t *types.Type). It uniquely identifies one API violation within the validator namespace.
	violationID string
}

func newAPILinter(sanitizedName, filename string, apiValidators []APIValidator) generator.Generator {
	return &apiLinter{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		validators:        apiValidators,
		violations:        []apiViolation{},
		whitelist:         map[apiViolation]bool{},
		whitelistFilename: filename,
	}
}

func (l *apiLinter) Init(c *generator.Context, w io.Writer) error {
	if l.whitelistFilename != "" {
		glog.V(1).Infof("Reading whitelist file: %s", l.whitelistFilename)
		// Read whitelist file
		f, err := os.Open(l.whitelistFilename)
		if err != nil {
			return err
		}
		defer f.Close()

		// Init whitelist
		r := csv.NewReader(bufio.NewReader(f))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				return err
			}
			// Assert whitelist entries in the format of: linterName,typeName,violationID
			if len(record) != 3 {
				return fmt.Errorf("unexpected whitelist entry length, want: 3, got: %d, %v", len(record), record)
			}

			// Add whitelist entry
			l.whitelist[apiViolation{
				linterName:  record[0],
				typeName:    record[1],
				violationID: record[2],
			}] = true

		}
		glog.V(1).Info("Read whitelist file successfully")
	} else {
		glog.V(1).Info("No whitelist file provided")
	}
	return nil
}

func (l *apiLinter) Finalize(c *generator.Context, w io.Writer) error {
	hasViolation := false
	for _, v := range l.violations {
		if !l.whitelist[v] {
			hasViolation = true
			glog.Errorf("Linter API conventions violation: %s,%s,%s", v.linterName, v.typeName, v.violationID)
		}
	}
	if hasViolation {
		return fmt.Errorf("linter API conventions violations exist; see stderr for details")
	}
	return nil
}

// APIValidator validates one API convention on an API type.
type APIValidator interface {
	// Validate(t *types.Type) evaluates API convention on type t and returns a list of
	// API violation identifier. API violation identifier should not contain ','.
	Validate(t *types.Type) ([]string, error)

	// Name() returns the name of APIValidator: linterName. linterName should not contain
	// ','.
	Name() string
}

func (l *apiLinter) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	for _, v := range l.validators {
		err := l.validate(t, v)
		if err != nil {
			return err
		}
	}
	return nil
}

func (l *apiLinter) validate(t *types.Type, v APIValidator) error {
	violationIDs, err := v.Validate(t)
	if err != nil {
		return err
	}

	for _, id := range violationIDs {
		l.violations = append(l.violations, apiViolation{
			linterName:  v.Name(),
			typeName:    util.ToCanonicalName(t.Name.String()),
			violationID: id,
		})
	}
	return nil
}

func (l *apiLinter) FileType() string {
	return "apiLinterReport"
}

func (l *apiLinter) Filename() string {
	return l.OptionalName + ".report"

}

// NewReportFile holds Format and Assemble functions for API linter report
// file type.
func NewReportFile() *generator.DefaultFileType {
	return &generator.DefaultFileType{
		Format:   formatReportFile,
		Assemble: assembleReportFile,
	}
}

func formatReportFile(source []byte) ([]byte, error) {
	return source, nil
}

func assembleReportFile(w io.Writer, f *generator.File) {
	// NOTE: writing/verifying report file won't get called if executeBody
	//       returns error, so we don't use f.Body to write violation
	//       details here.
	fmt.Fprint(w, "See stdout/stderr for API linter results.")
}
