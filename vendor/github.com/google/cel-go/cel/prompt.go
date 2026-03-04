// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cel

import (
	_ "embed"
	"sort"
	"strings"
	"text/template"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
)

//go:embed templates/authoring.tmpl
var authoringPrompt string

// AuthoringPrompt creates a prompt template from a CEL environment for the purpose of AI-assisted authoring.
func AuthoringPrompt(env *Env) (*Prompt, error) {
	funcMap := template.FuncMap{
		"split": func(str string) []string { return strings.Split(str, "\n") },
	}
	tmpl := template.New("cel").Funcs(funcMap)
	tmpl, err := tmpl.Parse(authoringPrompt)
	if err != nil {
		return nil, err
	}
	return &Prompt{
		Persona:      defaultPersona,
		FormatRules:  defaultFormatRules,
		GeneralUsage: defaultGeneralUsage,
		tmpl:         tmpl,
		env:          env,
	}, nil
}

// Prompt represents the core components of an LLM prompt based on a CEL environment.
//
// All fields of the prompt may be overwritten / modified with support for rendering the
// prompt to a human-readable string.
type Prompt struct {
	// Persona indicates something about the kind of user making the request
	Persona string

	// FormatRules indicate how the LLM should generate its output
	FormatRules string

	// GeneralUsage specifies additional context on how CEL should be used.
	GeneralUsage string

	// tmpl is the text template base-configuration for rendering text.
	tmpl *template.Template

	// env reference used to collect variables, functions, and macros available to the prompt.
	env *Env
}

type promptInst struct {
	*Prompt

	Variables  []*common.Doc
	Macros     []*common.Doc
	Functions  []*common.Doc
	UserPrompt string
}

// Render renders the user prompt with the associated context from the prompt template
// for use with LLM generators.
func (p *Prompt) Render(userPrompt string) string {
	var buffer strings.Builder
	vars := make([]*common.Doc, len(p.env.Variables()))
	for i, v := range p.env.Variables() {
		vars[i] = v.Documentation()
	}
	sort.SliceStable(vars, func(i, j int) bool {
		return vars[i].Name < vars[j].Name
	})
	macs := make([]*common.Doc, len(p.env.Macros()))
	for i, m := range p.env.Macros() {
		macs[i] = m.(common.Documentor).Documentation()
	}
	funcs := make([]*common.Doc, 0, len(p.env.Functions()))
	for _, f := range p.env.Functions() {
		if _, hidden := hiddenFunctions[f.Name()]; hidden {
			continue
		}
		funcs = append(funcs, f.Documentation())
	}
	sort.SliceStable(funcs, func(i, j int) bool {
		return funcs[i].Name < funcs[j].Name
	})
	inst := &promptInst{
		Prompt:     p,
		Variables:  vars,
		Macros:     macs,
		Functions:  funcs,
		UserPrompt: userPrompt}
	p.tmpl.Execute(&buffer, inst)
	return buffer.String()
}

const (
	defaultPersona = `You are a software engineer with expertise in networking and application security
authoring boolean Common Expression Language (CEL) expressions to ensure firewall,
networking, authentication, and data access is only permitted when all conditions
are satisfied.`

	defaultFormatRules = `Output your response as a CEL expression.

Write the expression with the comment on the first line and the expression on the
subsequent lines. Format the expression using 80-character line limits commonly
found in C++ or Java code.`

	defaultGeneralUsage = `CEL supports Protocol Buffer and JSON types, as well as simple types and aggregate types.

Simple types include bool, bytes, double, int, string, and uint:

* double literals must always include a decimal point: 1.0, 3.5, -2.2
* uint literals must be positive values suffixed with a 'u': 42u
* byte literals are strings prefixed with a 'b': b'1235'
* string literals can use either single quotes or double quotes: 'hello', "world"
* string literals can also be treated as raw strings that do not require any
  escaping within the string by using the 'R' prefix: R"""quote: "hi" """

Aggregate types include list and map:

* list literals consist of zero or more values between brackets: "['a', 'b', 'c']"
* map literal consist of colon-separated key-value pairs within braces: "{'key1': 1, 'key2': 2}"
* Only int, uint, string, and bool types are valid map keys.
* Maps containing HTTP headers must always use lower-cased string keys.

Comments start with two-forward slashes followed by text and a newline.`
)

var (
	hiddenFunctions = map[string]bool{
		overloads.DeprecatedIn:        true,
		operators.OldIn:               true,
		operators.OldNotStrictlyFalse: true,
		operators.NotStrictlyFalse:    true,
	}
)
