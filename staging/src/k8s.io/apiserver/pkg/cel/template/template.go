/*
Copyright 2024 The Kubernetes Authors.

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

package template

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/types"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
)

const (
	// ObjectName is the variable name for the input data in a CEL expression.
	ObjectName = "object"
)

var (
	placeholderRe = regexp.MustCompile(`{{.*?}}`)
	stringType    = reflect.TypeOf("")
)

// NewTemplate parses the template string and all CEL expressions embedded in
// it. CEL expressions are enclosed with {{<expression>}}. That placeholder
// gets replaced by the value of that expression. By default, the value must be
// a string or support conversion to a string. A conversion function can
// be configured which gets the Go `any` value that corresponds to the CEL
// type and then returns a string or error, see [ToStringConversion].
//
// If there are parse errors, then the returned error contains one error for
// each expression that failed to parse. All errors from the underlying CEL
// libraries are wrapped.
//
// Parsing is more permissive when using [environment.StoredExpression] as
// environment type. Use [environment.NewExpression] when dealing with
// templates before persisting them to storage. See [EnvironmentType].
func NewTemplate(template string, options ...Option) (*Template, error) {
	// TODO: move to init() to save repeated costs. Errors become panics?!
	ver := environment.DefaultCompatibilityVersion()
	envSet := environment.MustBaseEnvSet(ver)

	objectOpt := cel.Variable(ObjectName, types.MapType)
	versionedObjectOpt := environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 29), // TODO: not correct (?), but necessary to use it with environment.DefaultCompatibilityVersion() = 1.29.
		EnvOptions:        []cel.EnvOption{objectOpt},
	}
	envSet, err := envSet.Extend(versionedObjectOpt)
	if err != nil {
		return nil, fmt.Errorf("extending CEL environment with object variable: %w", err)
	}

	c := config{
		envType: environment.NewExpressions,
	}
	for _, opt := range options {
		if err := opt(&c); err != nil {
			return nil, err
		}
	}
	env, err := envSet.Env(c.envType)
	if err != nil {
		return nil, fmt.Errorf("create CEL environment: %w", err)
	}

	t := &Template{
		config:   c,
		env:      env,
		template: template,
	}

	var errs []error
	indices := placeholderRe.FindAllStringIndex(template, -1)
	t.expressions = make([]expression, len(indices))
	for i, offsets := range indices {
		t.expressions[i].start = offsets[0]
		t.expressions[i].end = offsets[1]

		// TODO: provide a custom common.Location which handles mapping offset back to the full template.
		ast, issues := env.CompileSource(common.NewStringSource(template[offsets[0]+2:offsets[1]-2], fmt.Sprintf("placeholder at #%d", offsets[0])))
		if issues != nil && issues.Err() != nil {
			errs = append(errs, fmt.Errorf("CEL type-check error: %w", issues.Err()))
			continue
		}
		program, err := env.Program(ast)
		if err != nil {
			errs = append(errs, fmt.Errorf("CEL program construction error: %w", err))
			continue
		}
		t.expressions[i].program = program
	}
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	return t, nil
}

type ToStringConversionFunction func(in any) (string, error)

func ToJSON(in any) (string, error) {
	out, err := json.Marshal(in)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func ToStringConversion(conv ToStringConversionFunction) Option {
	return func(c *config) error {
		c.conv = conv
		return nil
	}
}

func EnvironmentType(envType environment.Type) Option {
	return func(c *config) error {
		c.envType = envType
		return nil
	}
}

type Option func(c *config) error

type config struct {
	envType environment.Type
	conv    ToStringConversionFunction
}

type Template struct {
	config      config
	env         *cel.Env
	template    string
	expressions []expression
}

type expression struct {
	// Start and  end character (exclusive) of the {{...}} in the template string.
	start, end int

	program cel.Program
}

func (t *Template) Expand(ctx context.Context, object map[string]any) (string, error) {
	if len(t.expressions) == 0 {
		return t.template, nil
	}

	replacements := make([]string, 0, len(t.expressions))
	lenReplaced := 0
	lenReplacements := 0

	value := types.NewStringInterfaceMap(t.env.CELTypeAdapter(), object)

	var errs []error
	for _, expr := range t.expressions {
		lenReplaced += expr.end - expr.start
		replacementVal, _, err := expr.program.ContextEval(ctx,
			map[string]any{
				ObjectName: value,
			},
		)
		if err != nil {
			errs = append(errs, fmt.Errorf("CEL evaluation error: %w", err))
			continue
		}
		var replacementStr string
		if t.config.conv == nil {
			replacementAny, err := replacementVal.ConvertToNative(stringType)
			if err != nil {
				errs = append(errs, fmt.Errorf("CEL result of type %s for placeholder at #%d could not be converted to string: %w", replacementVal.Type().TypeName(), expr.start, err))
				continue
			}
			r, ok := replacementAny.(string)
			if !ok {
				errs = append(errs, fmt.Errorf("CEL native result value should have been a string, got instead: %T", replacementAny))
				continue
			}
			replacementStr = r
		} else {
			replacementAny := replacementVal.Value()
			r, err := t.config.conv(replacementAny)
			if err != nil {
				errs = append(errs, fmt.Errorf("CEL result value of type %T could not be converted to string: %w", replacementAny, err))
				continue
			}
			replacementStr = r
		}
		lenReplacements += len(replacementStr)
		replacements = append(replacements, replacementStr)
	}
	if len(errs) > 0 {
		return "", errors.Join(errs...)
	}

	var result strings.Builder
	result.Grow(len(t.template) + lenReplacements - lenReplaced)
	start := 0
	for i, expr := range t.expressions {
		result.WriteString(t.template[start:expr.start])
		result.WriteString(replacements[i])
		start = expr.end
	}
	result.WriteString(t.template[start:])

	return result.String(), nil
}
