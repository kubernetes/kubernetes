/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"regexp"
	"strings"

	"k8s.io/kubernetes/pkg/runtime"
)

// GenericGenerator is an implementation of Generator that loads its generator from a template file.
// Files are expected to have parameters enclosed with double curly braces, e.g.
//    field1: {{ param1 }}
//    field2:
//      - {{param2}} : {{param3}}
//
//  The template must be structured as either YAML or JSON
type GenericGenerator struct {
	Codec        runtime.Codec
	TemplateData []byte
	Params       []GeneratorParam
}

var paramRE = regexp.MustCompile("\\{\\{[^\\}]+\\}\\}")

// ValidationError is an error that occurs if template
type ValidationError struct {
	LineNumber int
	LineValue  string
}

func (v *ValidationError) Error() string {
	return fmt.Sprintf("mis-matched braces in line %d: %s", v.LineNumber, v.LineValue)
}

func validateTemplate(data []byte) error {
	scanner := bufio.NewScanner(bytes.NewBuffer(data))
	line := 1
	for scanner.Scan() {
		str := scanner.Text()
		// Simplistic validation, do a better job here...
		opens := strings.Count(str, "{{")
		closes := strings.Count(str, "}}")

		if opens != closes {
			return &ValidationError{
				LineNumber: line,
				LineValue:  str,
			}
		}
		line++
	}
	return nil
}

func getParams(data []byte) []GeneratorParam {
	params := []GeneratorParam{}
	matches := paramRE.FindAll(data, -1)
	for ix := range matches {
		params = append(params, GeneratorParam{
			Name:     extractParamName(matches[ix]),
			Required: true,
		})
	}
	return params
}

func NewGenericGeneratorFromFile(filename string, codec runtime.Codec) (Generator, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return NewGenericGeneratorFromBytes(data, codec)
}

func NewGenericGeneratorFromBytes(data []byte, codec runtime.Codec) (Generator, error) {
	if err := validateTemplate(data); err != nil {
		return nil, err
	}
	return &GenericGenerator{
		TemplateData: data,
		Params:       getParams(data),
		Codec:        runtime.YAMLDecoder(codec),
	}, nil
}

func extractParamName(match []byte) string {
	str := string(match)
	return strings.TrimSpace(str[2 : len(str)-2])
}

func regexpReplace(template []byte, params map[string]interface{}) ([]byte, error) {
	missing := []string{}
	data := paramRE.ReplaceAllFunc(template, func(match []byte) []byte {
		name := extractParamName(match)
		val, found := params[name]
		if !found {
			missing = append(missing, name)
			return []byte("!MISSING!")
		}
		return []byte(fmt.Sprintf("%v", val))
	})
	if len(missing) > 0 {
		return nil, fmt.Errorf("missing parameters: %v", missing)
	}
	return data, nil
}

func (g *GenericGenerator) Generate(params map[string]interface{}) (runtime.Object, error) {
	data, err := regexpReplace(g.TemplateData, params)
	if err != nil {
		return nil, err
	}
	return g.Codec.Decode(data)
}

func (g *GenericGenerator) ParamNames() []GeneratorParam {
	return g.Params
}
