/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/errors"
)

// GeneratorParam is a parameter for a generator
// TODO: facilitate structured json generator input schemes
type GeneratorParam struct {
	Name     string
	Required bool
}

// Generator is an interface for things that can generate API objects from input parameters.
type Generator interface {
	// Generate creates an API object given a set of parameters
	Generate(params map[string]interface{}) (runtime.Object, error)
	// ParamNames returns the list of parameters that this generator uses
	ParamNames() []GeneratorParam
}

func IsZero(i interface{}) bool {
	if i == nil {
		return true
	}
	return reflect.DeepEqual(i, reflect.Zero(reflect.TypeOf(i)).Interface())
}

// ValidateParams ensures that all required params are present in the params map
func ValidateParams(paramSpec []GeneratorParam, params map[string]interface{}) error {
	allErrs := []error{}
	for ix := range paramSpec {
		if paramSpec[ix].Required {
			value, found := params[paramSpec[ix].Name]
			if !found || IsZero(value) {
				allErrs = append(allErrs, fmt.Errorf("Parameter: %s is required", paramSpec[ix].Name))
			}
		}
	}
	return errors.NewAggregate(allErrs)
}

// MakeParams is a utility that creates generator parameters from a command line
func MakeParams(cmd *cobra.Command, params []GeneratorParam) map[string]interface{} {
	result := map[string]interface{}{}
	for ix := range params {
		f := cmd.Flags().Lookup(params[ix].Name)
		if f != nil {
			result[params[ix].Name] = f.Value.String()
		}
	}
	return result
}

func MakeLabels(labels map[string]string) string {
	out := []string{}
	for key, value := range labels {
		out = append(out, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(out, ",")
}

// ParseLabels turns a string representation of a label set into a map[string]string
func ParseLabels(labelSpec interface{}) (map[string]string, error) {
	labelString, isString := labelSpec.(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %v", labelSpec)
	}
	if len(labelString) == 0 {
		return nil, fmt.Errorf("no label spec passed")
	}
	labels := map[string]string{}
	labelSpecs := strings.Split(labelString, ",")
	for ix := range labelSpecs {
		labelSpec := strings.Split(labelSpecs[ix], "=")
		if len(labelSpec) != 2 {
			return nil, fmt.Errorf("unexpected label spec: %s", labelSpecs[ix])
		}
		labels[labelSpec[0]] = labelSpec[1]
	}
	return labels, nil
}

func GetBool(params map[string]string, key string, defValue bool) (bool, error) {
	if val, found := params[key]; !found {
		return defValue, nil
	} else {
		return strconv.ParseBool(val)
	}
}
