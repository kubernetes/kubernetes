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

package generate

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// GeneratorFunc returns the generators for the provided command
type GeneratorFunc func(cmdName string) map[string]Generator

// GeneratorParam is a parameter for a generator
// TODO: facilitate structured json generator input schemes
type GeneratorParam struct {
	Name     string
	Required bool
}

// Generator is an interface for things that can generate API objects from input
// parameters. One example is the "expose" generator that is capable of exposing
// new replication controllers and services, among other things.
type Generator interface {
	// Generate creates an API object given a set of parameters
	Generate(params map[string]interface{}) (runtime.Object, error)
	// ParamNames returns the list of parameters that this generator uses
	ParamNames() []GeneratorParam
}

// StructuredGenerator is an interface for things that can generate API objects not using parameter injection
type StructuredGenerator interface {
	// StructuredGenerator creates an API object using pre-configured parameters
	StructuredGenerate() (runtime.Object, error)
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
	return utilerrors.NewAggregate(allErrs)
}

// AnnotateFlags annotates all flags that are used by generators.
func AnnotateFlags(cmd *cobra.Command, generators map[string]Generator) {
	// Iterate over all generators and mark any flags used by them.
	for name, generator := range generators {
		generatorParams := map[string]struct{}{}
		for _, param := range generator.ParamNames() {
			generatorParams[param.Name] = struct{}{}
		}

		cmd.Flags().VisitAll(func(flag *pflag.Flag) {
			if _, found := generatorParams[flag.Name]; !found {
				// This flag is not used by the current generator
				// so skip it.
				return
			}
			if flag.Annotations == nil {
				flag.Annotations = map[string][]string{}
			}
			if annotations := flag.Annotations["generator"]; annotations == nil {
				flag.Annotations["generator"] = []string{}
			}
			flag.Annotations["generator"] = append(flag.Annotations["generator"], name)
		})
	}
}

// EnsureFlagsValid ensures that no invalid flags are being used against a
func EnsureFlagsValid(cmd *cobra.Command, generators map[string]Generator, generatorInUse string) error {
	AnnotateFlags(cmd, generators)

	allErrs := []error{}
	cmd.Flags().VisitAll(func(flag *pflag.Flag) {
		// If the flag hasn't changed, don't validate it.
		if !flag.Changed {
			return
		}
		// Look into the flag annotations for the generators that can use it.
		if annotations := flag.Annotations["generator"]; len(annotations) > 0 {
			annotationMap := map[string]struct{}{}
			for _, ann := range annotations {
				annotationMap[ann] = struct{}{}
			}
			// If the current generator is not annotated, then this flag shouldn't
			// be used with it.
			if _, found := annotationMap[generatorInUse]; !found {
				allErrs = append(allErrs, fmt.Errorf("cannot use --%s with --generator=%s", flag.Name, generatorInUse))
			}
		}
	})
	return utilerrors.NewAggregate(allErrs)
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

func MakeProtocols(protocols map[string]string) string {
	out := []string{}
	for key, value := range protocols {
		out = append(out, fmt.Sprintf("%s/%s", key, value))
	}
	return strings.Join(out, ",")
}

func ParseProtocols(protocols interface{}) (map[string]string, error) {
	protocolsString, isString := protocols.(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %v", protocols)
	}
	if len(protocolsString) == 0 {
		return nil, fmt.Errorf("no protocols passed")
	}
	portProtocolMap := map[string]string{}
	protocolsSlice := strings.Split(protocolsString, ",")
	for ix := range protocolsSlice {
		portProtocol := strings.Split(protocolsSlice[ix], "/")
		if len(portProtocol) != 2 {
			return nil, fmt.Errorf("unexpected port protocol mapping: %s", protocolsSlice[ix])
		}
		if len(portProtocol[0]) == 0 {
			return nil, fmt.Errorf("unexpected empty port")
		}
		if len(portProtocol[1]) == 0 {
			return nil, fmt.Errorf("unexpected empty protocol")
		}
		portProtocolMap[portProtocol[0]] = portProtocol[1]
	}
	return portProtocolMap, nil
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
		if len(labelSpec[0]) == 0 {
			return nil, fmt.Errorf("unexpected empty label key")
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
