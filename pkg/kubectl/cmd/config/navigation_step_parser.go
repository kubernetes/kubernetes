/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"reflect"
	"strings"

	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/sets"
)

type navigationSteps struct {
	steps            []navigationStep
	currentStepIndex int
}

type navigationStep struct {
	stepValue string
	stepType  reflect.Type
}

func newNavigationSteps(path string) (*navigationSteps, error) {
	steps := []navigationStep{}
	individualParts := strings.Split(path, ".")

	currType := reflect.TypeOf(clientcmdapi.Config{})
	currPartIndex := 0
	for currPartIndex < len(individualParts) {
		switch currType.Kind() {
		case reflect.Map:
			// if we're in a map, we need to locate a name.  That name may contain dots, so we need to know what tokens are legal for the map's value type
			// for example, we could have a set request like: `set clusters.10.10.12.56.insecure-skip-tls-verify true`.  We enter this case with
			// steps representing 10, 10, 12, 56, insecure-skip-tls-verify.  The name is "10.10.12.56", so we want to collect all those parts together and
			// store them as a single step.  In order to do that, we need to determine what set of tokens is a legal step AFTER the name of the map key
			// This set of reflective code pulls the type of the map values, uses that type to look up the set of legal tags.  Those legal tags are used to
			// walk the list of remaining parts until we find a match to a legal tag or the end of the string.  That name is used to burn all the used parts.
			mapValueType := currType.Elem().Elem()
			mapValueOptions, err := getPotentialTypeValues(mapValueType)
			if err != nil {
				return nil, err
			}
			nextPart := findNameStep(individualParts[currPartIndex:], sets.StringKeySet(mapValueOptions))

			steps = append(steps, navigationStep{nextPart, mapValueType})
			currPartIndex += len(strings.Split(nextPart, "."))
			currType = mapValueType

		case reflect.Struct:
			nextPart := individualParts[currPartIndex]

			options, err := getPotentialTypeValues(currType)
			if err != nil {
				return nil, err
			}
			fieldType, exists := options[nextPart]
			if !exists {
				return nil, fmt.Errorf("unable to parse %v after %v at %v", path, steps, currType)
			}

			steps = append(steps, navigationStep{nextPart, fieldType})
			currPartIndex += len(strings.Split(nextPart, "."))
			currType = fieldType
		}
	}

	return &navigationSteps{steps, 0}, nil
}

func (s *navigationSteps) pop() navigationStep {
	if s.moreStepsRemaining() {
		s.currentStepIndex++
		return s.steps[s.currentStepIndex-1]
	}
	return navigationStep{}
}

func (s *navigationSteps) peek() navigationStep {
	if s.moreStepsRemaining() {
		return s.steps[s.currentStepIndex]
	}
	return navigationStep{}
}

func (s *navigationSteps) moreStepsRemaining() bool {
	return len(s.steps) > s.currentStepIndex
}

// findNameStep takes the list of parts and a set of valid tags that can be used after the name.  It then walks the list of parts
// until it find a valid "next" tag or until it reaches the end of the parts and then builds the name back up out of the individual parts
func findNameStep(parts []string, typeOptions sets.String) string {
	if len(parts) == 0 {
		return ""
	}

	numberOfPartsInStep := findKnownValue(parts[1:], typeOptions) + 1
	// if we didn't find a known value, then the entire thing must be a name
	if numberOfPartsInStep == 0 {
		numberOfPartsInStep = len(parts)
	}
	nextParts := parts[0:numberOfPartsInStep]

	return strings.Join(nextParts, ".")
}

// getPotentialTypeValues takes a type and looks up the tags used to represent its fields when serialized.
func getPotentialTypeValues(typeValue reflect.Type) (map[string]reflect.Type, error) {
	if typeValue.Kind() == reflect.Ptr {
		typeValue = typeValue.Elem()
	}

	if typeValue.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%v is not of type struct", typeValue)
	}

	ret := make(map[string]reflect.Type)

	for fieldIndex := 0; fieldIndex < typeValue.NumField(); fieldIndex++ {
		fieldType := typeValue.Field(fieldIndex)
		yamlTag := fieldType.Tag.Get("json")
		yamlTagName := strings.Split(yamlTag, ",")[0]

		ret[yamlTagName] = fieldType.Type
	}

	return ret, nil
}

func findKnownValue(parts []string, valueOptions sets.String) int {
	for i := range parts {
		if valueOptions.Has(parts[i]) {
			return i
		}
	}

	return -1
}
