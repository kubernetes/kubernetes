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

package validation

import (
	"fmt"
	"log"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	//"k8s.io/kubernetes/pkg/apis/core/validation"

	internalapi "k8s.io/kubernetes/plugin/pkg/admission/podnodeselector/apis/podnodeselector"
)

func validateFlatNodeSelector(selector string, fldpath *field.Path) []*field.Error {
	_, err := labels.ConvertSelectorToLabelsMap(selector)
	if err != nil {
		log.Println("complaining about selector", selector, fldpath)
		return []*field.Error{field.Invalid(fldpath, selector, "the selector is invalid")}
	}
	return nil
}

// ValidateConfiguration validates the configuration.
func ValidateConfiguration(config *internalapi.Configuration) error {
	allErrs := field.ErrorList{}
	fldpath := field.NewPath("podnodeselector")

	// validate the default
	allErrs = append(allErrs,
		validateFlatNodeSelector(config.ClusterDefaultNodeSelectors, fldpath.Child("clusterDefaultNodeSelectors"))...)

	// validate each namespace specific selector
	fldpath = fldpath.Child("namespaceSelectorsWhitelists")
	for k, v := range config.NamespaceSelectorsWhitelists {
		allErrs = append(allErrs,
			validateFlatNodeSelector(v, fldpath.Child(k))...)
	}

	log.Println(len(allErrs))
	if len(allErrs) > 0 {
		return fmt.Errorf("invalid config: %v", allErrs)
	}
	return nil
}
