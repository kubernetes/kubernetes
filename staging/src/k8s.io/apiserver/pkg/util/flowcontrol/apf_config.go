/*
Copyright 2022 The Kubernetes Authors.

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

package flowcontrol

import (
	"fmt"
	"io/ioutil"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/install"
	"k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	"sigs.k8s.io/yaml"
)

var cfgScheme = runtime.NewScheme()

func init() {
	install.Install(cfgScheme)
}

// DefaultConfiguration return default Priority and Fairness configuration.
func DefaultConfiguration() apiserver.PriorityAndFairnessConfiguration {
	cfg, _ := ApplyConfigFromFileToDefaultConfiguration("")
	return *cfg
}

// ApplyConfigFromFileToDefaultConfiguration parses configuration stored under configFilePath and
// overrides PriorityAndFairnessConfiguration cfg object with defined values, leaving undefined ones
// unchanged.
func ApplyConfigFromFileToDefaultConfiguration(cfgFilePath string) (*apiserver.PriorityAndFairnessConfiguration, error) {
	decodedConfig := &v1alpha1.PriorityAndFairnessConfiguration{}
	cfgScheme.Default(decodedConfig)

	if cfgFilePath != "" {
		data, err := ioutil.ReadFile(cfgFilePath)
		if err != nil {
			return nil, fmt.Errorf("unable to read P&F configuration from %q [%v]", cfgFilePath, err)
		}

		if err = yaml.Unmarshal(data, &decodedConfig); err != nil {
			return nil, err
		}

		if decodedConfig.Kind != "PriorityAndFairnessConfiguration" {
			return nil, fmt.Errorf("invalid service configuration object %q", decodedConfig.Kind)
		}
	}

	internalConfig := &apiserver.PriorityAndFairnessConfiguration{}
	if err := cfgScheme.Convert(decodedConfig, internalConfig, nil); err != nil {
		return nil, err
	}

	return internalConfig, nil
}

// ValidatePriorityAndFairnessConfiguration checks the v1alpha1.PriorityAndFairnessConfiguration for
// common configuration errors. It will return error if either objectsPerSeat or watchesPerSeat
// have value lower or equal to 0.
func ValidatePriorityAndFairnessConfiguration(cfg *apiserver.PriorityAndFairnessConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	if cfg == nil {
		return allErrs // Treating a nil configuration as valid
	}

	objectsPerSeat := cfg.WorkEstimator.ListWorkEstimator.ObjectsPerSeat
	if objectsPerSeat <= 0 {
		fldPath := field.NewPath("workEstimator", "listRequests")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("objectsPerSeat"),
			objectsPerSeat,
			"objectsPerSeat can't be less than 0"))
	}

	watchesPerSeat := cfg.WorkEstimator.MutatingWorkEstimator.WatchesPerSeat
	if watchesPerSeat <= 0 {
		fldPath := field.NewPath("workEstimator", "mutatingRequests")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("watchesPerSeat"),
			watchesPerSeat,
			"watchesPerSeat can't be less than 0"))
	}

	return allErrs
}
