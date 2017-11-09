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

	"k8s.io/apimachinery/pkg/util/validation/field"
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources"
)

// ValidateConfiguration validates the configuration.
func ValidateConfiguration(config *internalapi.Configuration) error {
	allErrs := field.ErrorList{}
	fldpath := field.NewPath("initialresources").Child("datasourceinfo")
	allErrs = append(allErrs, ValidateDataSource(config.DataSourceInfo.DataSource, fldpath.Child("datasource"))...)
	if len(allErrs) > 0 {
		return fmt.Errorf("invalid config: %v", allErrs)
	}
	return nil
}

// ValidateDataSource validates data source type
func ValidateDataSource(dataSource internalapi.DataSourceType, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	allSupportDataSource := []string{string(internalapi.Influxdb), string(internalapi.Gcm), string(internalapi.Hawkular)}
	switch dataSource {
	case internalapi.Influxdb, internalapi.Gcm, internalapi.Hawkular:
		break
	default:
		allErrors = append(allErrors, field.NotSupported(fldPath, dataSource, allSupportDataSource))
	}
	return allErrors
}
