/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"fmt"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/featuregate"
)

var (
	maxSamplingRatePerMillion = int32(1000000)
)

// ValidateTracingConfiguration validates the tracing configuration
func ValidateTracingConfiguration(traceConfig *TracingConfiguration, featureGate featuregate.FeatureGate, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if traceConfig == nil {
		return allErrs
	}
	if traceConfig.SamplingRatePerMillion != nil {
		allErrs = append(allErrs, validateSamplingRate(*traceConfig.SamplingRatePerMillion, fldPath.Child("samplingRatePerMillion"))...)
	}
	if traceConfig.Endpoint != nil {
		allErrs = append(allErrs, validateEndpoint(*traceConfig.Endpoint, fldPath.Child("endpoint"))...)
	}
	return allErrs
}

func validateSamplingRate(rate int32, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if rate < 0 {
		errs = append(errs, field.Invalid(
			fldPath, rate,
			"sampling rate must be positive",
		))
	}
	if rate > maxSamplingRatePerMillion {
		errs = append(errs, field.Invalid(
			fldPath, rate,
			"sampling rate per million must be less than or equal to one million",
		))
	}
	return errs
}

func validateEndpoint(endpoint string, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if !strings.Contains(endpoint, "//") {
		endpoint = "dns://" + endpoint
	}
	url, err := url.Parse(endpoint)
	if err != nil {
		errs = append(errs, field.Invalid(
			fldPath, endpoint,
			err.Error(),
		))
		return errs
	}
	switch url.Scheme {
	case "dns":
	case "unix":
	case "unix-abstract":
	default:
		errs = append(errs, field.Invalid(
			fldPath, endpoint,
			fmt.Sprintf("unsupported scheme: %v.  Options are none, dns, unix, or unix-abstract.  See https://github.com/grpc/grpc/blob/master/doc/naming.md", url.Scheme),
		))
	}
	return errs
}
