/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// ValidateConditionalService validates conditionally valid fields.
func ValidateConditionalService(service, oldService *api.Service) field.ErrorList {
	var errs field.ErrorList

	errs = append(errs, validateMixedProtocolLBService(service, oldService)...)

	return errs
}

// validateMixedProtocolLBService checks if the old Service has type=LoadBalancer and whether the Service has different Protocols
// on its ports. If the MixedProtocolLBService feature flag is disabled the usage of different Protocols in the new Service is
// valid only if the old Service has different Protocols, too.
func validateMixedProtocolLBService(service, oldService *api.Service) (errs field.ErrorList) {
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		return
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.MixedProtocolLBService) {
		return
	}

	if serviceHasMixedProtocols(service) && !serviceHasMixedProtocols(oldService) {
		errs = append(errs, field.Invalid(field.NewPath("spec", "ports"), service.Spec.Ports, "may not contain more than 1 protocol when type is 'LoadBalancer'"))
	}
	return
}

func serviceHasMixedProtocols(service *api.Service) bool {
	if service == nil {
		return false
	}
	protos := map[string]bool{}
	for _, port := range service.Spec.Ports {
		protos[string(port.Protocol)] = true
	}
	return len(protos) > 1
}
