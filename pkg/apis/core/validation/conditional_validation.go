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
	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(serviceSCTPFields(oldService)) == 0 {
		for _, f := range serviceSCTPFields(service) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}
	return errs
}

func serviceSCTPFields(service *api.Service) []*field.Path {
	if service == nil {
		return nil
	}
	fields := []*field.Path{}
	for pIndex, p := range service.Spec.Ports {
		if p.Protocol == api.ProtocolSCTP {
			fields = append(fields, field.NewPath("spec.ports").Index(pIndex).Child("protocol"))
		}
	}
	return fields
}

// ValidateConditionalEndpoints validates conditionally valid fields.
func ValidateConditionalEndpoints(endpoints, oldEndpoints *api.Endpoints) field.ErrorList {
	var errs field.ErrorList
	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(endpointsSCTPFields(oldEndpoints)) == 0 {
		for _, f := range endpointsSCTPFields(endpoints) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}
	return errs
}

func endpointsSCTPFields(endpoints *api.Endpoints) []*field.Path {
	if endpoints == nil {
		return nil
	}
	fields := []*field.Path{}
	for sIndex, s := range endpoints.Subsets {
		for pIndex, p := range s.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, field.NewPath("subsets").Index(sIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	return fields
}

// ValidateConditionalPodTemplate validates conditionally valid fields.
// This should be called from Validate/ValidateUpdate for all resources containing a PodTemplateSpec
func ValidateConditionalPodTemplate(podTemplate, oldPodTemplate *api.PodTemplateSpec, fldPath *field.Path) field.ErrorList {
	var (
		podSpec    *api.PodSpec
		oldPodSpec *api.PodSpec
	)
	if podTemplate != nil {
		podSpec = &podTemplate.Spec
	}
	if oldPodTemplate != nil {
		oldPodSpec = &oldPodTemplate.Spec
	}
	return validateConditionalPodSpec(podSpec, oldPodSpec, fldPath.Child("spec"))
}

// ValidateConditionalPod validates conditionally valid fields.
// This should be called from Validate/ValidateUpdate for all resources containing a Pod
func ValidateConditionalPod(pod, oldPod *api.Pod, fldPath *field.Path) field.ErrorList {
	var (
		podSpec    *api.PodSpec
		oldPodSpec *api.PodSpec
	)
	if pod != nil {
		podSpec = &pod.Spec
	}
	if oldPod != nil {
		oldPodSpec = &oldPod.Spec
	}
	return validateConditionalPodSpec(podSpec, oldPodSpec, fldPath.Child("spec"))
}

func validateConditionalPodSpec(podSpec, oldPodSpec *api.PodSpec, fldPath *field.Path) field.ErrorList {
	// Always make sure we have a non-nil current pod spec
	if podSpec == nil {
		podSpec = &api.PodSpec{}
	}

	errs := field.ErrorList{}

	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(podSCTPFields(oldPodSpec, nil)) == 0 {
		for _, f := range podSCTPFields(podSpec, fldPath) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}

	return errs
}

func podSCTPFields(podSpec *api.PodSpec, fldPath *field.Path) []*field.Path {
	if podSpec == nil {
		return nil
	}
	fields := []*field.Path{}
	for cIndex, c := range podSpec.InitContainers {
		for pIndex, p := range c.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, fldPath.Child("initContainers").Index(cIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	for cIndex, c := range podSpec.Containers {
		for pIndex, p := range c.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, fldPath.Child("containers").Index(cIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	return fields
}
