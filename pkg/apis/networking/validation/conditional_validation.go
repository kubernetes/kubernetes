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
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"
)

// ValidateConditionalNetworkPolicy validates conditionally valid fields.
func ValidateConditionalNetworkPolicy(np, oldNP *networking.NetworkPolicy) field.ErrorList {
	var errs field.ErrorList
	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(sctpFields(oldNP)) == 0 {
		for _, f := range sctpFields(np) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}
	return errs
}

func sctpFields(np *networking.NetworkPolicy) []*field.Path {
	if np == nil {
		return nil
	}
	fields := []*field.Path{}
	for iIndex, e := range np.Spec.Ingress {
		for pIndex, p := range e.Ports {
			if p.Protocol != nil && *p.Protocol == api.ProtocolSCTP {
				fields = append(fields, field.NewPath("spec.ingress").Index(iIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	for eIndex, e := range np.Spec.Egress {
		for pIndex, p := range e.Ports {
			if p.Protocol != nil && *p.Protocol == api.ProtocolSCTP {
				fields = append(fields, field.NewPath("spec.egress").Index(eIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	return fields
}
