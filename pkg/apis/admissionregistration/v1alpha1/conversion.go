/*
Copyright 2023 The Kubernetes Authors.

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

package v1alpha1

import (
	"errors"

	v1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func Convert_admissionregistration_ValidatingAdmissionPolicyBindingSpec_To_v1alpha1_ValidatingAdmissionPolicyBindingSpec(src *admissionregistration.ValidatingAdmissionPolicyBindingSpec, dst *v1alpha1.ValidatingAdmissionPolicyBindingSpec, scope conversion.Scope) error {
	if err := autoConvert_admissionregistration_ValidatingAdmissionPolicyBindingSpec_To_v1alpha1_ValidatingAdmissionPolicyBindingSpec(src, dst, scope); err != nil {
		return err
	}

	dst.ParamRef = nil
	dst.NamespaceParamRef = nil

	if src.ParamRef != nil {
		switch src.ParamRef.ParamKind {
		case admissionregistration.BindingParamKindClusterWide:
			if p := src.ParamRef.ClusterWide; p != nil {
				// Conversion was not automatically generated due to name change?
				dst.ParamRef = &v1alpha1.ParamRef{
					Name:      p.Name,
					Namespace: p.Namespace,
				}
			}

		case admissionregistration.BindingParamKindPerNamespace:
			var v1alpha1ParamRef v1alpha1.NamespaceParamRef
			err := Convert_admissionregistration_NamespaceParamRef_To_v1alpha1_NamespaceParamRef(src.ParamRef.PerNamespace, &v1alpha1ParamRef, scope)
			if err != nil {
				return err
			}
			dst.NamespaceParamRef = &v1alpha1ParamRef
		default:
			return errors.New("invalid Spec.ParamRef.ParamKind")
		}
	}

	return nil
}

func Convert_v1alpha1_ValidatingAdmissionPolicyBindingSpec_To_admissionregistration_ValidatingAdmissionPolicyBindingSpec(src *v1alpha1.ValidatingAdmissionPolicyBindingSpec, dst *admissionregistration.ValidatingAdmissionPolicyBindingSpec, scope conversion.Scope) error {
	if err := autoConvert_v1alpha1_ValidatingAdmissionPolicyBindingSpec_To_admissionregistration_ValidatingAdmissionPolicyBindingSpec(src, dst, scope); err != nil {
		return err
	}

	dst.ParamRef = nil

	if src.ParamRef != nil {
		dst.ParamRef = &admissionregistration.BindingParamRef{
			ParamKind: admissionregistration.BindingParamKindClusterWide,
			ClusterWide: &admissionregistration.ClusterWideParamRef{
				Name:      src.ParamRef.Name,
				Namespace: src.ParamRef.Namespace,
			},
		}
	} else if src.NamespaceParamRef != nil {
		var internalParamRef admissionregistration.NamespaceParamRef
		err := Convert_v1alpha1_NamespaceParamRef_To_admissionregistration_NamespaceParamRef(src.NamespaceParamRef, &internalParamRef, scope)
		if err != nil {
			return err
		}
		dst.ParamRef = &admissionregistration.BindingParamRef{
			ParamKind:    admissionregistration.BindingParamKindPerNamespace,
			PerNamespace: &internalParamRef,
		}
	}
	return nil
}

func Convert_admissionregistration_BindingParamRef_To_v1alpha1_ParamRef(src *admissionregistration.BindingParamRef, dst *v1alpha1.ParamRef, scope conversion.Scope) error {
	// Do nothing. Conversion handled by Spec converter
	return nil
}

func Convert_v1alpha1_ParamRef_To_admissionregistration_BindingParamRef(src *v1alpha1.ParamRef, dst *admissionregistration.BindingParamRef, scope conversion.Scope) error {
	// Do nothing. Conversion handled by Spec converter
	return nil
}
