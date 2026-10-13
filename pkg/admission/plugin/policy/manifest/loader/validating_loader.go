/*
Copyright The Kubernetes Authors.

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

// Package loader provides functionality to load VAP/MAP configurations from
// manifest files with scheme-based defaulting and validation.
package loader

import (
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	policyloader "k8s.io/apiserver/pkg/admission/plugin/policy/manifest/loader"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
	admissionregistrationinstall "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme, serializer.EnableStrict)
)

func init() {
	admissionregistrationinstall.Install(scheme)
	scheme.AddUnversionedTypes(metav1.SchemeGroupVersion, &metav1.List{}, &metav1.Status{})
}

// LoadedValidatingPolicyManifests holds the VAP configurations loaded from manifest files.
type LoadedValidatingPolicyManifests struct {
	// Policies is the list of loaded ValidatingAdmissionPolicy resources.
	Policies []*admissionregistrationv1.ValidatingAdmissionPolicy
	// Bindings is the list of loaded ValidatingAdmissionPolicyBinding resources.
	Bindings []*admissionregistrationv1.ValidatingAdmissionPolicyBinding
	// Hash is the sha256 hash of all loaded files, used for change detection.
	Hash string
}

// LoadManifestsFromDirectory reads all YAML and JSON files from the specified directory
// and parses them as ValidatingAdmissionPolicy or ValidatingAdmissionPolicyBinding resources.
// Files are processed in alphabetical order for deterministic behavior.
func LoadValidatingManifestsFromDirectory(dir string) (*LoadedValidatingPolicyManifests, error) {
	policies, bindings, hash, err := policyloader.LoadPolicyManifests(
		dir,
		codecs.UniversalDeserializer(),
		acceptVAP,
		acceptVAPB,
		func(b *admissionregistrationv1.ValidatingAdmissionPolicyBinding) string {
			return b.Spec.PolicyName
		},
	)
	if err != nil {
		return nil, err
	}
	return &LoadedValidatingPolicyManifests{
		Policies: policies,
		Bindings: bindings,
		Hash:     hash,
	}, nil
}

// acceptVAP extracts, defaults, and validates ValidatingAdmissionPolicy items.
func acceptVAP(obj runtime.Object) ([]*admissionregistrationv1.ValidatingAdmissionPolicy, error) {
	switch config := obj.(type) {
	case *admissionregistrationv1.ValidatingAdmissionPolicy:
		if err := defaultAndValidateVAP(config); err != nil {
			return nil, err
		}
		return []*admissionregistrationv1.ValidatingAdmissionPolicy{config}, nil
	case *admissionregistrationv1.ValidatingAdmissionPolicyList:
		items := make([]*admissionregistrationv1.ValidatingAdmissionPolicy, len(config.Items))
		for i := range config.Items {
			items[i] = &config.Items[i]
			if err := defaultAndValidateVAP(items[i]); err != nil {
				return nil, err
			}
		}
		return items, nil
	default:
		return nil, policyloader.ErrUnrecognizedType
	}
}

// acceptVAPB extracts, defaults, and validates ValidatingAdmissionPolicyBinding items.
func acceptVAPB(obj runtime.Object) ([]*admissionregistrationv1.ValidatingAdmissionPolicyBinding, error) {
	switch config := obj.(type) {
	case *admissionregistrationv1.ValidatingAdmissionPolicyBinding:
		if err := defaultAndValidateVAPB(config); err != nil {
			return nil, err
		}
		return []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{config}, nil
	case *admissionregistrationv1.ValidatingAdmissionPolicyBindingList:
		items := make([]*admissionregistrationv1.ValidatingAdmissionPolicyBinding, len(config.Items))
		for i := range config.Items {
			items[i] = &config.Items[i]
			if err := defaultAndValidateVAPB(items[i]); err != nil {
				return nil, err
			}
		}
		return items, nil
	default:
		return nil, policyloader.ErrUnrecognizedType
	}
}

// defaultAndValidateVAP applies scheme defaults and standard validation.
func defaultAndValidateVAP(policy *admissionregistrationv1.ValidatingAdmissionPolicy) error {
	scheme.Default(policy)

	internalObj := &admissionregistration.ValidatingAdmissionPolicy{}
	if err := scheme.Convert(policy, internalObj, nil); err != nil {
		return fmt.Errorf("ValidatingAdmissionPolicy %q: conversion error: %w", policy.Name, err)
	}
	if errs := validation.ValidateValidatingAdmissionPolicy(internalObj); len(errs) > 0 {
		return fmt.Errorf("ValidatingAdmissionPolicy %q: %w", policy.Name, errs.ToAggregate())
	}
	resultObj := &admissionregistrationv1.ValidatingAdmissionPolicy{}
	if err := scheme.Convert(internalObj, resultObj, nil); err != nil {
		return fmt.Errorf("ValidatingAdmissionPolicy %q: back-conversion error: %w", policy.Name, err)
	}
	*policy = *resultObj
	return nil
}

// defaultAndValidateVAPB applies scheme defaults and standard validation.
func defaultAndValidateVAPB(binding *admissionregistrationv1.ValidatingAdmissionPolicyBinding) error {
	scheme.Default(binding)

	internalObj := &admissionregistration.ValidatingAdmissionPolicyBinding{}
	if err := scheme.Convert(binding, internalObj, nil); err != nil {
		return fmt.Errorf("ValidatingAdmissionPolicyBinding %q: conversion error: %w", binding.Name, err)
	}
	if errs := validation.ValidateValidatingAdmissionPolicyBinding(internalObj); len(errs) > 0 {
		return fmt.Errorf("ValidatingAdmissionPolicyBinding %q: %w", binding.Name, errs.ToAggregate())
	}
	resultObj := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{}
	if err := scheme.Convert(internalObj, resultObj, nil); err != nil {
		return fmt.Errorf("ValidatingAdmissionPolicyBinding %q: back-conversion error: %w", binding.Name, err)
	}
	*binding = *resultObj
	return nil
}
