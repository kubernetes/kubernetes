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

// Package loader provides functionality to load policy configurations from manifest files.
// This file handles MutatingAdmissionPolicy and MutatingAdmissionPolicyBinding resources.
// It uses the package-level scheme and codecs variables defined in validating_loader.go for
// decoding, defaulting, and validation.
package loader

import (
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime"
	policyloader "k8s.io/apiserver/pkg/admission/plugin/policy/manifest/loader"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// LoadedMutatingPolicyManifests holds the MAP configurations loaded from manifest files.
type LoadedMutatingPolicyManifests struct {
	// Policies is the list of loaded MutatingAdmissionPolicy resources.
	Policies []*admissionregistrationv1.MutatingAdmissionPolicy
	// Bindings is the list of loaded MutatingAdmissionPolicyBinding resources.
	Bindings []*admissionregistrationv1.MutatingAdmissionPolicyBinding
	// Hash is the sha256 hash of all loaded files, used for change detection.
	Hash string
}

// LoadMutatingManifestsFromDirectory reads all YAML and JSON files from the specified directory
// and parses them as MutatingAdmissionPolicy or MutatingAdmissionPolicyBinding resources.
// Files are processed in alphabetical order for deterministic behavior.
func LoadMutatingManifestsFromDirectory(dir string) (*LoadedMutatingPolicyManifests, error) {
	policies, bindings, hash, err := policyloader.LoadPolicyManifests(
		dir,
		codecs.UniversalDeserializer(),
		acceptMAP,
		acceptMAPB,
		func(b *admissionregistrationv1.MutatingAdmissionPolicyBinding) string {
			return b.Spec.PolicyName
		},
	)
	if err != nil {
		return nil, err
	}
	return &LoadedMutatingPolicyManifests{
		Policies: policies,
		Bindings: bindings,
		Hash:     hash,
	}, nil
}

// acceptMAP extracts, defaults, and validates MutatingAdmissionPolicy items.
func acceptMAP(obj runtime.Object) ([]*admissionregistrationv1.MutatingAdmissionPolicy, error) {
	switch config := obj.(type) {
	case *admissionregistrationv1.MutatingAdmissionPolicy:
		if err := defaultAndValidateMAP(config); err != nil {
			return nil, err
		}
		return []*admissionregistrationv1.MutatingAdmissionPolicy{config}, nil
	case *admissionregistrationv1.MutatingAdmissionPolicyList:
		items := make([]*admissionregistrationv1.MutatingAdmissionPolicy, len(config.Items))
		for i := range config.Items {
			items[i] = &config.Items[i]
			if err := defaultAndValidateMAP(items[i]); err != nil {
				return nil, err
			}
		}
		return items, nil
	default:
		return nil, policyloader.ErrUnrecognizedType
	}
}

// acceptMAPB extracts, defaults, and validates MutatingAdmissionPolicyBinding items.
func acceptMAPB(obj runtime.Object) ([]*admissionregistrationv1.MutatingAdmissionPolicyBinding, error) {
	switch config := obj.(type) {
	case *admissionregistrationv1.MutatingAdmissionPolicyBinding:
		if err := defaultAndValidateMAPB(config); err != nil {
			return nil, err
		}
		return []*admissionregistrationv1.MutatingAdmissionPolicyBinding{config}, nil
	case *admissionregistrationv1.MutatingAdmissionPolicyBindingList:
		items := make([]*admissionregistrationv1.MutatingAdmissionPolicyBinding, len(config.Items))
		for i := range config.Items {
			items[i] = &config.Items[i]
			if err := defaultAndValidateMAPB(items[i]); err != nil {
				return nil, err
			}
		}
		return items, nil
	default:
		return nil, policyloader.ErrUnrecognizedType
	}
}

// defaultAndValidateMAP applies scheme defaults and standard validation.
func defaultAndValidateMAP(policy *admissionregistrationv1.MutatingAdmissionPolicy) error {
	scheme.Default(policy)

	internalObj := &admissionregistration.MutatingAdmissionPolicy{}
	if err := scheme.Convert(policy, internalObj, nil); err != nil {
		return fmt.Errorf("MutatingAdmissionPolicy %q: conversion error: %w", policy.Name, err)
	}
	if errs := validation.ValidateMutatingAdmissionPolicy(internalObj); len(errs) > 0 {
		return fmt.Errorf("MutatingAdmissionPolicy %q: %w", policy.Name, errs.ToAggregate())
	}
	resultObj := &admissionregistrationv1.MutatingAdmissionPolicy{}
	if err := scheme.Convert(internalObj, resultObj, nil); err != nil {
		return fmt.Errorf("MutatingAdmissionPolicy %q: back-conversion error: %w", policy.Name, err)
	}
	*policy = *resultObj
	return nil
}

// defaultAndValidateMAPB applies scheme defaults and standard validation.
func defaultAndValidateMAPB(binding *admissionregistrationv1.MutatingAdmissionPolicyBinding) error {
	scheme.Default(binding)

	internalObj := &admissionregistration.MutatingAdmissionPolicyBinding{}
	if err := scheme.Convert(binding, internalObj, nil); err != nil {
		return fmt.Errorf("MutatingAdmissionPolicyBinding %q: conversion error: %w", binding.Name, err)
	}
	if errs := validation.ValidateMutatingAdmissionPolicyBinding(internalObj); len(errs) > 0 {
		return fmt.Errorf("MutatingAdmissionPolicyBinding %q: %w", binding.Name, errs.ToAggregate())
	}
	resultObj := &admissionregistrationv1.MutatingAdmissionPolicyBinding{}
	if err := scheme.Convert(internalObj, resultObj, nil); err != nil {
		return fmt.Errorf("MutatingAdmissionPolicyBinding %q: back-conversion error: %w", binding.Name, err)
	}
	*binding = *resultObj
	return nil
}
