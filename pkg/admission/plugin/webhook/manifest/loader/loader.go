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

// Package loader provides kube-apiserver-specific webhook manifest loading with
// scheme-based defaulting and validation. The generic load loop lives in the
// staging loader package; this package injects the kube-apiserver scheme, internal
// type conversion, and API validation.
package loader

import (
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	stagingloader "k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/loader"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
	admissionregistrationinstall "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// Re-export staging types so existing callers keep working.
type ValidatingLoadResult = stagingloader.ValidatingLoadResult
type MutatingLoadResult = stagingloader.MutatingLoadResult

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme, serializer.EnableStrict)
)

func init() {
	admissionregistrationinstall.Install(scheme)
	scheme.AddUnversionedTypes(metav1.SchemeGroupVersion, &metav1.List{}, &metav1.Status{})
}

// LoadValidatingManifests reads all YAML and JSON files from the specified directory
// and parses them as ValidatingWebhookConfiguration resources.
// Files containing MutatingWebhookConfiguration cause an error.
// Files are processed in alphabetical order for deterministic behavior.
func LoadValidatingManifests(dir string) (*ValidatingLoadResult, error) {
	configs, hash, err := stagingloader.LoadManifests(dir, codecs.UniversalDeserializer(), func(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
		return acceptObject(obj, extractValidating, defaultAndValidateVWC)
	})
	if err != nil {
		return nil, err
	}
	return &ValidatingLoadResult{Configurations: configs, Hash: hash}, nil
}

// LoadMutatingManifests reads all YAML and JSON files from the specified directory
// and parses them as MutatingWebhookConfiguration resources.
// Files containing ValidatingWebhookConfiguration cause an error.
// Files are processed in alphabetical order for deterministic behavior.
func LoadMutatingManifests(dir string) (*MutatingLoadResult, error) {
	configs, hash, err := stagingloader.LoadManifests(dir, codecs.UniversalDeserializer(), func(obj runtime.Object) ([]*admissionregistrationv1.MutatingWebhookConfiguration, error) {
		return acceptObject(obj, extractMutating, defaultAndValidateMWC)
	})
	if err != nil {
		return nil, err
	}
	return &MutatingLoadResult{Configurations: configs, Hash: hash}, nil
}

// acceptObject is a generic helper that handles type dispatch for decoded objects.
// extract attempts to get typed items from a single object or typed List.
// process applies defaulting and API validation to each individual item.
// Returns (items, nil) when the type is recognized, (nil, nil) when not
// recognized, or (nil, err) when recognized but processing fails.
func acceptObject[T runtime.Object](
	obj runtime.Object,
	extract func(runtime.Object) ([]T, bool),
	process func(T) error,
) ([]T, error) {
	items, ok := extract(obj)
	if !ok {
		return nil, fmt.Errorf("unsupported resource type %T", obj)
	}
	for _, item := range items {
		if err := process(item); err != nil {
			return nil, err
		}
	}
	return items, nil
}

func extractValidating(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, bool) {
	if c, ok := obj.(*admissionregistrationv1.ValidatingWebhookConfiguration); ok {
		return []*admissionregistrationv1.ValidatingWebhookConfiguration{c}, true
	}
	if l, ok := obj.(*admissionregistrationv1.ValidatingWebhookConfigurationList); ok {
		items := make([]*admissionregistrationv1.ValidatingWebhookConfiguration, len(l.Items))
		for i := range l.Items {
			items[i] = &l.Items[i]
		}
		return items, true
	}
	return nil, false
}

func extractMutating(obj runtime.Object) ([]*admissionregistrationv1.MutatingWebhookConfiguration, bool) {
	if c, ok := obj.(*admissionregistrationv1.MutatingWebhookConfiguration); ok {
		return []*admissionregistrationv1.MutatingWebhookConfiguration{c}, true
	}
	if l, ok := obj.(*admissionregistrationv1.MutatingWebhookConfigurationList); ok {
		items := make([]*admissionregistrationv1.MutatingWebhookConfiguration, len(l.Items))
		for i := range l.Items {
			items[i] = &l.Items[i]
		}
		return items, true
	}
	return nil, false
}

// defaultAndValidateVWC applies scheme defaults and runs standard API validation
// on a ValidatingWebhookConfiguration.
func defaultAndValidateVWC(config *admissionregistrationv1.ValidatingWebhookConfiguration) error {
	scheme.Default(config)

	internalObj := &admissionregistration.ValidatingWebhookConfiguration{}
	if err := scheme.Convert(config, internalObj, nil); err != nil {
		return fmt.Errorf("ValidatingWebhookConfiguration %q: conversion error: %w", config.Name, err)
	}
	if errs := validation.ValidateValidatingWebhookConfiguration(internalObj); len(errs) > 0 {
		return fmt.Errorf("ValidatingWebhookConfiguration %q: %w", config.Name, errs.ToAggregate())
	}
	resultConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	if err := scheme.Convert(internalObj, resultConfig, nil); err != nil {
		return fmt.Errorf("ValidatingWebhookConfiguration %q: back-conversion error: %w", config.Name, err)
	}
	*config = *resultConfig
	return nil
}

// defaultAndValidateMWC applies scheme defaults and runs standard API validation
// on a MutatingWebhookConfiguration.
func defaultAndValidateMWC(config *admissionregistrationv1.MutatingWebhookConfiguration) error {
	scheme.Default(config)

	internalObj := &admissionregistration.MutatingWebhookConfiguration{}
	if err := scheme.Convert(config, internalObj, nil); err != nil {
		return fmt.Errorf("MutatingWebhookConfiguration %q: conversion error: %w", config.Name, err)
	}
	if errs := validation.ValidateMutatingWebhookConfiguration(internalObj); len(errs) > 0 {
		return fmt.Errorf("MutatingWebhookConfiguration %q: %w", config.Name, errs.ToAggregate())
	}
	resultConfig := &admissionregistrationv1.MutatingWebhookConfiguration{}
	if err := scheme.Convert(internalObj, resultConfig, nil); err != nil {
		return fmt.Errorf("MutatingWebhookConfiguration %q: back-conversion error: %w", config.Name, err)
	}
	*config = *resultConfig
	return nil
}
