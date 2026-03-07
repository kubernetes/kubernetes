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

// Package loader provides generic functionality to load policy and binding
// configurations from manifest files. It handles file reading, YAML/JSON
// decoding, manifest name validation (suffix + uniqueness), policy-specific
// manifest constraints, and binding-reference checking. Type-specific
// defaulting and validation (e.g., scheme-based defaulting via internal
// types) are injected by callers through the AcceptObjectFunc callbacks.
package loader

import (
	"fmt"
	"sort"
	"strings"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
	webhookloader "k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/loader"
)

// AcceptObjectFunc processes a decoded runtime.Object and returns zero or more
// typed results. It returns (items, true, nil) when the object type is recognized
// and successfully processed (defaulting and API validation applied),
// (nil, false, nil) when the type is not recognized, or (nil, true, err) when
// the type is recognized but processing fails.
type AcceptObjectFunc[T metav1.Object] func(obj runtime.Object) ([]T, bool, error)

// LoadPolicyManifests is a generic helper that loads policy and binding manifests
// from a directory. It handles file I/O, decoding, v1.List unwrapping, manifest
// name validation, policy-specific manifest constraints, binding-reference checking,
// and deterministic sorting by name. The decoder should be created by the caller
// with whatever scheme is appropriate (e.g., with full internal type install for
// kube-apiserver, or a minimal v1-only scheme for other consumers).
func LoadPolicyManifests[P, B metav1.Object](
	dir string,
	decoder runtime.Decoder,
	acceptPolicy AcceptObjectFunc[P],
	acceptBinding AcceptObjectFunc[B],
	getBindingPolicyName func(B) string,
) ([]P, []B, string, error) {
	fileDocs, hash, err := manifest.LoadFiles(dir)
	if err != nil {
		return nil, nil, "", err
	}

	policies := make([]P, 0)
	bindings := make([]B, 0)
	seenPolicyNames := map[string]string{}
	seenBindingNames := map[string]string{}

	for _, fd := range fileDocs {
		obj, gvk, err := decoder.Decode(fd.Doc, nil, nil)
		if err != nil {
			return nil, nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}

		newPolicies, newBindings, err := acceptFileObject(obj, gvk, fd.FilePath, seenPolicyNames, seenBindingNames, decoder, acceptPolicy, acceptBinding)
		if err != nil {
			return nil, nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}
		policies = append(policies, newPolicies...)
		bindings = append(bindings, newBindings...)
	}

	if err := ValidateBindingReferences(policies, bindings, getBindingPolicyName); err != nil {
		return nil, nil, "", err
	}

	sort.Slice(policies, func(i, j int) bool {
		return policies[i].GetName() < policies[j].GetName()
	})
	sort.Slice(bindings, func(i, j int) bool {
		return bindings[i].GetName() < bindings[j].GetName()
	})

	return policies, bindings, hash, nil
}

// acceptFileObject handles type dispatch for a decoded object, including
// generic v1.List unwrapping, manifest name validation, policy-specific
// manifest constraint validation, and the default unsupported-type error.
func acceptFileObject[P, B metav1.Object](
	obj runtime.Object,
	gvk *schema.GroupVersionKind,
	filePath string,
	seenPolicyNames, seenBindingNames map[string]string,
	decoder runtime.Decoder,
	acceptPolicy AcceptObjectFunc[P],
	acceptBinding AcceptObjectFunc[B],
) ([]P, []B, error) {
	acceptAndValidatePolicy := func(obj runtime.Object) ([]P, bool, error) {
		return acceptAndValidate(obj, filePath, seenPolicyNames, acceptPolicy)
	}
	acceptAndValidateBinding := func(obj runtime.Object) ([]B, bool, error) {
		return acceptAndValidate(obj, filePath, seenBindingNames, acceptBinding)
	}

	// Try typed extraction first
	if policies, ok, err := acceptAndValidatePolicy(obj); ok {
		return policies, nil, err
	}
	if bindings, ok, err := acceptAndValidateBinding(obj); ok {
		return nil, bindings, err
	}

	// Handle generic v1.List
	if list, ok := obj.(*metav1.List); ok {
		var policies []P
		var bindings []B
		for _, rawItem := range list.Items {
			itemObj, itemGVK, err := decoder.Decode(rawItem.Raw, nil, nil)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to decode list item: %w", err)
			}
			if p, ok, err := acceptAndValidatePolicy(itemObj); ok {
				if err != nil {
					return nil, nil, err
				}
				policies = append(policies, p...)
			} else if b, ok, err := acceptAndValidateBinding(itemObj); ok {
				if err != nil {
					return nil, nil, err
				}
				bindings = append(bindings, b...)
			} else {
				return nil, nil, fmt.Errorf("unsupported resource type %v in List", itemGVK)
			}
		}
		return policies, bindings, nil
	}

	return nil, nil, fmt.Errorf("unsupported resource type %v", gvk)
}

// acceptAndValidate tries to accept an object and validates each accepted item.
func acceptAndValidate[T metav1.Object](obj runtime.Object, filePath string, seenNames map[string]string, accept AcceptObjectFunc[T]) ([]T, bool, error) {
	items, ok, err := accept(obj)
	if !ok || err != nil {
		return items, ok, err
	}
	for _, item := range items {
		if err := validateAcceptedItem(item, filePath, seenNames); err != nil {
			return nil, true, err
		}
	}
	return items, true, nil
}

// validateAcceptedItem runs manifest name validation and policy-specific
// manifest constraint validation.
func validateAcceptedItem[T metav1.Object](item T, filePath string, seenNames map[string]string) error {
	name := item.GetName()
	if err := webhookloader.ValidateManifestName(name, filePath, seenNames); err != nil {
		return err
	}
	if obj, ok := any(item).(runtime.Object); ok {
		if err := validatePolicyManifestConstraints(obj, name); err != nil {
			return err
		}
	}
	return nil
}

// validatePolicyManifestConstraints validates manifest-specific constraints
// for policy and binding types (paramKind, paramRef, policyName suffix).
func validatePolicyManifestConstraints(obj runtime.Object, name string) error {
	switch c := obj.(type) {
	case *admissionregistrationv1.ValidatingAdmissionPolicy:
		if c.Spec.ParamKind != nil {
			return fmt.Errorf("ValidatingAdmissionPolicy %q: spec.paramKind is not supported for static manifests", name)
		}
	case *admissionregistrationv1.ValidatingAdmissionPolicyBinding:
		if err := validateBindingConstraints(name, c.Spec.PolicyName, c.Spec.ParamRef != nil); err != nil {
			return err
		}
	case *admissionregistrationv1.MutatingAdmissionPolicy:
		if c.Spec.ParamKind != nil {
			return fmt.Errorf("MutatingAdmissionPolicy %q: spec.paramKind is not supported for static manifests", name)
		}
	case *admissionregistrationv1.MutatingAdmissionPolicyBinding:
		if err := validateBindingConstraints(name, c.Spec.PolicyName, c.Spec.ParamRef != nil); err != nil {
			return err
		}
	}
	return nil
}

// validateBindingConstraints checks common binding manifest constraints:
// policyName must be non-empty, have the static suffix, and paramRef must be nil.
func validateBindingConstraints(bindingName, policyName string, hasParamRef bool) error {
	if len(policyName) == 0 {
		return fmt.Errorf("binding %q must reference a policy (spec.policyName)", bindingName)
	}
	if !strings.HasSuffix(policyName, manifest.StaticConfigSuffix) {
		return fmt.Errorf("binding %q: spec.policyName %q must end with %q", bindingName, policyName, manifest.StaticConfigSuffix)
	}
	if hasParamRef {
		return fmt.Errorf("binding %q: spec.paramRef is not supported for static manifests", bindingName)
	}
	return nil
}

// ValidateBindingReferences ensures all bindings reference policies that exist in the manifest set.
func ValidateBindingReferences[P, B metav1.Object](policies []P, bindings []B, getBindingPolicyName func(B) string) error {
	policyNames := sets.New[string]()
	for _, policy := range policies {
		policyNames.Insert(policy.GetName())
	}
	for _, binding := range bindings {
		if !policyNames.Has(getBindingPolicyName(binding)) {
			return fmt.Errorf("binding %q references policy %q which does not exist in the manifest directory", binding.GetName(), getBindingPolicyName(binding))
		}
	}
	return nil
}
