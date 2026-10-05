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
	"errors"
	"fmt"
	"sort"
	"strings"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
)

// ErrUnrecognizedType is returned by AcceptObjectFunc when the object type is
// not handled. This allows callers with multiple accept functions to try the next one.
var ErrUnrecognizedType = fmt.Errorf("unrecognized resource type")

// AcceptObjectFunc extracts typed items from a decoded runtime.Object, applying
// defaulting and validation. Returns ErrUnrecognizedType if the object type is
// not handled by this function.
type AcceptObjectFunc[T metav1.Object] func(obj runtime.Object) ([]T, error)

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
		obj, _, err := decoder.Decode(fd.Doc, nil, nil)
		if err != nil {
			return nil, nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}

		newPolicies, newBindings, err := acceptFileObject(obj, fd.FilePath, seenPolicyNames, seenBindingNames, decoder, acceptPolicy, acceptBinding)
		if err != nil {
			return nil, nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}
		policies = append(policies, newPolicies...)
		bindings = append(bindings, newBindings...)
	}

	if err := validateBindingReferences(policies, bindings, getBindingPolicyName); err != nil {
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
	filePath string,
	seenPolicyNames, seenBindingNames map[string]string,
	decoder runtime.Decoder,
	acceptPolicy AcceptObjectFunc[P],
	acceptBinding AcceptObjectFunc[B],
) ([]P, []B, error) {
	// Handle generic v1.List by recursing into each item
	if list, ok := obj.(*metav1.List); ok {
		var allPolicies []P
		var allBindings []B
		for _, rawItem := range list.Items {
			itemObj, _, err := decoder.Decode(rawItem.Raw, nil, nil)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to decode list item: %w", err)
			}
			p, b, err := acceptFileObject(itemObj, filePath, seenPolicyNames, seenBindingNames, decoder, acceptPolicy, acceptBinding)
			if err != nil {
				return nil, nil, err
			}
			allPolicies = append(allPolicies, p...)
			allBindings = append(allBindings, b...)
		}
		return allPolicies, allBindings, nil
	}

	// Try policy
	policies, pErr := acceptPolicy(obj)
	if pErr != nil && !errors.Is(pErr, ErrUnrecognizedType) {
		return nil, nil, pErr
	}
	if pErr == nil {
		for _, item := range policies {
			if err := validateAcceptedItem(item, filePath, seenPolicyNames); err != nil {
				return nil, nil, err
			}
		}
		return policies, nil, nil
	}

	// Try binding
	bindings, bErr := acceptBinding(obj)
	if bErr != nil && !errors.Is(bErr, ErrUnrecognizedType) {
		return nil, nil, bErr
	}
	if bErr == nil {
		for _, item := range bindings {
			if err := validateAcceptedItem(item, filePath, seenBindingNames); err != nil {
				return nil, nil, err
			}
		}
		return nil, bindings, nil
	}

	// Neither recognized it
	return nil, nil, fmt.Errorf("unsupported resource type %T", obj)
}

// validateAcceptedItem runs manifest name validation and policy-specific
// manifest constraint validation.
func validateAcceptedItem[T metav1.Object](item T, filePath string, seenNames map[string]string) error {
	name := item.GetName()
	if err := manifest.ValidateManifestName(name, filePath, seenNames); err != nil {
		return err
	}
	obj, ok := any(item).(runtime.Object)
	if !ok {
		return fmt.Errorf("type %T does not implement runtime.Object", item)
	}
	return validatePolicyManifestConstraints(obj, name)
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
	default:
		return fmt.Errorf("unsupported policy type %T", obj)
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

// validateBindingReferences ensures all bindings reference policies that exist in the manifest set.
func validateBindingReferences[P, B metav1.Object](policies []P, bindings []B, getBindingPolicyName func(B) string) error {
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
