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

package resource

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi3"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var _ Verifier = &queryParamVerifierV3{}

// NewQueryParamVerifierV3 returns a pointer to the created queryParamVerifier3 struct,
// which implements the Verifier interface. The caching characteristics of the
// OpenAPI V3 specs are determined by the passed oapiClient. For memory caching, the
// client should be wrapped beforehand as: cached.NewClient(oapiClient). The disk
// caching is determined by the discovery client the oapiClient is created from.
func NewQueryParamVerifierV3(dynamicClient dynamic.Interface, oapiClient openapi.Client, queryParam VerifiableQueryParam) Verifier {
	return &queryParamVerifierV3{
		finder:     NewCRDFinder(CRDFromDynamic(dynamicClient)),
		root:       openapi3.NewRoot(oapiClient),
		queryParam: queryParam,
	}
}

// queryParamVerifierV3 encapsulates info necessary to determine if
// the queryParam is a parameter for the Patch endpoint for a
// passed GVK.
type queryParamVerifierV3 struct {
	finder     CRDFinder
	root       openapi3.Root
	queryParam VerifiableQueryParam
}

var namespaceGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Namespace"}

// HasSupport returns nil error if the passed GVK supports the parameter
// (stored in struct; usually "fieldValidation") for Patch endpoint.
// Returns an error if the passed GVK does not support the query param,
// or if another error occurred. If the Open API V3 spec for a CRD is not
// found, then the spec for Namespace is checked for query param support instead.
func (v *queryParamVerifierV3) HasSupport(gvk schema.GroupVersionKind) error {
	if (gvk == schema.GroupVersionKind{Version: "v1", Kind: "List"}) {
		return NewParamUnsupportedError(gvk, v.queryParam)
	}
	gvSpec, err := v.root.GVSpec(gvk.GroupVersion())
	if err == nil {
		return supportsQueryParamV3(gvSpec, gvk, v.queryParam)
	}
	if _, isErr := err.(*openapi3.GroupVersionNotFoundError); !isErr {
		return err
	}
	// If the spec for the passed GVK is not found, then check if it is a CRD.
	// For CRD's substitute Namespace OpenAPI V3 spec to check if query param is supported.
	if found, _ := v.finder.HasCRD(gvk.GroupKind()); found {
		namespaceSpec, err := v.root.GVSpec(namespaceGVK.GroupVersion())
		if err != nil {
			// If error retrieving Namespace spec, propagate error.
			return err
		}
		return supportsQueryParamV3(namespaceSpec, namespaceGVK, v.queryParam)
	}
	return NewParamUnsupportedError(gvk, v.queryParam)
}

// hasGVKExtensionV3 returns true if the passed OpenAPI extensions map contains
// the passed GVK; false otherwise.
func hasGVKExtensionV3(extensions spec.Extensions, gvk schema.GroupVersionKind) bool {
	var oapiGVK map[string]string
	err := extensions.GetObject("x-kubernetes-group-version-kind", &oapiGVK)
	if err != nil {
		return false
	}
	if oapiGVK["group"] == gvk.Group &&
		oapiGVK["version"] == gvk.Version &&
		oapiGVK["kind"] == gvk.Kind {
		return true
	}
	return false
}

// supportsQueryParam is a method that let's us look in the OpenAPI if the
// specific group-version-kind supports the specific query parameter for
// the PATCH end-point. Returns nil if the passed GVK supports the passed
// query parameter; otherwise, a "paramUnsupportedError" is returned (except
// when an invalid document error is returned when an invalid OpenAPI V3
// is passed in).
func supportsQueryParamV3(doc *spec3.OpenAPI, gvk schema.GroupVersionKind, queryParam VerifiableQueryParam) error {
	if doc == nil || doc.Paths == nil {
		return fmt.Errorf("Invalid OpenAPI V3 document")
	}
	for _, path := range doc.Paths.Paths {
		// If operation is not PATCH, then continue.
		if path == nil {
			continue
		}
		op := path.PathProps.Patch
		if op == nil {
			continue
		}
		// Is this PATCH operation for the passed GVK?
		if !hasGVKExtensionV3(op.VendorExtensible.Extensions, gvk) {
			continue
		}
		// Now look for the query parameter among the parameters
		// for the PATCH operation.
		for _, param := range op.OperationProps.Parameters {
			if param.ParameterProps.Name == string(queryParam) && param.In == "query" {
				return nil
			}

			// lookup global parameters
			if ref := param.Refable.Ref.Ref.String(); strings.HasPrefix(ref, "#/parameters/") && doc.Components != nil {
				k := strings.TrimPrefix(ref, "#/parameters/")
				if globalParam, ok := doc.Components.Parameters[k]; ok && globalParam != nil {
					if globalParam.In == "query" && globalParam.Name == string(queryParam) {
						return nil
					}
				}
			}
		}
		return NewParamUnsupportedError(gvk, queryParam)
	}
	return fmt.Errorf("Path not found for GVK (%s) in OpenAPI V3 doc", gvk)
}
