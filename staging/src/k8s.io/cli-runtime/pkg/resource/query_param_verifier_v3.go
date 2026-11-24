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
	"errors"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi3"
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
		resources:  openapi3.NewRootAndResources(oapiClient),
		queryParam: queryParam,
	}
}

// queryParamVerifierV3 encapsulates info necessary to determine if
// the queryParam is a parameter for the Patch endpoint for a
// passed GVK.
type queryParamVerifierV3 struct {
	finder     CRDFinder
	resources  openapi3.Resources
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
	supported, err := v.resources.SupportsQueryParam(gvk, http.MethodPatch, string(v.queryParam))
	if err == nil {
		if !supported {
			return NewParamUnsupportedError(gvk, v.queryParam)
		}
		return nil
	}
	if isUnsupportedByOpenAPIV3Spec(err) {
		return NewParamUnsupportedError(gvk, v.queryParam)
	}
	if _, ok := errors.AsType[*openapi3.GroupVersionNotFoundError](err); !ok {
		return err
	}
	// If the spec for the passed GVK is not found, then check if it is a CRD.
	// For CRD's substitute Namespace OpenAPI V3 spec to check if query param is supported.
	if found, _ := v.finder.HasCRD(gvk.GroupKind()); found {
		nsSupported, err := v.resources.SupportsQueryParam(namespaceGVK, http.MethodPatch, string(v.queryParam))
		if err != nil {
			if isUnsupportedByOpenAPIV3Spec(err) {
				return NewParamUnsupportedError(namespaceGVK, v.queryParam)
			}
			// If error retrieving Namespace spec, propagate error.
			return err
		}
		if !nsSupported {
			return NewParamUnsupportedError(namespaceGVK, v.queryParam)
		}
		return nil
	}
	return NewParamUnsupportedError(gvk, v.queryParam)
}

func isUnsupportedByOpenAPIV3Spec(err error) bool {
	_, gvkMissing := errors.AsType[*openapi3.GroupVersionKindNotFoundError](err)
	_, opMissing := errors.AsType[*openapi3.OperationNotFoundError](err)
	return gvkMissing || opMissing
}
