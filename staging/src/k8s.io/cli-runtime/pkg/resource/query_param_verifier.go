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

package resource

import (
	"errors"
	"fmt"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"

	yaml "go.yaml.in/yaml/v2"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
)

func NewQueryParamVerifier(dynamicClient dynamic.Interface, openAPIGetter discovery.OpenAPISchemaInterface, queryParam VerifiableQueryParam) *QueryParamVerifier {
	return &QueryParamVerifier{
		finder:        NewCRDFinder(CRDFromDynamic(dynamicClient)),
		openAPIGetter: openAPIGetter,
		queryParam:    queryParam,
	}
}

// QueryParamVerifier verifies if a given group-version-kind supports a
// given VerifiableQueryParam against the current server.
//
// Currently supported query params are: fieldValidation
//
// Support for each of these query params needs to be verified because
// we determine whether or not to perform server-side or client-side
// schema validation based on whether the fieldValidation query param is
// supported or not.
//
// It reads the OpenAPI to see if the given GVK supports the given query param.
// If the GVK can not be found, we assume that CRDs will have the same level of
// support as "namespaces", and non-CRDs will not be supported. We
// delay the check for CRDs as much as possible though, since it
// requires an extra round-trip to the server.
type QueryParamVerifier struct {
	finder        CRDFinder
	openAPIGetter discovery.OpenAPISchemaInterface
	queryParam    VerifiableQueryParam
}

// Verifier is the generic verifier interface used for testing QueryParamVerifier
type Verifier interface {
	HasSupport(gvk schema.GroupVersionKind) error
}

// VerifiableQueryParam is a query parameter who's enablement on the
// apiserver can be determined by evaluating the OpenAPI for a specific
// GVK.
type VerifiableQueryParam string

const (
	QueryParamFieldValidation VerifiableQueryParam = "fieldValidation"
)

// HasSupport checks if the given gvk supports the query param configured on v
func (v *QueryParamVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	if (gvk == schema.GroupVersionKind{Version: "v1", Kind: "List"}) {
		return NewParamUnsupportedError(gvk, v.queryParam)
	}

	oapi, err := v.openAPIGetter.OpenAPISchema()
	if err != nil {
		return fmt.Errorf("failed to download openapi: %v", err)
	}
	supports, err := supportsQueryParam(oapi, gvk, v.queryParam)
	if err != nil {
		// We assume that we couldn't find the type, then check for namespace:
		supports, _ = supportsQueryParam(oapi, schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Namespace"}, v.queryParam)
		// If namespace supports the query param, then we will support the query param for CRDs only.
		if supports {
			supports, err = v.finder.HasCRD(gvk.GroupKind())
			if err != nil {
				return fmt.Errorf("failed to check CRD: %v", err)
			}
		}
	}
	if !supports {
		return NewParamUnsupportedError(gvk, v.queryParam)
	}
	return nil
}

type paramUnsupportedError struct {
	gvk   schema.GroupVersionKind
	param VerifiableQueryParam
}

func NewParamUnsupportedError(gvk schema.GroupVersionKind, param VerifiableQueryParam) error {
	return &paramUnsupportedError{
		gvk:   gvk,
		param: param,
	}
}

func (e *paramUnsupportedError) Error() string {
	return fmt.Sprintf("%v doesn't support %s", e.gvk, e.param)
}

func IsParamUnsupportedError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*paramUnsupportedError)
	return ok
}

func hasGVKExtension(extensions []*openapi_v2.NamedAny, gvk schema.GroupVersionKind) bool {
	for _, extension := range extensions {
		if extension.GetValue().GetYaml() == "" ||
			extension.GetName() != "x-kubernetes-group-version-kind" {
			continue
		}
		var value map[string]string
		err := yaml.Unmarshal([]byte(extension.GetValue().GetYaml()), &value)
		if err != nil {
			continue
		}

		if value["group"] == gvk.Group && value["kind"] == gvk.Kind && value["version"] == gvk.Version {
			return true
		}
		return false
	}
	return false
}

// supportsQueryParam is a method that let's us look in the OpenAPI if the
// specific group-version-kind supports the specific query parameter for
// the PATCH end-point.
func supportsQueryParam(doc *openapi_v2.Document, gvk schema.GroupVersionKind, queryParam VerifiableQueryParam) (bool, error) {
	globalParams := map[string]*openapi_v2.NamedParameter{}
	for _, p := range doc.GetParameters().GetAdditionalProperties() {
		globalParams["#/parameters/"+p.GetName()] = p
	}

	for _, path := range doc.GetPaths().GetPath() {
		// Is this describing the gvk we're looking for?
		if !hasGVKExtension(path.GetValue().GetPatch().GetVendorExtension(), gvk) {
			continue
		}
		for _, param := range path.GetValue().GetPatch().GetParameters() {
			if param.GetParameter().GetNonBodyParameter().GetQueryParameterSubSchema().GetName() == string(queryParam) {
				return true, nil
			}

			// lookup global parameters
			if ref := param.GetJsonReference().GetXRef(); ref != "" {
				if globalParam, ok := globalParams[ref]; ok && globalParam != nil && globalParam.GetValue().GetNonBodyParameter().GetQueryParameterSubSchema().GetName() == string(queryParam) {
					return true, nil
				}
			}
		}
		return false, nil
	}

	return false, errors.New("couldn't find GVK in openapi")
}
