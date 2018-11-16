/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"crypto/sha512"
	"encoding/json"
	"fmt"

	"github.com/go-openapi/spec"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	deleteOptionsSchemaRef = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.DeleteOptions"
	objectMetaSchemaRef    = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"
	scaleSpecSchemaRef     = "#/definitions/io.k8s.api.autoscaling.v1.ScaleSpec"
	scaleStatusSchemaRef   = "#/definitions/io.k8s.api.autoscaling.v1.ScaleStatus"
)

var swaggerTypeMetaDescriptions = metav1.TypeMeta{}.SwaggerDoc()
var swaggerDeleteOptionsDescriptions = metav1.DeleteOptions{}.SwaggerDoc()
var swaggerListDescriptions = metav1.List{}.SwaggerDoc()
var swaggerListOptionsDescriptions = metav1.ListOptions{}.SwaggerDoc()
var swaggerScaleDescriptions = autoscalingv1.Scale{}.SwaggerDoc()
var swaggerScaleSpecDescriptions = autoscalingv1.ScaleSpec{}.SwaggerDoc()
var swaggerScaleStatusDescriptions = autoscalingv1.ScaleStatus{}.SwaggerDoc()

// calcSwaggerEtag calculates an etag of the OpenAPI swagger (spec)
func calcSwaggerEtag(openAPISpec *spec.Swagger) (string, error) {
	specBytes, err := json.MarshalIndent(openAPISpec, " ", " ")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("\"%X\"", sha512.Sum512(specBytes)), nil
}

// pathParameters constructs the Parameter used by all paths in the CRD swagger (spec)
func pathParameters() []spec.Parameter {
	return []spec.Parameter{
		*spec.QueryParam("pretty").
			Typed("string", "").
			UniqueValues().
			WithDescription("If 'true', then the output is pretty printed."),
	}
}

// addDeleteOperationParameters add the body&query parameters used by a delete operation
func addDeleteOperationParameters(op *spec.Operation) *spec.Operation {
	return op.
		AddParam((&spec.Parameter{ParamProps: spec.ParamProps{Schema: spec.RefSchema(deleteOptionsSchemaRef)}}).
			Named("body").
			WithLocation("body").
			AsRequired()).
		AddParam(spec.QueryParam("dryRun").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerDeleteOptionsDescriptions["dryRun"])).
		AddParam(spec.QueryParam("gracePeriodSeconds").
			Typed("integer", "").
			UniqueValues().
			WithDescription(swaggerDeleteOptionsDescriptions["gracePeriodSeconds"])).
		AddParam(spec.QueryParam("orphanDependents").
			Typed("boolean", "").
			UniqueValues().
			WithDescription(swaggerDeleteOptionsDescriptions["orphanDependents"])).
		AddParam(spec.QueryParam("propagationPolicy").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerDeleteOptionsDescriptions["propagationPolicy"]))
}

// addCollectionOperationParameters adds the query parameters used by list and deletecollection
// operations
func addCollectionOperationParameters(op *spec.Operation) *spec.Operation {
	return op.
		AddParam(spec.QueryParam("continue").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["continue"])).
		AddParam(spec.QueryParam("fieldSelector").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["fieldSelector"])).
		AddParam(spec.QueryParam("includeUninitialized").
			Typed("boolean", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["includeUninitialized"])).
		AddParam(spec.QueryParam("labelSelector").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["labelSelector"])).
		AddParam(spec.QueryParam("limit").
			Typed("integer", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["limit"])).
		AddParam(spec.QueryParam("resourceVersion").
			Typed("string", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["resourceVersion"])).
		AddParam(spec.QueryParam("timeoutSeconds").
			Typed("integer", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["timeoutSeconds"])).
		AddParam(spec.QueryParam("watch").
			Typed("boolean", "").
			UniqueValues().
			WithDescription(swaggerListOptionsDescriptions["watch"]))
}

// okResponse constructs a 200 OK response with the input object schema reference
func okResponse(ref string) *spec.Response {
	return spec.NewResponse().
		WithDescription("OK").
		WithSchema(spec.RefSchema(ref))
}

// createdResponse constructs a 201 Created response with the input object schema reference
func createdResponse(ref string) *spec.Response {
	return spec.NewResponse().
		WithDescription("Created").
		WithSchema(spec.RefSchema(ref))
}

// acceptedResponse constructs a 202 Accepted response with the input object schema reference
func acceptedResponse(ref string) *spec.Response {
	return spec.NewResponse().
		WithDescription("Accepted").
		WithSchema(spec.RefSchema(ref))
}

// unauthorizedResponse constructs a 401 Unauthorized response
func unauthorizedResponse() *spec.Response {
	return spec.NewResponse().
		WithDescription("Unauthorized")
}

// scaleSchema constructs the OpenAPI schema for io.k8s.api.autoscaling.v1.Scale objects
// TODO(roycaihw): this is a hack to let apiExtension apiserver and generic kube-apiserver
// to have the same io.k8s.api.autoscaling.v1.Scale definition, so that aggregator server won't
// detect name conflict and create a duplicate io.k8s.api.autoscaling.v1.Scale_V2 schema
// when aggregating the openapi spec. It would be better if apiExtension apiserver serves
// identical definition through the same code path (using routes) as generic kube-apiserver.
func scaleSchema() *spec.Schema {
	s := new(spec.Schema).
		WithDescription(swaggerScaleDescriptions[""]).
		SetProperty("apiVersion", *spec.StringProperty().
			WithDescription(swaggerTypeMetaDescriptions["apiVersion"])).
		SetProperty("kind", *spec.StringProperty().
			WithDescription(swaggerTypeMetaDescriptions["kind"])).
		SetProperty("metadata", *spec.RefSchema(objectMetaSchemaRef).
			WithDescription(swaggerScaleDescriptions["metadata"])).
		SetProperty("spec", *spec.RefSchema(scaleSpecSchemaRef).
			WithDescription(swaggerScaleDescriptions["spec"])).
		SetProperty("status", *spec.RefSchema(scaleStatusSchemaRef).
			WithDescription(swaggerScaleDescriptions["status"]))

	s.AddExtension("x-kubernetes-group-version-kind", []map[string]string{
		{
			"group":   "autoscaling",
			"kind":    "Scale",
			"version": "v1",
		},
	})
	return s
}

// scaleSchema constructs the OpenAPI schema for io.k8s.api.autoscaling.v1.ScaleSpec objects
func scaleSpecSchema() *spec.Schema {
	return new(spec.Schema).
		WithDescription(swaggerScaleSpecDescriptions[""]).
		SetProperty("replicas", *spec.Int32Property().
			WithDescription(swaggerScaleSpecDescriptions["replicas"]))
}

// scaleSchema constructs the OpenAPI schema for io.k8s.api.autoscaling.v1.ScaleStatus objects
func scaleStatusSchema() *spec.Schema {
	return new(spec.Schema).
		WithDescription(swaggerScaleStatusDescriptions[""]).
		WithRequired("replicas").
		SetProperty("replicas", *spec.Int32Property().
			WithDescription(swaggerScaleStatusDescriptions["replicas"])).
		SetProperty("selector", *spec.StringProperty().
			WithDescription(swaggerScaleStatusDescriptions["selector"]))
}

// CustomResourceDefinitionOpenAPISpec constructs the OpenAPI spec (swagger) and calculates
// etag for a given CustomResourceDefinitionSpec.
// NOTE: in apiserver we general operates on internal types. We are using versioned (v1beta1)
// validation schema here because we need the json tags to properly marshal the object to
// JSON.
func CustomResourceDefinitionOpenAPISpec(crdSpec *apiextensions.CustomResourceDefinitionSpec, version string, validationSchema *apiextensions.CustomResourceValidation) (*spec.Swagger, string, error) {
	schema := &spec.Schema{}
	if validationSchema != nil && validationSchema.OpenAPIV3Schema != nil {
		var err error
		schema, err = ConvertJSONSchemaPropsToOpenAPIv2Schema(validationSchema.OpenAPIV3Schema)
		if err != nil {
			return nil, "", err
		}
	}
	crdSwaggerConstructor, err := NewSwaggerConstructor(schema, crdSpec, version)
	if err != nil {
		return nil, "", err
	}
	crdOpenAPISpec := crdSwaggerConstructor.ConstructCRDOpenAPISpec()
	etag, err := calcSwaggerEtag(crdOpenAPISpec)
	if err != nil {
		return nil, "", err
	}
	return crdOpenAPISpec, etag, nil
}
