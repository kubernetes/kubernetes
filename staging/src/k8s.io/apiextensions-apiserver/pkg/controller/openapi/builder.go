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

package openapi

import (
	"fmt"
	"strings"
	"sync"

	restful "github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints"
	"k8s.io/apiserver/pkg/endpoints/openapi"
	openapibuilder "k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/util"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	generatedopenapi "k8s.io/apiextensions-apiserver/pkg/generated/openapi"
)

const (
	// Reference and Go types for built-in metadata
	objectMetaSchemaRef = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"
	listMetaType        = "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta"
	typeMetaType        = "k8s.io/apimachinery/pkg/apis/meta/v1.TypeMeta"

	definitionPrefix = "#/definitions/"
)

var (
	swaggerPartialObjectMetadataDescriptions = metav1beta1.PartialObjectMetadata{}.SwaggerDoc()
)

var definitions map[string]common.OpenAPIDefinition
var buildDefinitions sync.Once
var namer *openapi.DefinitionNamer

// BuildSwagger builds swagger for the given crd in the given version
func BuildSwagger(crd *apiextensions.CustomResourceDefinition, version string) (*spec.Swagger, error) {
	var schema *spec.Schema
	s, err := apiextensions.GetSchemaForVersion(crd, version)
	if err != nil {
		return nil, err
	}
	if s != nil && s.OpenAPIV3Schema != nil {
		schema, err = ConvertJSONSchemaPropsToOpenAPIv2Schema(s.OpenAPIV3Schema)
		if err != nil {
			return nil, err
		}
	}
	b := newBuilder(crd, version, schema)

	// Sample response types for building web service
	sample := &CRDCanonicalTypeNamer{
		group:   b.group,
		version: b.version,
		kind:    b.kind,
	}
	sampleList := &CRDCanonicalTypeNamer{
		group:   b.group,
		version: b.version,
		kind:    b.listKind,
	}
	status := &metav1.Status{}
	// patch := &metav1.Patch{}
	// scale := &v1.Scale{}

	actions := []*endpoints.Action{}
	params := []*restful.Parameter{}

	root := fmt.Sprintf("/apis/%s/%s/%s", b.group, b.version, b.plural)
	if b.namespaced {
		actions = append(actions, endpoints.NewAction("LIST", root, params, true))
		root = fmt.Sprintf("/apis/%s/%s/namespaces/{namespace}/%s", b.group, b.version, b.plural)
	}
	actions = append(actions, endpoints.NewAction("LIST", root, params, false))
	actions = append(actions, endpoints.NewAction("POST", root, params, false))
	actions = append(actions, endpoints.NewAction("DELETECOLLECTION", root, params, false))

	actions = append(actions, endpoints.NewAction("GET", root, params, false))
	actions = append(actions, endpoints.NewAction("PUT", root, params, false))
	actions = append(actions, endpoints.NewAction("DELETE", root, params, false))
	actions = append(actions, endpoints.NewAction("PATCH", root, params, false))

	subresources, err := apiextensions.GetSubresourcesForVersion(crd, version)
	if err != nil {
		return nil, err
	}
	if subresources != nil && subresources.Status != nil {
		// TODO(roycaihw): distinguish subresources
		root := root + "/{name}/status"
		actions = append(actions, endpoints.NewAction("GET", root, params, false))
		actions = append(actions, endpoints.NewAction("PUT", root, params, false))
		actions = append(actions, endpoints.NewAction("PATCH", root, params, false))
	}
	if subresources != nil && subresources.Scale != nil {
		// TODO(roycaihw): distinguish subresources
		root := root + "/{name}/scale"
		actions = append(actions, endpoints.NewAction("GET", root, params, false))
		actions = append(actions, endpoints.NewAction("PUT", root, params, false))
		actions = append(actions, endpoints.NewAction("PATCH", root, params, false))
	}

	// complete actions
	for _, action := range actions {
		action = action.AssignProducedMIMETypes([]string{"application/json", "application/yaml"}, nil, nil)

		action = action.AssignConsumedMIMETypes()

		action = action.AssignWriteSample(sample, sampleList, status, nil)

		action = action.AssignReadSample(sample, &metav1.DeleteOptions{})

		// TODO(roycaihw): generalize parameters without creater
		// action, err = action.AssignParameters(a.group.Creater, a.group.Typer, optionsExternalVersion, a.group.GroupVersion, storage)
		// if err != nil {
		// 	return nil, err
		// }
	}

	ws := &restful.WebService{}
	var routes []*restful.RouteBuilder
	for _, action := range actions {
		doc, err := endpoints.DocumentationAction(action, b.kind, "", false, true, true)
		if err != nil {
			return nil, err
		}
		routes, err = endpoints.RegisterActionsToWebService(action, ws, b.namespaced, true, b.kind, "", endpoints.ToRESTMethod[action.Verb], doc, endpoints.ToOperationPrefix[action.Verb])
		if err != nil {
			return nil, err
		}
		for _, route := range routes {
			route.Metadata(endpoints.ROUTE_META_GVK, metav1.GroupVersionKind{
				Group:   b.group,
				Version: b.version,
				Kind:    b.kind,
			})
			route.Metadata(endpoints.ROUTE_META_ACTION, strings.ToLower(action.Verb))
			b.ws.Route(route)
		}
	}

	openAPISpec, err := openapibuilder.BuildOpenAPISpec([]*restful.WebService{b.ws}, b.getOpenAPIConfig())
	if err != nil {
		return nil, err
	}

	return openAPISpec, nil
}

// Implements CanonicalTypeNamer
var _ = util.OpenAPICanonicalTypeNamer(&CRDCanonicalTypeNamer{})

// CRDCanonicalTypeNamer implements CanonicalTypeNamer interface for CRDs to
// seed kube-openapi canonical type name without Go types
type CRDCanonicalTypeNamer struct {
	group   string
	version string
	kind    string
}

// OpenAPICanonicalTypeName returns canonical type name for given CRD
func (c *CRDCanonicalTypeNamer) OpenAPICanonicalTypeName() string {
	return fmt.Sprintf("%s/%s.%s", c.group, c.version, c.kind)
}

// builder contains validation schema and basic naming information for a CRD in
// one version. The builder works to build a WebService that kube-openapi can
// consume.
type builder struct {
	schema     *spec.Schema
	listSchema *spec.Schema
	ws         *restful.WebService

	group    string
	version  string
	kind     string
	listKind string
	plural   string

	namespaced bool
}

// buildKubeNative builds input schema with Kubernetes' native object meta, type meta and
// extensions
func (b *builder) buildKubeNative(schema *spec.Schema) *spec.Schema {
	// only add properties if we have a schema. Otherwise, kubectl would (wrongly) assume additionalProperties=false
	// and forbid anything outside of apiVersion, kind and metadata. We have to fix kubectl to stop doing this, e.g. by
	// adding additionalProperties=true support to explicitly allow additional fields.
	// TODO: fix kubectl to understand additionalProperties=true
	if schema == nil {
		schema = &spec.Schema{
			SchemaProps: spec.SchemaProps{Type: []string{"object"}},
		}
		// no, we cannot add more properties here, not even TypeMeta/ObjectMeta because kubectl will complain about
		// unknown fields for anything else.
	} else {
		schema.SetProperty("metadata", *spec.RefSchema(objectMetaSchemaRef).
			WithDescription(swaggerPartialObjectMetadataDescriptions["metadata"]))
		addTypeMetaProperties(schema)
	}
	schema.AddExtension(endpoints.ROUTE_META_GVK, []interface{}{
		map[string]interface{}{
			"group":   b.group,
			"version": b.version,
			"kind":    b.kind,
		},
	})

	return schema
}

// getDefinition gets definition for given Kubernetes type. This function is extracted from
// kube-openapi builder logic
func getDefinition(name string) spec.Schema {
	buildDefinitions.Do(buildDefinitionsFunc)
	return definitions[name].Schema
}

func buildDefinitionsFunc() {
	namer = openapi.NewDefinitionNamer(runtime.NewScheme())
	definitions = generatedopenapi.GetOpenAPIDefinitions(func(name string) spec.Ref {
		defName, _ := namer.GetDefinitionName(name)
		return spec.MustCreateRef(definitionPrefix + common.EscapeJsonPointer(defName))
	})
}

// addTypeMetaProperties adds Kubernetes-specific type meta properties to input schema:
//     apiVersion and kind
func addTypeMetaProperties(s *spec.Schema) {
	s.SetProperty("apiVersion", getDefinition(typeMetaType).SchemaProps.Properties["apiVersion"])
	s.SetProperty("kind", getDefinition(typeMetaType).SchemaProps.Properties["kind"])
}

// buildListSchema builds the list kind schema for the CRD
func (b *builder) buildListSchema() *spec.Schema {
	name := definitionPrefix + util.ToRESTFriendlyName(fmt.Sprintf("%s/%s/%s", b.group, b.version, b.kind))
	doc := fmt.Sprintf("List of %s. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md", b.plural)
	s := new(spec.Schema).WithDescription(fmt.Sprintf("%s is a list of %s", b.listKind, b.kind)).
		WithRequired("items").
		SetProperty("items", *spec.ArrayProperty(spec.RefSchema(name)).WithDescription(doc)).
		SetProperty("metadata", getDefinition(listMetaType))
	addTypeMetaProperties(s)
	s.AddExtension(endpoints.ROUTE_META_GVK, []map[string]string{
		{
			"group":   b.group,
			"version": b.version,
			"kind":    b.listKind,
		},
	})
	return s
}

// getOpenAPIConfig builds config which wires up generated definitions for kube-openapi to consume
func (b *builder) getOpenAPIConfig() *common.Config {
	return &common.Config{
		ProtocolList: []string{"https"},
		Info: &spec.Info{
			InfoProps: spec.InfoProps{
				Title:   "Kubernetes CRD Swagger",
				Version: "v0.1.0",
			},
		},
		CommonResponses: map[int]spec.Response{
			401: {
				ResponseProps: spec.ResponseProps{
					Description: "Unauthorized",
				},
			},
		},
		GetOperationIDAndTags: openapi.GetOperationIDAndTags,
		GetDefinitionName: func(name string) (string, spec.Extensions) {
			buildDefinitions.Do(buildDefinitionsFunc)
			return namer.GetDefinitionName(name)
		},
		GetDefinitions: func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
			def := generatedopenapi.GetOpenAPIDefinitions(ref)
			def[fmt.Sprintf("%s/%s.%s", b.group, b.version, b.kind)] = common.OpenAPIDefinition{
				Schema: *b.schema,
			}
			def[fmt.Sprintf("%s/%s.%s", b.group, b.version, b.listKind)] = common.OpenAPIDefinition{
				Schema: *b.listSchema,
			}
			return def
		},
	}
}

func newBuilder(crd *apiextensions.CustomResourceDefinition, version string, schema *spec.Schema) *builder {
	b := &builder{
		schema: &spec.Schema{
			SchemaProps: spec.SchemaProps{Type: []string{"object"}},
		},
		listSchema: &spec.Schema{},
		ws:         &restful.WebService{},

		group:    crd.Spec.Group,
		version:  version,
		kind:     crd.Spec.Names.Kind,
		listKind: crd.Spec.Names.ListKind,
		plural:   crd.Spec.Names.Plural,
	}
	if crd.Spec.Scope == apiextensions.NamespaceScoped {
		b.namespaced = true
	}

	// Pre-build schema with Kubernetes native properties
	b.schema = b.buildKubeNative(schema)
	b.listSchema = b.buildListSchema()

	return b
}
