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
	"fmt"
	"strings"

	"github.com/go-openapi/spec"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
)

// ResourceKind determines the scope of an API object: if it's the parent resource,
// scale subresource or status subresource.
type ResourceKind string

const (
	// Resource specifies an object of custom resource kind
	Resource ResourceKind = "Resource"
	// Scale specifies an object of custom resource's scale subresource kind
	Scale ResourceKind = "Scale"
	// Status specifies an object of custom resource's status subresource kind
	Status ResourceKind = "Status"

	scaleSchemaRef    = "#/definitions/io.k8s.api.autoscaling.v1.Scale"
	statusSchemaRef   = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.Status"
	listMetaSchemaRef = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ListMeta"
	patchSchemaRef    = "#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.Patch"
)

// SwaggerConstructor takes in CRD OpenAPI schema and CustomResourceDefinitionSpec, and
// constructs the OpenAPI swagger that an apiserver serves.
type SwaggerConstructor struct {
	// schema is the CRD's OpenAPI v2 schema
	schema *spec.Schema

	status, scale bool

	group    string
	version  string
	kind     string
	listKind string
	plural   string
	scope    apiextensions.ResourceScope
}

// NewSwaggerConstructor creates a new SwaggerConstructor using the CRD OpenAPI schema
// and CustomResourceDefinitionSpec
func NewSwaggerConstructor(schema *spec.Schema, crdSpec *apiextensions.CustomResourceDefinitionSpec, version string) (*SwaggerConstructor, error) {
	ret := &SwaggerConstructor{
		schema:   schema,
		group:    crdSpec.Group,
		version:  version,
		kind:     crdSpec.Names.Kind,
		listKind: crdSpec.Names.ListKind,
		plural:   crdSpec.Names.Plural,
		scope:    crdSpec.Scope,
	}

	sub, err := getSubresourcesForVersion(crdSpec, version)
	if err != nil {
		return nil, err
	}
	if sub != nil {
		ret.status = sub.Status != nil
		ret.scale = sub.Scale != nil
	}

	return ret, nil
}

// ConstructCRDOpenAPISpec constructs the complete OpenAPI swagger (spec).
func (c *SwaggerConstructor) ConstructCRDOpenAPISpec() *spec.Swagger {
	basePath := fmt.Sprintf("/apis/%s/%s/%s", c.group, c.version, c.plural)
	if c.scope == apiextensions.NamespaceScoped {
		basePath = fmt.Sprintf("/apis/%s/%s/namespaces/{namespace}/%s", c.group, c.version, c.plural)
	}

	model := fmt.Sprintf("%s.%s.%s", c.group, c.version, c.kind)
	listModel := fmt.Sprintf("%s.%s.%s", c.group, c.version, c.listKind)

	var schema spec.Schema
	if c.schema != nil {
		schema = *c.schema
	}

	ret := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					basePath: {
						PathItemProps: spec.PathItemProps{
							Get:        c.listOperation(),
							Post:       c.createOperation(),
							Delete:     c.deleteCollectionOperation(),
							Parameters: pathParameters(),
						},
					},
					fmt.Sprintf("%s/{name}", basePath): {
						PathItemProps: spec.PathItemProps{
							Get:        c.readOperation(Resource),
							Put:        c.replaceOperation(Resource),
							Delete:     c.deleteOperation(),
							Patch:      c.patchOperation(Resource),
							Parameters: pathParameters(),
						},
					},
				},
			},
			Definitions: spec.Definitions{
				model:     schema,
				listModel: *c.listSchema(),
			},
		},
	}

	if c.status {
		ret.SwaggerProps.Paths.Paths[fmt.Sprintf("%s/{name}/status", basePath)] = spec.PathItem{
			PathItemProps: spec.PathItemProps{
				Get:        c.readOperation(Status),
				Put:        c.replaceOperation(Status),
				Patch:      c.patchOperation(Status),
				Parameters: pathParameters(),
			},
		}
	}

	if c.scale {
		ret.SwaggerProps.Paths.Paths[fmt.Sprintf("%s/{name}/scale", basePath)] = spec.PathItem{
			PathItemProps: spec.PathItemProps{
				Get:        c.readOperation(Scale),
				Put:        c.replaceOperation(Scale),
				Patch:      c.patchOperation(Scale),
				Parameters: pathParameters(),
			},
		}
		// TODO(roycaihw): this is a hack to let apiExtension apiserver and generic kube-apiserver
		// to have the same io.k8s.api.autoscaling.v1.Scale definition, so that aggregator server won't
		// detect name conflict and create a duplicate io.k8s.api.autoscaling.v1.Scale_V2 schema
		// when aggregating the openapi spec. It would be better if apiExtension apiserver serves
		// identical definition through the same code path (using routes) as generic kube-apiserver.
		ret.SwaggerProps.Definitions["io.k8s.api.autoscaling.v1.Scale"] = *scaleSchema()
		ret.SwaggerProps.Definitions["io.k8s.api.autoscaling.v1.ScaleSpec"] = *scaleSpecSchema()
		ret.SwaggerProps.Definitions["io.k8s.api.autoscaling.v1.ScaleStatus"] = *scaleStatusSchema()
	}

	return ret
}

// baseOperation initializes a base operation that all operations build upon
func (c *SwaggerConstructor) baseOperation(kind ResourceKind, action string) *spec.Operation {
	op := spec.NewOperation(c.operationID(kind, action)).
		WithConsumes(
			"application/json",
			"application/yaml",
		).
		WithProduces(
			"application/json",
			"application/yaml",
		).
		WithTags(fmt.Sprintf("%s_%s", c.group, c.version)).
		RespondsWith(401, unauthorizedResponse())
	op.Schemes = []string{"https"}
	op.AddExtension("x-kubernetes-action", action)

	// Add x-kubernetes-group-version-kind extension
	// For CRD scale subresource, the x-kubernetes-group-version-kind is autoscaling.v1.Scale
	switch kind {
	case Scale:
		op.AddExtension("x-kubernetes-group-version-kind", []map[string]string{
			{
				"group":   "autoscaling",
				"kind":    "Scale",
				"version": "v1",
			},
		})
	default:
		op.AddExtension("x-kubernetes-group-version-kind", []map[string]string{
			{
				"group":   c.group,
				"kind":    c.kind,
				"version": c.version,
			},
		})
	}
	return op
}

// listOperation constructs a list operation for a CRD
func (c *SwaggerConstructor) listOperation() *spec.Operation {
	op := c.baseOperation(Resource, "list").
		WithDescription(fmt.Sprintf("list or watch objects of kind %s", c.kind)).
		RespondsWith(200, okResponse(fmt.Sprintf("#/definitions/%s.%s.%s", c.group, c.version, c.listKind)))
	return addCollectionOperationParameters(op)
}

// createOperation constructs a create operation for a CRD
func (c *SwaggerConstructor) createOperation() *spec.Operation {
	ref := c.constructSchemaRef(Resource)
	return c.baseOperation(Resource, "create").
		WithDescription(fmt.Sprintf("create a %s", c.kind)).
		RespondsWith(200, okResponse(ref)).
		RespondsWith(201, createdResponse(ref)).
		RespondsWith(202, acceptedResponse(ref)).
		AddParam((&spec.Parameter{ParamProps: spec.ParamProps{Schema: spec.RefSchema(ref)}}).
			Named("body").
			WithLocation("body").
			AsRequired())
}

// deleteOperation constructs a delete operation for a CRD
func (c *SwaggerConstructor) deleteOperation() *spec.Operation {
	op := c.baseOperation(Resource, "delete").
		WithDescription(fmt.Sprintf("delete a %s", c.kind)).
		RespondsWith(200, okResponse(statusSchemaRef)).
		RespondsWith(202, acceptedResponse(statusSchemaRef))
	return addDeleteOperationParameters(op)
}

// deleteCollectionOperation constructs a deletecollection operation for a CRD
func (c *SwaggerConstructor) deleteCollectionOperation() *spec.Operation {
	op := c.baseOperation(Resource, "deletecollection").
		WithDescription(fmt.Sprintf("delete collection of %s", c.kind))
	return addCollectionOperationParameters(op)
}

// readOperation constructs a read operation for a CRD, CRD's scale subresource
// or CRD's status subresource
func (c *SwaggerConstructor) readOperation(kind ResourceKind) *spec.Operation {
	ref := c.constructSchemaRef(kind)
	action := "read"
	return c.baseOperation(kind, action).
		WithDescription(c.constructDescription(kind, action)).
		RespondsWith(200, okResponse(ref))
}

// replaceOperation constructs a replace operation for a CRD, CRD's scale subresource
// or CRD's status subresource
func (c *SwaggerConstructor) replaceOperation(kind ResourceKind) *spec.Operation {
	ref := c.constructSchemaRef(kind)
	action := "replace"
	return c.baseOperation(kind, action).
		WithDescription(c.constructDescription(kind, action)).
		RespondsWith(200, okResponse(ref)).
		RespondsWith(201, createdResponse(ref)).
		AddParam((&spec.Parameter{ParamProps: spec.ParamProps{Schema: spec.RefSchema(ref)}}).
			Named("body").
			WithLocation("body").
			AsRequired())
}

// patchOperation constructs a patch operation for a CRD, CRD's scale subresource
// or CRD's status subresource
func (c *SwaggerConstructor) patchOperation(kind ResourceKind) *spec.Operation {
	ref := c.constructSchemaRef(kind)
	action := "patch"
	return c.baseOperation(kind, action).
		WithDescription(c.constructDescription(kind, "partially update")).
		RespondsWith(200, okResponse(ref)).
		AddParam((&spec.Parameter{ParamProps: spec.ParamProps{Schema: spec.RefSchema(patchSchemaRef)}}).
			Named("body").
			WithLocation("body").
			AsRequired())
}

// listSchema constructs the OpenAPI schema for a list of CRD objects
func (c *SwaggerConstructor) listSchema() *spec.Schema {
	ref := c.constructSchemaRef(Resource)
	s := new(spec.Schema).
		WithDescription(fmt.Sprintf("%s is a list of %s", c.listKind, c.kind)).
		WithRequired("items").
		SetProperty("apiVersion", *spec.StringProperty().
			WithDescription(swaggerTypeMetaDescriptions["apiVersion"])).
		SetProperty("items", *spec.ArrayProperty(spec.RefSchema(ref)).
			WithDescription(fmt.Sprintf("List of %s. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md", c.plural))).
		SetProperty("kind", *spec.StringProperty().
			WithDescription(swaggerTypeMetaDescriptions["kind"])).
		SetProperty("metadata", *spec.RefSchema(listMetaSchemaRef).
			WithDescription(swaggerListDescriptions["metadata"]))
	s.AddExtension("x-kubernetes-group-version-kind", map[string]string{
		"group":   c.group,
		"kind":    c.listKind,
		"version": c.version,
	})
	return s
}

// operationID generates the ID for an operation
func (c *SwaggerConstructor) operationID(kind ResourceKind, action string) string {
	var collectionTemplate, namespacedTemplate, subresourceTemplate string
	if action == "deletecollection" {
		action = "delete"
		collectionTemplate = "Collection"
	}
	if c.scope == apiextensions.NamespaceScoped {
		namespacedTemplate = "Namespaced"
	}
	switch kind {
	case Status:
		subresourceTemplate = "Status"
	case Scale:
		subresourceTemplate = "Scale"
	}
	return fmt.Sprintf("%s%s%s%s%s%s%s", action, strings.Title(c.group), strings.Title(c.version), collectionTemplate, namespacedTemplate, c.kind, subresourceTemplate)
}

// constructSchemaRef generates a reference to an object schema, based on the ResourceKind
// used by an operation
func (c *SwaggerConstructor) constructSchemaRef(kind ResourceKind) string {
	var ref string
	switch kind {
	case Scale:
		ref = scaleSchemaRef
	default:
		ref = fmt.Sprintf("#/definitions/%s.%s.%s", c.group, c.version, c.kind)
	}
	return ref
}

// constructDescription generates a description for READ, REPLACE and PATCH operations, based on
// the ResourceKind used by the operation
func (c *SwaggerConstructor) constructDescription(kind ResourceKind, action string) string {
	var descriptionTemplate string
	switch kind {
	case Status:
		descriptionTemplate = "status of "
	case Scale:
		descriptionTemplate = "scale of "
	}
	return fmt.Sprintf("%s %sthe specified %s", action, descriptionTemplate, c.kind)
}

// hasPerVersionSubresources returns true if a CRD spec uses per-version subresources.
func hasPerVersionSubresources(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Subresources != nil {
			return true
		}
	}
	return false
}

// getSubresourcesForVersion returns the subresources for given version in given CRD spec.
func getSubresourcesForVersion(spec *apiextensions.CustomResourceDefinitionSpec, version string) (*apiextensions.CustomResourceSubresources, error) {
	if !hasPerVersionSubresources(spec.Versions) {
		return spec.Subresources, nil
	}
	if spec.Subresources != nil {
		return nil, fmt.Errorf("malformed CustomResourceDefinitionSpec version %s: top-level and per-version subresources must be mutual exclusive", version)
	}
	for _, v := range spec.Versions {
		if version == v.Name {
			return v.Subresources, nil
		}
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinitionSpec", version)
}
