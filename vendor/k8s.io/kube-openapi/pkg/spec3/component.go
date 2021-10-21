package spec3

import "k8s.io/kube-openapi/pkg/validation/spec"

// Components holds a set of reusable objects for different aspects of the OAS.
// All objects defined within the components object will have no effect on the API
// unless they are explicitly referenced from properties outside the components object.
//
// more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#componentsObject
type Components struct {
	// Schemas holds reusable Schema Objects
	Schemas map[string]*spec.Schema `json:"schemas,omitempty"`

	// the following fields are missing:
	// securitySchemes
	// responses	Map[string, Response Object | Reference Object]
	// parameters	Map[string, Parameter Object | Reference Object]
	// examples	Map[string, Example Object | Reference Object]
	// requestBodies	Map[string, Request Body Object | Reference Object]
	// headers	Map[string, Header Object | Reference Object]
	// links	Map[string, Link Object | Reference Object]
	// callbacks	Map[string, Callback Object | Reference Object]
	//
	// all fields are defined at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#componentsObject
}

// Schemas holds reusable Schema Objects
// type Schemas map[string]*spec.Schema
