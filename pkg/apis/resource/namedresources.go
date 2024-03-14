/*
Copyright 2024 The Kubernetes Authors.

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

import "k8s.io/apimachinery/pkg/api/resource"

// NamedResourcesResources is used in ResourceModel.
type NamedResourcesResources struct {
	// The list of all individual resources instances currently available.
	Instances []NamedResourcesInstance
}

// NamedResourcesInstance represents one individual hardware instance that can be selected based
// on its attributes.
type NamedResourcesInstance struct {
	// Name is unique identifier among all resource instances managed by
	// the driver on the node. It must be a DNS subdomain.
	Name string

	// Attributes defines the attributes of this resource instance.
	// The name of each attribute must be unique.
	Attributes []NamedResourcesAttribute
}

// NamedResourcesAttribute is a combination of an attribute name and its value.
type NamedResourcesAttribute struct {
	// Name is unique identifier among all resource instances managed by
	// the driver on the node. It must be a DNS subdomain.
	Name string

	NamedResourcesAttributeValue
}

// NamedResourcesAttributeValue must have one and only one field set.
type NamedResourcesAttributeValue struct {
	// QuantityValue is a quantity.
	QuantityValue *resource.Quantity
	// BoolValue is a true/false value.
	BoolValue *bool
	// IntValue is a 64-bit integer.
	IntValue *int64
	// IntSliceValue is an array of 64-bit integers.
	IntSliceValue *NamedResourcesIntSlice
	// StringValue is a string.
	StringValue *string
	// StringSliceValue is an array of strings.
	StringSliceValue *NamedResourcesStringSlice
	// VersionValue is a semantic version according to semver.org spec 2.0.0.
	VersionValue *string
}

// NamedResourcesIntSlice contains a slice of 64-bit integers.
type NamedResourcesIntSlice struct {
	// Ints is the slice of 64-bit integers.
	Ints []int64
}

// NamedResourcesStringSlice contains a slice of strings.
type NamedResourcesStringSlice struct {
	// Strings is the slice of strings.
	Strings []string
}

// NamedResourcesRequest is used in ResourceRequestModel.
type NamedResourcesRequest struct {
	// Selector is a CEL expression which must evaluate to true if a
	// resource instance is suitable. The language is as defined in
	// https://kubernetes.io/docs/reference/using-api/cel/
	//
	// In addition, for each type NamedResourcesin AttributeValue there is a map that
	// resolves to the corresponding value of the instance under evaluation.
	// For example:
	//
	//    attributes.quantity["a"].isGreaterThan(quantity("0")) &&
	//    attributes.stringslice["b"].isSorted()
	Selector string
}

// NamedResourcesFilter is used in ResourceFilterModel.
type NamedResourcesFilter struct {
	// Selector is a selector like the one in Request. It must be true for
	// a resource instance to be suitable for a claim using the class.
	Selector string
}

// NamedResourcesAllocationResult is used in AllocationResultModel.
type NamedResourcesAllocationResult struct {
	// Name is the name of the selected resource instance.
	Name string
}
