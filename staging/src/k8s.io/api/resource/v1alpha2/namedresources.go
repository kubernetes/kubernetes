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

package v1alpha2

import (
	"k8s.io/apimachinery/pkg/api/resource"
)

// NamedResourcesResources is used in ResourceModel.
type NamedResourcesResources struct {
	// The list of all individual resources instances currently available.
	//
	// +listType=atomic
	Instances []NamedResourcesInstance `json:"instances" protobuf:"bytes,1,name=instances"`
}

// NamedResourcesInstance represents one individual hardware instance that can be selected based
// on its attributes.
type NamedResourcesInstance struct {
	// Name is unique identifier among all resource instances managed by
	// the driver on the node. It must be a DNS subdomain.
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// Attributes defines the attributes of this resource instance.
	// The name of each attribute must be unique.
	//
	// +listType=atomic
	// +optional
	Attributes []NamedResourcesAttribute `json:"attributes,omitempty" protobuf:"bytes,2,opt,name=attributes"`
}

// NamedResourcesAttribute is a combination of an attribute name and its value.
type NamedResourcesAttribute struct {
	// Name is unique identifier among all resource instances managed by
	// the driver on the node. It must be a DNS subdomain.
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	NamedResourcesAttributeValue `json:",inline" protobuf:"bytes,2,opt,name=attributeValue"`
}

// The Go field names below have a Value suffix to avoid a conflict between the
// field "String" and the corresponding method. That method is required.
// The Kubernetes API is defined without that suffix to keep it more natural.

// NamedResourcesAttributeValue must have one and only one field set.
type NamedResourcesAttributeValue struct {
	// QuantityValue is a quantity.
	QuantityValue *resource.Quantity `json:"quantity,omitempty" protobuf:"bytes,6,opt,name=quantity"`
	// BoolValue is a true/false value.
	BoolValue *bool `json:"bool,omitempty" protobuf:"bytes,2,opt,name=bool"`
	// IntValue is a 64-bit integer.
	IntValue *int64 `json:"int,omitempty" protobuf:"varint,7,opt,name=int"`
	// IntSliceValue is an array of 64-bit integers.
	IntSliceValue *NamedResourcesIntSlice `json:"intSlice,omitempty" protobuf:"varint,8,rep,name=intSlice"`
	// StringValue is a string.
	StringValue *string `json:"string,omitempty" protobuf:"bytes,5,opt,name=string"`
	// StringSliceValue is an array of strings.
	StringSliceValue *NamedResourcesStringSlice `json:"stringSlice,omitempty" protobuf:"bytes,9,rep,name=stringSlice"`
	// VersionValue is a semantic version according to semver.org spec 2.0.0.
	VersionValue *string `json:"version,omitempty" protobuf:"bytes,10,opt,name=version"`
}

// NamedResourcesIntSlice contains a slice of 64-bit integers.
type NamedResourcesIntSlice struct {
	// Ints is the slice of 64-bit integers.
	//
	// +listType=atomic
	Ints []int64 `json:"ints" protobuf:"bytes,1,opt,name=ints"`
}

// NamedResourcesStringSlice contains a slice of strings.
type NamedResourcesStringSlice struct {
	// Strings is the slice of strings.
	//
	// +listType=atomic
	Strings []string `json:"strings" protobuf:"bytes,1,opt,name=strings"`
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
	Selector string `json:"selector" protobuf:"bytes,1,name=selector"`
}

// NamedResourcesFilter is used in ResourceFilterModel.
type NamedResourcesFilter struct {
	// Selector is a CEL expression which must evaluate to true if a
	// resource instance is suitable. The language is as defined in
	// https://kubernetes.io/docs/reference/using-api/cel/
	//
	// In addition, for each type in NamedResourcesAttributeValue there is a map that
	// resolves to the corresponding value of the instance under evaluation.
	// For example:
	//
	//    attributes.quantity["a"].isGreaterThan(quantity("0")) &&
	//    attributes.stringslice["b"].isSorted()
	Selector string `json:"selector" protobuf:"bytes,1,name=selector"`
}

// NamedResourcesAllocationResult is used in AllocationResultModel.
type NamedResourcesAllocationResult struct {
	// Name is the name of the selected resource instance.
	Name string `json:"name" protobuf:"bytes,1,name=name"`
}
