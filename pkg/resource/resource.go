/*
Copyright 2014 Google Inc. All rights reserved.

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

// ResourceType is used to describe a type of resource.
// This data structure should be immutable and shared between resources.
type ResourceType struct {
	// The name of this type of resource
	Name string

	// A human readable string to describe the resource type
	Description string

	// true if the resource is categorical.
	// Categorical resources are like NICs, CPU cores.
	Categorical bool

	// Examples of compressible resources are CPU, network bandwidth.
	Compressible bool
}

// An instance of a resource with a given type
type Resource struct {
	Type *ResourceType
	Name string
	// An optional field to specify additional values for a resource
	Attr map[string]string
}

// Resources whose values are categorical
type CategoricalResource struct {
	Resource
	Value []string
}

type QuantitativeResource struct {
	Resource
	Value uint64
}
