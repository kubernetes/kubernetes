/*
Copyright 2017 The Kubernetes Authors.

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

// package v1alpha1 is alpha objects from meta that will be introduced.
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// Table is a tabular representation of a set of API resources. The server transforms the
// object into a set of preferred columns for quickly reviewing the objects.
// +protobuf=false
type Table struct {
	v1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	// +optional
	v1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// columnDefinitions describes each column in the returned items array. The number of cells per row
	// will always match the number of column definitions.
	ColumnDefinitions []TableColumnDefinitions `json:"columnDefinitions"`
	// rows is the list of items in the table.
	Rows []TableRow `json:"rows"`
}

// TableColumnDefinitions contains information about a column returned in the Table.
// +protobuf=false
type TableColumnDefinitions struct {
	// name is a human readable name for the column.
	Name string `json:"name"`
	// type is an OpenAPI type definition for this column.
	// See https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#data-types for more.
	Type string `json:"type"`
	// format is an optional OpenAPI type definition for this column.
	// See https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#data-types for more.
	Format string `json:"format"`
	// description is a human readable description of this column.
	Description string `json:"description"`
}

// TableRow is an individual row in a table.
// +protobuf=false
type TableRow struct {
	// cells will be as wide as headers and may contain strings, numbers, booleans, simple maps, or lists, or
	// null. See the type field of the column definition for a more detailed description.
	Cells []interface{} `json:"cells"`
	// This field contains the requested additional information about each object based on the includeObject
	// policy when requesting the Table. If "None", this field is empty, if "Object" this will be the
	// default serialization of the object for the current API version, and if "Metadata" (the default) will
	// contain the object metadata. Check the returned kind and apiVersion of the object before parsing.
	Object runtime.RawExtension `json:"object"`
}

// IncludeObjectPolicy controls which portion of the object is returned with a Table.
type IncludeObjectPolicy string

const (
	// IncludeNone returns no object.
	IncludeNone IncludeObjectPolicy = "None"
	// IncludeMetadata serializes the object containing only its metadata field.
	IncludeMetadata IncludeObjectPolicy = "Metadata"
	// IncludeObject contains the full object.
	IncludeObject IncludeObjectPolicy = "Object"
)

// TableOptions are used when a Table is requested by the caller.
type TableOptions struct {
	v1.TypeMeta `json:",inline"`
	// includeObject decides whether to include each object along with its columnar information.
	// Specifying "None" will return no object, specifying "Object" will return the full object contents, and
	// specifying "Metadata" (the default) will return the object's metadata in the PartialObjectMetadata kind
	// in version v1alpha1 of the meta.k8s.io API group.
	IncludeObject IncludeObjectPolicy `json:"includeObject,omitempty" protobuf:"bytes,1,opt,name=includeObject"`
}

// PartialObjectMetadata is a generic representation of any object with ObjectMeta. It allows clients
// to get access to a particular ObjectMeta schema without knowing the details of the version.
type PartialObjectMetadata struct {
	v1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	// +optional
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}
