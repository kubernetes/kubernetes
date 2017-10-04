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

// TODO: Table does not generate to protobuf because of the interface{} - fix protobuf
//   generation to support a meta type that can accept any valid JSON.

// Table is a tabular representation of a set of API resources. The server transforms the
// object into a set of preferred columns for quickly reviewing the objects.
// +protobuf=false
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type Table struct {
	v1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	// +optional
	v1.ListMeta `json:"metadata,omitempty"`

	// columnDefinitions describes each column in the returned items array. The number of cells per row
	// will always match the number of column definitions.
	ColumnDefinitions []TableColumnDefinition `json:"columnDefinitions"`
	// rows is the list of items in the table.
	Rows []TableRow `json:"rows"`
}

// TableColumnDefinition contains information about a column returned in the Table.
// +protobuf=false
type TableColumnDefinition struct {
	// name is a human readable name for the column.
	Name string `json:"name"`
	// type is an OpenAPI type definition for this column.
	// See https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#data-types for more.
	Type string `json:"type"`
	// format is an optional OpenAPI type definition for this column. The 'name' format is applied
	// to the primary identifier column to assist in clients identifying column is the resource name.
	// See https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#data-types for more.
	Format string `json:"format"`
	// description is a human readable description of this column.
	Description string `json:"description"`
	// priority is an integer defining the relative importance of this column compared to others. Lower
	// numbers are considered higher priority. Columns that may be omitted in limited space scenarios
	// should be given a higher priority.
	Priority int32 `json:"priority"`
}

// TableRow is an individual row in a table.
// +protobuf=false
type TableRow struct {
	// cells will be as wide as headers and may contain strings, numbers, booleans, simple maps, or lists, or
	// null. See the type field of the column definition for a more detailed description.
	Cells []interface{} `json:"cells"`
	// conditions describe additional status of a row that are relevant for a human user.
	// +optional
	Conditions []TableRowCondition `json:"conditions,omitempty"`
	// This field contains the requested additional information about each object based on the includeObject
	// policy when requesting the Table. If "None", this field is empty, if "Object" this will be the
	// default serialization of the object for the current API version, and if "Metadata" (the default) will
	// contain the object metadata. Check the returned kind and apiVersion of the object before parsing.
	// +optional
	Object runtime.RawExtension `json:"object,omitempty"`
}

// TableRowCondition allows a row to be marked with additional information.
// +protobuf=false
type TableRowCondition struct {
	// Type of row condition.
	Type RowConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status ConditionStatus `json:"status"`
	// (brief) machine readable reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty"`
}

type RowConditionType string

// These are valid conditions of a row. This list is not exhaustive and new conditions may be
// included by other resources.
const (
	// RowCompleted means the underlying resource has reached completion and may be given less
	// visual priority than other resources.
	RowCompleted RowConditionType = "Completed"
)

type ConditionStatus string

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition.
// "ConditionFalse" means a resource is not in the condition. "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

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
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type TableOptions struct {
	v1.TypeMeta `json:",inline"`
	// includeObject decides whether to include each object along with its columnar information.
	// Specifying "None" will return no object, specifying "Object" will return the full object contents, and
	// specifying "Metadata" (the default) will return the object's metadata in the PartialObjectMetadata kind
	// in version v1alpha1 of the meta.k8s.io API group.
	IncludeObject IncludeObjectPolicy `json:"includeObject,omitempty" protobuf:"bytes,1,opt,name=includeObject,casttype=IncludeObjectPolicy"`
}

// PartialObjectMetadata is a generic representation of any object with ObjectMeta. It allows clients
// to get access to a particular ObjectMeta schema without knowing the details of the version.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type PartialObjectMetadata struct {
	v1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

// PartialObjectMetadataList contains a list of objects containing only their metadata
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type PartialObjectMetadataList struct {
	v1.TypeMeta `json:",inline"`

	// items contains each of the included items.
	Items []*PartialObjectMetadata `json:"items" protobuf:"bytes,1,rep,name=items"`
}
