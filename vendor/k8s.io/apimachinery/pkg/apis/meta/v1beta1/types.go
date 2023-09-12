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

// package v1beta1 is alpha objects from meta that will be introduced.
package v1beta1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Table is a tabular representation of a set of API resources. The server transforms the
// object into a set of preferred columns for quickly reviewing the objects.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +protobuf=false
type Table = v1.Table

// TableColumnDefinition contains information about a column returned in the Table.
// +protobuf=false
type TableColumnDefinition = v1.TableColumnDefinition

// TableRow is an individual row in a table.
// +protobuf=false
type TableRow = v1.TableRow

// TableRowCondition allows a row to be marked with additional information.
// +protobuf=false
type TableRowCondition = v1.TableRowCondition

type RowConditionType = v1.RowConditionType

type ConditionStatus = v1.ConditionStatus

type IncludeObjectPolicy = v1.IncludeObjectPolicy

// TableOptions are used when a Table is requested by the caller.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type TableOptions = v1.TableOptions

// PartialObjectMetadata is a generic representation of any object with ObjectMeta. It allows clients
// to get access to a particular ObjectMeta schema without knowing the details of the version.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type PartialObjectMetadata = v1.PartialObjectMetadata

// IMPORTANT: PartialObjectMetadataList has different protobuf field ids in v1beta1 than
// v1 because ListMeta was accidentally omitted prior to 1.15. Therefore this type must
// remain independent of v1.PartialObjectMetadataList to preserve mappings.

// PartialObjectMetadataList contains a list of objects containing only their metadata.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type PartialObjectMetadataList struct {
	v1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	v1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,2,opt,name=metadata"`

	// items contains each of the included items.
	Items []v1.PartialObjectMetadata `json:"items" protobuf:"bytes,1,rep,name=items"`
}

const (
	RowCompleted = v1.RowCompleted

	ConditionTrue    = v1.ConditionTrue
	ConditionFalse   = v1.ConditionFalse
	ConditionUnknown = v1.ConditionUnknown

	IncludeNone     = v1.IncludeNone
	IncludeMetadata = v1.IncludeMetadata
	IncludeObject   = v1.IncludeObject
)
