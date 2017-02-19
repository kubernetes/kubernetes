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
)

type TableList struct {
	v1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	// +optional
	v1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Headers []TableListHeader `json:"headers"`

	Items []TableListItem `json:"items"`
}

type TableListHeader struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
}

type TableListItem struct {
	// ObjectMeta?
	// Cells will be as wide as Headers and may contain strings, numbers, booleans, simple maps, or lists
	Cells []interface{} `json:"cells"`
}
