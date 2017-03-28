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

package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// FlunderList is a list of Flunder objects.
type FlunderList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []Flunder `json:"items" protobuf:"bytes,2,rep,name=items"`
}

type FlunderSpec struct {
}

type FlunderStatus struct {
}

// +genclient=true

type Flunder struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Spec   FlunderSpec   `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	Status FlunderStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}
