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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const ExampleResourcePlural = "examples"

type Example struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`
	Spec              ExampleSpec   `json:"spec"`
	Status            ExampleStatus `json:"status,omitempty"`
}

type ExampleSpec struct {
	Foo string `json:"foo"`
	Bar bool   `json:"bar"`
}

type ExampleStatus struct {
	State   ExampleState `json:"state,omitempty"`
	Message string       `json:"message,omitempty"`
}

type ExampleState string

const (
	ExampleStateCreated   ExampleState = "Created"
	ExampleStateProcessed ExampleState = "Processed"
)

type ExampleList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []Example `json:"items"`
}
