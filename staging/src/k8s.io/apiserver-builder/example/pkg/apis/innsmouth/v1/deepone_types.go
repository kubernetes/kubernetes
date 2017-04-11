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

// +genclient=true

// +k8s:openapi-gen=true
// +resource=deepones
// DeepOne defines a resident of innsmouth
type DeepOne struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DeepOneSpec   `json:"spec,omitempty"`
	Status DeepOneStatus `json:"status,omitempty"`
}

// DeepOnesSpec defines the desired state of DeepOne
type DeepOneSpec struct {
	// fish_required defines the number of fish required by the DeepOne.
	FishRequired int `json:"fish_required,omitempty"`
}

// DeepOneStatus defines the observed state of DeepOne
type DeepOneStatus struct {
	// actual_fish defines the number of fish caught by the DeepOne.
	ActualFish int `json:"actual_fish,omitempty"`
}
