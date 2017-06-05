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

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Generating code from university_types.go file will generate storage and status REST endpoints for
// University.

// +genclient=true

// +k8s:openapi-gen=true
// +resource=universities
// +subresource=scale,University,Scale,ScaleUniversityREST
type University struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   UniversitySpec   `json:"spec,omitempty"`
	Status UniversityStatus `json:"status,omitempty"`
}

// UniversitySpec defines the desired state of University
type UniversitySpec struct {
	// faculty_size defines the desired faculty size of the university.  Defaults to 15.
	FacultySize int `json:"faculty_size,omitempty"`

	// max_students defines the maximum number of enrolled students.  Defaults to 300.
	// +optional
	MaxStudents *int `json:"max_students,omitempty"`
}

// UniversityStatus defines the observed state of University
type UniversityStatus struct {
	// enrolled_students is the number of currently enrolled students
	EnrolledStudents []string `json:"enrolled_students,omitempty"`

	// statusfield provides status information about University
	FacultyEmployed []string `json:"faculty_employed,omitempty"`
}

// GetDefaultingFunctions returns functions for defaulting v1beta1.University values
func (UniversitySchemeFns) GetDefaultingFunctions() []interface{} {
	return []interface{}{func(obj *University) {
		if obj.Spec.MaxStudents == nil {
			n := 15
			obj.Spec.MaxStudents = &n
		}
	}}
}

// +genclient=true

// +subresource-request
type Scale struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Faculty int `json:"faculty,omitempty"`
}
