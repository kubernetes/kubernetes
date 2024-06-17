/*
Copyright 2020 The Kubernetes Authors.

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

package nonstructuralschema

import (
	"fmt"
	"reflect"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test_calculateCondition(t *testing.T) {
	tests := []struct {
		name string
		args *apiextensionsv1.CustomResourceDefinition
		want *apiextensionsv1.CustomResourceDefinitionCondition
	}{
		{
			name: "preserve unknown fields is false",
			args: &apiextensionsv1.CustomResourceDefinition{
				Spec: apiextensionsv1.CustomResourceDefinitionSpec{
					PreserveUnknownFields: false,
				},
			},
		},
		{
			name: "preserve unknown fields is true",
			args: &apiextensionsv1.CustomResourceDefinition{
				Spec: apiextensionsv1.CustomResourceDefinitionSpec{
					PreserveUnknownFields: true,
				},
			},
			want: &apiextensionsv1.CustomResourceDefinitionCondition{
				Type:   apiextensionsv1.NonStructuralSchema,
				Status: apiextensionsv1.ConditionTrue,
				Reason: "Violations",
				Message: field.Invalid(field.NewPath("spec", "preserveUnknownFields"),
					true,
					fmt.Sprint("must be false")).Error(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := calculateCondition(tt.args); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("calculateCondition() = %v, want %v", got, tt.want)
			}
		})
	}
}
