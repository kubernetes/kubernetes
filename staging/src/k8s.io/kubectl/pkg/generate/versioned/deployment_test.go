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

package versioned

import (
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDeploymentBasicGenerate(t *testing.T) {
	one := int32(1)
	tests := []struct {
		name           string
		deploymentName string
		images         []string
		expected       *appsv1.Deployment
		expectErr      bool
	}{
		{
			name:           "deployment name and images ok",
			deploymentName: "images-name-ok",
			images:         []string{"nn/image1", "registry/nn/image2", "nn/image3:tag", "nn/image4@digest", "nn/image5@sha256:digest"},
			expected: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "images-name-ok",
					Labels: map[string]string{"app": "images-name-ok"},
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: &one,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "images-name-ok"},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"app": "images-name-ok"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{Name: "image1", Image: "nn/image1"},
								{Name: "image2", Image: "registry/nn/image2"},
								{Name: "image3", Image: "nn/image3:tag"},
								{Name: "image4", Image: "nn/image4@digest"},
								{Name: "image5", Image: "nn/image5@sha256:digest"},
							},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			name:           "empty images",
			deploymentName: "images-empty",
			images:         []string{},
			expectErr:      true,
		},
		{
			name:           "no images",
			deploymentName: "images-missing",
			expectErr:      true,
		},
		{
			name:      "no deployment name and images",
			expectErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := &DeploymentBasicAppsGeneratorV1{
				BaseDeploymentGenerator{
					Name:   tt.deploymentName,
					Images: tt.images,
				},
			}
			obj, err := generator.StructuredGenerate()
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*appsv1.Deployment), tt.expected) {
				t.Errorf("test: %v\nexpected:\n%#v\nsaw:\n%#v", tt.name, tt.expected, obj.(*appsv1.Deployment))
			}
		})
	}
}
