/*
Copyright 2016 The Kubernetes Authors.

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

package kubectl

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestDeploymentGenerate(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *extensions.Deployment
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name":  "foo",
				"image": []string{"abc/app:v4"},
			},
			expected: &extensions.Deployment{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"app": "foo"},
				},
				Spec: extensions.DeploymentSpec{
					Replicas: 1,
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					Template: api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{"app": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{{Name: "app", Image: "abc/app:v4"}},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":  "foo",
				"image": []string{"abc/app:v4", "zyx/ape"},
			},
			expected: &extensions.Deployment{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"app": "foo"},
				},
				Spec: extensions.DeploymentSpec{
					Replicas: 1,
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					Template: api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{"app": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{{Name: "app", Image: "abc/app:v4"},
								{Name: "ape", Image: "zyx/ape"}},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			params:    map[string]interface{}{},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name": 1,
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name": nil,
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":  "foo",
				"image": []string{},
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"NAME": "some_value",
			},
			expectErr: true,
		},
	}
	generator := DeploymentBasicGeneratorV1{}
	for index, test := range tests {
		t.Logf("running scenario %d", index)
		obj, err := generator.Generate(test.params)
		switch {
		case test.expectErr && err != nil:
			continue // loop, since there's no output to check
		case test.expectErr && err == nil:
			t.Errorf("expected error and didn't get one")
			continue // loop, no expected output object
		case !test.expectErr && err != nil:
			t.Errorf("unexpected error %v", err)
			continue // loop, no output object
		case !test.expectErr && err == nil:
			// do nothing and drop through
		}
		if !reflect.DeepEqual(obj.(*extensions.Deployment), test.expected) {
			t.Errorf("expected:\n%#v\nsaw:\n%#v", test.expected, obj.(*extensions.Deployment))
		}
	}
}
