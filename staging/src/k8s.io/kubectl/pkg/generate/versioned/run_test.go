/*
Copyright 2014 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGeneratePod(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *v1.Pod
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test2",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"env":   []string{"a", "c"},
			},

			expected:  nil,
			expectErr: true,
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"env":               []string{"a=b", "c=d"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullAlways,
							Env: []v1.EnvVar{
								{
									Name:  "a",
									Value: "b",
								},
								{
									Name:  "c",
									Value: "d",
								},
							},
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test4",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 80,
								},
							},
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test5",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"port":     "80",
				"hostport": "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 80,
									HostPort:      80,
								},
							},
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test6",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"hostport": "80",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			name: "test7",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test8",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"stdin":    "true",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:      "foo",
							Image:     "someimage",
							Stdin:     true,
							StdinOnce: true,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test9",
			params: map[string]interface{}{
				"name":             "foo",
				"image":            "someimage",
				"replicas":         "1",
				"labels":           "foo=bar,baz=blah",
				"stdin":            "true",
				"leave-stdin-open": "true",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:      "foo",
							Image:     "someimage",
							Stdin:     true,
							StdinOnce: false,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test10: privileged mode",
			params: map[string]interface{}{
				"name":       "foo",
				"image":      "someimage",
				"replicas":   "1",
				"privileged": "true",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							SecurityContext: securityContextWithPrivilege(true),
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test11: check annotations",
			params: map[string]interface{}{
				"name":        "foo",
				"image":       "someimage",
				"replicas":    "1",
				"labels":      "foo=bar,baz=blah",
				"annotations": []string{"foo=bar1", "baz=blah1"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "foo",
					Labels:      map[string]string{"foo": "bar", "baz": "blah"},
					Annotations: map[string]string{"foo": "bar1", "baz": "blah1"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}
	generator := BasicPod{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.Pod), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*v1.Pod))
			}
		})
	}
}

func TestParseEnv(t *testing.T) {
	tests := []struct {
		name      string
		envArray  []string
		expected  []v1.EnvVar
		expectErr bool
		test      string
	}{
		{
			name: "test1",
			envArray: []string{
				"THIS_ENV=isOK",
				"this.dotted.env=isOKToo",
				"HAS_COMMAS=foo,bar",
				"HAS_EQUALS=jJnro54iUu75xNy==",
			},
			expected: []v1.EnvVar{
				{
					Name:  "THIS_ENV",
					Value: "isOK",
				},
				{
					Name:  "this.dotted.env",
					Value: "isOKToo",
				},
				{
					Name:  "HAS_COMMAS",
					Value: "foo,bar",
				},
				{
					Name:  "HAS_EQUALS",
					Value: "jJnro54iUu75xNy==",
				},
			},
			expectErr: false,
			test:      "test case 1",
		},
		{
			name: "test2",
			envArray: []string{
				"WITH_OUT_EQUALS",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 2",
		},
		{
			name: "test3",
			envArray: []string{
				"WITH_OUT_VALUES=",
			},
			expected: []v1.EnvVar{
				{
					Name:  "WITH_OUT_VALUES",
					Value: "",
				},
			},
			expectErr: false,
			test:      "test case 3",
		},
		{
			name: "test4",
			envArray: []string{
				"=WITH_OUT_NAME",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			envs, err := parseEnvs(tt.envArray)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v (%s)", err, tt.test)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(envs, tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v (%s)", tt.expected, envs, tt.test)
			}
		})
	}
}

func securityContextWithPrivilege(privileged bool) *v1.SecurityContext {
	return &v1.SecurityContext{
		Privileged: &privileged,
	}
}
