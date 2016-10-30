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

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestClusterGenerate(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *federationapi.Cluster
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &federationapi.Cluster{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ServerAddress: "",
						},
					},
					SecretRef: &v1.LocalObjectReference{
						Name: "",
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":          "foo",
				"serverAddress": "10.20.30.40",
			},
			expected: &federationapi.Cluster{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ServerAddress: "10.20.30.40",
						},
					},
					SecretRef: &v1.LocalObjectReference{
						Name: "",
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":   "foo",
				"secret": "foo-credentials",
			},
			expected: &federationapi.Cluster{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ServerAddress: "",
						},
					},
					SecretRef: &v1.LocalObjectReference{
						Name: "foo-credentials",
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":          "foo",
				"serverAddress": "https://foo.example.com",
				"secret":        "foo-credentials",
			},
			expected: &federationapi.Cluster{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ServerAddress: "https://foo.example.com",
						},
					},
					SecretRef: &v1.LocalObjectReference{
						Name: "foo-credentials",
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":          "bar-cluster",
				"serverAddress": "http://10.20.30.40",
				"secret":        "credentials",
			},
			expected: &federationapi.Cluster{
				ObjectMeta: v1.ObjectMeta{
					Name: "bar-cluster",
				},
				Spec: federationapi.ClusterSpec{
					ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
						{
							ServerAddress: "http://10.20.30.40",
						},
					},
					SecretRef: &v1.LocalObjectReference{
						Name: "credentials",
					},
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"serverAddress": "https://10.20.30.40",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"secret": "baz-credentials",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"serverAddress": "http://foo.example.com",
				"secret":        "foo-credentials",
			},
			expected:  nil,
			expectErr: true,
		},
	}
	generator := ClusterGeneratorV1Beta1{}
	for i, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*federationapi.Cluster), test.expected) {
			t.Errorf("\n[%d] want:\n%#v\n[%d] got:\n%#v", i, test.expected, i, obj.(*federationapi.Cluster))
		}
	}
}
