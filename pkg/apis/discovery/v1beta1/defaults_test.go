/*
Copyright 2019 The Kubernetes Authors.

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

package v1beta1_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/discovery/install"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultEndpointPort(t *testing.T) {
	emptyStr := ""
	fooStr := "foo"
	protoTCP := v1.ProtocolTCP
	protoUDP := v1.ProtocolUDP

	tests := map[string]struct {
		original *discovery.EndpointSlice
		expected *discovery.EndpointSlice
	}{
		"should set appropriate defaults": {
			original: &discovery.EndpointSlice{Ports: []discovery.EndpointPort{{
				Port: utilpointer.Int32(80),
			}}},
			expected: &discovery.EndpointSlice{
				Ports: []discovery.EndpointPort{{
					Name:     &emptyStr,
					Protocol: &protoTCP,
					Port:     utilpointer.Int32(80),
				}},
			},
		},
		"should not overwrite values with defaults when set": {
			original: &discovery.EndpointSlice{
				Ports: []discovery.EndpointPort{{
					Name:     &fooStr,
					Protocol: &protoUDP,
				}},
			},
			expected: &discovery.EndpointSlice{
				Ports: []discovery.EndpointPort{{
					Name:     &fooStr,
					Protocol: &protoUDP,
				}},
			},
		},
	}

	for _, test := range tests {
		actual := test.original
		expected := test.expected
		legacyscheme.Scheme.Default(actual)
		if !apiequality.Semantic.DeepEqual(actual, expected) {
			t.Error(cmp.Diff(expected, actual))
		}
	}
}
