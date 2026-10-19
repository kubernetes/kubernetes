/*
Copyright The Kubernetes Authors.

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

package types_test

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/types"
)

func TestNamespacedNameString(t *testing.T) {
	testCases := []struct {
		name string
		nn   types.NamespacedName
		want string
	}{
		{
			name: "namespace and name set",
			nn:   types.NamespacedName{Namespace: "kube-system", Name: "coredns"},
			want: "kube-system/coredns",
		},
		{
			name: "cluster-scoped object has empty namespace",
			nn:   types.NamespacedName{Name: "worker-node-1"},
			want: "/worker-node-1",
		},
		{
			name: "empty name",
			nn:   types.NamespacedName{Namespace: "default"},
			want: "default/",
		},
		{
			name: "zero value",
			nn:   types.NamespacedName{},
			want: "/",
		},
		{
			// Names are validated to RFC 1123 at the API entry point and can
			// never contain '/' in practice; this only pins down String()'s
			// literal behavior if that invariant were ever violated.
			name: "name itself contains the separator",
			nn:   types.NamespacedName{Namespace: "default", Name: "a/b"},
			want: "default/a/b",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.nn.String(); got != tc.want {
				t.Errorf("String() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestNamespacedNameMarshalLog(t *testing.T) {
	testCases := []struct {
		name     string
		nn       types.NamespacedName
		wantJSON string
	}{
		{
			name:     "namespace and name set",
			nn:       types.NamespacedName{Namespace: "kube-system", Name: "coredns"},
			wantJSON: `{"name":"coredns","namespace":"kube-system"}`,
		},
		{
			name:     "cluster-scoped object omits namespace",
			nn:       types.NamespacedName{Name: "worker-node-1"},
			wantJSON: `{"name":"worker-node-1"}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.nn.MarshalLog())
			if err != nil {
				t.Fatalf("unexpected error marshaling MarshalLog() result: %v", err)
			}
			if diff := cmp.Diff(tc.wantJSON, string(got)); diff != "" {
				t.Errorf("unexpected JSON (-want +got):\n%s", diff)
			}
		})
	}
}
