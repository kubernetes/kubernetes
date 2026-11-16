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

package testing

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestPodReverseKeyFunc(t *testing.T) {
	for _, tc := range []struct {
		name       string
		namespaced bool
		namespace  string
	}{
		{name: "cluster scoped"},
		{name: "namespaced", namespaced: true, namespace: "ns"},
		{name: "empty namespace in storage tests", namespaced: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: tc.namespace}}
			var key string
			var err error
			if tc.namespaced {
				key, err = storage.NamespaceKeyFunc("/pods/", pod)
			} else {
				key, err = storage.NoNamespaceKeyFunc("/pods/", pod)
			}
			if err != nil {
				t.Fatal(err)
			}
			reverse := PodReverseKeyFunc("/pods/", tc.namespaced)
			name, namespace, err := reverse(key)
			if err != nil || name != pod.Name || namespace != pod.Namespace {
				t.Fatalf("reverse(%q) = (%q, %q, %v)", key, name, namespace, err)
			}
			for _, invalid := range []string{"/other/ns/pod", "/pods", "/pods/", "/pods/ns/", "/pods/ns/pod/extra"} {
				if _, _, err := reverse(invalid); err == nil {
					t.Errorf("expected error for key %q", invalid)
				}
			}
		})
	}
}
