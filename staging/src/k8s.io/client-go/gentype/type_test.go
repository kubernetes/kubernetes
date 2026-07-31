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

package gentype

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPrefersProtobuf(t *testing.T) {
	for _, tc := range []struct {
		name  string
		set   bool
		value string
		want  bool
	}{
		{name: "unset defaults to true", want: true},
		{name: "true", set: true, value: "true", want: true},
		{name: "false", set: true, value: "false", want: false},
		{name: "1", set: true, value: "1", want: true},
		{name: "0", set: true, value: "0", want: false},
		{name: "invalid falls back to true", set: true, value: "bogus", want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv(prefersProtobufEnvVar, tc.value)
			} else {
				t.Setenv(prefersProtobufEnvVar, "")
			}

			c := &Client[*metav1.PartialObjectMetadata]{}
			PrefersProtobuf[*metav1.PartialObjectMetadata]()(c)
			if c.prefersProtobuf != tc.want {
				t.Errorf("prefersProtobuf = %v, want %v", c.prefersProtobuf, tc.want)
			}
		})
	}
}
