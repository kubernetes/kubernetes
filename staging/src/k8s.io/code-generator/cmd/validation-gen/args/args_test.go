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

package args

import (
	"strings"
	"testing"
)

func TestValidateTagPrefix(t *testing.T) {
	cases := []struct {
		prefix  string
		wantErr bool
	}{
		{prefix: "k8s:"},
		{prefix: "xyz:"},
		{prefix: ""},
		{prefix: "a:b:"},
		{prefix: "my-project.v1:"},
		{prefix: "k8s", wantErr: true},
		{prefix: ":", wantErr: true},
		{prefix: "k8s::", wantErr: true},
		{prefix: "1k8s:", wantErr: true},
		{prefix: "k8s: ", wantErr: true},
		{prefix: "+k8s:", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.prefix, func(t *testing.T) {
			args := New()
			args.OutputFile = "zz_generated.validations.go"
			args.TagPrefix = tc.prefix
			err := args.Validate()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Validate() = nil, want error")
				}
				if !strings.Contains(err.Error(), "--tag-prefix") {
					t.Errorf("Validate() = %q, want it to mention --tag-prefix", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Validate() = %v, want nil", err)
			}
		})
	}
}
