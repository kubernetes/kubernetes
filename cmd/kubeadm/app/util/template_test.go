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

package util

import (
	"testing"
)

const (
	validTmpl    = "image: {{ .ImageRepository }}/pause:3.7"
	validTmplOut = "image: registry.k8s.io/pause:3.7"
	doNothing    = "image: registry.k8s.io/pause:3.7"
	invalidTmpl1 = "{{ .baz }/d}"
	invalidTmpl2 = "{{ !foobar }}"
)

func TestParseTemplate(t *testing.T) {
	var tmplTests = []struct {
		name        string
		template    string
		data        interface{}
		output      string
		errExpected bool
	}{
		{
			name:     "should parse a valid template and set the right values",
			template: validTmpl,
			data: struct{ ImageRepository, Arch string }{
				ImageRepository: "registry.k8s.io",
				Arch:            "amd64",
			},
			output:      validTmplOut,
			errExpected: false,
		},
		{
			name:     "should noop if there aren't any {{ .foo }} present",
			template: doNothing,
			data: struct{ ImageRepository, Arch string }{
				ImageRepository: "registry.k8s.io",
				Arch:            "amd64",
			},
			output:      doNothing,
			errExpected: false,
		},
		{
			name:        "invalid syntax, passing nil",
			template:    invalidTmpl1,
			data:        nil,
			output:      "",
			errExpected: true,
		},
		{
			name:        "invalid syntax",
			template:    invalidTmpl2,
			data:        struct{}{},
			output:      "",
			errExpected: true,
		},
	}
	for _, tt := range tmplTests {
		t.Run(tt.name, func(t *testing.T) {
			outbytes, err := ParseTemplate(tt.template, tt.data)
			if tt.errExpected != (err != nil) {
				t.Errorf(
					"failed TestParseTemplate:\n\texpected err: %t\n\t  actual: %s",
					tt.errExpected,
					err,
				)
			}
			if tt.output != string(outbytes) {
				t.Errorf(
					"failed TestParseTemplate:\n\texpected bytes: %s\n\t  actual: %s",
					tt.output,
					outbytes,
				)
			}
		})
	}
}
