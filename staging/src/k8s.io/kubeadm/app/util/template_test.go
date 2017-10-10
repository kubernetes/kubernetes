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
	validTmpl    = "image: {{ .ImageRepository }}/pause-{{ .Arch }}:3.0"
	validTmplOut = "image: gcr.io/google_containers/pause-amd64:3.0"
	doNothing    = "image: gcr.io/google_containers/pause-amd64:3.0"
	invalidTmpl1 = "{{ .baz }/d}"
	invalidTmpl2 = "{{ !foobar }}"
)

func TestParseTemplate(t *testing.T) {
	var tmplTests = []struct {
		template    string
		data        interface{}
		output      string
		errExpected bool
	}{
		// should parse a valid template and set the right values
		{
			template: validTmpl,
			data: struct{ ImageRepository, Arch string }{
				ImageRepository: "gcr.io/google_containers",
				Arch:            "amd64",
			},
			output:      validTmplOut,
			errExpected: false,
		},
		// should noop if there aren't any {{ .foo }} present
		{
			template: doNothing,
			data: struct{ ImageRepository, Arch string }{
				ImageRepository: "gcr.io/google_containers",
				Arch:            "amd64",
			},
			output:      doNothing,
			errExpected: false,
		},
		// invalid syntax, passing nil
		{
			template:    invalidTmpl1,
			data:        nil,
			output:      "",
			errExpected: true,
		},
		// invalid syntax
		{
			template:    invalidTmpl2,
			data:        struct{}{},
			output:      "",
			errExpected: true,
		},
	}
	for _, tt := range tmplTests {
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
	}
}
