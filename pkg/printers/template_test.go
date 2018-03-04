/*
Copyright 2018 The Kubernetes Authors.

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

package printers

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestTemplate(t *testing.T) {
	testCase := map[string]struct {
		template  string
		obj       runtime.Object
		expectOut string
		expectErr func(error) (string, bool)
	}{
		"support base64 decoding of secret data": {
			template: "{{ .Data.username | base64decode }}",
			obj: &api.Secret{
				Data: map[string][]byte{
					"username": []byte("hunter"),
				},
			},
			expectOut: "hunter",
		},
		"test error path for base64 decoding": {
			template: "{{ .Data.username | base64decode }}",
			obj:      &badlyMarshaledSecret{},
			expectErr: func(err error) (string, bool) {
				matched := strings.Contains(err.Error(), "base64 decode")
				return "a base64 decode error", matched
			},
		},
	}
	for name, test := range testCase {
		buffer := &bytes.Buffer{}

		p, err := NewTemplatePrinter([]byte(test.template))
		if err != nil {
			if test.expectErr == nil {
				t.Errorf("[%s]expected success but got:\n %v\n", name, err)
				continue
			}
			if expected, ok := test.expectErr(err); !ok {
				t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", name, expected, err)
			}
			continue
		}

		err = p.PrintObj(test.obj, buffer)
		if err != nil {
			if test.expectErr == nil {
				t.Errorf("[%s]expected success but got:\n %v\n", name, err)
				continue
			}
			if expected, ok := test.expectErr(err); !ok {
				t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", name, expected, err)
			}
			continue
		}

		if test.expectErr != nil {
			t.Errorf("[%s]expect:\n error\n but got:\n no error\n", name)
			continue
		}

		if test.expectOut != buffer.String() {
			t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", name, test.expectOut, buffer.String())
		}
	}
}

type badlyMarshaledSecret struct {
	api.Secret
}

func (a badlyMarshaledSecret) MarshalJSON() ([]byte, error) {
	return []byte(`{"apiVersion":"v1","Data":{"username":"--THIS IS NOT BASE64--"},"kind":"Secret"}`), nil
}
