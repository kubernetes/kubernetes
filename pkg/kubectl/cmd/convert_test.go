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

package cmd

import (
	"bytes"
	"fmt"
	"net/http"
	"testing"

	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testcase struct {
	name          string
	file          string
	outputVersion string
	fields        []checkField
}

type checkField struct {
	template string
	expected string
}

func TestConvertObject(t *testing.T) {
	testcases := []testcase{
		{
			name:          "apps deployment to extensions deployment",
			file:          "../../../test/fixtures/pkg/kubectl/cmd/convert/appsdeployment.yaml",
			outputVersion: "extensions/v1beta1",
			fields: []checkField{
				{
					template: "{{.apiVersion}}",
					expected: "extensions/v1beta1",
				},
			},
		},
		{
			name:          "extensions deployment to apps deployment",
			file:          "../../../test/fixtures/pkg/kubectl/cmd/convert/extensionsdeployment.yaml",
			outputVersion: "apps/v1beta2",
			fields: []checkField{
				{
					template: "{{.apiVersion}}",
					expected: "apps/v1beta2",
				},
			},
		},
		{
			name:          "v1 HPA to v2beta1 HPA",
			file:          "../../../test/fixtures/pkg/kubectl/cmd/convert/v1HPA.yaml",
			outputVersion: "autoscaling/v2beta1",
			fields: []checkField{
				{
					template: "{{.apiVersion}}",
					expected: "autoscaling/v2beta1",
				},
				{
					template: "{{(index .spec.metrics 0).resource.name}}",
					expected: "cpu",
				},
				{
					template: "{{(index .spec.metrics 0).resource.targetAverageUtilization}}",
					expected: "50",
				},
			},
		},
		{
			name:          "v2beta1 HPA to v1 HPA",
			file:          "../../../test/fixtures/pkg/kubectl/cmd/convert/v2beta1HPA.yaml",
			outputVersion: "autoscaling/v1",
			fields: []checkField{
				{
					template: "{{.apiVersion}}",
					expected: "autoscaling/v1",
				},
				{
					template: "{{.spec.targetCPUUtilizationPercentage}}",
					expected: "50",
				},
			},
		},
	}

	for _, tc := range testcases {
		for _, field := range tc.fields {
			t.Run(fmt.Sprintf("%s %s", tc.name, field), func(t *testing.T) {
				tf := cmdtesting.NewTestFactory()
				tf.UnstructuredClient = &fake.RESTClient{
					Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}),
				}
				tf.Namespace = "test"

				buf := bytes.NewBuffer([]byte{})
				cmd := NewCmdConvert(tf, buf)
				cmd.Flags().Set("filename", tc.file)
				cmd.Flags().Set("output-version", tc.outputVersion)
				cmd.Flags().Set("local", "true")
				cmd.Flags().Set("output", "go-template="+field.template)
				cmd.Run(cmd, []string{})
				if buf.String() != field.expected {
					t.Errorf("unexpected output when converting %s to %q, expected: %q, but got %q", tc.file, tc.outputVersion, field.expected, buf.String())
				}
			})
		}
	}
}
