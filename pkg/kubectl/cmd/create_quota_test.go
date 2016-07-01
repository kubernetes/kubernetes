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

package cmd

import (
	"bytes"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
)

func TestCreateQuota(t *testing.T) {
	resourceQuotaObject := &api.ResourceQuota{}
	resourceQuotaObject.Name = "my-quota"
	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/resourcequotas" && m == "POST":
				return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: objBody(codec, resourceQuotaObject)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"

	tests := map[string]struct {
		flags          map[string]string
		expectedOutput string
	}{
		"single resource": {
			flags:          map[string]string{"hard": "cpu=1", "output": "name"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"single resource with a scope": {
			flags:          map[string]string{"hard": "cpu=1", "output": "name", "scopes": "BestEffort"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"multiple resources": {
			flags:          map[string]string{"hard": "cpu=1,pods=42", "output": "name", "scopes": "BestEffort"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
		"single resource with multiple scopes": {
			flags:          map[string]string{"hard": "cpu=1", "output": "name", "scopes": "BestEffort,NotTerminating"},
			expectedOutput: "resourcequota/" + resourceQuotaObject.Name + "\n",
		},
	}
	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateQuota(f, buf)
		cmd.Flags().Set("hard", "cpu=1")
		cmd.Flags().Set("output", "name")
		cmd.Run(cmd, []string{resourceQuotaObject.Name})

		if buf.String() != test.expectedOutput {
			t.Errorf("%s: expected output: %s, but got: %s", name, test.expectedOutput, buf.String())
		}
	}
}
