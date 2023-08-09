/*
Copyright 2020 The Kubernetes Authors.

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

package config

import (
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestGetUsersRun(t *testing.T) {
	var tests = []struct {
		name     string
		config   clientcmdapi.Config
		expected string
	}{
		{
			name:     "no users",
			config:   clientcmdapi.Config{},
			expected: "NAME\n",
		},
		{
			name: "some users",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {Username: "minikube"},
					"admin":    {Username: "admin"},
				},
			},
			expected: `NAME
admin
minikube
`,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			ioStreams, _, out, _ := genericclioptions.NewTestIOStreams()
			pathOptions, err := tf.PathOptionsWithConfig(test.config)
			if err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}
			options := NewGetUsersOptions(ioStreams, pathOptions)

			if err = options.Run(); err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			if got := out.String(); got != test.expected {
				t.Fatalf("expected: %s but got %s", test.expected, got)
			}
		})
	}
}
