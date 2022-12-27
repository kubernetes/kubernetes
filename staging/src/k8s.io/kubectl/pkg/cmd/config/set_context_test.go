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

package config

import (
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setContextTest struct {
	name           string
	testContext    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestCreateContext(t *testing.T) {
	for _, test := range []setContextTest{
		{
			name:        "CreateNewContext",
			testContext: "shaker-context",
			config:      clientcmdapi.Config{},
			args:        []string{"shaker-context"},
			flags: []string{
				"--cluster=cluster_nickname",
				"--user=user_nickname",
				"--namespace=namespace",
			},
			expected: `Context "shaker-context" created.` + "\n",
			expectedConfig: clientcmdapi.Config{
				Contexts: map[string]*clientcmdapi.Context{
					"shaker-context": {
						AuthInfo:  "user_nickname",
						Cluster:   "cluster_nickname",
						Namespace: "namespace",
					},
				},
			},
		},
		{
			name:        "ModifyExistingContext",
			testContext: "shaker-context",
			config: clientcmdapi.Config{
				Contexts: map[string]*clientcmdapi.Context{
					"shaker-context": {AuthInfo: "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
					"not-this": {
						AuthInfo:  "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
				},
			},
			args: []string{"shaker-context"},
			flags: []string{
				"--cluster=cluster_nickname",
				"--user=user_nickname",
				"--namespace=namespace",
			},
			expected: `Context "shaker-context" modified.` + "\n",
			expectedConfig: clientcmdapi.Config{
				Contexts: map[string]*clientcmdapi.Context{
					"shaker-context": {
						AuthInfo:  "user_nickname",
						Cluster:   "cluster_nickname",
						Namespace: "namespace",
					},
					"not-this": {
						AuthInfo:  "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
				},
			},
		},
		{
			name:        "ModifyCurrentContext",
			testContext: "shaker-context",
			config: clientcmdapi.Config{
				CurrentContext: "shaker-context",
				Contexts: map[string]*clientcmdapi.Context{
					"shaker-context": {
						AuthInfo:  "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
					"not-this": {
						AuthInfo:  "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
				},
			},
			args: []string{},
			flags: []string{
				"--current",
				"--cluster=cluster_nickname",
				"--user=user_nickname",
				"--namespace=namespace",
			},
			expected: `Context "shaker-context" modified.` + "\n",
			expectedConfig: clientcmdapi.Config{
				Contexts: map[string]*clientcmdapi.Context{
					"shaker-context": {
						AuthInfo:  "user_nickname",
						Cluster:   "cluster_nickname",
						Namespace: "namespace",
					},
					"not-this": {
						AuthInfo:  "blue-user",
						Cluster:   "big-cluster",
						Namespace: "saw-ns",
					},
				},
			},
		},
	} {
		test.run(t)
	}
}

func (test setContextTest) run(t *testing.T) {
	fakeKubeFile, err := generateTestKubeConfig(test.config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	streams, _, buffOut, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdConfigSetContext(streams, pathOptions)
	cmd.SetArgs(test.args)
	if err := cmd.Flags().Parse(test.flags); err != nil {
		t.Fatalf("unexpected error parsing flags: %v", err)
	}
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v,kubectl set-context args: %v,flags: %v", err, test.args, test.flags)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buffOut.String() != test.expected {
			t.Errorf("Fail in %q:\n expected %v\n but got %v\n", test.name, test.expected, buffOut.String())
		}
	}
	if test.expectedConfig.Contexts != nil {
		expectContext := test.expectedConfig.Contexts[test.testContext]
		actualContext := config.Contexts[test.testContext]
		if expectContext.AuthInfo != actualContext.AuthInfo || expectContext.Cluster != actualContext.Cluster ||
			expectContext.Namespace != actualContext.Namespace {
			t.Errorf("Fail in %q:\n expected Context %v\n but found %v in kubeconfig\n", test.name, expectContext, actualContext)
		}
	}
}
