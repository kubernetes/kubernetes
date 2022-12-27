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

package config

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type deleteClusterTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	completeError  string
	runError       string
}

func TestDeleteCluster(t *testing.T) {
	t.Parallel()

	startingConf := clientcmdapi.NewConfig()
	startingConf.Clusters = map[string]*clientcmdapi.Cluster{
		"minikube":  {Server: "https://192.168.0.99"},
		"otherkube": {Server: "https://192.168.0.100"},
	}

	resultConfDeleteMinikube := clientcmdapi.NewConfig()
	resultConfDeleteMinikube.Clusters = map[string]*clientcmdapi.Cluster{
		"otherkube": {Server: "https://192.168.0.100"},
	}

	for _, test := range []deleteClusterTest{
		{
			name:           "DeleteCluster",
			startingConfig: startingConf,
			args:           []string{"minikube"},
			expectedConfig: resultConfDeleteMinikube,
			expectedOut:    "deleted cluster \"minikube\" from",
		}, {
			name:           "ErrorMultipleArgs",
			startingConfig: startingConf,
			args:           []string{"minikube", "test"},
			expectedConfig: startingConf,
			completeError:  "unexpected args: [minikube test]",
		}, {
			name:           "ErrorNonexistentCluster",
			startingConfig: startingConf,
			args:           []string{"test"},
			expectedConfig: startingConf,
			runError:       "cannot delete cluster \"test\", not in file",
		},
	} {
		test := test
		t.Run(test.name, func(f *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*test.startingConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, buffOut, _ := genericclioptions.NewTestIOStreams()

			options := NewDeleteClusterOptions(streams, pathOptions)

			err = options.Complete(test.args)
			if len(test.completeError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.completeError)
				checkOutputConfig(t, options.ConfigAccess, test.expectedConfig, cmp.Options{})
				return
			} else if len(test.completeError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.completeError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			err = options.RunDeleteCluster()
			if len(test.runError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.runError)
				checkOutputConfig(t, options.ConfigAccess, test.expectedConfig, cmp.Options{})
				return
			} else if len(test.runError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.runError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
				checkOutputConfig(t, options.ConfigAccess, test.expectedConfig, cmp.Options{})
			}
		})
	}
}
