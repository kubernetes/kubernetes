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

type deleteContextTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	completeError  string
	runError       string
}

func TestDeleteContext(t *testing.T) {
	t.Parallel()
	startConf := clientcmdapi.NewConfig()
	startConf.CurrentContext = "otherkube"
	startConf.Contexts = map[string]*clientcmdapi.Context{
		"minikube":  {Cluster: "minikube"},
		"otherkube": {Cluster: "otherkube"},
	}

	resultConfDeleteMinikube := clientcmdapi.NewConfig()
	resultConfDeleteMinikube.CurrentContext = "otherkube"
	resultConfDeleteMinikube.Contexts = map[string]*clientcmdapi.Context{
		"otherkube": {Cluster: "otherkube"},
	}

	resultConfDeleteOtherkube := clientcmdapi.NewConfig()
	resultConfDeleteOtherkube.CurrentContext = "otherkube"
	resultConfDeleteOtherkube.Contexts = map[string]*clientcmdapi.Context{
		"minikube": {Cluster: "minikube"},
	}

	for _, test := range []deleteContextTest{
		{
			name:           "DeleteContext",
			startingConfig: startConf,
			args:           []string{"minikube"},
			expectedConfig: resultConfDeleteMinikube,
			expectedOut:    "deleted context \"minikube\" from",
		}, {
			name:           "ErrorMultipleArgs",
			startingConfig: startConf,
			args:           []string{"minikube", "test"},
			expectedConfig: startConf,
			completeError:  "unexpected args: [minikube test]",
		}, {
			name:           "ErrorNonexistentContext",
			startingConfig: startConf,
			args:           []string{"test"},
			expectedConfig: startConf,
			runError:       "context \"test\" does not exist in config file",
		}, {
			name:           "WarningRemovedCurrentContext",
			startingConfig: startConf,
			args:           []string{"otherkube"},
			expectedConfig: resultConfDeleteOtherkube,
			expectedOut: `warning: this removed your active context, use "kubectl config use-context" to select a different one
deleted context "otherkube" from`,
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*test.startingConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, buffOut, _ := genericclioptions.NewTestIOStreams()

			options := NewDeleteContextOptions(streams, pathOptions)

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

			err = options.RunDeleteContext()
			if len(test.runError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.completeError)
				checkOutputConfig(t, options.ConfigAccess, test.expectedConfig, cmp.Options{})
				return
			} else if len(test.runError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.completeError)
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
