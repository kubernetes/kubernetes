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

type useContextTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	completeError  string
	runError       string
}

func TestUseContext(t *testing.T) {
	t.Parallel()

	startConfig := clientcmdapi.NewConfig()
	startConfig.CurrentContext = "otherkube"
	testContext1 := clientcmdapi.NewContext()
	testContext1.AuthInfo = "mu-cluster"
	testContext1.Cluster = "my-cluster"
	testContext2 := clientcmdapi.NewContext()
	testContext2.AuthInfo = "mu-cluster"
	testContext2.Cluster = "my-cluster"
	startConfig.Contexts = map[string]*clientcmdapi.Context{
		"otherkube": testContext1,
		"minikube":  testContext2,
	}

	expectedConfig := clientcmdapi.NewConfig()
	expectedConfig.CurrentContext = "minikube"
	testContext3 := clientcmdapi.NewContext()
	testContext3.AuthInfo = "mu-cluster"
	testContext3.Cluster = "my-cluster"
	testContext4 := clientcmdapi.NewContext()
	testContext4.AuthInfo = "mu-cluster"
	testContext4.Cluster = "my-cluster"
	expectedConfig.Contexts = map[string]*clientcmdapi.Context{
		"otherkube": testContext3,
		"minikube":  testContext4,
	}

	for _, test := range []useContextTest{
		{
			name:           "UseContext",
			startingConfig: startConfig,
			args:           []string{"minikube"},
			expectedConfig: expectedConfig,
			expectedOut:    "Switched to context \"minikube\"",
		}, {
			name:           "ErrorNonexistentContext",
			startingConfig: startConfig,
			args:           []string{"foo"},
			expectedConfig: startConfig,
			runError:       "can not set current-context to \"foo\", context does not exist",
		}, {
			name:           "ErrorZeroArgs",
			startingConfig: startConfig,
			args:           []string{},
			expectedConfig: startConfig,
			completeError:  "unexpected args: ",
		}, {
			name:           "ErrorTwoArgs",
			startingConfig: startConfig,
			args:           []string{"context1", "context2"},
			expectedConfig: startConfig,
			completeError:  "unexpected args: [context1 context2]",
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

			options := NewUseContextOptions(streams, pathOptions)

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

			err = options.RunUseContext()
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
