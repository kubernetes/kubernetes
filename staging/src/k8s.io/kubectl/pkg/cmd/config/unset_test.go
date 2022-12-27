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

	"github.com/google/go-cmp/cmp"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type unsetConfigTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	completeError  string
	runError       string
}

func TestUnset(t *testing.T) {
	t.Parallel()

	initConfigCurrentContext := clientcmdapi.NewConfig()
	initConfigCurrentContext.CurrentContext = "otherkube"

	expectedConfigEmpty := clientcmdapi.NewConfig()

	initConfigClusters := clientcmdapi.NewConfig()
	testClusters := clientcmdapi.NewCluster()
	testClusters.Server = "https://192.168.99.100:8443"
	initConfigClusters.Clusters = map[string]*clientcmdapi.Cluster{
		"minikube": testClusters,
	}

	initConfigContexts := clientcmdapi.NewConfig()
	testContext := clientcmdapi.NewContext()
	testContext.Cluster = "minikube"
	testContext.AuthInfo = "minikube"
	initConfigContexts.Contexts = map[string]*clientcmdapi.Context{
		"minikube": testContext,
	}

	initConfigAuthInfos := clientcmdapi.NewConfig()
	testAuthInfo := clientcmdapi.NewAuthInfo()
	testAuthInfo.ClientKey = "./fake-client-key"
	testAuthInfo.ImpersonateUserExtra = nil // Required because set nils out this value if it doesn't get set
	initConfigAuthInfos.AuthInfos = map[string]*clientcmdapi.AuthInfo{
		"test-user": testAuthInfo,
	}

	for _, test := range []unsetConfigTest{
		{
			name:           "CurrentContext",
			args:           []string{"current-context"},
			startingConfig: initConfigCurrentContext,
			expectedConfig: expectedConfigEmpty,
			expectedOut:    "Property \"current-context\" unset.\n",
		}, {
			name:           "Clusters",
			args:           []string{"clusters"},
			startingConfig: initConfigClusters,
			expectedConfig: expectedConfigEmpty,
			expectedOut:    "Property \"clusters\" unset.\n",
		}, {
			name:           "NonexistentCluster",
			args:           []string{"clusters.foo.namespace"},
			startingConfig: initConfigClusters,
			expectedConfig: initConfigClusters,
			// TODO: this is not actually the expected error but this isn't meant to be a user facing change
			expectedOut: "Property \"clusters.foo.namespace\" unset.\n",
		}, {
			name:           "Contexts",
			args:           []string{"contexts"},
			startingConfig: initConfigContexts,
			expectedConfig: expectedConfigEmpty,
			expectedOut:    "Property \"contexts\" unset.\n",
		}, {
			name:           "NonexistentContext",
			args:           []string{"contexts.foo.namespace"},
			startingConfig: initConfigContexts,
			expectedConfig: initConfigContexts,
			runError:       "current map key `foo` is invalid",
		}, {
			name:           "AuthInfos",
			args:           []string{"users"},
			startingConfig: initConfigAuthInfos,
			expectedConfig: expectedConfigEmpty,
			expectedOut:    "Property \"users\" unset.\n",
		}, {
			name:           "NonexistentAuthInfos",
			args:           []string{"users.foo.username"},
			startingConfig: initConfigAuthInfos,
			expectedConfig: initConfigAuthInfos,
			runError:       "current map key `foo` is invalid",
		}, {
			name:           "ErrorEmptyArgs",
			args:           []string{},
			startingConfig: initConfigContexts,
			expectedConfig: initConfigContexts,
			completeError:  "unexpected args: ",
		}, {
			name:           "ErrorTwoArgs",
			args:           []string{"clusters.my-cluster", "clusters.mu-cluster"},
			startingConfig: initConfigContexts,
			expectedConfig: initConfigContexts,
			completeError:  "unexpected args: [clusters.my-cluster clusters.mu-cluster]",
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

			options := NewUnsetOptions(streams, pathOptions)

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

			err = options.RunUnset()
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
