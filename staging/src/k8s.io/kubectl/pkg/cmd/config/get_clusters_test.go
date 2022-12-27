/*
Copyright 2014 The Kubernetes Authors.

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

type getClustersTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
}

func TestGetClusters(t *testing.T) {
	t.Parallel()

	startingConf := clientcmdapi.NewConfig()
	startingConf.Clusters = map[string]*clientcmdapi.Cluster{
		"minikube": {Server: "https://192.168.0.99"},
	}

	startingConfEmpty := clientcmdapi.NewConfig()

	for _, test := range []getClustersTest{
		{
			name:           "WithClusters",
			startingConfig: startingConf,
			args:           []string{},
			expectedConfig: startingConf,
			expectedOut: `NAME
minikube`,
		}, {
			name:           "EmptyClusters",
			startingConfig: startingConfEmpty,
			args:           []string{},
			expectedConfig: startingConfEmpty,
			expectedOut:    "NAME\n",
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

			options := NewGetClusterOptions(streams, pathOptions)

			err = options.RunGetClusters()
			if err != nil {
				t.Errorf("Unexpected error running command: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
				checkOutputConfig(t, options.ConfigAccess, test.expectedConfig, cmp.Options{})
			}
		})
	}
}
