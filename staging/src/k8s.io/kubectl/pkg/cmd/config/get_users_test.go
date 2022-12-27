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

	"k8s.io/client-go/tools/clientcmd"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type testGetUser struct {
	name           string
	startingConfig *clientcmdapi.Config
	expectedOut    string
}

func TestGetUsersRun(t *testing.T) {
	t.Parallel()

	startingConfEmpty := clientcmdapi.NewConfig()

	startingConfMultipleUsers := clientcmdapi.NewConfig()
	startingConfMultipleUsers.AuthInfos = map[string]*clientcmdapi.AuthInfo{
		"minikube": {Username: "minikube"},
		"admin":    {Username: "admin"},
	}

	for _, test := range []testGetUser{
		{
			name:           "NoUsers",
			startingConfig: startingConfEmpty,
			expectedOut:    "NAME\n",
		},
		{
			name:           "Users",
			startingConfig: startingConfMultipleUsers,
			expectedOut: `NAME
admin
minikube
`,
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

			options := NewGetUsersOptions(streams, pathOptions)

			err = options.RunGetUsers()
			if err != nil {
				t.Errorf("Unexpected error running command: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
			}
		})
	}
}
