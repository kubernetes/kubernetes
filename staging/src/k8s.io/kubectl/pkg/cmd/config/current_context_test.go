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

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type currentContextTest struct {
	name           string
	startingConfig clientcmdapi.Config
	expectedOutput string
	expectedError  string
}

func TestCurrentContext(t *testing.T) {
	t.Parallel()
	for _, test := range []currentContextTest{
		{
			name: "WithSetContext",
			startingConfig: clientcmdapi.Config{
				CurrentContext: "federal-context",
			},
			expectedOutput: "federal-context\n",
		}, {
			name:           "WithNoSetContext",
			startingConfig: clientcmdapi.Config{},
			expectedError:  "current-context is not set",
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(test.startingConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, buffOut, _ := genericclioptions.NewTestIOStreams()

			options := NewCurrentContextOptions(streams, pathOptions)
			if err := options.Complete([]string{}); err != nil {
				t.Fatalf("unexpected error completing options:\n%v\n", err)
			}
			err = options.RunCurrentContext()
			if len(test.expectedError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.expectedError)
				return
			} else if len(test.expectedError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.expectedError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			if len(test.expectedOutput) != 0 {
				if buffOut.String() != test.expectedOutput {
					t.Fatalf("expected out: %v\ngot: %v\n", test.expectedOutput, buffOut.String())
				}
			}
		})
	}
}
