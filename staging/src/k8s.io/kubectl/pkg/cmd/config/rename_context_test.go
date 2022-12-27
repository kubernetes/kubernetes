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

type renameContextTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	args           []string
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	completeError  string
	runError       string
}

func TestRenameContext(t *testing.T) {
	t.Parallel()
	startingConfigSingleContext := clientcmdapi.NewConfig()
	startingConfigSingleContext.CurrentContext = "current-context"
	startingConfigSingleContext.Contexts = map[string]*clientcmdapi.Context{
		"current-context": clientcmdapi.NewContext(),
	}

	startingConfigMultipleContext := clientcmdapi.NewConfig()
	startingConfigMultipleContext.CurrentContext = "current-context"
	startingConfigMultipleContext.Contexts = map[string]*clientcmdapi.Context{
		"current-context":      clientcmdapi.NewContext(),
		"existent-new-context": clientcmdapi.NewContext(),
	}

	resultConfig := clientcmdapi.NewConfig()
	resultConfig.CurrentContext = "new-context"
	resultConfig.Contexts = map[string]*clientcmdapi.Context{
		"new-context": clientcmdapi.NewContext(),
	}

	for _, test := range []renameContextTest{
		{
			name:           "RenameCurrentContext",
			args:           []string{"current-context", "new-context"},
			startingConfig: startingConfigSingleContext,
			expectedConfig: resultConfig,
			expectedOut:    "Context \"current-context\" renamed to \"new-context\"",
		}, {
			name:           "ErrorRenameNonexistentContext",
			args:           []string{"fake-context", "new-context"},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			runError:       "cannot rename the context \"fake-context\", it's not in",
		}, {
			name:           "ErrorRenameToExistingContext",
			args:           []string{"current-context", "existent-new-context"},
			startingConfig: startingConfigMultipleContext,
			expectedConfig: startingConfigMultipleContext,
			runError:       "cannot rename the context \"current-context\", the context \"existent-new-context\" already exists",
		}, {
			name:           "ErrorZeroArgs",
			args:           []string{},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			completeError:  "unexpected args:",
		}, {
			name:           "ErrorOneArg",
			args:           []string{"current-context"},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			completeError:  "unexpected args: [current-context]",
		}, {
			name:           "ErrorThreeArg",
			args:           []string{"current-context", "existing-context", "fake-context"},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			completeError:  "unexpected args: [current-context existing-context fake-context]",
		}, {
			name:           "ErrorEmptyExistingName",
			args:           []string{"", "fake-context"},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			completeError:  "you must specify an original non-empty context name",
		}, {
			name:           "ErrorEmptyNewName",
			args:           []string{"fake-context", ""},
			startingConfig: startingConfigSingleContext,
			expectedConfig: startingConfigSingleContext,
			completeError:  "you must specify a new non-empty context name",
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

			options := NewRenameContextOptions(streams, pathOptions)

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

			err = options.RunRenameContext()
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
