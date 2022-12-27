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
	"bytes"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type getContextsRunTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	options        *GetContextsOptions
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	runError       string
}

type getContextsToOptionsTest struct {
	name            string
	args            []string
	flags           *GetContextsFlags
	expectedOptions *GetContextsOptions
	expectedError   string
}

func TestRunGetContexts(t *testing.T) {
	t.Parallel()

	startingConfigSingleContextNoCurrent := clientcmdapi.NewConfig()
	startingConfigSingleContextNoCurrent.Contexts = map[string]*clientcmdapi.Context{
		"shaker-context": {
			AuthInfo:  "blue-user",
			Cluster:   "big-cluster",
			Namespace: "saw-ns",
		},
	}

	startingConfigSingleContextWithCurrent := clientcmdapi.NewConfig()
	startingConfigSingleContextWithCurrent.CurrentContext = "shaker-context"
	startingConfigSingleContextWithCurrent.Contexts = map[string]*clientcmdapi.Context{
		"shaker-context": {
			AuthInfo:  "blue-user",
			Cluster:   "big-cluster",
			Namespace: "saw-ns",
		},
	}

	startingConfigMultipleContextWithCurrent := clientcmdapi.NewConfig()
	startingConfigMultipleContextWithCurrent.CurrentContext = "shaker-context"
	startingConfigMultipleContextWithCurrent.Contexts = map[string]*clientcmdapi.Context{
		"shaker-context": {
			AuthInfo:  "blue-user",
			Cluster:   "big-cluster",
			Namespace: "saw-ns",
		},
		"abc": {
			AuthInfo:  "blue-user",
			Cluster:   "abc-cluster",
			Namespace: "kube-system",
		},
		"xyz": {
			AuthInfo:  "blue-user",
			Cluster:   "xyz-cluster",
			Namespace: "default",
		},
	}

	for _, test := range []getContextsRunTest{
		{
			name:           "WithHeadersSingleNoCurrentContext",
			startingConfig: startingConfigSingleContextNoCurrent,
			options:        &GetContextsOptions{},
			expectedConfig: startingConfigSingleContextNoCurrent,
			expectedOut: `CURRENT   NAME             CLUSTER       AUTHINFO    NAMESPACE
          shaker-context   big-cluster   blue-user   saw-ns`,
		}, {
			name:           "WithHeadersSingleWithCurrentContext",
			startingConfig: startingConfigSingleContextWithCurrent,
			options:        &GetContextsOptions{},
			expectedConfig: startingConfigSingleContextWithCurrent,
			expectedOut: `CURRENT   NAME             CLUSTER       AUTHINFO    NAMESPACE
*         shaker-context   big-cluster   blue-user   saw-ns
`,
		}, {
			name:           "WithoutHeadersSingle",
			startingConfig: startingConfigSingleContextWithCurrent,
			options: &GetContextsOptions{
				NotShowHeaders: true,
				NameOnly:       true,
			},
			expectedConfig: startingConfigSingleContextWithCurrent,
			expectedOut:    "shaker-context",
		}, {
			name:           "WithHeadersSorted",
			startingConfig: startingConfigMultipleContextWithCurrent,
			options:        &GetContextsOptions{},
			expectedConfig: startingConfigMultipleContextWithCurrent,
			expectedOut: `CURRENT   NAME             CLUSTER       AUTHINFO    NAMESPACE
          abc              abc-cluster   blue-user   kube-system
*         shaker-context   big-cluster   blue-user   saw-ns
          xyz              xyz-cluster   blue-user   default
`,
		}, {
			name:           "WithoutHeadersSorted",
			startingConfig: startingConfigMultipleContextWithCurrent,
			options: &GetContextsOptions{
				NotShowHeaders: true,
			},
			expectedConfig: startingConfigMultipleContextWithCurrent,
			expectedOut: `      abc              abc-cluster   blue-user   kube-system
*     shaker-context   big-cluster   blue-user   saw-ns
      xyz              xyz-cluster   blue-user   default`,
		}, {
			name:           "WithoutHeadersSortedInputContexts",
			startingConfig: startingConfigMultipleContextWithCurrent,
			options: &GetContextsOptions{
				NotShowHeaders: true,
				ContextNames:   []string{"abc", "xyz"},
			},
			expectedConfig: startingConfigMultipleContextWithCurrent,
			expectedOut: `      abc   abc-cluster   blue-user   kube-system
      xyz   xyz-cluster   blue-user   default`,
		}, {
			name:           "WitHeadersSortedInputContexts",
			startingConfig: startingConfigMultipleContextWithCurrent,
			options: &GetContextsOptions{
				ContextNames: []string{"abc", "xyz"},
			},
			expectedConfig: startingConfigMultipleContextWithCurrent,
			expectedOut: `CURRENT   NAME   CLUSTER       AUTHINFO    NAMESPACE
          abc    abc-cluster   blue-user   kube-system
          xyz    xyz-cluster   blue-user   default`,
		}, {
			name:           "ErrorNoContextFound",
			startingConfig: startingConfigMultipleContextWithCurrent,
			options: &GetContextsOptions{
				ContextNames: []string{"efg"},
			},
			expectedConfig: startingConfigMultipleContextWithCurrent,
			runError:       "context \"efg\" not found",
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

			test.options.IOStreams = streams
			test.options.ConfigAccess = pathOptions

			err = test.options.RunGetContexts()
			if len(test.runError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.runError)
				checkOutputConfig(t, test.options.ConfigAccess, test.expectedConfig, cmp.Options{})
				return
			} else if len(test.runError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.runError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
				checkOutputConfig(t, test.options.ConfigAccess, test.expectedConfig, cmp.Options{})
			}
		})
	}
}

func TestGetContextToOptions(t *testing.T) {
	t.Parallel()

	for _, test := range []getContextsToOptionsTest{
		{
			name:  "DefaultFlagsNoArgs",
			args:  []string{},
			flags: &GetContextsFlags{},
			expectedOptions: &GetContextsOptions{
				ContextNames:   []string{},
				NameOnly:       false,
				NotShowHeaders: false,
			},
		}, {
			name: "NonDefaultFlagsNoArgs",
			args: []string{},
			flags: &GetContextsFlags{
				notShowHeaders: true,
				output:         "name",
			},
			expectedOptions: &GetContextsOptions{
				ContextNames:   []string{},
				NameOnly:       true,
				NotShowHeaders: true,
			},
		}, {
			name: "NonDefaultFlagsArgs",
			args: []string{"test1", "test2"},
			flags: &GetContextsFlags{
				notShowHeaders: true,
				output:         "name",
			},
			expectedOptions: &GetContextsOptions{
				ContextNames:   []string{"test1", "test2"},
				NameOnly:       true,
				NotShowHeaders: true,
			},
		}, {
			name: "DefaultFlagsArgs",
			args: []string{"test1", "test2"},
			flags: &GetContextsFlags{
				notShowHeaders: false,
				output:         "",
			},
			expectedOptions: &GetContextsOptions{
				ContextNames:   []string{"test1", "test2"},
				NameOnly:       false,
				NotShowHeaders: false,
			},
		}, {
			name: "InvalidOutputType",
			args: []string{},
			flags: &GetContextsFlags{
				notShowHeaders: false,
				output:         "fake-output-type",
			},
			expectedError: "output must be one of '' or 'name': fake-output-type",
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*clientcmdapi.NewConfig())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, _, _ := genericclioptions.NewTestIOStreams()

			test.flags.ioStreams = streams
			test.flags.configAccess = pathOptions

			options, err := test.flags.ToOptions(test.args)
			if len(test.expectedError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.expectedError)
				return
			} else if len(test.expectedError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.expectedError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			// finish options for proper comparison
			test.expectedOptions.IOStreams = streams
			test.expectedOptions.ConfigAccess = pathOptions
			cmpOptions := cmpopts.IgnoreUnexported(bytes.Buffer{})
			if cmp.Diff(test.expectedOptions, options, cmpOptions) != "" {
				t.Errorf("expected options did not match actual options (-want, +got):\n%v", cmp.Diff(test.expectedOptions, options, cmpOptions))
			}
		})
	}
}
