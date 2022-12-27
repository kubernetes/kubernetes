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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setRunTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	options        *SetOptions
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	toOptionsError string
	runError       string
}

type setToOptionsTest struct {
	name            string
	args            []string
	flags           *SetFlags
	expectedOptions *SetOptions
	expectedError   string
}

func TestRunSet(t *testing.T) {
	t.Parallel()

	startingConfigEmpty := clientcmdapi.NewConfig()

	expectedConfigCurrentContext := clientcmdapi.NewConfig()
	expectedConfigCurrentContext.CurrentContext = "my-cluster"

	expectedConfigCluster := clientcmdapi.NewConfig()
	myCluster := clientcmdapi.NewCluster()
	myCluster.Server = "https://1.2.3.4"
	expectedConfigCluster.Clusters = map[string]*clientcmdapi.Cluster{
		"my-cluster": myCluster,
	}

	expectedConfigContext := clientcmdapi.NewConfig()
	myContext := clientcmdapi.NewContext()
	myContext.Cluster = "my-cluster"
	expectedConfigContext.Contexts = map[string]*clientcmdapi.Context{
		"my-context": myContext,
	}

	expectedConfigUser := clientcmdapi.NewConfig()
	myAuthInfo := clientcmdapi.NewAuthInfo()
	myAuthInfo.ClientKey = "./fake-client-key"
	myAuthInfo.ImpersonateUserExtra = nil // Required because set nils out this value if it doesn't get set
	expectedConfigUser.AuthInfos = map[string]*clientcmdapi.AuthInfo{
		"cluster-admin": myAuthInfo,
	}

	expectedConfigUserRawBytes := clientcmdapi.NewConfig()
	myAuthInfoRawBytes := clientcmdapi.NewAuthInfo()
	myAuthInfoRawBytes.ClientKeyData = []byte("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
	myAuthInfoRawBytes.ImpersonateUserExtra = nil // Required because set nils out this value if it doesn't get set
	expectedConfigUserRawBytes.AuthInfos = map[string]*clientcmdapi.AuthInfo{
		"cluster-admin": myAuthInfoRawBytes,
	}

	for _, test := range []setRunTest{
		{
			name:           "CurrentContext",
			startingConfig: startingConfigEmpty,
			options: &SetOptions{
				PropertyName:  "current-context",
				PropertyValue: "my-cluster",
				SetRawBytes:   false,
			},
			expectedConfig: expectedConfigCurrentContext,
			expectedOut:    "Property \"current-context\" set.\n",
		}, {
			name:           "ClusterServer",
			startingConfig: startingConfigEmpty,
			options: &SetOptions{
				PropertyName:  "clusters.my-cluster.server",
				PropertyValue: "https://1.2.3.4",
				SetRawBytes:   false,
			},
			expectedConfig: expectedConfigCluster,
			expectedOut:    "Property \"clusters.my-cluster.server\" set.\n",
		}, {
			name:           "ContextCluster",
			startingConfig: startingConfigEmpty,
			options: &SetOptions{
				PropertyName:  "contexts.my-context.cluster",
				PropertyValue: "my-cluster",
				SetRawBytes:   false,
			},
			expectedConfig: expectedConfigContext,
			expectedOut:    "Property \"contexts.my-context.cluster\" set.\n",
		}, {
			name:           "UserClientKey",
			startingConfig: startingConfigEmpty,
			options: &SetOptions{
				PropertyName:  "users.cluster-admin.client-key",
				PropertyValue: "./fake-client-key",
				SetRawBytes:   false,
			},
			expectedConfig: expectedConfigUser,
			expectedOut:    "Property \"users.cluster-admin.client-key\" set.\n",
		}, {
			name:           "UserClientKeyData",
			startingConfig: startingConfigEmpty,
			options: &SetOptions{
				PropertyName:  "users.cluster-admin.client-key-data",
				PropertyValue: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
				SetRawBytes:   true,
			},
			expectedConfig: expectedConfigUserRawBytes,
			expectedOut:    "Property \"users.cluster-admin.client-key-data\" set.\n",
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

			err = test.options.RunSet()
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

func TestSetToOptions(t *testing.T) {
	t.Parallel()

	for _, test := range []setToOptionsTest{
		{
			name:  "DefaultFlagsTwoArgs",
			args:  []string{"current-context", "my-cluster"},
			flags: &SetFlags{},
			expectedOptions: &SetOptions{
				PropertyName:  "current-context",
				PropertyValue: "my-cluster",
				SetRawBytes:   false,
			},
		}, {
			name:          "ErrorDefaultFlagsZeroArgs",
			args:          []string{},
			flags:         &SetFlags{},
			expectedError: "unexpected args:",
		}, {
			name:          "ErrorDefaultFlagsOneArg",
			args:          []string{"current-context"},
			flags:         &SetFlags{},
			expectedError: "unexpected args: [current-context]",
		}, {
			name:          "ErrorDefaultFlagsThreeArg",
			args:          []string{"current-context", "my-cluster", "fake-test"},
			flags:         &SetFlags{},
			expectedError: "unexpected args: [current-context my-cluster fake-test]",
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

			test.flags.configAccess = pathOptions
			test.flags.ioStreams = streams

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
			if !reflect.DeepEqual(test.expectedOptions, options) {
				t.Errorf("expected options did not match actual options (-want, +got):\n%v", cmp.Diff(test.expectedOptions, options))
			}
		})
	}
}
