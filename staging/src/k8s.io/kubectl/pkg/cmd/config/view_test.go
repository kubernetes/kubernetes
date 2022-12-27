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
	"bytes"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
)

type viewRunTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	options        *ViewOptions
	expectedOut    string
	toOptionsError string
	runError       string
}

type viewToOptionsTest struct {
	name            string
	args            []string
	flags           *ViewFlags
	expectedOptions *ViewOptions
	expectedError   string
}

func TestRunView(t *testing.T) {
	t.Parallel()

	startingConfig := clientcmdapi.NewConfig()
	startingConfig.Kind = "Config"
	startingConfig.APIVersion = "v1"
	startingConfig.CurrentContext = "minikube"
	cluster1 := clientcmdapi.NewCluster()
	cluster1.Server = "https://192.168.99.100:8443"
	cluster2 := clientcmdapi.NewCluster()
	cluster2.Server = "https://192.168.0.1:3434"
	startingConfig.Clusters = map[string]*clientcmdapi.Cluster{
		"minikube":   cluster1,
		"my-cluster": cluster2,
	}
	context1 := clientcmdapi.NewContext()
	context1.AuthInfo = "minikube"
	context1.Cluster = "minikube"
	context2 := clientcmdapi.NewContext()
	context2.AuthInfo = "mu-cluster"
	context2.Cluster = "my-cluster"
	startingConfig.Contexts = map[string]*clientcmdapi.Context{
		"minikube":   context1,
		"my-cluster": context2,
	}
	authInfo1 := clientcmdapi.NewAuthInfo()
	authInfo1.Token = "REDACTED"
	authInfo2 := clientcmdapi.NewAuthInfo()
	authInfo2.Token = "REDACTED"
	startingConfig.AuthInfos = map[string]*clientcmdapi.AuthInfo{
		"minikube":   authInfo1,
		"mu-cluster": authInfo2,
	}

	for _, test := range []viewRunTest{
		{
			name:           "DefaultOptions",
			startingConfig: startingConfig,
			options: &ViewOptions{
				Merge:        true,
				Flatten:      false,
				Minify:       false,
				RawByteData:  false,
				Context:      cliflag.StringFlag{},
				OutputFormat: cliflag.StringFlag{},
			},
			expectedOut: `apiVersion: v1
clusters:
- cluster:
    server: https://192.168.99.100:8443
  name: minikube
- cluster:
    server: https://192.168.0.1:3434
  name: my-cluster
contexts:
- context:
    cluster: minikube
    user: minikube
  name: minikube
- context:
    cluster: my-cluster
    user: mu-cluster
  name: my-cluster
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    token: REDACTED
- name: mu-cluster
  user:
    token: REDACTED
`,
		}, {
			name:           "Minify",
			startingConfig: startingConfig,
			options: &ViewOptions{
				Merge:        true,
				Flatten:      false,
				Minify:       true,
				RawByteData:  false,
				Context:      cliflag.StringFlag{},
				OutputFormat: cliflag.StringFlag{},
			},
			expectedOut: `apiVersion: v1
clusters:
- cluster:
    server: https://192.168.99.100:8443
  name: minikube
contexts:
- context:
    cluster: minikube
    user: minikube
  name: minikube
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    token: REDACTED
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

			test.options.IOStreams = streams
			test.options.ConfigAccess = pathOptions

			// set printer manually for test
			printer, err := genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml").ToPrinter()
			if err != nil {
				t.Fatalf("unexpected error getting printer: %v", err)
			}
			test.options.PrintObject = printer.PrintObj

			err = test.options.RunView()
			if len(test.runError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.runError)
				return
			} else if len(test.runError) != 0 && err == nil {
				t.Fatalf("expected error %q running command but non received", test.runError)
			} else if err != nil {
				t.Fatalf("unexpected error running to options: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
			}
		})
	}
}

func TestViewToOptions(t *testing.T) {
	t.Parallel()

	for _, test := range []viewToOptionsTest{
		{
			name: "DefaultFlagsNoArgs",
			args: []string{},
			flags: &ViewFlags{
				printFlags:   genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml"),
				merge:        true,
				flatten:      false,
				minify:       false,
				rawByteData:  false,
				context:      cliflag.StringFlag{},
				outputFormat: cliflag.StringFlag{},
			},
			expectedOptions: &ViewOptions{
				Merge:        true,
				Flatten:      false,
				Minify:       false,
				RawByteData:  false,
				Context:      cliflag.StringFlag{},
				OutputFormat: cliflag.StringFlag{},
			},
		}, {
			name: "ErrorDefaultFlagsOneArg",
			args: []string{"./.kube/config"},
			flags: &ViewFlags{
				printFlags:   genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml"),
				merge:        true,
				flatten:      false,
				minify:       false,
				rawByteData:  false,
				context:      cliflag.StringFlag{},
				outputFormat: cliflag.StringFlag{},
			},
			expectedError: "received unepxected argument: [./.kube/config]",
		}, {
			name: "ErrorMergeFalse",
			args: []string{},
			flags: &ViewFlags{
				printFlags:   genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml"),
				merge:        false,
				flatten:      false,
				minify:       false,
				rawByteData:  false,
				context:      cliflag.StringFlag{},
				outputFormat: cliflag.StringFlag{},
			},
			expectedError: "if merge==false a precise file must be specified",
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

			// set print object to nil for proper comparison
			test.expectedOptions.PrintObject = nil
			options.PrintObject = nil

			cmpOptions := cmpopts.IgnoreUnexported(
				cliflag.StringFlag{},
				bytes.Buffer{})
			if cmp.Diff(test.expectedOptions, options, cmpOptions) != "" {
				t.Errorf("expected options did not match actual options (-want, +got):\n%v", cmp.Diff(test.expectedOptions, options, cmpOptions))
			}
		})
	}
}
