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
	"reflect"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestDeleteUserComplete(t *testing.T) {
	var tests = []struct {
		name string
		args []string
		err  string
	}{
		{
			name: "no args",
			args: []string{},
			err:  "user to delete is required",
		},
		{
			name: "user provided",
			args: []string{"minikube"},
			err:  "",
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			ioStreams, _, out, _ := genericiooptions.NewTestIOStreams()
			pathOptions, err := tf.PathOptionsWithConfig(clientcmdapi.Config{})
			if err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			cmd := NewCmdConfigDeleteUser(ioStreams, pathOptions)
			cmd.SetOut(out)
			options := NewDeleteUserOptions(ioStreams, pathOptions)

			if err := options.Complete(cmd, test.args); err != nil {
				if test.err == "" {
					t.Fatalf("unexpected error executing command: %v", err)
				}

				if !strings.Contains(err.Error(), test.err) {
					t.Fatalf("expected error to contain %v, got %v", test.err, err.Error())
				}

				return
			}

			if options.configFile != pathOptions.GlobalFile {
				t.Fatalf("expected configFile to be %v, got %v", pathOptions.GlobalFile, options.configFile)
			}
		})
	}
}

func TestDeleteUserValidate(t *testing.T) {
	var tests = []struct {
		name   string
		user   string
		config clientcmdapi.Config
		err    string
	}{
		{
			name: "user not in config",
			user: "kube",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {Username: "minikube"},
				},
			},
			err: "cannot delete user kube",
		},
		{
			name: "user in config",
			user: "kube",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {Username: "minikube"},
					"kube":     {Username: "kube"},
				},
			},
			err: "",
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
			pathOptions, err := tf.PathOptionsWithConfig(test.config)
			if err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			options := NewDeleteUserOptions(ioStreams, pathOptions)
			options.config = &test.config
			options.user = test.user

			if err := options.Validate(); err != nil {
				if !strings.Contains(err.Error(), test.err) {
					t.Fatalf("expected: %s but got %s", test.err, err.Error())
				}

				return
			}
		})
	}
}

func TestDeleteUserRun(t *testing.T) {
	var tests = []struct {
		name          string
		user          string
		config        clientcmdapi.Config
		expectedUsers []string
		out           string
	}{
		{
			name: "delete user",
			user: "kube",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {Username: "minikube"},
					"kube":     {Username: "kube"},
				},
			},
			expectedUsers: []string{"minikube"},
			out:           "deleted user kube from",
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			ioStreams, _, out, _ := genericiooptions.NewTestIOStreams()
			pathOptions, err := tf.PathOptionsWithConfig(test.config)
			if err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			options := NewDeleteUserOptions(ioStreams, pathOptions)
			options.config = &test.config
			options.configFile = pathOptions.GlobalFile
			options.user = test.user

			if err := options.Run(); err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			if got := out.String(); !strings.Contains(got, test.out) {
				t.Fatalf("expected: %s but got %s", test.out, got)
			}

			config, err := clientcmd.LoadFromFile(options.configFile)
			if err != nil {
				t.Fatalf("unexpected error executing command: %v", err)
			}

			users := make([]string, 0, len(config.AuthInfos))
			for user := range config.AuthInfos {
				users = append(users, user)
			}

			if !reflect.DeepEqual(test.expectedUsers, users) {
				t.Fatalf("expected %v, got %v", test.expectedUsers, users)
			}
		})
	}
}
