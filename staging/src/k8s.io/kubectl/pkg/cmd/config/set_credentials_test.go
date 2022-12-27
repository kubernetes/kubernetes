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
	"os"
	"os/exec"
	"reflect"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestSetCredentials(t *testing.T) {
	var tests = []struct {
		name       string
		initConfig clientcmdapi.Config
		args       []string
		flags      []string
		expected   string

		expectedConfig *clientcmdapi.Config
	}{
		{
			name:       "Set credential name",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags:      []string{},
			expected:   "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": new(clientcmdapi.AuthInfo),
				},
			},
		},
		{
			name:       "Set credential token",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--token=foo",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Token: "foo",
					},
				},
			},
		},
		{
			name:       "Set credential username and password",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--username=jane",
				"--password=bar",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Username: "jane",
						Password: "bar",
					},
				},
			},
		},
		{
			name:       "Set credential auth provider and auth provider args",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id=foo",
				"--auth-provider-arg=client-secret=bar",
			},
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "oidc",
							Config: map[string]string{
								"client-id":     "foo",
								"client-secret": "bar",
							},
						},
					},
				},
			},
		},
		{
			name: "Remove credential auth provider args with oidc",
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "oidc",
							Config: map[string]string{
								"cliend-id":     "foo",
								"client-secret": "bar",
							},
						},
					},
				},
			},
			args: []string{"me"},
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id-",
				"--auth-provider-arg=client-secret-",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name:   "oidc",
							Config: map[string]string{},
						},
					},
				},
			},
		},
		{
			name: "Remove credential auth provider args without oidc",
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "oidc",
							Config: map[string]string{
								"cliend-id":     "foo",
								"client-secret": "bar",
							},
						},
					},
				},
			},
			args: []string{"me"},
			flags: []string{
				"--auth-provider-arg=client-id-", // auth provider name not required
				"--auth-provider-arg=client-secret-",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name:   "oidc",
							Config: map[string]string{},
						},
					},
				},
			},
		},
		{
			name:       "Set exec command",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command: "example-client-go-exec-plugin",
						},
					},
				},
			},
		},
		{
			name:       "Set exec command with exec args",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-arg=arg1",
				"--exec-arg=arg2",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command: "example-client-go-exec-plugin",
							Args: []string{
								"arg1",
								"arg2",
							},
						},
					},
				},
			},
		},
		{
			name: "Set exec command, set exec env vars, and remove exec env vars",
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Env: []clientcmdapi.ExecEnvVar{
								{
									Name:  "env-remove1",
									Value: "val1",
								},
								{
									Name:  "env-remove2",
									Value: "val2",
								},
							},
						},
					},
				},
			},
			args: []string{"me"},
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-env=key1=val1",
				"--exec-env=key2=val2",
				"--exec-env=env-remove1-",
				"--exec-env=env-remove2-",
			},
			expected: "User \"me\" set.\n",
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command: "example-client-go-exec-plugin",
							Env: []clientcmdapi.ExecEnvVar{
								{
									Name:  "key1",
									Value: "val1",
								},
								{
									Name:  "key2",
									Value: "val2",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "Update exec args",
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "example-client-go-exec-plugin",
							APIVersion: "client.authentication.k8s.io/v1beta1",
							Args:       []string{"existing-arg1", "existing-arg2"},
						},
					},
				},
			},
			args: []string{"me"},
			flags: []string{
				"--exec-arg=new-arg1",
				"--exec-arg=new-arg2",
			},
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "example-client-go-exec-plugin",
							APIVersion: "client.authentication.k8s.io/v1beta1",
							Args:       []string{"new-arg1", "new-arg2"},
						},
					},
				},
			},
		},
		{
			name: "Delete exec args",
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "example-client-go-exec-plugin",
							APIVersion: "client.authentication.k8s.io/v1beta1",
							Args:       []string{"existing-arg1", "existing-arg2"},
						},
					},
				},
			},
			args: []string{"me"},
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
			},
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "example-client-go-exec-plugin",
							APIVersion: "client.authentication.k8s.io/v1beta1",
						},
					},
				},
			},
		},
		{
			name: "Update existing exec env variables",
			args: []string{"me"},
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-env=name1=value1000",
				"--exec-env=name3=value3",
				"--exec-env=name2-",
				"--exec-env=non-existing-",
			},
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "existing-command",
							APIVersion: "client.authentication.k8s.io/v1beta1",
							Env: []clientcmdapi.ExecEnvVar{
								{Name: "name1", Value: "value1"},
								{Name: "name2", Value: "value2"},
							},
						},
					},
				},
			},
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						Exec: &clientcmdapi.ExecConfig{
							Command:    "existing-command",
							APIVersion: "client.authentication.k8s.io/v1beta1",
							Env: []clientcmdapi.ExecEnvVar{
								{Name: "name1", Value: "value1000"},
								{Name: "name3", Value: "value3"},
							},
						},
					},
				},
			},
		},
		{
			name: "Update auth provider arguments",
			args: []string{"me"},
			flags: []string{
				"--auth-provider=new-auth-provider",
				"--auth-provider-arg=key1=val1000",
				"--auth-provider-arg=key3=val3",
				"--auth-provider-arg=key2-",
				"--auth-provider-arg=non-existing-",
			},
			initConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "auth-provider",
							Config: map[string]string{
								"key1": "val1",
								"key2": "val2",
							},
						},
					},
				},
			},
			expectedConfig: &clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"me": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "new-auth-provider",
							Config: map[string]string{
								"key1": "val1000",
								"key3": "val3",
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeKubeFile, err := os.CreateTemp(os.TempDir(), "")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer removeTempFile(t, fakeKubeFile.Name())
			err = clientcmd.WriteToFile(tt.initConfig, fakeKubeFile.Name())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""
			streams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdConfigSetCredentials(streams, pathOptions)
			cmd.SetArgs(tt.args)
			if err := cmd.Flags().Parse(tt.flags); err != nil {
				t.Fatalf("unexpected error parsing flags: %v", err)
			}
			err = cmd.Execute()
			if err != nil {
				t.Fatalf("unexpected error executing command: %v,kubectl set-context args: %v,flags: %v", err, tt.args, tt.flags)
			}
			config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
			if err != nil {
				t.Fatalf("unexpected error loading kubeconfig file: %v", err)
			}
			if tt.expected != "" {
				if buf.String() != tt.expected {
					t.Errorf("Fail in %q:\n expected %v\n but got %v\n", tt.name, tt.expected, buf.String())
				}
			}
			if tt.expectedConfig != nil {
				if reflect.DeepEqual(tt.expectedConfig, config) {
					t.Errorf("Fail in %q:\n expected %v\n but found %v in kubeconfig\n", tt.name, tt.expectedConfig, config)
				}
			}
		})
	}
}

func TestSetCredentialsErrors(t *testing.T) {
	var tests = []struct {
		name        string
		initConfig  clientcmdapi.Config
		args        []string
		flags       []string
		expected    string
		expectedErr bool

		expectedConfig *clientcmdapi.Config
	}{
		{
			name:       "Error: Malformed auth provider arg",
			initConfig: clientcmdapi.Config{},
			args:       []string{"me"},
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id", // values must be of form 'key=value' or 'key-'
			},
			expected:    "",
			expectedErr: true,
		},
		{
			name:        "Error: No name provided",
			initConfig:  clientcmdapi.Config{},
			args:        []string{},
			flags:       []string{},
			expected:    "",
			expectedErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(tt.initConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""
			streams, _, _, _ := genericclioptions.NewTestIOStreams()

			cmdSetCredentials := NewCmdConfigSetCredentials(streams, pathOptions)
			cmdSetCredentials.SetArgs(tt.args)
			if err := cmdSetCredentials.Flags().Parse(tt.flags); err != nil {
				t.Fatalf("unexpected error parsing flags: %v", err)
			}

			// Disable exit codes causing testing errors when we expect them
			if os.Getenv("CRASH") == "1" {
				cmdSetCredentials.Execute()
				return
			}
			cmd := exec.Command(os.Args[0], "-test.run=^TestSetCredentialsErrors$")
			cmd.Env = append(os.Environ(), "CRASH=1")
			err = cmd.Run()
			if execErr, ok := err.(*exec.ExitError); ok && !execErr.Success() {
				return
			}
			t.Fatalf("expected an error but found none")
		})
	}
}
