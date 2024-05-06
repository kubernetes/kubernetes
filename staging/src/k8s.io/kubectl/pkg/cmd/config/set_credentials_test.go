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
	"bytes"
	utiltesting "k8s.io/client-go/util/testing"
	"os"
	"reflect"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
)

func stringFlagFor(s string) cliflag.StringFlag {
	var f cliflag.StringFlag
	f.Set(s)
	return f
}

func TestSetCredentialsOptions(t *testing.T) {
	tests := []struct {
		name            string
		flags           []string
		wantParseErr    bool
		wantCompleteErr bool
		wantValidateErr bool

		wantOptions *setCredentialsOptions
	}{
		{
			name: "test1",
			flags: []string{
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name: "me",
			},
		},
		{
			name: "test2",
			flags: []string{
				"me",
				"--token=foo",
			},
			wantOptions: &setCredentialsOptions{
				name:  "me",
				token: stringFlagFor("foo"),
			},
		},
		{
			name: "test3",
			flags: []string{
				"me",
				"--username=jane",
				"--password=bar",
			},
			wantOptions: &setCredentialsOptions{
				name:     "me",
				username: stringFlagFor("jane"),
				password: stringFlagFor("bar"),
			},
		},
		{
			name: "test4",
			// Cannot provide both token and basic auth.
			flags: []string{
				"me",
				"--token=foo",
				"--username=jane",
				"--password=bar",
			},
			wantValidateErr: true,
		},
		{
			name: "test5",
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id=foo",
				"--auth-provider-arg=client-secret=bar",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:         "me",
				authProvider: stringFlagFor("oidc"),
				authProviderArgs: map[string]string{
					"client-id":     "foo",
					"client-secret": "bar",
				},
				authProviderArgsToRemove: []string{},
			},
		},
		{
			name: "test6",
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id-",
				"--auth-provider-arg=client-secret-",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:             "me",
				authProvider:     stringFlagFor("oidc"),
				authProviderArgs: map[string]string{},
				authProviderArgsToRemove: []string{
					"client-id",
					"client-secret",
				},
			},
		},
		{
			name: "test7",
			flags: []string{
				"--auth-provider-arg=client-id-", // auth provider name not required
				"--auth-provider-arg=client-secret-",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:             "me",
				authProviderArgs: map[string]string{},
				authProviderArgsToRemove: []string{
					"client-id",
					"client-secret",
				},
			},
		},
		{
			name: "test8",
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id", // values must be of form 'key=value' or 'key-'
				"me",
			},
			wantCompleteErr: true,
		},
		{
			name:  "test9",
			flags: []string{
				// No name for authinfo provided.
			},
			wantCompleteErr: true,
		},
		{
			name: "test10",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:        "me",
				execCommand: stringFlagFor("example-client-go-exec-plugin"),
			},
		},
		{
			name: "test11",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-arg=arg1",
				"--exec-arg=arg2",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:        "me",
				execCommand: stringFlagFor("example-client-go-exec-plugin"),
				execArgs:    []string{"arg1", "arg2"},
			},
		},
		{
			name: "test12",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-env=key1=val1",
				"--exec-env=key2=val2",
				"--exec-env=env-remove1-",
				"--exec-env=env-remove2-",
				"me",
			},
			wantOptions: &setCredentialsOptions{
				name:            "me",
				execCommand:     stringFlagFor("example-client-go-exec-plugin"),
				execEnv:         map[string]string{"key1": "val1", "key2": "val2"},
				execEnvToRemove: []string{"env-remove1", "env-remove2"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buff := new(bytes.Buffer)

			opts := new(setCredentialsOptions)
			cmd := newCmdConfigSetCredentials(buff, opts)
			if err := cmd.ParseFlags(tt.flags); err != nil {
				if !tt.wantParseErr {
					t.Errorf("case %s: parsing error for flags %q: %v: %s", tt.name, tt.flags, err, buff)
				}
				return
			}
			if tt.wantParseErr {
				t.Errorf("case %s: expected parsing error for flags %q: %s", tt.name, tt.flags, buff)
				return
			}

			if err := opts.complete(cmd); err != nil {
				if !tt.wantCompleteErr {
					t.Errorf("case %s: complete() error for flags %q: %s", tt.name, tt.flags, buff)
				}
				return
			}
			if tt.wantCompleteErr {
				t.Errorf("case %s: complete() expected errors for flags %q: %s", tt.name, tt.flags, buff)
				return
			}

			if err := opts.validate(); err != nil {
				if !tt.wantValidateErr {
					t.Errorf("case %s: flags %q: validate failed: %v", tt.name, tt.flags, err)
				}
				return
			}

			if tt.wantValidateErr {
				t.Errorf("case %s: flags %q: expected validate to fail", tt.name, tt.flags)
				return
			}

			if !reflect.DeepEqual(opts, tt.wantOptions) {
				t.Errorf("case %s: flags %q: mis-matched options,\nwanted=%#v\ngot=   %#v", tt.name, tt.flags, tt.wantOptions, opts)
			}
		})
	}
}

func TestModifyExistingAuthInfo(t *testing.T) {
	tests := []struct {
		name            string
		flags           []string
		wantParseErr    bool
		wantCompleteErr bool
		wantValidateErr bool

		existingAuthInfo clientcmdapi.AuthInfo
		wantAuthInfo     clientcmdapi.AuthInfo
	}{
		{
			name: "1. create new exec config",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-api-version=client.authentication.k8s.io/v1",
				"me",
			},
			existingAuthInfo: clientcmdapi.AuthInfo{},
			wantAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1",
				},
			},
		},
		{
			name: "2. redefine exec args",
			flags: []string{
				"--exec-arg=new-arg1",
				"--exec-arg=new-arg2",
				"me",
			},
			existingAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1beta1",
					Args:       []string{"existing-arg1", "existing-arg2"},
				},
			},
			wantAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1beta1",
					Args:       []string{"new-arg1", "new-arg2"},
				},
			},
		},
		{
			name: "3. reset exec args",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"me",
			},
			existingAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1beta1",
					Args:       []string{"existing-arg1", "existing-arg2"},
				},
			},
			wantAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1beta1",
				},
			},
		},
		{
			name: "4. modify exec env variables",
			flags: []string{
				"--exec-command=example-client-go-exec-plugin",
				"--exec-env=name1=value1000",
				"--exec-env=name3=value3",
				"--exec-env=name2-",
				"--exec-env=non-existing-",
				"me",
			},
			existingAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "existing-command",
					APIVersion: "client.authentication.k8s.io/v1beta1",
					Env: []clientcmdapi.ExecEnvVar{
						{Name: "name1", Value: "value1"},
						{Name: "name2", Value: "value2"},
					},
				},
			},
			wantAuthInfo: clientcmdapi.AuthInfo{
				Exec: &clientcmdapi.ExecConfig{
					Command:    "example-client-go-exec-plugin",
					APIVersion: "client.authentication.k8s.io/v1beta1",
					Env: []clientcmdapi.ExecEnvVar{
						{Name: "name1", Value: "value1000"},
						{Name: "name3", Value: "value3"},
					},
				},
			},
		},
		{
			name: "5. modify auth provider arguments",
			flags: []string{
				"--auth-provider=new-auth-provider",
				"--auth-provider-arg=key1=val1000",
				"--auth-provider-arg=key3=val3",
				"--auth-provider-arg=key2-",
				"--auth-provider-arg=non-existing-",
				"me",
			},
			existingAuthInfo: clientcmdapi.AuthInfo{
				AuthProvider: &clientcmdapi.AuthProviderConfig{
					Name: "auth-provider",
					Config: map[string]string{
						"key1": "val1",
						"key2": "val2",
					},
				},
			},
			wantAuthInfo: clientcmdapi.AuthInfo{
				AuthProvider: &clientcmdapi.AuthProviderConfig{
					Name: "new-auth-provider",
					Config: map[string]string{
						"key1": "val1000",
						"key3": "val3",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buff := new(bytes.Buffer)

			opts := new(setCredentialsOptions)
			cmd := newCmdConfigSetCredentials(buff, opts)
			if err := cmd.ParseFlags(tt.flags); err != nil {
				if !tt.wantParseErr {
					t.Errorf("case %s: parsing error for flags %q: %v: %s", tt.name, tt.flags, err, buff)
				}
				return
			}
			if tt.wantParseErr {
				t.Errorf("case %s: expected parsing error for flags %q: %s", tt.name, tt.flags, buff)
				return
			}

			if err := opts.complete(cmd); err != nil {
				if !tt.wantCompleteErr {
					t.Errorf("case %s: complete() error for flags %q: %s", tt.name, tt.flags, buff)
				}
				return
			}
			if tt.wantCompleteErr {
				t.Errorf("case %s: complete() expected errors for flags %q: %s", tt.name, tt.flags, buff)
				return
			}

			if err := opts.validate(); err != nil {
				if !tt.wantValidateErr {
					t.Errorf("case %s: flags %q: validate failed: %v", tt.name, tt.flags, err)
				}
				return
			}

			if tt.wantValidateErr {
				t.Errorf("case %s: flags %q: expected validate to fail", tt.name, tt.flags)
				return
			}

			modifiedAuthInfo := opts.modifyAuthInfo(tt.existingAuthInfo)

			if !reflect.DeepEqual(modifiedAuthInfo, tt.wantAuthInfo) {
				t.Errorf("case %s: flags %q: mis-matched auth info,\nwanted=%#v\ngot=   %#v", tt.name, tt.flags, tt.wantAuthInfo, modifiedAuthInfo)
			}
		})
	}
}

type setCredentialsTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestSetCredentials(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := setCredentialsTest{
		description: "Testing set credentials",
		config:      conf,
		args:        []string{"cluster-admin"},
		flags: []string{
			"--username=admin",
			"--password=uXFGweU9l35qcif",
		},
		expected: `User "cluster-admin" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			AuthInfos: map[string]*clientcmdapi.AuthInfo{
				"cluster-admin": {Username: "admin", Password: "uXFGweU9l35qcif"}},
		},
	}
	test.run(t)
}
func (test setCredentialsTest) run(t *testing.T) {
	fakeKubeFile, err := os.CreateTemp(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, fakeKubeFile)
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigSetCredentials(buf, pathOptions)
	cmd.SetArgs(test.args)
	cmd.Flags().Parse(test.flags)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v,kubectl config set-credentials  args: %v,flags: %v", err, test.args, test.flags)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Fail in %q:\n expected %v\n but got %v\n", test.description, test.expected, buf.String())
		}
	}
	if test.expectedConfig.AuthInfos != nil {
		expectAuthInfo := test.expectedConfig.AuthInfos[test.args[0]]
		actualAuthInfo := config.AuthInfos[test.args[0]]
		if expectAuthInfo.Username != actualAuthInfo.Username || expectAuthInfo.Password != actualAuthInfo.Password {
			t.Errorf("Fail in %q:\n expected AuthInfo%v\n but found %v in kubeconfig\n", test.description, expectAuthInfo, actualAuthInfo)
		}
	}
}
