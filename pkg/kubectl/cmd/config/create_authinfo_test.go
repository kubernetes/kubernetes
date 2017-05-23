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
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func stringFlagFor(s string) flag.StringFlag {
	var f flag.StringFlag
	f.Set(s)
	return f
}

func TestCreateAuthInfoOptions(t *testing.T) {
	tests := []struct {
		flags           []string
		wantParseErr    bool
		wantCompleteErr bool
		wantValidateErr bool

		wantOptions *createAuthInfoOptions
	}{
		{
			flags: []string{
				"me",
			},
			wantOptions: &createAuthInfoOptions{
				name: "me",
			},
		},
		{
			flags: []string{
				"me",
				"--token=foo",
			},
			wantOptions: &createAuthInfoOptions{
				name:  "me",
				token: stringFlagFor("foo"),
			},
		},
		{
			flags: []string{
				"me",
				"--username=jane",
				"--password=bar",
			},
			wantOptions: &createAuthInfoOptions{
				name:     "me",
				username: stringFlagFor("jane"),
				password: stringFlagFor("bar"),
			},
		},
		{
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
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id=foo",
				"--auth-provider-arg=client-secret=bar",
				"me",
			},
			wantOptions: &createAuthInfoOptions{
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
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id-",
				"--auth-provider-arg=client-secret-",
				"me",
			},
			wantOptions: &createAuthInfoOptions{
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
			flags: []string{
				"--auth-provider-arg=client-id-", // auth provider name not required
				"--auth-provider-arg=client-secret-",
				"me",
			},
			wantOptions: &createAuthInfoOptions{
				name:             "me",
				authProviderArgs: map[string]string{},
				authProviderArgsToRemove: []string{
					"client-id",
					"client-secret",
				},
			},
		},
		{
			flags: []string{
				"--auth-provider=oidc",
				"--auth-provider-arg=client-id", // values must be of form 'key=value' or 'key-'
				"me",
			},
			wantCompleteErr: true,
		},
		{
			flags: []string{
			// No name for authinfo provided.
			},
			wantCompleteErr: true,
		},
	}

	for i, test := range tests {
		buff := new(bytes.Buffer)

		opts := new(createAuthInfoOptions)
		cmd := newCmdConfigSetAuthInfo(buff, opts)
		if err := cmd.ParseFlags(test.flags); err != nil {
			if !test.wantParseErr {
				t.Errorf("case %d: parsing error for flags %q: %v: %s", i, test.flags, err, buff)
			}
			continue
		}
		if test.wantParseErr {
			t.Errorf("case %d: expected parsing error for flags %q: %s", i, test.flags, buff)
			continue
		}

		if err := opts.complete(cmd, buff); err != nil {
			if !test.wantCompleteErr {
				t.Errorf("case %d: complete() error for flags %q: %s", i, test.flags, buff)
			}
			continue
		}
		if test.wantCompleteErr {
			t.Errorf("case %d: complete() expected errors for flags %q: %s", i, test.flags, buff)
			continue
		}

		if err := opts.validate(); err != nil {
			if !test.wantValidateErr {
				t.Errorf("case %d: flags %q: validate failed: %v", i, test.flags, err)
			}
			continue
		}

		if test.wantValidateErr {
			t.Errorf("case %d: flags %q: expected validate to fail", i, test.flags)
			continue
		}

		if !reflect.DeepEqual(opts, test.wantOptions) {
			t.Errorf("case %d: flags %q: mis-matched options,\nwanted=%#v\ngot=   %#v", i, test.flags, test.wantOptions, opts)
		}
	}
}

type createAuthInfoTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestCreateAuthInfo(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := createAuthInfoTest{
		description: "Testing for create aythinfo",
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
func (test createAuthInfoTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigSetAuthInfo(buf, pathOptions)
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
