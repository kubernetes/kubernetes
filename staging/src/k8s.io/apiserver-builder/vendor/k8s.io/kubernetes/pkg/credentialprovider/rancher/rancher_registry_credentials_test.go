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

package rancher_credentials

import (
	"path"
	"testing"

	"github.com/rancher/go-rancher/client"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

const username = "foo"
const password = "qwerty"
const email = "foo@bar.baz"

var serverAddresses = []string{"quay.io", "192.168.5.0"}

type testCredentialsGetter struct {
	client *client.RancherClient
}

func (p *testCredentialsGetter) getCredentials() []registryCredential {
	var registryCreds []registryCredential

	for _, serverAddress := range serverAddresses {
		cred := &client.RegistryCredential{
			PublicValue: username,
			SecretValue: password,
			Email:       email,
		}
		registryCred := registryCredential{
			credential: cred,
			serverIP:   serverAddress,
		}
		registryCreds = append(registryCreds, registryCred)
	}

	return registryCreds
}

func TestRancherCredentialsProvide(t *testing.T) {
	image := "foo/bar"

	url := "http://localhost:8080"
	accessKey := "B481F55E0C48C546E094"
	secretKey := "dND2fBcytWWvCRJ8LvqnYcjyNfEkaikvfVxk2C5r"
	conf := rConfig{
		Global: configGlobal{
			CattleURL:       url,
			CattleAccessKey: accessKey,
			CattleSecretKey: secretKey,
		},
	}

	rancherClient, _ := client.NewRancherClient(&client.ClientOpts{
		Url:       conf.Global.CattleURL,
		AccessKey: conf.Global.CattleAccessKey,
		SecretKey: conf.Global.CattleSecretKey,
	})

	testGetter := &testCredentialsGetter{
		client: rancherClient,
	}

	provider := &rancherProvider{
		credGetter: testGetter,
	}

	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(provider.Provide())

	for _, registry := range serverAddresses {
		fullImagePath := path.Join(registry, image)
		creds, ok := keyring.Lookup(fullImagePath)
		if !ok {
			t.Errorf("Didn't find expected image: %s", fullImagePath)
			return
		}

		if len(creds) > 1 {
			t.Errorf("Expected 1 result, received %v", len(creds))
		}

		val := creds[0]

		if username != val.Username {
			t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
		}
		if password != val.Password {
			t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
		}
		if email != val.Email {
			t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
		}
	}

	// try to fetch non-existing registry
	fullImagePath := path.Join("1.1.1.1", image)
	_, ok := keyring.Lookup(fullImagePath)
	if ok {
		t.Errorf("Found non-existing image: %s", fullImagePath)
	}

	return
}
