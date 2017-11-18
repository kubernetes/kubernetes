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

package credentials

import (
	"encoding/base64"
	"fmt"
	"path"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ecr"

	"k8s.io/kubernetes/pkg/credentialprovider"
)

const user = "foo"
const password = "1234567890abcdef"
const email = "not@val.id"

// Mock implementation
type testTokenGetter struct {
	user     string
	password string
	endpoint string
}

func (p *testTokenGetter) GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error) {

	expiration := time.Now().Add(1 * time.Hour)
	creds := []byte(fmt.Sprintf("%s:%s", p.user, p.password))
	data := &ecr.AuthorizationData{
		AuthorizationToken: aws.String(base64.StdEncoding.EncodeToString(creds)),
		ExpiresAt:          &expiration,
		ProxyEndpoint:      aws.String(p.endpoint),
	}
	output := &ecr.GetAuthorizationTokenOutput{
		AuthorizationData: []*ecr.AuthorizationData{data},
	}

	return output, nil //p.svc.GetAuthorizationToken(input)
}

func TestEcrProvide(t *testing.T) {
	registry := "123456789012.dkr.ecr.lala-land-1.amazonaws.com"
	otherRegistries := []string{
		"123456789012.dkr.ecr.cn-foo-1.amazonaws.com.cn",
		"private.registry.com",
		"gcr.io",
	}
	image := "foo/bar"

	provider := newEcrProvider("lala-land-1",
		&testTokenGetter{
			user:     user,
			password: password,
			endpoint: registry,
		})

	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(provider.Provide())

	// Verify that we get the expected username/password combo for
	// an ECR image name.
	fullImage := path.Join(registry, image)
	creds, ok := keyring.Lookup(fullImage)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", fullImage)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if user != val.Username {
		t.Errorf("Unexpected username value, want: _token, got: %s", val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}

	// Verify that we get an error for other images.
	for _, otherRegistry := range otherRegistries {
		fullImage = path.Join(otherRegistry, image)
		creds, ok = keyring.Lookup(fullImage)
		if ok {
			t.Errorf("Unexpectedly found image: %s", fullImage)
			return
		}
	}
}

func TestChinaEcrProvide(t *testing.T) {
	registry := "123456789012.dkr.ecr.cn-foo-1.amazonaws.com.cn"
	otherRegistries := []string{
		"123456789012.dkr.ecr.lala-land-1.amazonaws.com",
		"private.registry.com",
		"gcr.io",
	}
	image := "foo/bar"

	provider := newEcrProvider("cn-foo-1",
		&testTokenGetter{
			user:     user,
			password: password,
			endpoint: registry,
		})

	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(provider.Provide())

	// Verify that we get the expected username/password combo for
	// an ECR image name.
	fullImage := path.Join(registry, image)
	creds, ok := keyring.Lookup(fullImage)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", fullImage)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if user != val.Username {
		t.Errorf("Unexpected username value, want: _token, got: %s", val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}

	// Verify that we get an error for other images.
	for _, otherRegistry := range otherRegistries {
		fullImage = path.Join(otherRegistry, image)
		creds, ok = keyring.Lookup(fullImage)
		if ok {
			t.Errorf("Unexpectedly found image: %s", fullImage)
			return
		}
	}
}
