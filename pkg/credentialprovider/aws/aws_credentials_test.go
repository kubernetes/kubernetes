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
	"math/rand"
	"path"
	"strconv"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ecr"

	"k8s.io/kubernetes/pkg/credentialprovider"
)

const user = "foo"
const password = "1234567890abcdef" // Fake value for testing.
const email = "not@val.id"

// Mock implementation
// randomizePassword is used to check for a cache hit to verify the password
// has not changed
type testTokenGetter struct {
	user              string
	password          string
	endpoint          string
	randomizePassword bool
}

type testTokenGetterFactory struct {
	getter tokenGetter
}

func (f *testTokenGetterFactory) GetTokenGetterForRegion(region string) (tokenGetter, error) {
	return f.getter, nil
}

func (p *testTokenGetter) GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error) {
	if p.randomizePassword {
		rand.Seed(int64(time.Now().Nanosecond()))
		p.password = strconv.Itoa(rand.Int())
	}
	expiration := time.Now().Add(1 * time.Hour)
	// expiration := time.Now().Add(5 * time.Second) //for testing with the cache expiring
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

func TestRegistryPatternMatch(t *testing.T) {
	grid := []struct {
		Registry string
		Expected bool
	}{
		{"123456789012.dkr.ecr.lala-land-1.amazonaws.com", true},
		// fips
		{"123456789012.dkr.ecr-fips.lala-land-1.amazonaws.com", true},
		// .cn
		{"123456789012.dkr.ecr.lala-land-1.amazonaws.com.cn", true},
		// registry ID too long
		{"1234567890123.dkr.ecr.lala-land-1.amazonaws.com", false},
		// registry ID too short
		{"12345678901.dkr.ecr.lala-land-1.amazonaws.com", false},
		// registry ID has invalid chars
		{"12345678901A.dkr.ecr.lala-land-1.amazonaws.com", false},
		// region has invalid chars
		{"123456789012.dkr.ecr.lala-land-1!.amazonaws.com", false},
		// region starts with invalid char
		{"123456789012.dkr.ecr.#lala-land-1.amazonaws.com", false},
		// invalid host suffix
		{"123456789012.dkr.ecr.lala-land-1.amazonaws.hacker.com", false},
		// invalid host suffix
		{"123456789012.dkr.ecr.lala-land-1.hacker.com", false},
		// invalid host suffix
		{"123456789012.dkr.ecr.lala-land-1.amazonaws.lol", false},
		// without dkr
		{"123456789012.dog.ecr.lala-land-1.amazonaws.com", false},
		// without ecr
		{"123456789012.dkr.cat.lala-land-1.amazonaws.com", false},
		// without amazonaws
		{"123456789012.dkr.cat.lala-land-1.awsamazon.com", false},
		// too short
		{"123456789012.lala-land-1.amazonaws.com", false},
	}
	for _, g := range grid {
		actual := ecrPattern.MatchString(g.Registry)
		if actual != g.Expected {
			t.Errorf("unexpected pattern match value, want %v for %s", g.Expected, g.Registry)
		}
	}
}

func TestParseRepoURLPass(t *testing.T) {
	registryID := "123456789012"
	region := "lala-land-1"
	port := "9001"
	registry := "123456789012.dkr.ecr.lala-land-1.amazonaws.com"
	image := path.Join(registry, port, "foo/bar")
	parsedURL, err := parseRepoURL(image)

	if err != nil {
		t.Errorf("Could not parse URL: %s, err: %v", image, err)
	}
	if registryID != parsedURL.registryID {
		t.Errorf("Unexpected registryID value, want: %s, got: %s", registryID, parsedURL.registryID)
	}
	if region != parsedURL.region {
		t.Errorf("Unexpected region value, want: %s, got: %s", region, parsedURL.region)
	}
	if registry != parsedURL.registry {
		t.Errorf("Unexpected registry value, want: %s, got: %s", registry, parsedURL.registry)
	}
}

func TestParseRepoURLFail(t *testing.T) {
	registry := "123456789012.foo.bar.baz"
	image := path.Join(registry, "foo/bar")
	parsedURL, err := parseRepoURL(image)
	expectedErr := "123456789012.foo.bar.baz is not a valid ECR repository URL"

	if err == nil {
		t.Errorf("Should fail to parse URL %s", image)
	}
	if err.Error() != expectedErr {
		t.Errorf("Unexpected error, want: %s, got: %v", expectedErr, err)
	}
	if parsedURL != nil {
		t.Errorf("Expected parsedURL to be nil")
	}
}

func TestECRProvide(t *testing.T) {
	registry := "123456789012.dkr.ecr.lala-land-1.amazonaws.com"
	otherRegistries := []string{
		"123456789012.dkr.ecr.cn-foo-1.amazonaws.com.cn",
		"private.registry.com",
		"gcr.io",
	}
	image := path.Join(registry, "foo/bar")
	p := newECRProvider(&testTokenGetterFactory{
		getter: &testTokenGetter{
			user:     user,
			password: password,
			endpoint: registry,
		},
	})
	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(p.Provide(image))

	// Verify that we get the expected username/password combo for
	// an ECR image name.
	creds, ok := keyring.Lookup(image)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", image)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	cred := creds[0]
	if user != cred.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", user, cred.Username)
	}
	if password != creds[0].Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, cred.Password)
	}
	if email != creds[0].Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, cred.Email)
	}

	// Verify that we get an error for other images.
	for _, otherRegistry := range otherRegistries {
		image = path.Join(otherRegistry, "foo/bar")
		_, ok = keyring.Lookup(image)
		if ok {
			t.Errorf("Unexpectedly found image: %s", image)
			return
		}
	}
}

func TestECRProvideCached(t *testing.T) {
	registry := "123456789012.dkr.ecr.lala-land-1.amazonaws.com"
	p := newECRProvider(&testTokenGetterFactory{
		getter: &testTokenGetter{
			user:              user,
			password:          password,
			endpoint:          registry,
			randomizePassword: true,
		},
	})
	image1 := path.Join(registry, "foo/bar")
	image2 := path.Join(registry, "bar/baz")
	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(p.Provide(image1))
	// time.Sleep(6 * time.Second) //for testing with the cache expiring
	keyring.Add(p.Provide(image2))
	// Verify that we get the credentials from the
	// cache the second time
	creds1, ok := keyring.Lookup(image1)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", image1)
		return
	}
	if len(creds1) != 2 {
		t.Errorf("Got more hits than expected: %s", creds1)
	}

	if creds1[0].Password != creds1[1].Password {
		t.Errorf("cached credentials do not match")
	}

	creds2, ok := keyring.Lookup(image2)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", image1)
		return
	}
	if len(creds2) != 2 {
		t.Errorf("Got more hits than expected: %s", creds2)
	}

	if creds2[0].Password != creds2[1].Password {
		t.Errorf("cached credentials do not match")
	}
	if creds1[0].Password != creds2[0].Password {
		t.Errorf("cached credentials do not match")
	}
}

func TestChinaECRProvide(t *testing.T) {
	registry := "123456789012.dkr.ecr.cn-foo-1.amazonaws.com.cn"
	otherRegistries := []string{
		"123456789012.dkr.ecr.lala-land-1.amazonaws.com",
		"private.registry.com",
		"gcr.io",
	}
	image := path.Join(registry, "foo/bar")
	p := newECRProvider(&testTokenGetterFactory{
		getter: &testTokenGetter{
			user:     user,
			password: password,
			endpoint: registry,
		},
	})
	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(p.Provide(image))
	// Verify that we get the expected username/password combo for
	// an ECR image name.
	creds, ok := keyring.Lookup(image)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", image)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	cred := creds[0]
	if user != cred.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", user, cred.Username)
	}
	if password != cred.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, cred.Password)
	}
	if email != cred.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, cred.Email)
	}

	// Verify that we get an error for other images.
	for _, otherRegistry := range otherRegistries {
		image = path.Join(otherRegistry, image)
		_, ok = keyring.Lookup(image)
		if ok {
			t.Errorf("Unexpectedly found image: %s", image)
			return
		}
	}
}

func TestChinaECRProvideCached(t *testing.T) {
	registry := "123456789012.dkr.ecr.cn-foo-1.amazonaws.com.cn"
	p := newECRProvider(&testTokenGetterFactory{
		getter: &testTokenGetter{
			user:              user,
			password:          password,
			endpoint:          registry,
			randomizePassword: true,
		},
	})
	image := path.Join(registry, "foo/bar")
	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(p.Provide(image))
	// time.Sleep(6 * time.Second) //for testing with the cache expiring
	keyring.Add(p.Provide(image))
	// Verify that we get the credentials from the
	// cache the second time
	creds, ok := keyring.Lookup(image)
	if !ok {
		t.Errorf("Didn't find expected URL: %s", image)
		return
	}
	if len(creds) != 2 {
		t.Errorf("Got more hits than expected: %s", creds)
	}

	if creds[0].Password != creds[1].Password {
		t.Errorf("cached credentials do not match")
	}
}
