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

package validation

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateSecret(t *testing.T) {
	// Opaque secret validation
	validSecret := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Data: map[string][]byte{
				"data-1": []byte("bar"),
			},
		}
	}

	var (
		emptyName     = validSecret()
		invalidName   = validSecret()
		emptyNs       = validSecret()
		invalidNs     = validSecret()
		overMaxSize   = validSecret()
		invalidKey    = validSecret()
		leadingDotKey = validSecret()
		dotKey        = validSecret()
		doubleDotKey  = validSecret()
	)

	emptyName.Name = ""
	invalidName.Name = "NoUppercaseOrSpecialCharsLike=Equals"
	emptyNs.Namespace = ""
	invalidNs.Namespace = "NoUppercaseOrSpecialCharsLike=Equals"
	overMaxSize.Data = map[string][]byte{
		"over": make([]byte, core.MaxSecretSize+1),
	}
	invalidKey.Data["a*b"] = []byte("whoops")
	leadingDotKey.Data[".key"] = []byte("bar")
	dotKey.Data["."] = []byte("bar")
	doubleDotKey.Data[".."] = []byte("bar")

	// kubernetes.io/service-account-token secret validation
	validServiceAccountTokenSecret := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: "bar",
				Annotations: map[string]string{
					core.ServiceAccountNameKey: "foo",
				},
			},
			Type: core.SecretTypeServiceAccountToken,
			Data: map[string][]byte{
				"data-1": []byte("bar"),
			},
		}
	}

	var (
		emptyTokenAnnotation    = validServiceAccountTokenSecret()
		missingTokenAnnotation  = validServiceAccountTokenSecret()
		missingTokenAnnotations = validServiceAccountTokenSecret()
	)
	emptyTokenAnnotation.Annotations[core.ServiceAccountNameKey] = ""
	delete(missingTokenAnnotation.Annotations, core.ServiceAccountNameKey)
	missingTokenAnnotations.Annotations = nil

	tests := map[string]struct {
		secret core.Secret
		valid  bool
	}{
		"valid":                                     {validSecret(), true},
		"empty name":                                {emptyName, false},
		"invalid name":                              {invalidName, false},
		"empty namespace":                           {emptyNs, false},
		"invalid namespace":                         {invalidNs, false},
		"over max size":                             {overMaxSize, false},
		"invalid key":                               {invalidKey, false},
		"valid service-account-token secret":        {validServiceAccountTokenSecret(), true},
		"empty service-account-token annotation":    {emptyTokenAnnotation, false},
		"missing service-account-token annotation":  {missingTokenAnnotation, false},
		"missing service-account-token annotations": {missingTokenAnnotations, false},
		"leading dot key":                           {leadingDotKey, true},
		"dot key":                                   {dotKey, false},
		"double dot key":                            {doubleDotKey, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateDockerConfigSecret(t *testing.T) {
	validDockerSecret := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       core.SecretTypeDockercfg,
			Data: map[string][]byte{
				core.DockerConfigKey: []byte(`{"https://index.docker.io/v1/": {"auth": "Y2x1ZWRyb29sZXIwMDAxOnBhc3N3b3Jk","email": "fake@example.com"}}`),
			},
		}
	}
	validDockerSecret2 := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       core.SecretTypeDockerConfigJson,
			Data: map[string][]byte{
				core.DockerConfigJsonKey: []byte(`{"auths":{"https://index.docker.io/v1/": {"auth": "Y2x1ZWRyb29sZXIwMDAxOnBhc3N3b3Jk","email": "fake@example.com"}}}`),
			},
		}
	}

	var (
		missingDockerConfigKey  = validDockerSecret()
		emptyDockerConfigKey    = validDockerSecret()
		invalidDockerConfigKey  = validDockerSecret()
		missingDockerConfigKey2 = validDockerSecret2()
		emptyDockerConfigKey2   = validDockerSecret2()
		invalidDockerConfigKey2 = validDockerSecret2()
	)

	delete(missingDockerConfigKey.Data, core.DockerConfigKey)
	emptyDockerConfigKey.Data[core.DockerConfigKey] = []byte("")
	invalidDockerConfigKey.Data[core.DockerConfigKey] = []byte("bad")
	delete(missingDockerConfigKey2.Data, core.DockerConfigJsonKey)
	emptyDockerConfigKey2.Data[core.DockerConfigJsonKey] = []byte("")
	invalidDockerConfigKey2.Data[core.DockerConfigJsonKey] = []byte("bad")

	tests := map[string]struct {
		secret core.Secret
		valid  bool
	}{
		"valid dockercfg":     {validDockerSecret(), true},
		"missing dockercfg":   {missingDockerConfigKey, false},
		"empty dockercfg":     {emptyDockerConfigKey, false},
		"invalid dockercfg":   {invalidDockerConfigKey, false},
		"valid config.json":   {validDockerSecret2(), true},
		"missing config.json": {missingDockerConfigKey2, false},
		"empty config.json":   {emptyDockerConfigKey2, false},
		"invalid config.json": {invalidDockerConfigKey2, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateBasicAuthSecret(t *testing.T) {
	validBasicAuthSecret := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       core.SecretTypeBasicAuth,
			Data: map[string][]byte{
				core.BasicAuthUsernameKey: []byte("username"),
				core.BasicAuthPasswordKey: []byte("password"),
			},
		}
	}

	var (
		missingBasicAuthUsernamePasswordKeys = validBasicAuthSecret()
	)

	delete(missingBasicAuthUsernamePasswordKeys.Data, core.BasicAuthUsernameKey)
	delete(missingBasicAuthUsernamePasswordKeys.Data, core.BasicAuthPasswordKey)

	tests := map[string]struct {
		secret core.Secret
		valid  bool
	}{
		"valid": {validBasicAuthSecret(), true},
		"missing username and password": {missingBasicAuthUsernamePasswordKeys, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateSSHAuthSecret(t *testing.T) {
	validSSHAuthSecret := func() core.Secret {
		return core.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       core.SecretTypeSSHAuth,
			Data: map[string][]byte{
				core.SSHAuthPrivateKey: []byte("foo-bar-baz"),
			},
		}
	}

	missingSSHAuthPrivateKey := validSSHAuthSecret()

	delete(missingSSHAuthPrivateKey.Data, core.SSHAuthPrivateKey)

	tests := map[string]struct {
		secret core.Secret
		valid  bool
	}{
		"valid":               {validSSHAuthSecret(), true},
		"missing private key": {missingSSHAuthPrivateKey, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateTLSSecret(t *testing.T) {
	successCases := map[string]core.Secret{
		"empty certificate chain": {
			ObjectMeta: metav1.ObjectMeta{Name: "tls-cert", Namespace: "namespace"},
			Data: map[string][]byte{
				core.TLSCertKey:       []byte("public key"),
				core.TLSPrivateKeyKey: []byte("private key"),
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateSecret(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}
	errorCases := map[string]struct {
		secrets     core.Secret
		errorType   field.ErrorType
		errorDetail string
	}{
		"missing public key": {
			secrets: core.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "tls-cert"},
				Data: map[string][]byte{
					core.TLSCertKey: []byte("public key"),
				},
			},
			errorType: "FieldValueRequired",
		},
		"missing private key": {
			secrets: core.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "tls-cert"},
				Data: map[string][]byte{
					core.TLSCertKey: []byte("public key"),
				},
			},
			errorType: "FieldValueRequired",
		},
	}
	for k, v := range errorCases {
		if errs := ValidateSecret(&v.secrets); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %s with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}
