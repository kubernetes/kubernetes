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

package clientcmd

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestConfirmUsableBadInfoButOkConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["missing ca"] = &clientcmdapi.Cluster{
		Server:               "anything",
		CertificateAuthority: "missing",
	}
	config.AuthInfos["error"] = &clientcmdapi.AuthInfo{
		Username: "anything",
		Token:    "here",
	}
	config.Contexts["dirty"] = &clientcmdapi.Context{
		Cluster:  "missing ca",
		AuthInfo: "error",
	}
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "anything",
	}
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		Token: "here",
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}

	badValidation := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read certificate-authority"},
	}
	okTest := configValidationTest{
		config: config,
	}

	okTest.testConfirmUsable("clean", t)
	badValidation.testConfig(t)
}

func TestConfirmUsableMissingObjects(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["kind-cluster"] = &clientcmdapi.Cluster{
		Server: "anything",
	}
	config.AuthInfos["kind-user"] = &clientcmdapi.AuthInfo{
		Token: "any-value",
	}
	config.Contexts["missing-user"] = &clientcmdapi.Context{
		Cluster:  "kind-cluster",
		AuthInfo: "garbage",
	}
	config.Contexts["missing-cluster"] = &clientcmdapi.Context{
		Cluster:  "garbage",
		AuthInfo: "kind-user",
	}

	missingUser := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`user "garbage" was not found for context "missing-user"`,
		},
	}
	missingUser.testConfirmUsable("missing-user", t)
	missingUser.testConfig(t)

	missingCluster := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`cluster "garbage" was not found for context "missing-cluster"`,
		},
	}
	missingCluster.testConfirmUsable("missing-cluster", t)
	missingCluster.testConfig(t)
}

func TestConfirmUsableBadInfoConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["missing ca"] = &clientcmdapi.Cluster{
		Server:               "anything",
		CertificateAuthority: "missing",
	}
	config.AuthInfos["error"] = &clientcmdapi.AuthInfo{
		Username: "anything",
		Token:    "here",
	}
	config.Contexts["first"] = &clientcmdapi.Context{
		Cluster:  "missing ca",
		AuthInfo: "error",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read certificate-authority"},
	}

	test.testConfirmUsable("first", t)
}

func TestConfirmUsableEmptyConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"invalid configuration: no configuration has been provided"},
	}

	test.testConfirmUsable("", t)
}

func TestConfirmUsableMissingConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"invalid configuration: no configuration has been provided"},
	}

	test.testConfirmUsable("not-here", t)
}

func TestValidateEmptyConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"invalid configuration: no configuration has been provided"},
	}

	test.testConfig(t)
}

func TestValidateMissingCurrentContextConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.CurrentContext = "anything"
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"context was not found for specified "},
	}

	test.testConfig(t)
}

func TestIsContextNotFound(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.CurrentContext = "anything"

	err := Validate(*config)
	if !IsContextNotFound(err) {
		t.Errorf("Expected context not found, but got %v", err)
	}
	if !IsConfigurationInvalid(err) {
		t.Errorf("Expected configuration invalid, but got %v", err)
	}
}

func TestIsEmptyConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()

	err := Validate(*config)
	if !IsEmptyConfig(err) {
		t.Errorf("Expected context not found, but got %v", err)
	}
	if !IsConfigurationInvalid(err) {
		t.Errorf("Expected configuration invalid, but got %v", err)
	}
}

func TestIsConfigurationInvalid(t *testing.T) {
	if newErrConfigurationInvalid([]error{}) != nil {
		t.Errorf("unexpected error")
	}
	if newErrConfigurationInvalid([]error{ErrNoContext}) == ErrNoContext {
		t.Errorf("unexpected error")
	}
	if newErrConfigurationInvalid([]error{ErrNoContext, ErrNoContext}) == nil {
		t.Errorf("unexpected error")
	}
	if !IsConfigurationInvalid(newErrConfigurationInvalid([]error{ErrNoContext, ErrNoContext})) {
		t.Errorf("unexpected error")
	}
}

func TestValidateMissingReferencesConfig(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.CurrentContext = "anything"
	config.Contexts["anything"] = &clientcmdapi.Context{Cluster: "missing", AuthInfo: "missing"}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"user \"missing\" was not found for context \"anything\"", "cluster \"missing\" was not found for context \"anything\""},
	}

	test.testContext("anything", t)
	test.testConfig(t)
}

func TestValidateEmptyContext(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.CurrentContext = "anything"
	config.Contexts["anything"] = &clientcmdapi.Context{}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"user was not specified for context \"anything\"", "cluster was not specified for context \"anything\""},
	}

	test.testContext("anything", t)
	test.testConfig(t)
}

func TestValidateEmptyContextName(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.CurrentContext = "anything"
	config.Contexts[""] = &clientcmdapi.Context{Cluster: "missing", AuthInfo: "missing"}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"empty context name", "is not allowed"},
	}

	test.testContext("", t)
	test.testConfig(t)
}

func TestValidateEmptyClusterInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["empty"] = clientcmdapi.NewCluster()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"cluster has no server defined"},
	}

	test.testCluster("empty", t)
	test.testConfig(t)
}

func TestValidateClusterInfoErrEmptyCluster(t *testing.T) {
	cluster := clientcmdapi.NewCluster()
	errs := validateClusterInfo("", *cluster)

	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs[0] != ErrEmptyCluster {
		t.Errorf("unexpected error: %v", errs[0])
	}
}

func TestValidateMissingCAFileClusterInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["missing ca"] = &clientcmdapi.Cluster{
		Server:               "anything",
		CertificateAuthority: "missing",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read certificate-authority"},
	}

	test.testCluster("missing ca", t)
	test.testConfig(t)
}

func TestValidateCleanClusterInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "anything",
	}
	test := configValidationTest{
		config: config,
	}

	test.testCluster("clean", t)
	test.testConfig(t)
}

func TestValidateCleanWithCAClusterInfo(t *testing.T) {
	tempFile, _ := os.CreateTemp("", "")
	defer os.Remove(tempFile.Name())

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server:               "anything",
		CertificateAuthority: tempFile.Name(),
	}
	test := configValidationTest{
		config: config,
	}

	test.testCluster("clean", t)
	test.testConfig(t)
}

func TestValidateEmptyAuthInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["error"] = &clientcmdapi.AuthInfo{}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("error", t)
	test.testConfig(t)
}

func TestValidateCertFilesNotFoundAuthInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["error"] = &clientcmdapi.AuthInfo{
		ClientCertificate: "missing",
		ClientKey:         "missing",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read client-cert", "unable to read client-key"},
	}

	test.testAuthInfo("error", t)
	test.testConfig(t)
}

func TestValidateCertDataOverridesFiles(t *testing.T) {
	tempFile, _ := os.CreateTemp("", "")
	defer os.Remove(tempFile.Name())

	config := clientcmdapi.NewConfig()
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		ClientCertificate:     tempFile.Name(),
		ClientCertificateData: []byte("certdata"),
		ClientKey:             tempFile.Name(),
		ClientKeyData:         []byte("keydata"),
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"client-cert-data and client-cert are both specified", "client-key-data and client-key are both specified"},
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}

func TestValidateCleanCertFilesAuthInfo(t *testing.T) {
	tempFile, _ := os.CreateTemp("", "")
	defer os.Remove(tempFile.Name())

	config := clientcmdapi.NewConfig()
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		ClientCertificate: tempFile.Name(),
		ClientKey:         tempFile.Name(),
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}

func TestValidateCleanTokenAuthInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		Token: "any-value",
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}

func TestValidateMultipleMethodsAuthInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["error"] = &clientcmdapi.AuthInfo{
		Token:    "token",
		Username: "username",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"more than one authentication method", "token", "basicAuth"},
	}

	test.testAuthInfo("error", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExec(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			Command:    "/bin/example",
			APIVersion: "clientauthentication.k8s.io/v1alpha1",
			Args:       []string{"hello", "world"},
			Env: []clientcmdapi.ExecEnvVar{
				{Name: "foo", Value: "bar"},
			},
			InteractiveMode: clientcmdapi.IfAvailableExecInteractiveMode,
		},
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecNoVersion(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			Command:         "/bin/example",
			InteractiveMode: clientcmdapi.IfAvailableExecInteractiveMode,
		},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			"apiVersion must be specified for user to use exec authentication plugin",
		},
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecNoCommand(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			APIVersion:      "clientauthentication.k8s.io/v1alpha1",
			InteractiveMode: clientcmdapi.IfAvailableExecInteractiveMode,
		},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			"command must be specified for user to use exec authentication plugin",
		},
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecWithAuthProvider(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		AuthProvider: &clientcmdapi.AuthProviderConfig{
			Name: "oidc",
		},
		Exec: &clientcmdapi.ExecConfig{
			Command:         "/bin/example",
			APIVersion:      "clientauthentication.k8s.io/v1alpha1",
			InteractiveMode: clientcmdapi.IfAvailableExecInteractiveMode,
		},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			"authProvider cannot be provided in combination with an exec plugin for user",
		},
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecNoEnv(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			Command:    "/bin/example",
			APIVersion: "clientauthentication.k8s.io/v1alpha1",
			Env: []clientcmdapi.ExecEnvVar{
				{Name: "foo", Value: ""},
			},
			InteractiveMode: clientcmdapi.IfAvailableExecInteractiveMode,
		},
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecInteractiveModeMissing(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			Command:    "/bin/example",
			APIVersion: "clientauthentication.k8s.io/v1alpha1",
		},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			"interactiveMode must be specified for user to use exec authentication plugin",
		},
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoExecInteractiveModeInvalid(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Exec: &clientcmdapi.ExecConfig{
			Command:         "/bin/example",
			APIVersion:      "clientauthentication.k8s.io/v1alpha1",
			InteractiveMode: "invalid",
		},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`invalid interactiveMode for user: "invalid"`,
		},
	}

	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoImpersonateUser(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Impersonate: "user",
	}
	test := configValidationTest{
		config: config,
	}
	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoImpersonateEverything(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		Impersonate:          "user",
		ImpersonateUID:       "abc123",
		ImpersonateGroups:    []string{"group-1", "group-2"},
		ImpersonateUserExtra: map[string][]string{"key": {"val1", "val2"}},
	}
	test := configValidationTest{
		config: config,
	}
	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoImpersonateGroupsWithoutUserInvalid(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		ImpersonateGroups: []string{"group-1", "group-2"},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`requesting uid, groups or user-extra for user without impersonating a user`,
		},
	}
	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoImpersonateExtraWithoutUserInvalid(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		ImpersonateUserExtra: map[string][]string{"key": {"val1", "val2"}},
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`requesting uid, groups or user-extra for user without impersonating a user`,
		},
	}
	test.testAuthInfo("user", t)
	test.testConfig(t)
}

func TestValidateAuthInfoImpersonateUIDWithoutUserInvalid(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.AuthInfos["user"] = &clientcmdapi.AuthInfo{
		ImpersonateUID: "abc123",
	}
	test := configValidationTest{
		config: config,
		expectedErrorSubstring: []string{
			`requesting uid, groups or user-extra for user without impersonating a user`,
		},
	}
	test.testAuthInfo("user", t)
	test.testConfig(t)
}

type configValidationTest struct {
	config                 *clientcmdapi.Config
	expectedErrorSubstring []string
}

func (c configValidationTest) testContext(contextName string, t *testing.T) {
	errs := validateContext(contextName, *c.config.Contexts[contextName], *c.config)

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, utilerrors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", utilerrors.NewAggregate(errs))
		}
	}
}

func (c configValidationTest) testConfirmUsable(contextName string, t *testing.T) {
	err := ConfirmUsable(*c.config, contextName)

	if len(c.expectedErrorSubstring) != 0 {
		if err == nil {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		} else {
			for _, curr := range c.expectedErrorSubstring {
				if err != nil && !strings.Contains(err.Error(), curr) {
					t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, err)
				}
			}
		}
	} else {
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}

func (c configValidationTest) testConfig(t *testing.T) {
	err := Validate(*c.config)

	if len(c.expectedErrorSubstring) != 0 {
		if err == nil {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		} else {
			for _, curr := range c.expectedErrorSubstring {
				if err != nil && !strings.Contains(err.Error(), curr) {
					t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, err)
				}
			}
			if !IsConfigurationInvalid(err) {
				t.Errorf("all errors should be configuration invalid: %v", err)
			}
		}
	} else {
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}

func (c configValidationTest) testCluster(clusterName string, t *testing.T) {
	errs := validateClusterInfo(clusterName, *c.config.Clusters[clusterName])

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, utilerrors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", utilerrors.NewAggregate(errs))
		}
	}
}

func (c configValidationTest) testAuthInfo(authInfoName string, t *testing.T) {
	errs := validateAuthInfo(authInfoName, *c.config.AuthInfos[authInfoName])

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, utilerrors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", utilerrors.NewAggregate(errs))
		}
	}
}

type alwaysMatchingError struct{}

func (_ alwaysMatchingError) Error() string {
	return "error"
}

func (_ alwaysMatchingError) Is(_ error) bool {
	return true
}

type someError struct{ msg string }

func (se someError) Error() string {
	if se.msg != "" {
		return se.msg
	}
	return "err"
}

func TestErrConfigurationInvalidWithErrorsIs(t *testing.T) {
	testCases := []struct {
		name         string
		err          error
		matchAgainst error
		expectMatch  bool
	}{
		{
			name:         "no match",
			err:          errConfigurationInvalid{errors.New("my-error"), errors.New("my-other-error")},
			matchAgainst: fmt.Errorf("no entry %s", "here"),
		},
		{
			name:         "match via .Is()",
			err:          errConfigurationInvalid{errors.New("forbidden"), alwaysMatchingError{}},
			matchAgainst: errors.New("unauthorized"),
			expectMatch:  true,
		},
		{
			name:         "match via equality",
			err:          errConfigurationInvalid{errors.New("err"), someError{}},
			matchAgainst: someError{},
			expectMatch:  true,
		},
		{
			name:         "match via nested aggregate",
			err:          errConfigurationInvalid{errors.New("closed today"), errConfigurationInvalid{errConfigurationInvalid{someError{}}}},
			matchAgainst: someError{},
			expectMatch:  true,
		},
		{
			name:         "match via wrapped aggregate",
			err:          fmt.Errorf("wrap: %w", errConfigurationInvalid{errors.New("err"), someError{}}),
			matchAgainst: someError{},
			expectMatch:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := errors.Is(tc.err, tc.matchAgainst)
			if result != tc.expectMatch {
				t.Errorf("expected match: %t, got match: %t", tc.expectMatch, result)
			}
		})
	}
}

type accessTrackingError struct {
	wasAccessed bool
}

func (accessTrackingError) Error() string {
	return "err"
}

func (ate *accessTrackingError) Is(_ error) bool {
	ate.wasAccessed = true
	return true
}

var _ error = &accessTrackingError{}

func TestErrConfigurationInvalidWithErrorsIsShortCircuitsOnFirstMatch(t *testing.T) {
	errC := errConfigurationInvalid{&accessTrackingError{}, &accessTrackingError{}}
	_ = errors.Is(errC, &accessTrackingError{})

	var numAccessed int
	for _, err := range errC {
		if ate := err.(*accessTrackingError); ate.wasAccessed {
			numAccessed++
		}
	}
	if numAccessed != 1 {
		t.Errorf("expected exactly one error to get accessed, got %d", numAccessed)
	}
}
