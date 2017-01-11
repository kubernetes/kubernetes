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
	"io/ioutil"
	"os"
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
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

func TestValidateEmptyClusterInfo(t *testing.T) {
	config := clientcmdapi.NewConfig()
	config.Clusters["empty"] = &clientcmdapi.Cluster{}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"cluster has no server defined"},
	}

	test.testCluster("empty", t)
	test.testConfig(t)
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
	tempFile, _ := ioutil.TempFile("", "")
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
	tempFile, _ := ioutil.TempFile("", "")
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
	tempFile, _ := ioutil.TempFile("", "")
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
