/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

func TestConfirmUsableBadInfoButOkConfig(t *testing.T) {
	config := NewConfig()
	config.Clusters["missing ca"] = Cluster{
		Server:               "anything",
		CertificateAuthority: "missing",
	}
	config.AuthInfos["error"] = AuthInfo{
		AuthPath: "anything",
		Token:    "here",
	}
	config.Contexts["dirty"] = Context{
		Cluster:  "missing ca",
		AuthInfo: "error",
	}
	config.Clusters["clean"] = Cluster{
		Server: "anything",
	}
	config.AuthInfos["clean"] = AuthInfo{
		Token: "here",
	}
	config.Contexts["clean"] = Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}

	badValidation := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read auth-path", "unable to read certificate-authority"},
	}
	okTest := configValidationTest{
		config: config,
	}

	okTest.testConfirmUsable("clean", t)
	badValidation.testConfig(t)
}
func TestConfirmUsableBadInfoConfig(t *testing.T) {
	config := NewConfig()
	config.Clusters["missing ca"] = Cluster{
		Server:               "anything",
		CertificateAuthority: "missing",
	}
	config.AuthInfos["error"] = AuthInfo{
		AuthPath: "anything",
		Token:    "here",
	}
	config.Contexts["first"] = Context{
		Cluster:  "missing ca",
		AuthInfo: "error",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read auth-path", "unable to read certificate-authority"},
	}

	test.testConfirmUsable("first", t)
}
func TestConfirmUsableEmptyConfig(t *testing.T) {
	config := NewConfig()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"no context chosen"},
	}

	test.testConfirmUsable("", t)
}
func TestConfirmUsableMissingConfig(t *testing.T) {
	config := NewConfig()
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"context was not found for"},
	}

	test.testConfirmUsable("not-here", t)
}
func TestValidateEmptyConfig(t *testing.T) {
	config := NewConfig()
	test := configValidationTest{
		config: config,
	}

	test.testConfig(t)
}
func TestValidateMissingCurrentContextConfig(t *testing.T) {
	config := NewConfig()
	config.CurrentContext = "anything"
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"context was not found for specified "},
	}

	test.testConfig(t)
}
func TestIsContextNotFound(t *testing.T) {
	config := NewConfig()
	config.CurrentContext = "anything"

	err := Validate(*config)
	if !IsContextNotFound(err) {
		t.Errorf("Expected context not found, but got %v", err)
	}
}
func TestValidateMissingReferencesConfig(t *testing.T) {
	config := NewConfig()
	config.CurrentContext = "anything"
	config.Contexts["anything"] = Context{Cluster: "missing", AuthInfo: "missing"}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"user, missing, was not found for Context anything", "cluster, missing, was not found for Context anything"},
	}

	test.testContext("anything", t)
	test.testConfig(t)
}
func TestValidateEmptyContext(t *testing.T) {
	config := NewConfig()
	config.CurrentContext = "anything"
	config.Contexts["anything"] = Context{}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"user was not specified for Context anything", "cluster was not specified for Context anything"},
	}

	test.testContext("anything", t)
	test.testConfig(t)
}

func TestValidateEmptyClusterInfo(t *testing.T) {
	config := NewConfig()
	config.Clusters["empty"] = Cluster{}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"no server found for"},
	}

	test.testCluster("empty", t)
	test.testConfig(t)
}
func TestValidateMissingCAFileClusterInfo(t *testing.T) {
	config := NewConfig()
	config.Clusters["missing ca"] = Cluster{
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
	config := NewConfig()
	config.Clusters["clean"] = Cluster{
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

	config := NewConfig()
	config.Clusters["clean"] = Cluster{
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
	config := NewConfig()
	config.AuthInfos["error"] = AuthInfo{}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("error", t)
	test.testConfig(t)
}
func TestValidatePathNotFoundAuthInfo(t *testing.T) {
	config := NewConfig()
	config.AuthInfos["error"] = AuthInfo{
		AuthPath: "missing",
	}
	test := configValidationTest{
		config:                 config,
		expectedErrorSubstring: []string{"unable to read auth-path"},
	}

	test.testAuthInfo("error", t)
	test.testConfig(t)
}
func TestValidateCertFilesNotFoundAuthInfo(t *testing.T) {
	config := NewConfig()
	config.AuthInfos["error"] = AuthInfo{
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
func TestValidateCleanCertFilesAuthInfo(t *testing.T) {
	tempFile, _ := ioutil.TempFile("", "")
	defer os.Remove(tempFile.Name())

	config := NewConfig()
	config.AuthInfos["clean"] = AuthInfo{
		ClientCertificate: tempFile.Name(),
		ClientKey:         tempFile.Name(),
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}
func TestValidateCleanPathAuthInfo(t *testing.T) {
	tempFile, _ := ioutil.TempFile("", "")
	defer os.Remove(tempFile.Name())

	config := NewConfig()
	config.AuthInfos["clean"] = AuthInfo{
		AuthPath: tempFile.Name(),
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}
func TestValidateCleanTokenAuthInfo(t *testing.T) {
	config := NewConfig()
	config.AuthInfos["clean"] = AuthInfo{
		Token: "any-value",
	}
	test := configValidationTest{
		config: config,
	}

	test.testAuthInfo("clean", t)
	test.testConfig(t)
}

type configValidationTest struct {
	config                 *Config
	expectedErrorSubstring []string
}

func (c configValidationTest) testContext(contextName string, t *testing.T) {
	errs := validateContext(contextName, c.config.Contexts[contextName], *c.config)

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(errors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, errors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", errors.NewAggregate(errs))
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
		}
	} else {
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}
func (c configValidationTest) testCluster(clusterName string, t *testing.T) {
	errs := validateClusterInfo(clusterName, c.config.Clusters[clusterName])

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(errors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, errors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", errors.NewAggregate(errs))
		}
	}
}

func (c configValidationTest) testAuthInfo(authInfoName string, t *testing.T) {
	errs := validateAuthInfo(authInfoName, c.config.AuthInfos[authInfoName])

	if len(c.expectedErrorSubstring) != 0 {
		if len(errs) == 0 {
			t.Errorf("Expected error containing: %v", c.expectedErrorSubstring)
		}
		for _, curr := range c.expectedErrorSubstring {
			if len(errs) != 0 && !strings.Contains(errors.NewAggregate(errs).Error(), curr) {
				t.Errorf("Expected error containing: %v, but got %v", c.expectedErrorSubstring, errors.NewAggregate(errs))
			}
		}

	} else {
		if len(errs) != 0 {
			t.Errorf("Unexpected error: %v", errors.NewAggregate(errs))
		}
	}
}
