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

package config

import (
	"bytes"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type currentContextTest struct {
	startingConfig clientcmdapi.Config
	expectedError  string
}

func newFederalContextConfig() clientcmdapi.Config {
	return clientcmdapi.Config{
		CurrentContext: "federal-context",
	}
}

func TestCurrentContextWithSetContext(t *testing.T) {
	test := currentContextTest{
		startingConfig: newFederalContextConfig(),
		expectedError:  "",
	}

	test.run(t)
}

func TestCurrentContextWithUnsetContext(t *testing.T) {
	test := currentContextTest{
		startingConfig: *clientcmdapi.NewConfig(),
		expectedError:  "current-context is not set",
	}

	test.run(t)
}

func (test currentContextTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.startingConfig, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	options := CurrentContextOptions{
		ConfigAccess: pathOptions,
	}

	buf := bytes.NewBuffer([]byte{})
	err = RunCurrentContext(buf, &options)
	if len(test.expectedError) != 0 {
		if err == nil {
			t.Errorf("Did not get %v", test.expectedError)
		} else {
			if !strings.Contains(err.Error(), test.expectedError) {
				t.Errorf("Expected %v, but got %v", test.expectedError, err)
			}
		}
		return
	}

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}
