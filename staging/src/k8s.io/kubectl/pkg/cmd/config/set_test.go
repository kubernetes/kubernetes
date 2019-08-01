/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	"reflect"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setConfigTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestSetConfigCurrentContext(t *testing.T) {
	conf := clientcmdapi.Config{
		Kind:           "Config",
		APIVersion:     "v1",
		CurrentContext: "minikube",
	}
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.CurrentContext = "my-cluster"
	test := setConfigTest{
		description:    "Testing for kubectl config set current-context",
		config:         conf,
		args:           []string{"current-context", "my-cluster"},
		expected:       `Property "current-context" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func (test setConfigTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile("", "")
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
	cmd := NewCmdConfigSet(buf, pathOptions)
	cmd.SetArgs(test.args)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in:%q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if !reflect.DeepEqual(*config, test.expectedConfig) {
		t.Errorf("Failed in: %q\n expected %v\n but got %v", test.description, *config, test.expectedConfig)
	}
}
