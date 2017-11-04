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
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type deleteContextTest struct {
	config           clientcmdapi.Config
	contextToDelete  string
	expectedContexts []string
	expectedOut      string
}

func TestDeleteContext(t *testing.T) {
	conf := clientcmdapi.Config{
		Contexts: map[string]*clientcmdapi.Context{
			"minikube":  {Cluster: "minikube"},
			"otherkube": {Cluster: "otherkube"},
		},
	}
	test := deleteContextTest{
		config:           conf,
		contextToDelete:  "minikube",
		expectedContexts: []string{"otherkube"},
		expectedOut:      "deleted context minikube from %s\n",
	}

	test.run(t)
}

func (test deleteContextTest) run(t *testing.T) {
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
	errBuf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigDeleteContext(buf, errBuf, pathOptions)
	cmd.SetArgs([]string{test.contextToDelete})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}

	expectedOutWithFile := fmt.Sprintf(test.expectedOut, fakeKubeFile.Name())
	if expectedOutWithFile != buf.String() {
		t.Errorf("expected output %s, but got %s", expectedOutWithFile, buf.String())
		return
	}

	// Verify context was removed from kubeconfig file
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	contexts := make([]string, 0, len(config.Contexts))
	for k := range config.Contexts {
		contexts = append(contexts, k)
	}

	if !reflect.DeepEqual(test.expectedContexts, contexts) {
		t.Errorf("expected contexts %v, but found %v in kubeconfig", test.expectedContexts, contexts)
	}
}
