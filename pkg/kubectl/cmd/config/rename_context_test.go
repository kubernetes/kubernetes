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
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

const (
	currentContext            = "current-context"
	newContext                = "new-context"
	nonexistentCurrentContext = "nonexistent-current-context"
	existentNewContext        = "existent-new-context"
)

var (
	contextData = clientcmdapi.NewContext()
)

type renameContextTest struct {
	description    string
	initialConfig  clientcmdapi.Config // initial config
	expectedConfig clientcmdapi.Config // expected config
	args           []string            // kubectl rename-context args
	expectedOut    string              // expected out message
	expectedErr    string              // expected error message
}

func TestRenameContext(t *testing.T) {
	initialConfig := clientcmdapi.Config{
		CurrentContext: currentContext,
		Contexts:       map[string]*clientcmdapi.Context{currentContext: contextData}}

	expectedConfig := clientcmdapi.Config{
		CurrentContext: newContext,
		Contexts:       map[string]*clientcmdapi.Context{newContext: contextData}}

	test := renameContextTest{
		description:    "Testing for kubectl config rename-context whose context to be renamed is the CurrentContext",
		initialConfig:  initialConfig,
		expectedConfig: expectedConfig,
		args:           []string{currentContext, newContext},
		expectedOut:    fmt.Sprintf("Context %q renamed to %q.\n", currentContext, newContext),
		expectedErr:    "",
	}
	test.run(t)
}

func TestRenameNonexistentContext(t *testing.T) {
	initialConfig := clientcmdapi.Config{
		CurrentContext: currentContext,
		Contexts:       map[string]*clientcmdapi.Context{currentContext: contextData}}

	test := renameContextTest{
		description:    "Testing for kubectl config rename-context whose context to be renamed no exists",
		initialConfig:  initialConfig,
		expectedConfig: initialConfig,
		args:           []string{nonexistentCurrentContext, newContext},
		expectedOut:    "",
		expectedErr:    fmt.Sprintf("cannot rename the context %q, it's not in", nonexistentCurrentContext),
	}
	test.run(t)
}

func TestRenameToAlreadyExistingContext(t *testing.T) {
	initialConfig := clientcmdapi.Config{
		CurrentContext: currentContext,
		Contexts: map[string]*clientcmdapi.Context{
			currentContext:     contextData,
			existentNewContext: contextData}}

	test := renameContextTest{
		description:    "Testing for kubectl config rename-context whose the new name is already in another context.",
		initialConfig:  initialConfig,
		expectedConfig: initialConfig,
		args:           []string{currentContext, existentNewContext},
		expectedOut:    "",
		expectedErr:    fmt.Sprintf("cannot rename the context %q, the context %q already exists", currentContext, existentNewContext),
	}
	test.run(t)
}

func (test renameContextTest) run(t *testing.T) {
	fakeKubeFile, _ := ioutil.TempFile("", "")
	defer os.Remove(fakeKubeFile.Name())
	err := clientcmd.WriteToFile(test.initialConfig, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	options := RenameContextOptions{
		configAccess: pathOptions,
		contextName:  test.args[0],
		newName:      test.args[1],
	}
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigRenameContext(buf, options.configAccess)

	options.Complete(cmd, test.args, buf)
	options.Validate()
	err = options.RunRenameContext(buf)

	if len(test.expectedErr) != 0 {
		if err == nil {
			t.Errorf("Did not get %v", test.expectedErr)
		} else {
			if !strings.Contains(err.Error(), test.expectedErr) {
				t.Errorf("Expected error %v, but got %v", test.expectedErr, err)
			}
		}
		return
	}

	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	_, oldExists := config.Contexts[currentContext]
	_, newExists := config.Contexts[newContext]

	if (!newExists) || (oldExists) || (config.CurrentContext != newContext) {
		t.Errorf("Failed in: %q\n expected %v\n but got %v", test.description, test.expectedConfig, *config)
	}

	if len(test.expectedOut) != 0 {
		if buf.String() != test.expectedOut {
			t.Errorf("Failed in:%q\n expected out %v\n but got %v", test.description, test.expectedOut, buf.String())
		}
	}
}
