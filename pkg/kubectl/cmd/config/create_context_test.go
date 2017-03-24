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

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type createContextTest struct {
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestCreateContext(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := createContextTest{
		config: conf,
		args:   []string{"shaker-context"},
		flags: []string{
			"--cluster=cluster_nickname",
			"--user=user_nickname",
			"--namespace=namespace",
		},
		expected: `Context "shaker-context" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Contexts: map[string]*clientcmdapi.Context{
				"shaker-context": {AuthInfo: "user_nickname", Cluster: "cluster_nickname", Namespace: "namespace"}},
		},
	}
	test.run(t)
}
func TestModifyContext(t *testing.T) {
	conf := clientcmdapi.Config{
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"},
			"not-this":       {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := createContextTest{
		config: conf,
		args:   []string{"shaker-context"},
		flags: []string{
			"--cluster=cluster_nickname",
			"--user=user_nickname",
			"--namespace=namespace",
		},
		expected: `Context "shaker-context" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Contexts: map[string]*clientcmdapi.Context{
				"shaker-context": {AuthInfo: "user_nickname", Cluster: "cluster_nickname", Namespace: "namespace"},
				"not-this":       {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}},
	}
	test.run(t)
}

func (test createContextTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
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
	cmd := NewCmdConfigSetContext(buf, pathOptions)
	cmd.SetArgs(test.args)
	cmd.Flags().Parse(test.flags)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v,kubectl set-context args: %v,flags: %v", err, test.args, test.flags)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Fatalf("expected %v, but got %v", test.expected, buf.String())
		}
	}
	if test.expectedConfig.Contexts != nil {
		expectContext := test.expectedConfig.Contexts[test.args[0]]
		actualContext := config.Contexts[test.args[0]]
		if expectContext.AuthInfo != actualContext.AuthInfo || expectContext.Cluster != actualContext.Cluster ||
			expectContext.Namespace != actualContext.Namespace {
			t.Errorf("expected Context %v, but found %v in kubeconfig", expectContext, actualContext)
		}
	}
}
