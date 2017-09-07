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
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type getContextsTest struct {
	startingConfig clientcmdapi.Config
	names          []string
	noHeader       string
	output         string
	expectedOut    string
}

func TestGetContextsAll(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{},
		noHeader:       "false",
		output:         "",
		expectedOut: `CURRENT   NAME             CLUSTER       AUTHINFO    NAMESPACE
*         shaker-context   big-cluster   blue-user   saw-ns
`,
	}
	test.run(t)
}

func TestGetContextsAllNoHeader(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{},
		noHeader:       "true",
		output:         "",
		expectedOut:    "*         shaker-context   big-cluster   blue-user   saw-ns\n",
	}
	test.run(t)
}

func TestGetContextsAllName(t *testing.T) {
	tconf := clientcmdapi.Config{
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{},
		noHeader:       "false",
		output:         "name",
		expectedOut:    "shaker-context\n",
	}
	test.run(t)
}

func TestGetContextsAllNameNoHeader(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{},
		noHeader:       "true",
		output:         "name",
		expectedOut:    "shaker-context\n",
	}
	test.run(t)
}

func TestGetContextsAllNone(t *testing.T) {
	test := getContextsTest{
		startingConfig: *clientcmdapi.NewConfig(),
		names:          []string{},
		noHeader:       "true",
		output:         "",
		expectedOut:    "",
	}
	test.run(t)
}

func TestGetContextsSelectOneOfTwo(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"},
			"not-this":       {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{"shaker-context"},
		noHeader:       "true",
		output:         "name",
		expectedOut:    "shaker-context\n",
	}
	test.run(t)
}

func TestGetContextsWithErrorOutput(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"},
			"not-this":       {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := getContextsTest{
		startingConfig: tconf,
		names:          []string{"shaker-context"},
		noHeader:       "true",
		output:         "unknow",
		expectedOut: `--output unknow is not available in kubectl config get-contexts; resetting to default output format
*         shaker-context   big-cluster   blue-user   saw-ns` + "\n",
	}
	test.run(t)
}

func (test getContextsTest) run(t *testing.T) {
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
	buf := bytes.NewBuffer([]byte{})
	options := GetContextsOptions{
		configAccess: pathOptions,
	}
	cmd := NewCmdConfigGetContexts(buf, options.configAccess)
	cmd.Flags().Set("output", test.output)
	cmd.Flags().Set("no-headers", test.noHeader)
	cmd.Run(cmd, test.names)
	if len(test.expectedOut) != 0 {
		if buf.String() != test.expectedOut {
			t.Errorf("Expected:\n %v\n but got:\n %v", test.expectedOut, buf.String())
		}
		return
	}
}
