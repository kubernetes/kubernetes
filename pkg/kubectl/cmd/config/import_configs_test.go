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
	notExistingFile   = "notexisting"
	targetContextName = "target-context"
)

type importConfigsTest struct {
	startingConfig   clientcmdapi.Config
	sourceConfig     clientcmdapi.Config
	createSourceFile bool
	createTargetFile bool
	target           string
	expectedOut      []string
	expectedErr      string
}

func TestSourceNotExists(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := importConfigsTest{
		startingConfig:   tconf,
		createSourceFile: false,
		target:           "",
		expectedErr:      "provided source file does not exist or is empty"}
	test.run(t)
}

func TestTargetNotExists(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     tconf,
		createSourceFile: true,
		createTargetFile: false,
		target:           "otherfile",
		expectedErr:      "provided target file does not exist or is empty"}
	test.run(t)
}

func TestProvidingTargetFile(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"}}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     tconf,
		createSourceFile: true,
		createTargetFile: true,
		target:           "targetfile",
		expectedOut: []string{
			fmt.Sprintf("Using provided target file instead of default kubeconfig.\n")},
		expectedErr: ""}
	test.run(t)
}

func TestImportingOnlyContexts(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "any-context",
		Contexts:       make(map[string]*clientcmdapi.Context)}
	source := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"},
			"other-context":  {AuthInfo: "red-user", Cluster: "bigger-cluster", Namespace: "saw-ns"}}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     source,
		createSourceFile: true,
		target:           "",
		expectedOut: []string{
			"There is no cluster entry in the source file.\n",
			"There is no user entry in the source file.\n",
			fmt.Sprintf("Copied context entry %q to target file.\n", "shaker-context"),
			fmt.Sprintf("Copied context entry %q to target file.\n", "other-context")},
		expectedErr: ""}
	test.run(t)
}

func TestImportingOnlyClusters(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "any-context",
		Clusters:       make(map[string]*clientcmdapi.Cluster)}
	source := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube":      {Server: "https://192.168.0.99"},
			"bigkube":       {Server: "https://192.168.0.100"},
			"federatedkube": {Server: "https://192.168.0.101"},
		}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     source,
		createSourceFile: true,
		target:           "",
		expectedOut: []string{
			"There is no context entry in the source file.\n",
			"There is no user entry in the source file.\n",
			fmt.Sprintf("Copied cluster entry %q to target file.\n", "minikube"),
			fmt.Sprintf("Copied cluster entry %q to target file.\n", "bigkube"),
			fmt.Sprintf("Copied cluster entry %q to target file.", "federatedkube")},
		expectedErr: ""}
	test.run(t)
}

func TestImportingOnlyAuthInfo(t *testing.T) {
	tconf := clientcmdapi.Config{
		CurrentContext: "any-context",
		AuthInfos:      make(map[string]*clientcmdapi.AuthInfo)}
	source := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"myuser": {Token: "somebiguuid"},
			"admin":  {Token: "anotherbiguuid"},
		}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     source,
		createSourceFile: true,
		target:           "",
		expectedOut: []string{
			"There is no cluster entry in the source file.\n",
			"There is no context entry in the source file.\n",
			fmt.Sprintf("Copied User entry %q to target file.\n", "myuser"),
			fmt.Sprintf("Copied User entry %q to target file.", "admin")},
		expectedErr: ""}
	test.run(t)
}

func TestImportingWithConflict(t *testing.T) {
	// In this test case, there is one context, one cluster and one user entry
	// that exist in both files. They should be ignored and the ones without
	// conflict, imported.
	tconf := clientcmdapi.Config{
		CurrentContext: "any-context",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.100"}},
		Contexts: map[string]*clientcmdapi.Context{
			"other-context": {AuthInfo: "red-user", Cluster: "bigger-cluster", Namespace: "saw-ns"}},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"admin": {Token: "anotherbiguuid"}}}
	source := clientcmdapi.Config{
		CurrentContext: "shaker-context",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube":      {Server: "https://192.168.0.99"},
			"federatedkube": {Server: "https://192.168.0.101"}},
		Contexts: map[string]*clientcmdapi.Context{
			"shaker-context": {AuthInfo: "blue-user", Cluster: "big-cluster", Namespace: "saw-ns"},
			"other-context":  {AuthInfo: "red-user", Cluster: "bigger-cluster", Namespace: "saw-ns"}},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"myuser": {Token: "somebiguuid"},
			"admin":  {Token: "anotherbiguuid"}}}
	test := importConfigsTest{
		startingConfig:   tconf,
		sourceConfig:     source,
		createSourceFile: true,
		target:           "",
		expectedOut: []string{
			fmt.Sprintf("Copied cluster entry %q to target file.\n", "federatedkube"),
			fmt.Sprintf("Cluster called %q exists in both files, so it will not be copied.", "minikube"),
			fmt.Sprintf("Copied context entry %q to target file.\n", "shaker-context"),
			fmt.Sprintf("Context called %q exists in both files, so it will not be copied.", "other-context"),
			fmt.Sprintf("Copied User entry %q to target file.\n", "myuser"),
			fmt.Sprintf("User called %q exists in both files, so it will not be copied.", "admin")},
		expectedErr: ""}
	test.run(t)
}

func (test importConfigsTest) run(t *testing.T) {
	fakeKubeFile, _ := ioutil.TempFile("", "")
	defer os.Remove(fakeKubeFile.Name())
	err := clientcmd.WriteToFile(test.startingConfig, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	options := ImportConfigsOptions{
		configAccess: pathOptions,
	}

	args := []string{notExistingFile}
	if test.createSourceFile {
		fakeSourceKubeFile, _ := ioutil.TempFile("", "")
		defer os.Remove(fakeSourceKubeFile.Name())
		err := clientcmd.WriteToFile(test.sourceConfig, fakeSourceKubeFile.Name())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		args[0] = fakeSourceKubeFile.Name()
	}

	cmd := NewCmdConfigImportConfigs(buf, options.configAccess)
	if test.target != "" {
		if test.createTargetFile {
			fakeTargetFile, _ := ioutil.TempFile("", "")
			defer os.Remove(fakeTargetFile.Name())
			err := clientcmd.WriteToFile(test.startingConfig, fakeTargetFile.Name())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			cmd.Flags().Set("target", fakeTargetFile.Name())
		} else {
			cmd.Flags().Set("target", "notexistingfile")
		}
	}
	options.Complete(cmd, args, buf)
	err = options.RunImportConfigs(buf)

	if len(test.expectedOut) != 0 {
		for _, o := range test.expectedOut {
			if !strings.Contains(buf.String(), o) {
				t.Errorf("could not find %q phrase in %q", o, buf.String())
			}
		}
		return
	}
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

	// Once we have finished checking the output, we now need to see
	// whether all entries were copied to the target file. We check
	// if all entries in the starting configs exist in the target file
	newTarget, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	for ctx := range test.sourceConfig.Contexts {
		if newTarget.Contexts[ctx] == nil {
			t.Errorf("Context %q missing in target file", ctx)
		}
	}
	for ctx := range test.startingConfig.Contexts {
		if newTarget.Contexts[ctx] == nil {
			t.Errorf("Context %q missing in target file", ctx)
		}
	}
	for clu := range test.sourceConfig.Clusters {
		if newTarget.Clusters[clu] == nil {
			t.Errorf("Cluster %q missing in target file", clu)
		}
	}
	for clu := range test.startingConfig.Clusters {
		if newTarget.Clusters[clu] == nil {
			t.Errorf("Cluster %q missing in target file", clu)
		}
	}
	for auth := range test.sourceConfig.AuthInfos {
		if newTarget.AuthInfos[auth] == nil {
			t.Errorf("Authinfo %q missing in target file", auth)
		}
	}
	for auth := range test.startingConfig.AuthInfos {
		if newTarget.AuthInfos[auth] == nil {
			t.Errorf("Authinfo %q missing in target file", auth)
		}
	}

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

}
