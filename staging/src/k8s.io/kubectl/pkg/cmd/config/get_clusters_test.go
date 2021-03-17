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
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type getClustersTest struct {
	config   clientcmdapi.Config
	expected string
}

func TestGetClusters(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
		},
	}
	test := getClustersTest{
		config: conf,
		expected: `NAME
minikube
`,
	}

	test.run(t)
}

func TestGetClustersEmpty(t *testing.T) {
	test := getClustersTest{
		config:   clientcmdapi.Config{},
		expected: "NAME\n",
	}

	test.run(t)
}

func (test getClustersTest) run(t *testing.T) {
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
	ioStreams, _, out, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdConfigGetClusters(ioStreams, pathOptions)
	cmd.SetOut(out)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}
	if len(test.expected) != 0 {
		if out.String() != test.expected {
			t.Errorf("expected %v, but got %v", test.expected, out.String())
		}
		return
	}
}
