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

type getClustersTest struct {
	config   clientcmdapi.Config
	names    []string
	noHeader string
	output   string
	expected string
}

func TestGetClustersAll(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
		},
	}
	test := getClustersTest{
		config:   conf,
		names:    []string{},
		noHeader: "false",
		output:   "",
		expected: `NAME       SERVER
minikube   https://192.168.0.99` + "\n",
	}

	test.run(t)
}

func TestGetClustersEmpty(t *testing.T) {
	test := getClustersTest{
		config:   clientcmdapi.Config{},
		expected: "NAME      SERVER\n",
	}

	test.run(t)
}

func TestGetClustersAllNoHeader(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
		},
	}
	test := getClustersTest{
		config:   conf,
		names:    []string{},
		noHeader: "true",
		output:   "",
		expected: "minikube   https://192.168.0.99\n",
	}

	test.run(t)
}

func TestGetClustersAllName(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
		},
	}
	test := getClustersTest{
		config:   conf,
		names:    []string{},
		noHeader: "false",
		output:   "name",
		expected: "minikube\n",
	}

	test.run(t)
}

func TestGetClustersAllNameNoHeader(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
		},
	}
	test := getClustersTest{
		config:   conf,
		names:    []string{},
		noHeader: "true",
		output:   "name",
		expected: "minikube\n",
	}

	test.run(t)
}

func TestGetClustersSelectOneOfTwo(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube": {Server: "https://192.168.0.99"},
			"dev":      {Server: "https://192.168.0.100"},
		},
	}
	test := getClustersTest{
		config:   conf,
		names:    []string{"minikube"},
		noHeader: "true",
		output:   "name",
		expected: "minikube\n",
	}

	test.run(t)
}

func (test getClustersTest) run(t *testing.T) {
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
	cmd := NewCmdConfigGetClusters(buf, pathOptions)
	cmd.Flags().Set("output", test.output)
	cmd.Flags().Set("no-headers", test.noHeader)
	cmd.Run(cmd, test.names)
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("expected:\n %v, \nbut got:\n %v", test.expected, buf.String())
		}
		return
	}
}
