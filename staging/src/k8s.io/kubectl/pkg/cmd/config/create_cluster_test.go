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

type createClusterTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestCreateCluster(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := createClusterTest{
		description: "Testing 'kubectl config set-cluster' with a new cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=http://192.168.0.1",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "http://192.168.0.1"},
			},
		},
	}
	test.run(t)
}

func TestModifyCluster(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1"},
		},
	}
	test := createClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99"},
			},
		},
	}
	test.run(t)
}

func (test createClusterTest) run(t *testing.T) {
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
	cmd := NewCmdConfigSetCluster(buf, pathOptions)
	cmd.SetArgs(test.args)
	cmd.Flags().Parse(test.flags)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v, args: %v, flags: %v", err, test.args, test.flags)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in %q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if len(test.args) > 0 {
		cluster, ok := config.Clusters[test.args[0]]
		if !ok {
			t.Errorf("expected cluster %v, but got nil", test.args[0])
			return
		}
		if cluster.Server != test.expectedConfig.Clusters[test.args[0]].Server {
			t.Errorf("Fail in %q\n expected cluster server %v\n but got %v\n ", test.description, test.expectedConfig.Clusters[test.args[0]].Server, cluster.Server)
		}
	}
}
