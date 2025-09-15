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
	"os"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setClusterTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedConfig clientcmdapi.Config
}

func TestCreateCluster(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := setClusterTest{
		description: "Testing 'kubectl config set-cluster' with a new cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=http://192.168.0.1",
			"--tls-server-name=my-cluster-name",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "http://192.168.0.1", TLSServerName: "my-cluster-name"},
			},
		},
	}
	test.run(t)
}

func TestCreateClusterWithProxy(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := setClusterTest{
		description: "Testing 'kubectl config set-cluster' with a new cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=http://192.168.0.1",
			"--tls-server-name=my-cluster-name",
			"--proxy-url=http://192.168.0.2",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {
					Server:        "http://192.168.0.1",
					TLSServerName: "my-cluster-name",
					ProxyURL:      "http://192.168.0.2",
				},
			},
		},
	}
	test.run(t)
}

func TestModifyCluster(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1", TLSServerName: "to-be-cleared"},
		},
	}
	test := setClusterTest{
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

// TestModifyClusterWithProxy tests setting proxy-url in kubeconfig
func TestModifyClusterWithProxy(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1", TLSServerName: "to-be-cleared"},
		},
	}
	test := setClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
			"--proxy-url=https://192.168.0.100",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99", ProxyURL: "https://192.168.0.100"},
			},
		},
	}
	test.run(t)
}

// TestModifyClusterWithProxyOverride tests updating proxy-url
// in kubeconfig which already exists
func TestModifyClusterWithProxyOverride(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {
				Server:        "https://192.168.0.1",
				TLSServerName: "to-be-cleared",
				ProxyURL:      "https://192.168.0.2",
			},
		},
	}
	test := setClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
			"--proxy-url=https://192.168.0.100",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99", ProxyURL: "https://192.168.0.100"},
			},
		},
	}
	test.run(t)
}

func TestModifyClusterServerAndTLS(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1"},
		},
	}
	test := setClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
			"--tls-server-name=my-cluster-name",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99", TLSServerName: "my-cluster-name"},
			},
		},
	}
	test.run(t)
}

func (test setClusterTest) run(t *testing.T) {
	fakeKubeFile, err := os.CreateTemp(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, fakeKubeFile)
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
		if cluster.TLSServerName != test.expectedConfig.Clusters[test.args[0]].TLSServerName {
			t.Errorf("Fail in %q\n expected cluster TLS server name %q\n but got %q\n ", test.description, test.expectedConfig.Clusters[test.args[0]].TLSServerName, cluster.TLSServerName)
		}
	}
}
