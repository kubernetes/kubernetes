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

type deleteClusterTest struct {
	config           clientcmdapi.Config
	clusterToDelete  string
	expectedClusters []string
	expectedOut      string
}

func TestDeleteCluster(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube":  {Server: "https://192.168.0.99"},
			"otherkube": {Server: "https://192.168.0.100"},
		},
	}
	test := deleteClusterTest{
		config:           conf,
		clusterToDelete:  "minikube",
		expectedClusters: []string{"otherkube"},
		expectedOut:      "deleted cluster minikube from %s\n",
	}

	test.run(t)
}

func (test deleteClusterTest) run(t *testing.T) {
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
	cmd := NewCmdConfigDeleteCluster(buf, pathOptions)
	cmd.SetArgs([]string{test.clusterToDelete})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}

	expectedOutWithFile := fmt.Sprintf(test.expectedOut, fakeKubeFile.Name())
	if expectedOutWithFile != buf.String() {
		t.Errorf("expected output %s, but got %s", expectedOutWithFile, buf.String())
		return
	}

	// Verify cluster was removed from kubeconfig file
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	clusters := make([]string, 0, len(config.Clusters))
	for k := range config.Clusters {
		clusters = append(clusters, k)
	}

	if !reflect.DeepEqual(test.expectedClusters, clusters) {
		t.Errorf("expected clusters %v, but found %v in kubeconfig", test.expectedClusters, clusters)
	}
}
