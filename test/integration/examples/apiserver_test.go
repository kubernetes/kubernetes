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

package apiserver

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	"k8s.io/sample-apiserver/pkg/cmd/server"
)

const securePort = "6444"

var groupVersion = v1alpha1.SchemeGroupVersion

var groupVersionForDiscovery = metav1.GroupVersionForDiscovery{
	GroupVersion: groupVersion.String(),
	Version:      groupVersion.Version,
}

func TestRunServer(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s := framework.RunAMaster(masterConfig)
	defer s.Close()

	adminKubeConfig := createKubeConfig(masterConfig.GenericConfig.LoopbackClientConfig)
	kubeconfigFile, _ := ioutil.TempFile("", "")
	defer os.Remove(kubeconfigFile.Name())
	clientcmd.WriteToFile(*adminKubeConfig, kubeconfigFile.Name())

	stopCh := make(chan struct{})
	defer close(stopCh)
	cmd := server.NewCommandStartWardleServer(os.Stdout, os.Stderr, stopCh)
	cmd.SetArgs([]string{
		"--secure-port", securePort,
		"--requestheader-username-headers", "",
		"--authentication-kubeconfig", kubeconfigFile.Name(),
		"--authorization-kubeconfig", kubeconfigFile.Name(),
		"--etcd-servers", framework.GetEtcdURLFromEnv(),
	})
	go cmd.Execute()

	serverLocation := fmt.Sprintf("https://localhost:%s", securePort)
	if err := waitForApiserverUp(serverLocation); err != nil {
		t.Fatalf("%v", err)
	}

	testAPIGroupList(t, serverLocation)
	testAPIGroup(t, serverLocation)
	testAPIResourceList(t, serverLocation)
}

func createKubeConfig(clientCfg *rest.Config) *clientcmdapi.Config {
	clusterNick := "cluster"
	userNick := "user"
	contextNick := "context"

	config := clientcmdapi.NewConfig()

	credentials := clientcmdapi.NewAuthInfo()
	credentials.Token = clientCfg.BearerToken
	credentials.ClientCertificate = clientCfg.TLSClientConfig.CertFile
	if len(credentials.ClientCertificate) == 0 {
		credentials.ClientCertificateData = clientCfg.TLSClientConfig.CertData
	}
	credentials.ClientKey = clientCfg.TLSClientConfig.KeyFile
	if len(credentials.ClientKey) == 0 {
		credentials.ClientKeyData = clientCfg.TLSClientConfig.KeyData
	}
	config.AuthInfos[userNick] = credentials

	cluster := clientcmdapi.NewCluster()
	cluster.Server = clientCfg.Host
	cluster.CertificateAuthority = clientCfg.CAFile
	if len(cluster.CertificateAuthority) == 0 {
		cluster.CertificateAuthorityData = clientCfg.CAData
	}
	cluster.InsecureSkipTLSVerify = clientCfg.Insecure
	if clientCfg.GroupVersion != nil {
		cluster.APIVersion = clientCfg.GroupVersion.String()
	}
	config.Clusters[clusterNick] = cluster

	context := clientcmdapi.NewContext()
	context.Cluster = clusterNick
	context.AuthInfo = userNick
	config.Contexts[contextNick] = context
	config.CurrentContext = contextNick

	return config
}

func waitForApiserverUp(serverLocation string) error {
	for start := time.Now(); time.Since(start) < 10*time.Second; time.Sleep(5 * time.Second) {
		glog.Errorf("Waiting for : %#v", serverLocation)
		tr := &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
		client := &http.Client{Transport: tr}
		_, err := client.Get(serverLocation)
		if err == nil {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

func readResponse(serverURL string) ([]byte, error) {
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	response, err := client.Get(serverURL)
	if err != nil {
		glog.Errorf("http get err code : %#v", err)
		return nil, fmt.Errorf("Error in fetching %s: %v", serverURL, err)
	}
	defer response.Body.Close()
	glog.Errorf("http get response code : %#v", response.StatusCode)
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d for URL: %s, expected status: %d", response.StatusCode, serverURL, http.StatusOK)
	}
	contents, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Error reading response from %s: %v", serverURL, err)
	}
	return contents, nil
}

func testAPIGroupList(t *testing.T, serverLocation string) {
	serverURL := serverLocation + "/apis"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroupList metav1.APIGroupList
	err = json.Unmarshal(contents, &apiGroupList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, 1, len(apiGroupList.Groups))
	assert.Equal(t, groupVersion.Group, apiGroupList.Groups[0].Name)
	assert.Equal(t, 1, len(apiGroupList.Groups[0].Versions))
	assert.Equal(t, groupVersionForDiscovery, apiGroupList.Groups[0].Versions[0])
	assert.Equal(t, groupVersionForDiscovery, apiGroupList.Groups[0].PreferredVersion)
}

func testAPIGroup(t *testing.T, serverLocation string) {
	serverURL := serverLocation + "/apis/wardle.k8s.io"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroup metav1.APIGroup
	err = json.Unmarshal(contents, &apiGroup)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, groupVersion.Group, apiGroup.Name)
	assert.Equal(t, 1, len(apiGroup.Versions))
	assert.Equal(t, groupVersion.String(), apiGroup.Versions[0].GroupVersion)
	assert.Equal(t, groupVersion.Version, apiGroup.Versions[0].Version)
	assert.Equal(t, apiGroup.PreferredVersion, apiGroup.Versions[0])
}

func testAPIResourceList(t *testing.T, serverLocation string) {
	serverURL := serverLocation + "/apis/wardle.k8s.io/v1alpha1"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiResourceList metav1.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, groupVersion.String(), apiResourceList.GroupVersion)
	assert.Equal(t, 1, len(apiResourceList.APIResources))
	assert.Equal(t, "flunders", apiResourceList.APIResources[0].Name)
	assert.True(t, apiResourceList.APIResources[0].Namespaced)
}
