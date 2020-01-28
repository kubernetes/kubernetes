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
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/discovery"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"
	apiregistrationv1beta1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	wardlev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	wardlev1beta1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1beta1"
	sampleserver "k8s.io/sample-apiserver/pkg/cmd/server"
)

func TestAggregatedAPIServer(t *testing.T) {
	// makes the kube-apiserver very responsive.  it's normally a minute
	dynamiccertificates.FileRefreshDuration = 1 * time.Second

	stopCh := make(chan struct{})
	defer close(stopCh)

	testServer := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true}, nil, framework.SharedEtcd())
	defer testServer.TearDownFn()
	kubeClientConfig := rest.CopyConfig(testServer.ClientConfig)
	// force json because everything speaks it
	kubeClientConfig.ContentType = ""
	kubeClientConfig.AcceptContentTypes = ""
	kubeClient := client.NewForConfigOrDie(kubeClientConfig)
	aggregatorClient := aggregatorclient.NewForConfigOrDie(kubeClientConfig)

	// start the wardle server to prove we can aggregate it
	wardleToKASKubeConfigFile := writeKubeConfigForWardleServerToKASConnection(t, rest.CopyConfig(kubeClientConfig))
	defer os.Remove(wardleToKASKubeConfigFile)
	wardleCertDir, _ := ioutil.TempDir("", "test-integration-wardle-server")
	defer os.RemoveAll(wardleCertDir)
	listener, wardlePort, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		o := sampleserver.NewWardleServerOptions(os.Stdout, os.Stderr)
		o.RecommendedOptions.SecureServing.Listener = listener
		o.RecommendedOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
		wardleCmd := sampleserver.NewCommandStartWardleServer(o, stopCh)
		wardleCmd.SetArgs([]string{
			"--authentication-kubeconfig", wardleToKASKubeConfigFile,
			"--authorization-kubeconfig", wardleToKASKubeConfigFile,
			"--etcd-servers", framework.GetEtcdURL(),
			"--cert-dir", wardleCertDir,
			"--kubeconfig", wardleToKASKubeConfigFile,
		})
		if err := wardleCmd.Execute(); err != nil {
			t.Fatal(err)
		}
	}()
	directWardleClientConfig, err := waitForWardleRunning(t, kubeClientConfig, wardleCertDir, wardlePort)
	if err != nil {
		t.Fatal(err)
	}

	// now we're finally ready to test. These are what's run by default now
	wardleClient, err := client.NewForConfig(directWardleClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	testAPIGroupList(t, wardleClient.Discovery().RESTClient())
	testAPIGroup(t, wardleClient.Discovery().RESTClient())
	testAPIResourceList(t, wardleClient.Discovery().RESTClient())

	wardleCA, err := ioutil.ReadFile(directWardleClientConfig.CAFile)
	if err != nil {
		t.Fatal(err)
	}
	_, err = aggregatorClient.ApiregistrationV1beta1().APIServices().Create(&apiregistrationv1beta1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1beta1.APIServiceSpec{
			Service: &apiregistrationv1beta1.ServiceReference{
				Namespace: "kube-wardle",
				Name:      "api",
			},
			Group:                "wardle.example.com",
			Version:              "v1alpha1",
			CABundle:             wardleCA,
			GroupPriorityMinimum: 200,
			VersionPriority:      200,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// wait for the unavailable API service to be processed with updated status
	err = wait.Poll(100*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		_, err = kubeClient.Discovery().ServerResources()
		hasExpectedError := checkWardleUnavailableDiscoveryError(t, err)
		return hasExpectedError, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// TODO figure out how to turn on enough of services and dns to run more

	// Now we want to verify that the client CA bundles properly reflect the values for the cluster-authentication
	firstKubeCANames, err := cert.GetClientCANamesForURL(kubeClientConfig.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(firstKubeCANames)
	firstWardleCANames, err := cert.GetClientCANamesForURL(directWardleClientConfig.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(firstWardleCANames)
	if !reflect.DeepEqual(firstKubeCANames, firstWardleCANames) {
		t.Fatal("names don't match")
	}

	// now we update the client-ca nd request-header-client-ca-file and the kas will consume it, update the configmap
	// and then the wardle server will detect and update too.
	if err := ioutil.WriteFile(path.Join(testServer.TmpDir, "client-ca.crt"), differentClientCA, 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(testServer.TmpDir, "proxy-ca.crt"), differentFrontProxyCA, 0644); err != nil {
		t.Fatal(err)
	}
	// wait for it to be picked up.  there's a test in certreload_test.go that ensure this works
	time.Sleep(4 * time.Second)

	// Now we want to verify that the client CA bundles properly updated to reflect the new values written for the kube-apiserver
	secondKubeCANames, err := cert.GetClientCANamesForURL(kubeClientConfig.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(secondKubeCANames)
	for i := range firstKubeCANames {
		if firstKubeCANames[i] == secondKubeCANames[i] {
			t.Errorf("ca bundles should change")
		}
	}
	secondWardleCANames, err := cert.GetClientCANamesForURL(directWardleClientConfig.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(secondWardleCANames)

	// second wardle should contain all the certs, first and last
	numMatches := 0
	for _, needle := range firstKubeCANames {
		for _, haystack := range secondWardleCANames {
			if needle == haystack {
				numMatches++
				break
			}
		}
	}
	for _, needle := range secondKubeCANames {
		for _, haystack := range secondWardleCANames {
			if needle == haystack {
				numMatches++
				break
			}
		}
	}
	if numMatches != 4 {
		t.Fatal("names don't match")
	}

}

func waitForWardleRunning(t *testing.T, wardleToKASKubeConfig *rest.Config, wardleCertDir string, wardlePort int) (*rest.Config, error) {
	directWardleClientConfig := rest.AnonymousClientConfig(rest.CopyConfig(wardleToKASKubeConfig))
	directWardleClientConfig.CAFile = path.Join(wardleCertDir, "apiserver.crt")
	directWardleClientConfig.CAData = nil
	directWardleClientConfig.ServerName = ""
	directWardleClientConfig.BearerToken = wardleToKASKubeConfig.BearerToken
	var wardleClient client.Interface
	lastHealthContent := []byte{}
	var lastHealthErr error
	err := wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		if _, err := os.Stat(directWardleClientConfig.CAFile); os.IsNotExist(err) { // wait until the file trust is created
			lastHealthErr = err
			return false, nil
		}
		directWardleClientConfig.Host = fmt.Sprintf("https://127.0.0.1:%d", wardlePort)
		wardleClient, err = client.NewForConfig(directWardleClientConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}
		healthStatus := 0
		result := wardleClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do(context.TODO()).StatusCode(&healthStatus)
		lastHealthContent, lastHealthErr = result.Raw()
		if healthStatus != http.StatusOK {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Log(string(lastHealthContent))
		t.Log(lastHealthErr)
		return nil, err
	}

	return directWardleClientConfig, nil
}

func writeKubeConfigForWardleServerToKASConnection(t *testing.T, kubeClientConfig *rest.Config) string {
	// write a kubeconfig out for starting other API servers with delegated auth.  remember, no in-cluster config
	// the loopback client config uses a loopback cert with different SNI.  We need to use the "real"
	// cert, so we'll hope we aren't hacked during a unit test and instead load it from the server we started.
	wardleToKASKubeClientConfig := rest.CopyConfig(kubeClientConfig)

	servingCerts, _, err := cert.GetServingCertificatesForURL(wardleToKASKubeClientConfig.Host, "")
	if err != nil {
		t.Fatal(err)
	}
	encodedServing, err := cert.EncodeCertificates(servingCerts...)
	if err != nil {
		t.Fatal(err)
	}
	wardleToKASKubeClientConfig.CAData = encodedServing

	for _, v := range servingCerts {
		t.Logf("Client: Server public key is %v\n", dynamiccertificates.GetHumanCertDetail(v))
	}
	certs, err := cert.ParseCertsPEM(wardleToKASKubeClientConfig.CAData)
	if err != nil {
		t.Fatal(err)
	}
	for _, curr := range certs {
		t.Logf("CA bundle %v\n", dynamiccertificates.GetHumanCertDetail(curr))
	}

	adminKubeConfig := createKubeConfig(wardleToKASKubeClientConfig)
	wardleToKASKubeConfigFile, _ := ioutil.TempFile("", "")
	if err := clientcmd.WriteToFile(*adminKubeConfig, wardleToKASKubeConfigFile.Name()); err != nil {
		t.Fatal(err)
	}

	return wardleToKASKubeConfigFile.Name()
}

func checkWardleUnavailableDiscoveryError(t *testing.T, err error) bool {
	if err == nil {
		t.Log("Discovery call expected to return failed unavailable service")
		return false
	}
	if !discovery.IsGroupDiscoveryFailedError(err) {
		t.Logf("Unexpected error: %T, %v", err, err)
		return false
	}
	discoveryErr := err.(*discovery.ErrGroupDiscoveryFailed)
	if len(discoveryErr.Groups) != 1 {
		t.Logf("Unexpected failed groups: %v", err)
		return false
	}
	groupVersion := schema.GroupVersion{Group: "wardle.example.com", Version: "v1alpha1"}
	groupVersionErr, ok := discoveryErr.Groups[groupVersion]
	if !ok {
		t.Logf("Unexpected failed group version: %v", err)
		return false
	}
	if !apierrors.IsServiceUnavailable(groupVersionErr) {
		t.Logf("Unexpected failed group version error: %v", err)
		return false
	}
	return true
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
	config.Clusters[clusterNick] = cluster

	context := clientcmdapi.NewContext()
	context.Cluster = clusterNick
	context.AuthInfo = userNick
	config.Contexts[contextNick] = context
	config.CurrentContext = contextNick

	return config
}

func readResponse(client rest.Interface, location string) ([]byte, error) {
	return client.Get().AbsPath(location).DoRaw(context.TODO())
}

func testAPIGroupList(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroupList metav1.APIGroupList
	err = json.Unmarshal(contents, &apiGroupList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis", err)
	}
	assert.Equal(t, 1, len(apiGroupList.Groups))
	assert.Equal(t, wardlev1alpha1.GroupName, apiGroupList.Groups[0].Name)
	assert.Equal(t, 2, len(apiGroupList.Groups[0].Versions))

	v1alpha1 := metav1.GroupVersionForDiscovery{
		GroupVersion: wardlev1alpha1.SchemeGroupVersion.String(),
		Version:      wardlev1alpha1.SchemeGroupVersion.Version,
	}
	v1beta1 := metav1.GroupVersionForDiscovery{
		GroupVersion: wardlev1beta1.SchemeGroupVersion.String(),
		Version:      wardlev1beta1.SchemeGroupVersion.Version,
	}

	assert.Equal(t, v1beta1, apiGroupList.Groups[0].Versions[0])
	assert.Equal(t, v1alpha1, apiGroupList.Groups[0].Versions[1])
	assert.Equal(t, v1beta1, apiGroupList.Groups[0].PreferredVersion)
}

func testAPIGroup(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis/wardle.example.com")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroup metav1.APIGroup
	err = json.Unmarshal(contents, &apiGroup)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis/wardle.example.com", err)
	}
	assert.Equal(t, wardlev1alpha1.SchemeGroupVersion.Group, apiGroup.Name)
	assert.Equal(t, 2, len(apiGroup.Versions))
	assert.Equal(t, wardlev1alpha1.SchemeGroupVersion.String(), apiGroup.Versions[1].GroupVersion)
	assert.Equal(t, wardlev1alpha1.SchemeGroupVersion.Version, apiGroup.Versions[1].Version)
	assert.Equal(t, apiGroup.PreferredVersion, apiGroup.Versions[0])
}

func testAPIResourceList(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis/wardle.example.com/v1alpha1")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiResourceList metav1.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis/wardle.example.com/v1alpha1", err)
	}
	assert.Equal(t, wardlev1alpha1.SchemeGroupVersion.String(), apiResourceList.GroupVersion)
	assert.Equal(t, 2, len(apiResourceList.APIResources))
	assert.Equal(t, "fischers", apiResourceList.APIResources[0].Name)
	assert.False(t, apiResourceList.APIResources[0].Namespaced)
	assert.Equal(t, "flunders", apiResourceList.APIResources[1].Name)
	assert.True(t, apiResourceList.APIResources[1].Namespaced)
}

var (
	// I have no idea what these certs are, they just need to be different
	differentClientCA = []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----
`)
	differentFrontProxyCA = []byte(`-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----

`)
)
