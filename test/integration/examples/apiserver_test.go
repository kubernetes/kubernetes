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
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	wardlev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	wardlev1beta1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1beta1"
	sampleserver "k8s.io/sample-apiserver/pkg/cmd/server"
	wardlev1alpha1client "k8s.io/sample-apiserver/pkg/generated/clientset/versioned/typed/wardle/v1alpha1"
	netutils "k8s.io/utils/net"
)

func TestAggregatedAPIServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	t.Cleanup(cancel)

	// makes the kube-apiserver very responsive.  it's normally a minute
	dynamiccertificates.FileRefreshDuration = 1 * time.Second

	stopCh := make(chan struct{})
	defer close(stopCh)

	// we need the wardle port information first to set up the service resolver
	listener, wardlePort, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0", net.ListenConfig{})
	if err != nil {
		t.Fatal(err)
	}
	// endpoints cannot have loopback IPs so we need to override the resolver itself
	t.Cleanup(app.SetServiceResolverForTests(staticURLServiceResolver(fmt.Sprintf("https://127.0.0.1:%d", wardlePort))))

	testServer := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true}, nil, framework.SharedEtcd())
	defer testServer.TearDownFn()
	kubeClientConfig := rest.CopyConfig(testServer.ClientConfig)
	// force json because everything speaks it
	kubeClientConfig.ContentType = ""
	kubeClientConfig.AcceptContentTypes = ""
	kubeClient := client.NewForConfigOrDie(kubeClientConfig)
	aggregatorClient := aggregatorclient.NewForConfigOrDie(kubeClientConfig)
	wardleClient := wardlev1alpha1client.NewForConfigOrDie(kubeClientConfig)

	// create the bare minimum resources required to be able to get the API service into an available state
	_, err = kubeClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-wardle",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = kubeClient.CoreV1().Services("kube-wardle").Create(ctx, &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "api",
		},
		Spec: corev1.ServiceSpec{
			ExternalName: "needs-to-be-non-empty",
			Type:         corev1.ServiceTypeExternalName,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// start the wardle server to prove we can aggregate it
	wardleToKASKubeConfigFile := writeKubeConfigForWardleServerToKASConnection(t, rest.CopyConfig(kubeClientConfig))
	defer os.Remove(wardleToKASKubeConfigFile)
	wardleCertDir, _ := os.MkdirTemp("", "test-integration-wardle-server")
	defer os.RemoveAll(wardleCertDir)
	go func() {
		o := sampleserver.NewWardleServerOptions(os.Stdout, os.Stderr)
		// ensure this is a SAN on the generated cert for service FQDN
		o.AlternateDNS = []string{
			"api.kube-wardle.svc",
		}
		o.RecommendedOptions.SecureServing.Listener = listener
		o.RecommendedOptions.SecureServing.BindAddress = netutils.ParseIPSloppy("127.0.0.1")
		wardleCmd := sampleserver.NewCommandStartWardleServer(o, stopCh)
		wardleCmd.SetArgs([]string{
			"--authentication-kubeconfig", wardleToKASKubeConfigFile,
			"--authorization-kubeconfig", wardleToKASKubeConfigFile,
			"--etcd-servers", framework.GetEtcdURL(),
			"--cert-dir", wardleCertDir,
			"--kubeconfig", wardleToKASKubeConfigFile,
		})
		if err := wardleCmd.Execute(); err != nil {
			t.Error(err)
		}
	}()
	directWardleClientConfig, err := waitForWardleRunning(ctx, t, kubeClientConfig, wardleCertDir, wardlePort)
	if err != nil {
		t.Fatal(err)
	}

	// now we're finally ready to test. These are what's run by default now
	wardleDirectClient := client.NewForConfigOrDie(directWardleClientConfig)
	testAPIGroupList(ctx, t, wardleDirectClient.Discovery().RESTClient())
	testAPIGroup(ctx, t, wardleDirectClient.Discovery().RESTClient())
	testAPIResourceList(ctx, t, wardleDirectClient.Discovery().RESTClient())

	wardleCA, err := os.ReadFile(directWardleClientConfig.CAFile)
	if err != nil {
		t.Fatal(err)
	}
	_, err = aggregatorClient.ApiregistrationV1().APIServices().Create(ctx, &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Service: &apiregistrationv1.ServiceReference{
				Namespace: "kube-wardle",
				Name:      "api",
			},
			Group:                "wardle.example.com",
			Version:              "v1alpha1",
			CABundle:             wardleCA,
			GroupPriorityMinimum: 200,
			VersionPriority:      200,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// wait for the API service to be available
	err = wait.Poll(time.Second, wait.ForeverTestTimeout, func() (done bool, err error) {
		apiService, err := aggregatorClient.ApiregistrationV1().APIServices().Get(ctx, "v1alpha1.wardle.example.com", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		var available bool
		for _, condition := range apiService.Status.Conditions {
			if condition.Type == apiregistrationv1.Available && condition.Status == apiregistrationv1.ConditionTrue {
				available = true
				break
			}
		}
		if !available {
			t.Log("api service is not available", apiService.Status.Conditions)
			return false, nil
		}

		// make sure discovery is healthy overall
		_, _, err = kubeClient.Discovery().ServerGroupsAndResources()
		if err != nil {
			t.Log("discovery failed", err)
			return false, nil
		}

		// make sure we have the wardle resources in discovery
		apiResources, err := kubeClient.Discovery().ServerResourcesForGroupVersion("wardle.example.com/v1alpha1")
		if err != nil {
			t.Log("wardle discovery failed", err)
			return false, nil
		}
		if len(apiResources.APIResources) != 2 {
			t.Log("wardle discovery has wrong resources", apiResources.APIResources)
			return false, nil
		}
		resources := make([]string, 0, 2)
		for _, resource := range apiResources.APIResources {
			resource := resource
			resources = append(resources, resource.Name)
		}
		sort.Strings(resources)
		if !reflect.DeepEqual([]string{"fischers", "flunders"}, resources) {
			return false, fmt.Errorf("unexpected resources: %v", resources)
		}

		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// perform simple CRUD operations against the wardle resources
	_, err = wardleClient.Fischers().Create(ctx, &wardlev1alpha1.Fischer{
		ObjectMeta: metav1.ObjectMeta{
			Name: "panda",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	fischersList, err := wardleClient.Fischers().List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(fischersList.Items) != 1 {
		t.Errorf("expected one fischer: %#v", fischersList.Items)
	}
	if len(fischersList.ResourceVersion) == 0 {
		t.Error("expected non-empty resource version for fischer list")
	}

	_, err = wardleClient.Flunders(metav1.NamespaceSystem).Create(ctx, &wardlev1alpha1.Flunder{
		ObjectMeta: metav1.ObjectMeta{
			Name: "panda",
		},
	}, metav1.CreateOptions{})
	flunderList, err := wardleClient.Flunders(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(flunderList.Items) != 1 {
		t.Errorf("expected one flunder: %#v", flunderList.Items)
	}
	if len(flunderList.ResourceVersion) == 0 {
		t.Error("expected non-empty resource version for flunder list")
	}

	// Since ClientCAs are provided by "client-ca::kube-system::extension-apiserver-authentication::client-ca-file" controller
	// we need to wait until it picks up the configmap (via a lister) otherwise the response might contain an empty result.
	// The following code waits up to ForeverTestTimeout seconds for ClientCA to show up otherwise it fails
	// maybe in the future this could be wired into the /readyz EP

	// Now we want to verify that the client CA bundles properly reflect the values for the cluster-authentication
	var firstKubeCANames []string
	err = wait.Poll(1*time.Second, wait.ForeverTestTimeout, func() (done bool, err error) {
		firstKubeCANames, err = cert.GetClientCANamesForURL(kubeClientConfig.Host)
		if err != nil {
			return false, err
		}
		return len(firstKubeCANames) != 0, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(firstKubeCANames)
	var firstWardleCANames []string
	err = wait.Poll(1*time.Second, wait.ForeverTestTimeout, func() (done bool, err error) {
		firstWardleCANames, err = cert.GetClientCANamesForURL(directWardleClientConfig.Host)
		if err != nil {
			return false, err
		}
		return len(firstWardleCANames) != 0, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(firstWardleCANames)
	// Now we want to verify that the client CA bundles properly reflect the values for the cluster-authentication
	if !reflect.DeepEqual(firstKubeCANames, firstWardleCANames) {
		t.Fatal("names don't match")
	}

	// now we update the client-ca nd request-header-client-ca-file and the kas will consume it, update the configmap
	// and then the wardle server will detect and update too.
	if err := os.WriteFile(path.Join(testServer.TmpDir, "client-ca.crt"), differentClientCA, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path.Join(testServer.TmpDir, "proxy-ca.crt"), differentFrontProxyCA, 0644); err != nil {
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

func waitForWardleRunning(ctx context.Context, t *testing.T, wardleToKASKubeConfig *rest.Config, wardleCertDir string, wardlePort int) (*rest.Config, error) {
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
		result := wardleClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do(ctx).StatusCode(&healthStatus)
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
	wardleToKASKubeConfigFile, _ := os.CreateTemp("", "")
	if err := clientcmd.WriteToFile(*adminKubeConfig, wardleToKASKubeConfigFile.Name()); err != nil {
		t.Fatal(err)
	}

	return wardleToKASKubeConfigFile.Name()
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

func readResponse(ctx context.Context, client rest.Interface, location string) ([]byte, error) {
	return client.Get().AbsPath(location).DoRaw(ctx)
}

func testAPIGroupList(ctx context.Context, t *testing.T, client rest.Interface) {
	contents, err := readResponse(ctx, client, "/apis")
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

func testAPIGroup(ctx context.Context, t *testing.T, client rest.Interface) {
	contents, err := readResponse(ctx, client, "/apis/wardle.example.com")
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

func testAPIResourceList(ctx context.Context, t *testing.T, client rest.Interface) {
	contents, err := readResponse(ctx, client, "/apis/wardle.example.com/v1alpha1")
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

type staticURLServiceResolver string

func (u staticURLServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return url.Parse(string(u))
}
