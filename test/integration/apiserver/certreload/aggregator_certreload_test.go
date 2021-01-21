/*
Copyright 2020 The Kubernetes Authors.

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
package certreload

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	apiserverintegration "k8s.io/kubernetes/test/integration/apiserver"
	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
	sampleserver "k8s.io/sample-apiserver/pkg/cmd/server"
)

func TestAggregatorCertReload(t *testing.T) {

	pkgPath, err := pkgPath(t)
	if err != nil {
		t.Fatal(err)
	}

	// Default testdata files locations
	servingCertPath := filepath.Join(pkgPath, "testdata")
	proxyClientCertPath := filepath.Join(servingCertPath, "proxy-client.pem")
	proxyCACertPath := filepath.Join(servingCertPath, "proxy-ca.pem")
	proxyClientKeyPath := filepath.Join(servingCertPath, "proxy-client.key")

	// Loads the client certificates from the servingCertPath
	proxyCACert, proxyClientKey, proxyClientCert, err := loadCAWithClientCert(servingCertPath, "original")
	if err != nil {
		t.Fatal(err)
	}

	// Writes original client certificates to use with kube-apiserver
	if err := ioutil.WriteFile(proxyClientCertPath, testutil.EncodeCertPEM(proxyClientCert), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(proxyClientKeyPath, proxyClientKey, 0644); err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	t.Log("STEP 1: Start Kube API Server")
	testServer := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, FileRefreshDuration: 1 * time.Second},
		[]string{"--cert-dir", servingCertPath,
			"--proxy-client-cert-file", proxyClientCertPath,
			"--proxy-client-key-file", proxyClientKeyPath,
			"--enable-aggregator-routing", "true"}, framework.SharedEtcd())
	defer testServer.TearDownFn()


	kubeClientConfig := rest.CopyConfig(testServer.ClientConfig)
	kubeClientConfig.ContentType = ""
	kubeClientConfig.AcceptContentTypes = ""
	kubeClient := client.NewForConfigOrDie(kubeClientConfig)
	aggregatorClient := aggregatorclient.NewForConfigOrDie(kubeClientConfig)

	t.Log("STEP 2: Starting and waiting for an Aggregated API Server")
	aggregatorIP, aggregatorPort, cleanUpFn, err := startAndWaitForAggregatedAPI(t, stopCh, kubeClientConfig, proxyCACertPath, proxyCACert)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanUpFn()

	t.Log("STEP 3: Registering the Aggregated API Server and waiting for it to show up in the discovery")
	if err := registerAggregatedAPIAndWaitForDiscovery(t, kubeClient, aggregatorClient, aggregatorIP, aggregatorPort); err != nil {
		t.Fatal(err)
	}

	t.Log("STEP 4: Swapping the proxy certificate and trying to connect to the aggregated API")
	{
		_, newProxyClientKey, newProxyClientCert, err := loadCAWithClientCert(servingCertPath, "new")
		if err != nil {
			t.Fatal(err)
		}
		if err := swapProxyClientCertAndConnect(kubeClient, proxyClientCertPath, proxyClientKeyPath, newProxyClientKey, newProxyClientCert, http.StatusUnauthorized); err != nil {
			t.Fatal(err)
		}
	}
	t.Log("STEP 5: Swapping the proxy certificate back to original and connect to the aggregated API")
	{
		_, proxyClientKey, proxyClientCert, err = loadCAWithClientCert(servingCertPath, "original")
		if err != nil {
			t.Fatal(err)
		}
		if err := swapProxyClientCertAndConnect(kubeClient, proxyClientCertPath, proxyClientKeyPath, proxyClientKey, proxyClientCert, http.StatusOK); err != nil {
			t.Fatal(err)
		}
	}

}

func startAndWaitForAggregatedAPI(t *testing.T, stopCh <-chan struct{}, clientConfig *rest.Config, proxyCAFilePath string, proxyCACert *x509.Certificate) (string, int, func(), error) {
	wardleToKASKubeConfigFile := apiserverintegration.WriteKubeConfigForWardleServerToKASConnection(t, rest.CopyConfig(clientConfig))
	wardleCertDir, _ := ioutil.TempDir("", "test-integration-wardle-server")

	// Writes the CA certificate to use with Aggregated API server
	if err := ioutil.WriteFile(proxyCAFilePath, testutil.EncodeCertPEM(proxyCACert), 0644); err != nil {
		t.Fatal(err)
	}


	cleanUpFn := func() {
		os.Remove(wardleToKASKubeConfigFile)
		os.RemoveAll(wardleCertDir)
	}

	handleErrFn := func(err error) (string, int, func(), error) {
		cleanUpFn()
		return "", 0, nil, err
	}

	wardleIP, err := getIPToListenOn()
	if err != nil {
		return handleErrFn(err)
	}

	wardleListener, wardlePort, err := genericapiserveroptions.CreateListener("tcp", fmt.Sprintf("%s:0", wardleIP), net.ListenConfig{})
	if err != nil {
		return handleErrFn(err)
	}

	go func() {
		o := sampleserver.NewWardleServerOptions(os.Stdout, os.Stderr)
		o.RecommendedOptions.SecureServing.Listener = wardleListener
		o.RecommendedOptions.SecureServing.BindAddress = net.ParseIP(wardleIP)
		wardleCmd := sampleserver.NewCommandStartWardleServer(o, stopCh)
		wardleCmd.SetArgs([]string{
			"--requestheader-client-ca-file", proxyCAFilePath,
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

	_, err = apiserverintegration.WaitForWardleRunning(t, clientConfig, wardleCertDir, wardleIP, wardlePort)
	if err != nil {
		return handleErrFn(err)
	}

	return wardleIP, wardlePort, cleanUpFn, nil
}

func registerAggregatedAPIAndWaitForDiscovery(t *testing.T, kubeClient *client.Clientset, aggregatorClient *aggregatorclient.Clientset, aggregatorIP string, aggregatorPort int) error {
	_, err := kubeClient.CoreV1().Namespaces().Create(context.Background(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "kube-wardle"}}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("unable to create a namespace, err: %v", err)
	}

	_, err = kubeClient.CoreV1().Services("kube-wardle").Create(context.Background(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "api"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 443}}}}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("unable to create object: %v", err)
	}

	ep := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name: "api",
		},
		Subsets: []v1.EndpointSubset{
			{
				Addresses: []v1.EndpointAddress{{IP: aggregatorIP}},
				Ports:     []v1.EndpointPort{{Port: int32(aggregatorPort), Protocol: v1.ProtocolTCP}},
			},
		},
	}
	_, err = kubeClient.CoreV1().Endpoints("kube-wardle").Create(context.Background(), ep, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	_, err = aggregatorClient.ApiregistrationV1().APIServices().Create(context.TODO(), &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Service: &apiregistrationv1.ServiceReference{
				Namespace: "kube-wardle",
				Name:      "api",
			},
			InsecureSkipTLSVerify: true, // this is okay as we don't want to validate wardle certificate
			Group:                 "wardle.example.com",
			Version:               "v1alpha1",
			GroupPriorityMinimum:  200,
			VersionPriority:       200,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	// wait until wardle is  available
	if err = wait.Poll(300*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		svc, err := aggregatorClient.ApiregistrationV1().APIServices().Get(context.Background(), "v1alpha1.wardle.example.com", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, cond := range svc.Status.Conditions {
			if cond.Type == apiregistrationv1.Available && cond.Status == apiregistrationv1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		return fmt.Errorf("failed waiting for wardle to become availiable, err = %v", err)
	}

	// check the are no discovery errors
	err = wait.Poll(300*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		if _, err = kubeClient.Discovery().ServerResources(); err != nil {
			t.Logf("discovery call returned an unexpected error %q, tyring one more time", err)
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		return fmt.Errorf("failed waiting for discovery, err = %v", err)
	}

	// try to hit the new endpoint
	res := kubeClient.RESTClient().Get().AbsPath("/apis/wardle.example.com/v1alpha1/flunders").Do(context.Background())
	st := 0
	res.StatusCode(&st)
	if st != http.StatusOK {
		return fmt.Errorf("unexpected status code %v while accessing the aggregated API server", st)
	}

	return nil
}

func swapProxyClientCertAndConnect(kubeClient *client.Clientset, proxyClientCertPath string, proxyClientKeyPath string, proxyClientKey []byte, proxyClientCert *x509.Certificate, expectedHTTPStatus int) error {
	if err := ioutil.WriteFile(proxyClientCertPath, testutil.EncodeCertPEM(proxyClientCert), 0644); err != nil {
		return err
	}

	if err := ioutil.WriteFile(proxyClientKeyPath, proxyClientKey, 0644); err != nil {
		return err
	}

	// give it a time to reload the cert
	time.Sleep(2 * time.Second)
	res := kubeClient.RESTClient().Get().AbsPath("/apis/wardle.example.com/v1alpha1/flunders").Do(context.Background())
	st := 0
	res.StatusCode(&st)
	if st != expectedHTTPStatus {
		return fmt.Errorf("unexpected status code %v returned (expected %v) from the aggregated API after swaping the proxy cert", expectedHTTPStatus, st)
	}
	return nil
}

func getIPToListenOn() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}
	for _, addr := range addrs {
		ipnet, ok := addr.(*net.IPNet)
		if !ok {
			continue
		}
		if ipnet.IP.IsLoopback() || ipnet.IP.IsUnspecified() || ipnet.IP.IsLinkLocalUnicast() || ipnet.IP.IsLinkLocalMulticast() {
			continue
		}
		return ipnet.IP.String(), nil
	}
	return "", errors.New("didn't find suitable IP address on a local machine to listen on")
}

func loadCAWithClientCert(servingCertPath, suffix string) (*x509.Certificate, []byte, *x509.Certificate, error) {

	proxyClientCertPath := filepath.Join(servingCertPath, "proxy-client_" + suffix + ".pem")
	proxyClientCertData, err := ioutil.ReadFile(proxyClientCertPath)
	if err != nil {
		return nil, nil, nil, err
	}
	proxyClientCertBlock, _ := pem.Decode(proxyClientCertData)
	proxyClientCert, err := x509.ParseCertificate(proxyClientCertBlock.Bytes)

	proxyCACertPath := filepath.Join(servingCertPath, "proxy-ca_" + suffix + ".pem")
	proxyCACertData, err := ioutil.ReadFile(proxyCACertPath)
	if err != nil {
		return nil, nil, nil, err
	}
	proxyCACertBlock, _ := pem.Decode(proxyCACertData)
	proxyCACert, err := x509.ParseCertificate(proxyCACertBlock.Bytes)


	proxyClientKeyPath := filepath.Join(servingCertPath, "proxy-client_" + suffix + ".key")
	proxyClientKeyPEM, err := ioutil.ReadFile(proxyClientKeyPath)
	if err != nil {
		return nil, nil, nil, err
	}

	return proxyCACert, proxyClientKeyPEM, proxyClientCert, nil
}

// use pkgPath to identify test package directory for the test
func pkgPath(t *testing.T) (string, error) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("failed to get current file")
	}

	pkgPath := filepath.Dir(thisFile)

	// If we find bazel env variables, then -trimpath was passed so we need to
	// construct the path from the environment.
	if testSrcdir, testWorkspace := os.Getenv("TEST_SRCDIR"), os.Getenv("TEST_WORKSPACE"); testSrcdir != "" && testWorkspace != "" {
		t.Logf("Detected bazel env varaiables: TEST_SRCDIR=%q TEST_WORKSPACE=%q", testSrcdir, testWorkspace)
		pkgPath = filepath.Join(testSrcdir, testWorkspace, pkgPath)
	}

	// If the path is still not absolute, something other than bazel compiled
	// with -trimpath.
	if !filepath.IsAbs(pkgPath) {
		return "", fmt.Errorf("can't construct an absolute path from %q", pkgPath)
	}

	t.Logf("Resolved testserver package path to: %q", pkgPath)

	return pkgPath, nil
}