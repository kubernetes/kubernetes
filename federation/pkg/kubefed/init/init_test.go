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

package init

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/coredns"
	kubefedtesting "k8s.io/kubernetes/federation/pkg/kubefed/testing"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"gopkg.in/gcfg.v1"
)

const (
	testNamespace    = "test-ns"
	testSvcName      = "test-service"
	testCertValidity = 1 * time.Hour

	helloMsg = "Hello, certificate test!"

	lbIP     = "10.20.30.40"
	nodeIP   = "10.20.30.50"
	nodePort = 32111

	testAPIGroup   = "testGroup"
	testAPIVersion = "testVersion"
)

func TestInitFederation(t *testing.T) {
	cmdErrMsg := ""
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		cmdErrMsg = str
	})

	fakeKubeFiles, err := kubefedtesting.FakeKubeconfigFiles()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer kubefedtesting.RemoveFakeKubeconfigFiles(fakeKubeFiles)

	testCases := []struct {
		federation                   string
		kubeconfigGlobal             string
		kubeconfigExplicit           string
		dnsZoneName                  string
		lbIP                         string
		apiserverServiceType         v1.ServiceType
		advertiseAddress             string
		serverImage                  string
		etcdImage                    string
		etcdPVCapacity               string
		etcdPVStorageClass           string
		etcdPersistence              string
		expectedErr                  string
		dnsProvider                  string
		dnsProviderConfig            string
		dryRun                       string
		apiserverArgOverrides        string
		cmArgOverrides               string
		apiserverEnableHTTPBasicAuth bool
		apiserverEnableTokenAuth     bool
		isRBACAPIAvailable           bool
	}{
		{
			federation:            "union",
			kubeconfigGlobal:      fakeKubeFiles[0],
			kubeconfigExplicit:    "",
			dnsZoneName:           "example.test.",
			lbIP:                  lbIP,
			apiserverServiceType:  v1.ServiceTypeLoadBalancer,
			serverImage:           "example.test/foo:bar",
			etcdPVCapacity:        "5Gi",
			etcdPersistence:       "true",
			expectedErr:           "",
			dnsProvider:           util.FedDNSProviderCoreDNS,
			dnsProviderConfig:     "dns-provider.conf",
			dryRun:                "",
			apiserverArgOverrides: "--client-ca-file=override,--log-dir=override",
			cmArgOverrides:        "--dns-provider=override,--log-dir=override",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[1],
			kubeconfigExplicit:   fakeKubeFiles[2],
			dnsZoneName:          "example.test.",
			lbIP:                 lbIP,
			apiserverServiceType: v1.ServiceTypeLoadBalancer,
			serverImage:          "example.test/foo:bar",
			etcdPVCapacity:       "", //test for default value of pvc-size
			etcdPersistence:      "true",
			expectedErr:          "",
			dryRun:               "",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[0],
			kubeconfigExplicit:   "",
			dnsZoneName:          "example.test.",
			lbIP:                 lbIP,
			apiserverServiceType: v1.ServiceTypeLoadBalancer,
			serverImage:          "example.test/foo:bar",
			etcdPVCapacity:       "",
			etcdPersistence:      "true",
			expectedErr:          "",
			dryRun:               "valid-run",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[0],
			kubeconfigExplicit:   "",
			dnsZoneName:          "example.test.",
			lbIP:                 lbIP,
			apiserverServiceType: v1.ServiceTypeLoadBalancer,
			serverImage:          "example.test/foo:bar",
			etcdPVCapacity:       "5Gi",
			etcdPersistence:      "false",
			expectedErr:          "",
			dryRun:               "",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[0],
			kubeconfigExplicit:   "",
			dnsZoneName:          "example.test.",
			apiserverServiceType: v1.ServiceTypeNodePort,
			serverImage:          "example.test/foo:bar",
			etcdPVCapacity:       "5Gi",
			etcdPersistence:      "true",
			expectedErr:          "",
			dryRun:               "",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[0],
			kubeconfigExplicit:   "",
			dnsZoneName:          "example.test.",
			apiserverServiceType: v1.ServiceTypeNodePort,
			advertiseAddress:     nodeIP,
			serverImage:          "example.test/foo:bar",
			etcdPVCapacity:       "5Gi",
			etcdPersistence:      "true",
			expectedErr:          "",
			dryRun:               "",
		},
		{
			federation:           "union",
			kubeconfigGlobal:     fakeKubeFiles[0],
			kubeconfigExplicit:   "",
			dnsZoneName:          "example.test.",
			apiserverServiceType: v1.ServiceTypeNodePort,
			advertiseAddress:     nodeIP,
			serverImage:          "example.test/foo:bar",
			etcdImage:            "gcr.io/google_containers/etcd:latest",
			etcdPVCapacity:       "5Gi",
			etcdPVStorageClass:   "fast",
			etcdPersistence:      "true",
			expectedErr:          "",
			dryRun:               "",
			apiserverEnableHTTPBasicAuth: true,
			apiserverEnableTokenAuth:     true,
			isRBACAPIAvailable:           true,
		},
	}

	defaultEtcdImage := "gcr.io/google_containers/etcd:3.0.17"

	//TODO: implement a negative case for dry run

	for i, tc := range testCases {
		cmdErrMsg = ""
		tmpDirPath := ""
		buf := bytes.NewBuffer([]byte{})

		if tc.dnsProvider == "" {
			tc.dnsProvider = "google-clouddns"
		}
		if tc.dnsProviderConfig != "" {
			tmpfile, err := ioutil.TempFile("", tc.dnsProviderConfig)
			if err != nil {
				t.Fatalf("[%d] unexpected error: %v", i, err)
			}
			tc.dnsProviderConfig = tmpfile.Name()
			defer os.Remove(tmpfile.Name())
		}

		// Check pkg/kubectl/cmd/testing/fake (fakeAPIFactory.DiscoveryClient()) for details of tmpDir
		// We want an unique discovery cache path for each test run, else the case from previous case would be used
		tmpDirPath, err = ioutil.TempDir("", "")
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}
		defer os.Remove(tmpDirPath)

		// If tc.etcdImage is set, setting the etcd image via the flag will be
		// validated.  If not set, the default value will be validated.
		if tc.etcdImage == "" {
			tc.etcdImage = defaultEtcdImage
		}

		hostFactory, err := fakeInitHostFactory(tc.apiserverServiceType, tc.federation, util.DefaultFederationSystemNamespace, tc.advertiseAddress, tc.lbIP, tc.dnsZoneName, tc.serverImage, tc.etcdImage, tc.dnsProvider, tc.dnsProviderConfig, tc.etcdPersistence, tc.etcdPVCapacity, tc.etcdPVStorageClass, tc.apiserverArgOverrides, tc.cmArgOverrides, tmpDirPath, tc.apiserverEnableHTTPBasicAuth, tc.apiserverEnableTokenAuth, tc.isRBACAPIAvailable)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		adminConfig, err := kubefedtesting.NewFakeAdminConfig(hostFactory, nil, "", tc.kubeconfigGlobal)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdInit(buf, adminConfig, "serverImage", defaultEtcdImage)

		cmd.Flags().Set("kubeconfig", tc.kubeconfigExplicit)
		cmd.Flags().Set("host-cluster-context", "substrate")
		cmd.Flags().Set("dns-zone-name", tc.dnsZoneName)
		cmd.Flags().Set("image", tc.serverImage)
		cmd.Flags().Set("etcd-image", tc.etcdImage)
		cmd.Flags().Set("dns-provider", tc.dnsProvider)
		cmd.Flags().Set("apiserver-arg-overrides", tc.apiserverArgOverrides)
		cmd.Flags().Set("controllermanager-arg-overrides", tc.cmArgOverrides)

		if tc.dnsProviderConfig != "" {
			cmd.Flags().Set("dns-provider-config", tc.dnsProviderConfig)
		}
		if tc.etcdPVCapacity != "" {
			cmd.Flags().Set("etcd-pv-capacity", tc.etcdPVCapacity)
		}
		if tc.etcdPVStorageClass != "" {
			cmd.Flags().Set("etcd-pv-storage-class", tc.etcdPVStorageClass)
		}
		if tc.etcdPersistence != "true" {
			cmd.Flags().Set("etcd-persistent-storage", tc.etcdPersistence)
		}
		if tc.apiserverServiceType != v1.ServiceTypeLoadBalancer {
			cmd.Flags().Set(apiserverServiceTypeFlag, string(tc.apiserverServiceType))
			cmd.Flags().Set(apiserverAdvertiseAddressFlag, tc.advertiseAddress)
		}
		if tc.dryRun == "valid-run" {
			cmd.Flags().Set("dry-run", "true")
		}
		if tc.apiserverEnableHTTPBasicAuth {
			cmd.Flags().Set("apiserver-enable-basic-auth", "true")
		}
		if tc.apiserverEnableTokenAuth {
			cmd.Flags().Set("apiserver-enable-token-auth", "true")
		}

		cmd.Run(cmd, []string{tc.federation})

		if tc.expectedErr == "" {
			// uses the name from the federation, not the response
			// Actual data passed are tested in the fake secret and cluster
			// REST clients.
			endpoint := getEndpoint(tc.apiserverServiceType, tc.lbIP, tc.advertiseAddress)
			wantedSuffix := fmt.Sprintf("Federation API server is running at: %s\n", endpoint)
			if tc.dryRun != "" {
				wantedSuffix = fmt.Sprintf("Federation control plane runs (dry run)\n")
			}

			if got := buf.String(); !strings.HasSuffix(got, wantedSuffix) {
				t.Errorf("[%d] unexpected output: got: %s, wanted suffix: %s", i, got, wantedSuffix)
				if cmdErrMsg != "" {
					t.Errorf("[%d] unexpected error message: %s", i, cmdErrMsg)
				}
			}
		} else {
			if cmdErrMsg != tc.expectedErr {
				t.Errorf("[%d] expected error: %s, got: %s, output: %s", i, tc.expectedErr, cmdErrMsg, buf.String())
			}
			return
		}

		testKubeconfigUpdate(t, tc.apiserverServiceType, tc.federation, tc.advertiseAddress, tc.lbIP, tc.kubeconfigGlobal, tc.kubeconfigExplicit, tc.apiserverEnableHTTPBasicAuth, tc.apiserverEnableTokenAuth)
	}
}

func TestMarshallAndMergeOverrides(t *testing.T) {
	testCases := []struct {
		overrideParams string
		expectedSet    sets.String
		expectedErr    string
	}{
		{
			overrideParams: "valid-format-param1=override1,valid-format-param2=override2",
			expectedSet:    sets.NewString("arg2=val2", "arg1=val1", "valid-format-param1=override1", "valid-format-param2=override2"),
			expectedErr:    "",
		},
		{
			overrideParams: "valid-format-param1=override1,arg1=override1",
			expectedSet:    sets.NewString("arg2=val2", "arg1=override1", "valid-format-param1=override1"),
			expectedErr:    "",
		},
		{
			overrideParams: "zero-value-arg=",
			expectedSet:    sets.NewString("arg2=val2", "arg1=val1", "zero-value-arg="),
			expectedErr:    "",
		},
		{
			overrideParams: "wrong-format-arg",
			expectedErr:    "wrong format for override arg: wrong-format-arg",
		},
		{
			// TODO: Multiple arg values separated by , are not supported yet
			overrideParams: "multiple-equalto-char=first-key=1",
			expectedSet:    sets.NewString("arg2=val2", "arg1=val1", "multiple-equalto-char=first-key=1"),
			expectedErr:    "",
		},
		{
			overrideParams: "=wrong-format-only-value",
			expectedErr:    "wrong format for override arg: =wrong-format-only-value, arg name cannot be empty",
		},
	}

	for i, tc := range testCases {
		args, err := marshallOverrides(tc.overrideParams)
		if tc.expectedErr == "" {
			origArgs := map[string]string{
				"arg1": "val1",
				"arg2": "val2",
			}
			merged := argMapsToArgStrings(origArgs, args)

			got := sets.NewString(merged...)
			want := tc.expectedSet

			if !got.Equal(want) {
				t.Errorf("[%d] unexpected output: got: %v, want: %v", i, got, want)
			}
		} else {
			if err.Error() != tc.expectedErr {
				t.Errorf("[%d] unexpected error output: got: %s, want: %s", i, err.Error(), tc.expectedErr)
			}
		}
	}
}

// TestCertsTLS tests TLS handshake with client authentication for any server
// name. There is a separate test below to test the certificate generation
// end-to-end over HTTPS.
// TODO(madhusudancs): Consider using a deterministic random number generator
// for generating certificates in tests.
func TestCertsTLS(t *testing.T) {
	params := []certParams{
		{
			cAddr:     "10.1.2.3",
			ips:       []string{"10.1.2.3", "10.2.3.4"},
			hostnames: []string{"federation.test", "federation2.test"},
		},
		{
			cAddr:     "10.10.20.30",
			ips:       []string{"10.20.30.40", "10.64.128.4"},
			hostnames: []string{"tls.federation.test"},
		},
	}

	tlsCfgs, err := tlsConfigs(params)
	if err != nil {
		t.Errorf("failed to generate tls configs: %v", err)
		// No point in proceeding further
		return
	}

	testCases := []struct {
		serverName string
		sCfg       *tls.Config
		cCfg       *tls.Config
		failType   string
	}{
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "10.2.3.4",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "federation.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "10.20.30.40",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[1].client,
		},
		{
			serverName: "tls.federation.test",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[1].client,
		},
		{
			serverName: "10.100.200.50",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "noexist.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "10.64.128.4",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "tls.federation.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[1].client,
			failType:   "UnknownAuthorityError",
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[1].client,
			failType:   "UnknownAuthorityError",
		},
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
	}

	for i, tc := range testCases {
		// Make a copy of the client config before modifying it.
		// We can't do a regular pointer deref shallow copy because
		// tls.Config contains an unexported sync.Once field which
		// must not be copied. This was pointed out by go vet.
		cCfg := copyTLSConfig(tc.cCfg)
		cCfg.ServerName = tc.serverName
		cCfg.BuildNameToCertificate()

		err := tlsHandshake(t, tc.sCfg, cCfg)
		if len(tc.failType) > 0 {
			switch tc.failType {
			case "HostnameError":
				if _, ok := err.(x509.HostnameError); !ok {
					t.Errorf("[%d] unexpected error: want x509.HostnameError, got: %T", i, err)
				}
			case "UnknownAuthorityError":
				if _, ok := err.(x509.UnknownAuthorityError); !ok {
					t.Errorf("[%d] unexpected error: want x509.UnknownAuthorityError, got: %T", i, err)
				}
			default:
				t.Errorf("cannot handle error type: %s", tc.failType)

			}
		} else if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
		}
	}
}

// TestCertsHTTPS cannot test client authentication for non-localhost server
// names, but it tests TLS handshake end-to-end over HTTPS.
func TestCertsHTTPS(t *testing.T) {
	params := []certParams{
		{
			// Unfortunately, due to the limitation in the way Go
			// net/http/httptest package sets up the test HTTPS/TLS server,
			// 127.0.0.1 is the only accepted server address. So, we need to
			// generate certificates for this address.
			cAddr:     "127.0.0.1",
			ips:       []string{"127.0.0.1"},
			hostnames: []string{},
		},
		{
			// Unfortunately, due to the limitation in the way Go
			// net/http/httptest package sets up the test HTTPS/TLS server,
			// 127.0.0.1 is the only accepted server address. So, we need to
			// generate certificates for this address.
			cAddr:     "localhost",
			ips:       []string{"127.0.0.1"},
			hostnames: []string{"localhost"},
		},
	}

	tlsCfgs, err := tlsConfigs(params)
	if err != nil {
		t.Errorf("failed to generate tls configs: %v", err)
		// No point in proceeding further
		return
	}

	testCases := []struct {
		sCfg *tls.Config
		cCfg *tls.Config
		fail bool
	}{
		{
			sCfg: tlsCfgs[0].server,
			cCfg: tlsCfgs[0].client,
			fail: false,
		},
		{
			sCfg: tlsCfgs[0].server,
			cCfg: tlsCfgs[1].client,
			fail: true,
		},
		{
			sCfg: tlsCfgs[1].server,
			cCfg: tlsCfgs[0].client,
			fail: true,
		},
	}

	for i, tc := range testCases {
		// Make a copy of the client config before modifying it.
		// We can't do a regular pointer deref shallow copy because
		// tls.Config contains an unexported sync.Once field which
		// must not be copied. This was pointed out by go vet.
		cCfg := copyTLSConfig(tc.cCfg)
		cCfg.BuildNameToCertificate()

		s, err := fakeHTTPSServer(tc.sCfg)
		if err != nil {
			t.Errorf("[%d] unexpected error starting TLS server: %v", i, err)
			// No point in proceeding
			continue
		}
		defer s.Close()

		tr := &http.Transport{
			TLSClientConfig: cCfg,
		}
		client := &http.Client{Transport: tr}
		resp, err := client.Get(s.URL)
		if tc.fail {
			_, ok := err.(*url.Error)
			if !ok || !strings.HasSuffix(err.Error(), "x509: certificate signed by unknown authority") {
				t.Errorf("[%d] unexpected error: want x509.HostnameError, got: %T", i, err)
			}
			// We are done for this test.
			continue
		} else if err != nil {
			t.Errorf("[%d] unexpected error while sending GET request to the server: %T", i, err)
			// No point in proceeding
			continue
		}
		defer resp.Body.Close()

		got, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("[%d] unexpected error reading server response: %v", i, err)
		} else if string(got) != helloMsg {
			t.Errorf("[%d] want %q, got %q", i, helloMsg, got)
		}
	}
}

func fakeInitHostFactory(apiserverServiceType v1.ServiceType, federationName, namespaceName, advertiseAddress, lbIp, dnsZoneName, serverImage, etcdImage, dnsProvider, dnsProviderConfig, etcdPersistence, etcdPVCapacity, etcdPVStorageClass, apiserverOverrideArg, cmOverrideArg, tmpDirPath string, apiserverEnableHTTPBasicAuth, apiserverEnableTokenAuth, isRBACAPIAvailable bool) (cmdutil.Factory, error) {
	svcName := federationName + "-apiserver"
	svcUrlPrefix := "/api/v1/namespaces/federation-system/services"
	credSecretName := svcName + "-credentials"
	cmKubeconfigSecretName := federationName + "-controller-manager-kubeconfig"
	pvCap := "10Gi"
	if etcdPVCapacity != "" {
		pvCap = etcdPVCapacity
	}

	capacity, err := resource.ParseQuantity(pvCap)
	if err != nil {
		return nil, err
	}
	pvcName := svcName + "-etcd-claim"
	replicas := int32(1)

	namespace := v1.Namespace{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Namespace",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: namespaceName,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
	}

	svc := v1.Service{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Service",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespaceName,
			Name:      svcName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Spec: v1.ServiceSpec{
			Type:     apiserverServiceType,
			Selector: apiserverSvcSelector,
			Ports: []v1.ServicePort{
				{
					Name:       "https",
					Protocol:   "TCP",
					Port:       443,
					TargetPort: intstr.FromString(apiServerSecurePortName),
				},
			},
		},
	}

	svcWithLB := svc
	svcWithLB.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{
					IP: lbIp,
				},
			},
		},
	}

	credSecret := v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      credSecretName,
			Namespace: namespaceName,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Data: nil,
	}

	cmKubeconfigSecret := v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmKubeconfigSecretName,
			Namespace: namespaceName,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Data: nil,
	}

	cmDNSProviderSecret := v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      dnsProviderSecretName,
			Namespace: namespaceName,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Data: nil,
	}

	var storageClassName *string
	if len(etcdPVStorageClass) > 0 {
		storageClassName = &etcdPVStorageClass
	}

	pvc := v1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind:       "PersistentVolumeClaim",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: capacity,
				},
			},
			StorageClassName: storageClassName,
		},
	}

	sa := v1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ServiceAccount",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "federation-controller-manager",
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
	}

	role := rbacv1beta1.Role{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Role",
			APIVersion: rbacv1beta1.SchemeGroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "federation-system:federation-controller-manager",
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Rules: []rbacv1beta1.PolicyRule{
			{
				Verbs:     []string{"get", "list", "watch"},
				APIGroups: []string{""},
				Resources: []string{"secrets"},
			},
		},
	}

	rolebinding := rbacv1beta1.RoleBinding{
		TypeMeta: metav1.TypeMeta{
			Kind:       "RoleBinding",
			APIVersion: rbacv1beta1.SchemeGroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "federation-system:federation-controller-manager",
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: federationName,
			},
		},
		Subjects: []rbacv1beta1.Subject{
			{
				Kind:      "ServiceAccount",
				APIGroup:  "",
				Name:      "federation-controller-manager",
				Namespace: "federation-system",
			},
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
			Name:     "federation-system:federation-controller-manager",
		},
	}

	node := v1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeIP,
		},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{
					Type:    v1.NodeExternalIP,
					Address: nodeIP,
				},
			},
		},
	}
	nodeList := v1.NodeList{}
	nodeList.Items = append(nodeList.Items, node)

	address := lbIp
	if apiserverServiceType == v1.ServiceTypeNodePort {
		if advertiseAddress != "" {
			address = advertiseAddress
		} else {
			address = nodeIP
		}
	}

	apiserverCommand := []string{
		"/hyperkube",
		"federation-apiserver",
	}
	apiserverArgs := []string{
		"--bind-address=0.0.0.0",
		"--etcd-servers=http://localhost:2379",
		fmt.Sprintf("--secure-port=%d", apiServerSecurePort),
		"--tls-cert-file=/etc/federation/apiserver/server.crt",
		"--tls-private-key-file=/etc/federation/apiserver/server.key",
		"--admission-control=NamespaceLifecycle",
		fmt.Sprintf("--advertise-address=%s", address),
	}

	if apiserverOverrideArg != "" {
		apiserverArgs = append(apiserverArgs, "--client-ca-file=override")
		apiserverArgs = append(apiserverArgs, "--log-dir=override")

	} else {
		apiserverArgs = append(apiserverArgs, "--client-ca-file=/etc/federation/apiserver/ca.crt")
	}
	if apiserverEnableHTTPBasicAuth {
		apiserverArgs = append(apiserverArgs, "--basic-auth-file=/etc/federation/apiserver/basicauth.csv")
	}
	if apiserverEnableTokenAuth {
		apiserverArgs = append(apiserverArgs, "--token-auth-file=/etc/federation/apiserver/token.csv")
	}
	sort.Strings(apiserverArgs)
	apiserverCommand = append(apiserverCommand, apiserverArgs...)

	apiserver := &v1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        svcName,
			Namespace:   namespaceName,
			Labels:      componentLabel,
			Annotations: map[string]string{federation.FederationNameAnnotation: federationName},
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: nil,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Name:        svcName,
					Labels:      apiserverPodLabels,
					Annotations: map[string]string{federation.FederationNameAnnotation: federationName},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "apiserver",
							Image:   serverImage,
							Command: apiserverCommand,
							Ports: []v1.ContainerPort{
								{
									Name:          apiServerSecurePortName,
									ContainerPort: apiServerSecurePort,
								},
								{
									Name:          "local",
									ContainerPort: 8080,
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      credSecretName,
									MountPath: "/etc/federation/apiserver",
									ReadOnly:  true,
								},
							},
						},
						{
							Name:  "etcd",
							Image: etcdImage,
							Command: []string{
								"/usr/local/bin/etcd",
								"--data-dir",
								"/var/etcd/data",
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: credSecretName,
							VolumeSource: v1.VolumeSource{
								Secret: &v1.SecretVolumeSource{
									SecretName: credSecretName,
								},
							},
						},
					},
				},
			},
		},
	}
	if etcdPersistence == "true" {
		dataVolumeName := "etcddata"
		etcdVolume := v1.Volume{
			Name: dataVolumeName,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
				},
			},
		}
		etcdVolumeMount := v1.VolumeMount{
			Name:      dataVolumeName,
			MountPath: "/var/etcd",
		}

		apiserver.Spec.Template.Spec.Volumes = append(apiserver.Spec.Template.Spec.Volumes, etcdVolume)
		for i, container := range apiserver.Spec.Template.Spec.Containers {
			if container.Name == "etcd" {
				apiserver.Spec.Template.Spec.Containers[i].VolumeMounts = append(apiserver.Spec.Template.Spec.Containers[i].VolumeMounts, etcdVolumeMount)
			}
		}
	}

	cmCommand := []string{
		"/hyperkube",
		"federation-controller-manager",
	}

	cmArgs := []string{
		"--kubeconfig=/etc/federation/controller-manager/kubeconfig",
		fmt.Sprintf("--federation-name=%s", federationName),
		fmt.Sprintf("--zone-name=%s", dnsZoneName),
		fmt.Sprintf("--master=https://%s", svcName),
	}

	if cmOverrideArg != "" {
		cmArgs = append(cmArgs, "--dns-provider=override")
		cmArgs = append(cmArgs, "--log-dir=override")
	} else {
		cmArgs = append(cmArgs, fmt.Sprintf("--dns-provider=%s", dnsProvider))
	}

	sort.Strings(cmArgs)
	cmCommand = append(cmCommand, cmArgs...)

	cmName := federationName + "-controller-manager"
	cm := &v1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmName,
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				util.FedDomainMapKey:                fmt.Sprintf("%s=%s", federationName, strings.TrimRight(dnsZoneName, ".")),
				federation.FederationNameAnnotation: federationName,
			},
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: nil,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Name:        cmName,
					Labels:      controllerManagerPodLabels,
					Annotations: map[string]string{federation.FederationNameAnnotation: federationName},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "controller-manager",
							Image:   serverImage,
							Command: cmCommand,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      cmKubeconfigSecretName,
									MountPath: "/etc/federation/controller-manager",
									ReadOnly:  true,
								},
							},
							Env: []v1.EnvVar{
								{
									Name: "POD_NAMESPACE",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "metadata.namespace",
										},
									},
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: cmKubeconfigSecretName,
							VolumeSource: v1.VolumeSource{
								Secret: &v1.SecretVolumeSource{
									SecretName: cmKubeconfigSecretName,
								},
							},
						},
					},
				},
			},
		},
	}
	if isRBACAPIAvailable {
		cm.Spec.Template.Spec.ServiceAccountName = "federation-controller-manager"
		cm.Spec.Template.Spec.DeprecatedServiceAccount = "federation-controller-manager"
	}
	if dnsProviderConfig != "" {
		cm = addDNSProviderConfigTest(cm, cmDNSProviderSecret.Name)
		if dnsProvider == util.FedDNSProviderCoreDNS {
			cm, err = addCoreDNSServerAnnotationTest(cm, dnsZoneName, dnsProviderConfig)
			if err != nil {
				return nil, err
			}
		}
	}

	podList := v1.PodList{}
	apiServerPod := v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      svcName,
			Namespace: namespaceName,
		},
		Status: v1.PodStatus{
			Phase: "Running",
		},
	}

	cmPod := v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmName,
			Namespace: namespaceName,
		},
		Status: v1.PodStatus{
			Phase: "Running",
		},
	}

	podList.Items = append(podList.Items, apiServerPod)
	podList.Items = append(podList.Items, cmPod)

	apiGroupList := &metav1.APIGroupList{}
	testGroup := metav1.APIGroup{
		Name: testAPIGroup,
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: testAPIGroup + "/" + testAPIVersion,
				Version:      testAPIVersion,
			},
		},
	}
	rbacGroup := metav1.APIGroup{
		Name: rbac.GroupName,
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: rbac.GroupName + "/v1beta1",
				Version:      "v1beta1",
			},
		},
	}

	apiGroupList.Groups = append(apiGroupList.Groups, testGroup)
	if isRBACAPIAvailable {
		apiGroupList.Groups = append(apiGroupList.Groups, rbacGroup)
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	extCodec := testapi.Extensions.Codec()
	rbacCodec := testapi.Rbac.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
	tf.TmpDir = tmpDirPath
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/healthz":
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte("ok")))}, nil
			case p == "/api" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &metav1.APIVersions{})}, nil
			case p == "/apis" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, apiGroupList)}, nil
			case p == "/api/v1/namespaces" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.Namespace
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !apiequality.Semantic.DeepEqual(got, namespace) {
					return nil, fmt.Errorf("unexpected namespace object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, namespace))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &namespace)}, nil
			case p == svcUrlPrefix && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.Service
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !apiequality.Semantic.DeepEqual(got, svc) {
					return nil, fmt.Errorf("unexpected service object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, svc))
				}
				if apiserverServiceType == v1.ServiceTypeNodePort {
					svc.Spec.Type = v1.ServiceTypeNodePort
					svc.Spec.Ports[0].NodePort = nodePort
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &svc)}, nil
			case strings.HasPrefix(p, svcUrlPrefix) && m == http.MethodGet:
				got := strings.TrimPrefix(p, svcUrlPrefix+"/")
				if got != svcName {
					return nil, errors.NewNotFound(api.Resource("services"), got)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &svcWithLB)}, nil
			case p == "/api/v1/namespaces/federation-system/secrets" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got, want v1.Secret
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}

				switch got.Name {
				case credSecretName:
					want = credSecret
					if apiserverEnableHTTPBasicAuth {
						if got.Data["basicauth.csv"] == nil {
							return nil, fmt.Errorf("expected secret data key 'basicauth.csv', but got nil")
						}
					} else {
						if got.Data["basicauth.csv"] != nil {
							return nil, fmt.Errorf("unexpected secret data key 'basicauth.csv'")
						}
					}
					if apiserverEnableTokenAuth {
						if got.Data["token.csv"] == nil {
							return nil, fmt.Errorf("expected secret data key 'token.csv', but got nil")
						}
					} else {
						if got.Data["token.csv"] != nil {
							return nil, fmt.Errorf("unexpected secret data key 'token.csv'")
						}
					}
				case cmKubeconfigSecretName:
					want = cmKubeconfigSecret
				case dnsProviderSecretName:
					want = cmDNSProviderSecret
				}
				got.Data = nil
				if !apiequality.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("unexpected secret object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, want))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &want)}, nil
			case p == "/api/v1/namespaces/federation-system/persistentvolumeclaims" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.PersistentVolumeClaim
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !apiequality.Semantic.DeepEqual(got, pvc) {
					return nil, fmt.Errorf("unexpected PVC object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, pvc))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &pvc)}, nil
			case p == "/apis/extensions/v1beta1/namespaces/federation-system/deployments" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got, want v1beta1.Deployment
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				switch got.Name {
				case svcName:
					want = *apiserver
				case cmName:
					want = *cm
				}
				//want = *cm
				if !apiequality.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("unexpected deployment object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, want))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(extCodec, &want)}, nil
			case p == "/api/v1/namespaces/federation-system/pods" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &podList)}, nil
			case p == "/api/v1/namespaces/federation-system/serviceaccounts" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.ServiceAccount
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !helper.Semantic.DeepEqual(got, sa) {
					return nil, fmt.Errorf("unexpected service account object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, sa))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &sa)}, nil
			case p == "/apis/rbac.authorization.k8s.io/v1beta1/namespaces/federation-system/roles" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got rbacv1beta1.Role
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !helper.Semantic.DeepEqual(got, role) {
					return nil, fmt.Errorf("unexpected role object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, role))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(rbacCodec, &role)}, nil
			case p == "/apis/rbac.authorization.k8s.io/v1beta1/namespaces/federation-system/rolebindings" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got rbacv1beta1.RoleBinding
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !helper.Semantic.DeepEqual(got, rolebinding) {
					return nil, fmt.Errorf("unexpected rolebinding object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, rolebinding))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(rbacCodec, &rolebinding)}, nil
			case p == "/api/v1/nodes" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &nodeList)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	return f, nil
}

func testKubeconfigUpdate(t *testing.T, apiserverServiceType v1.ServiceType, federationName, advertiseAddress, lbIP, kubeconfigGlobal, kubeconfigExplicit string, apiserverEnableHTTPBasicAuth, apiserverEnableTokenAuth bool) {
	filename := kubeconfigGlobal
	if kubeconfigExplicit != "" {
		filename = kubeconfigExplicit
	}
	config, err := clientcmd.LoadFromFile(filename)
	if err != nil {
		t.Errorf("Failed to open kubeconfig file: %v", err)
		return
	}

	cluster, ok := config.Clusters[federationName]
	if !ok {
		t.Errorf("No cluster info for %q", federationName)
		return
	}
	endpoint := getEndpoint(apiserverServiceType, lbIP, advertiseAddress)
	if !strings.HasSuffix(endpoint, "https://") {
		endpoint = fmt.Sprintf("https://%s", endpoint)
	}
	if cluster.Server != endpoint {
		t.Errorf("Want federation API server endpoint %q, got %q", endpoint, cluster.Server)
	}

	authInfo, ok := config.AuthInfos[federationName]
	if !ok {
		t.Errorf("No credentials for %q", federationName)
		return
	}
	if len(authInfo.ClientCertificateData) == 0 {
		t.Errorf("Expected client certificate to be non-empty")
		return
	}
	if len(authInfo.ClientKeyData) == 0 {
		t.Errorf("Expected client key to be non-empty")
		return
	}
	if !apiserverEnableTokenAuth && len(authInfo.Token) != 0 {
		t.Errorf("Expected token to be empty: got: %s", authInfo.Token)
	}
	if apiserverEnableTokenAuth && len(authInfo.Token) == 0 {
		t.Errorf("Expected token to be non-empty")
	}

	httpBasicAuthInfo, ok := config.AuthInfos[fmt.Sprintf("%s-basic-auth", federationName)]
	if !apiserverEnableHTTPBasicAuth && ok {
		t.Errorf("Expected basic auth AuthInfo entry not to exist: got %v", httpBasicAuthInfo)
		return
	}

	if apiserverEnableHTTPBasicAuth {
		if !ok {
			t.Errorf("Expected basic auth AuthInfo entry to exist")
			return
		}
		if httpBasicAuthInfo.Username != "admin" {
			t.Errorf("Unexpected username in basic auth AuthInfo entry: got %s, want admin", httpBasicAuthInfo.Username)
		}
		if len(httpBasicAuthInfo.Password) == 0 {
			t.Errorf("Expected basic auth AuthInfo entry to contain password")
		}
	}

	context, ok := config.Contexts[federationName]
	if !ok {
		t.Errorf("No context for %q", federationName)
		return
	}
	if context.Cluster != federationName {
		t.Errorf("Want context cluster name: %q, got: %q", federationName, context.Cluster)
	}
	if context.AuthInfo != federationName {
		t.Errorf("Want context auth info: %q, got: %q", federationName, context.AuthInfo)
	}
}

type clientServerTLSConfigs struct {
	server *tls.Config
	client *tls.Config
}

type certParams struct {
	cAddr     string
	ips       []string
	hostnames []string
}

func tlsHandshake(t *testing.T, sCfg, cCfg *tls.Config) error {
	// Tried to use net.Pipe() instead of TCP. But the connections returned by
	// net.Pipe() do a fully-synchronous reads and writes on both the ends.
	// So if a TLS handshake fails, they can't return the error until the
	// other side reads the message which it did not expect. Since the other
	// side does not read the message it did not expect, the server and
	// clients hang. Since TCP is non-blocking we use that as transport
	// instead. One could have as well used a Unix Domain Socket, but TCP is
	// more portable.
	s, err := tls.Listen("tcp", "", sCfg)
	if err != nil {
		return fmt.Errorf("failed to create a test TLS server: %v", err)
	}
	defer s.Close()

	errCh := make(chan error)
	go func() {
		for {
			conn, err := s.Accept()
			if err != nil {
				errCh <- fmt.Errorf("failed to accept a TLS connection: %v", err)
				return
			}
			gotByte := make([]byte, len(helloMsg))
			_, err = conn.Read(gotByte)
			if err != nil && err != io.EOF {
				errCh <- fmt.Errorf("failed to read input: %v", err)
			} else if got := string(gotByte); got != helloMsg {
				errCh <- fmt.Errorf("got %q, want %q", got, helloMsg)
			}
			errCh <- nil
			return
		}
	}()

	// workaround [::] not working in ipv4 only systems (https://github.com/golang/go/issues/18806)
	// TODO: remove with Golang 1.9 with https://go-review.googlesource.com/c/45088/
	addr := strings.TrimPrefix(s.Addr().String(), "[::]")

	c, err := tls.Dial("tcp", addr, cCfg)
	if err != nil {
		// Intentionally not serializing the error received because we want to
		// test for the failure case in the caller test function.
		return err
	}
	defer c.Close()
	if _, err := c.Write([]byte(helloMsg)); err != nil {
		return fmt.Errorf("failed to write to server: %v", err)
	}

	return <-errCh
}

func fakeHTTPSServer(sCfg *tls.Config) (*httptest.Server, error) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, helloMsg)
	}))

	s.TLS.Certificates = sCfg.Certificates
	s.TLS.RootCAs = sCfg.RootCAs
	s.TLS.ClientAuth = sCfg.ClientAuth
	s.TLS.ClientCAs = sCfg.ClientCAs
	s.TLS.InsecureSkipVerify = sCfg.InsecureSkipVerify
	return s, nil
}

func tlsConfigs(params []certParams) ([]clientServerTLSConfigs, error) {
	tlsCfgs := []clientServerTLSConfigs{}
	for i, p := range params {
		sCfg, cCfg, err := genServerClientTLSConfigs(testNamespace, p.cAddr, testSvcName, HostClusterLocalDNSZoneName, p.ips, p.hostnames)
		if err != nil {
			return nil, fmt.Errorf("[%d] failed to generate tls configs: %v", i, err)
		}
		tlsCfgs = append(tlsCfgs, clientServerTLSConfigs{sCfg, cCfg})
	}
	return tlsCfgs, nil
}

func genServerClientTLSConfigs(namespace, name, svcName, localDNSZoneName string, ips, hostnames []string) (*tls.Config, *tls.Config, error) {
	entKeyPairs, err := genCerts(namespace, name, svcName, localDNSZoneName, ips, hostnames)
	if err != nil {
		return nil, nil, fmt.Errorf("unexpected error generating certs: %v", err)
	}

	roots := x509.NewCertPool()
	roots.AddCert(entKeyPairs.ca.Cert)

	serverCert := tls.Certificate{
		Certificate: [][]byte{
			entKeyPairs.server.Cert.Raw,
		},
		PrivateKey: entKeyPairs.server.Key,
	}

	cmCert := tls.Certificate{
		Certificate: [][]byte{
			entKeyPairs.controllerManager.Cert.Raw,
		},
		PrivateKey: entKeyPairs.controllerManager.Key,
	}

	sCfg := &tls.Config{
		Certificates:       []tls.Certificate{serverCert},
		RootCAs:            roots,
		ClientAuth:         tls.RequireAndVerifyClientCert,
		ClientCAs:          roots,
		InsecureSkipVerify: false,
	}

	cCfg := &tls.Config{
		Certificates: []tls.Certificate{cmCert},
		RootCAs:      roots,
	}

	return sCfg, cCfg, nil
}

func copyTLSConfig(cfg *tls.Config) *tls.Config {
	// We are copying only the required fields.
	return &tls.Config{
		Certificates:       cfg.Certificates,
		RootCAs:            cfg.RootCAs,
		ClientAuth:         cfg.ClientAuth,
		ClientCAs:          cfg.ClientCAs,
		InsecureSkipVerify: cfg.InsecureSkipVerify,
	}
}

func getEndpoint(apiserverServiceType v1.ServiceType, lbIP, advertiseAddress string) string {
	endpoint := lbIP
	if apiserverServiceType == v1.ServiceTypeNodePort {
		if advertiseAddress != "" {
			endpoint = advertiseAddress + ":" + strconv.Itoa(nodePort)
		} else {
			endpoint = nodeIP + ":" + strconv.Itoa(nodePort)
		}
	}
	return endpoint
}

// TODO: Reuse the function addDNSProviderConfig once that function is converted to use versioned objects.
func addDNSProviderConfigTest(dep *v1beta1.Deployment, secretName string) *v1beta1.Deployment {
	const (
		dnsProviderConfigVolume    = "config-volume"
		dnsProviderConfigMountPath = "/etc/federation/dns-provider"
	)

	// Create a volume from dns-provider secret
	volume := v1.Volume{
		Name: dnsProviderConfigVolume,
		VolumeSource: v1.VolumeSource{
			Secret: &v1.SecretVolumeSource{
				SecretName: secretName,
			},
		},
	}
	dep.Spec.Template.Spec.Volumes = append(dep.Spec.Template.Spec.Volumes, volume)

	// Mount dns-provider secret volume to controller-manager container
	volumeMount := v1.VolumeMount{
		Name:      dnsProviderConfigVolume,
		MountPath: dnsProviderConfigMountPath,
		ReadOnly:  true,
	}
	dep.Spec.Template.Spec.Containers[0].VolumeMounts = append(dep.Spec.Template.Spec.Containers[0].VolumeMounts, volumeMount)
	dep.Spec.Template.Spec.Containers[0].Command = append(dep.Spec.Template.Spec.Containers[0].Command, fmt.Sprintf("--dns-provider-config=%s/%s", dnsProviderConfigMountPath, secretName))

	return dep
}

// TODO: Reuse the function addCoreDNSServerAnnotation once that function is converted to use versioned objects.
func addCoreDNSServerAnnotationTest(deployment *v1beta1.Deployment, dnsZoneName, dnsProviderConfig string) (*v1beta1.Deployment, error) {
	var cfg coredns.Config
	if err := gcfg.ReadFileInto(&cfg, dnsProviderConfig); err != nil {
		return nil, err
	}

	deployment.Annotations[util.FedDNSZoneName] = dnsZoneName
	deployment.Annotations[util.FedNameServer] = cfg.Global.CoreDNSEndpoints
	deployment.Annotations[util.FedDNSProvider] = util.FedDNSProviderCoreDNS
	return deployment, nil
}
