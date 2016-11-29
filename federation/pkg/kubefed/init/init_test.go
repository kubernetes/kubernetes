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
	"strings"
	"testing"
	"time"

	"k8s.io/client-go/pkg/util/diff"
	kubefedtesting "k8s.io/kubernetes/federation/pkg/kubefed/testing"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

const (
	testNamespace    = "test-ns"
	testSvcName      = "test-service"
	testCertValidity = 1 * time.Hour

	helloMsg = "Hello, certificate test!"
)

func TestInitFederation(t *testing.T) {
	cmdErrMsg := ""
	dnsProvider := ""
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		cmdErrMsg = str
	})

	fakeKubeFiles, err := kubefedtesting.FakeKubeconfigFiles()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer kubefedtesting.RemoveFakeKubeconfigFiles(fakeKubeFiles)

	testCases := []struct {
		federation         string
		kubeconfigGlobal   string
		kubeconfigExplicit string
		dnsZoneName        string
		lbIP               string
		image              string
		expectedErr        string
		dnsProvider        string
	}{
		{
			federation:         "union",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			dnsZoneName:        "example.test.",
			lbIP:               "10.20.30.40",
			image:              "example.test/foo:bar",
			expectedErr:        "",
			dnsProvider:        "test-dns-provider",
		},
		{
			federation:         "union",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			dnsZoneName:        "example.test.",
			lbIP:               "10.20.30.40",
			image:              "example.test/foo:bar",
			expectedErr:        "",
			dnsProvider:        "", //test for default value of dns provider
		},
	}

	for i, tc := range testCases {
		cmdErrMsg = ""
		dnsProvider = ""
		buf := bytes.NewBuffer([]byte{})

		if "" != tc.dnsProvider {
			dnsProvider = tc.dnsProvider
		} else {
			dnsProvider = "google-clouddns" //default value of dns-provider
		}
		hostFactory, err := fakeInitHostFactory(tc.federation, util.DefaultFederationSystemNamespace, tc.lbIP, tc.dnsZoneName, tc.image, dnsProvider)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		adminConfig, err := kubefedtesting.NewFakeAdminConfig(hostFactory, tc.kubeconfigGlobal)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdInit(buf, adminConfig)

		cmd.Flags().Set("kubeconfig", tc.kubeconfigExplicit)
		cmd.Flags().Set("host-cluster-context", "substrate")
		cmd.Flags().Set("dns-zone-name", tc.dnsZoneName)
		cmd.Flags().Set("image", tc.image)
		if "" != tc.dnsProvider {
			cmd.Flags().Set("dns-provider", tc.dnsProvider)
		}
		cmd.Run(cmd, []string{tc.federation})

		if tc.expectedErr == "" {
			// uses the name from the federation, not the response
			// Actual data passed are tested in the fake secret and cluster
			// REST clients.
			want := fmt.Sprintf("Federation API server is running at: %s\n", tc.lbIP)
			if got := buf.String(); got != want {
				t.Errorf("[%d] unexpected output: got: %s, want: %s", i, got, want)
				if cmdErrMsg != "" {
					t.Errorf("[%d] unexpected error message: %s", i, cmdErrMsg)
				}
			}
		} else {
			if cmdErrMsg != tc.expectedErr {
				t.Errorf("[%d] expected error: %s, got: %s, output: %s", i, tc.expectedErr, cmdErrMsg, buf.String())
			}
		}

		testKubeconfigUpdate(t, tc.federation, tc.lbIP, tc.kubeconfigGlobal, tc.kubeconfigExplicit)
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

func fakeInitHostFactory(federationName, namespaceName, ip, dnsZoneName, image, dnsProvider string) (cmdutil.Factory, error) {
	svcName := federationName + "-apiserver"
	svcUrlPrefix := "/api/v1/namespaces/federation-system/services"
	credSecretName := svcName + "-credentials"
	cmKubeconfigSecretName := federationName + "-controller-manager-kubeconfig"
	capacity, err := resource.ParseQuantity("10Gi")
	if err != nil {
		return nil, err
	}
	pvcName := svcName + "-etcd-claim"
	replicas := int32(1)

	namespace := v1.Namespace{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Namespace",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name: namespaceName,
		},
	}

	svc := v1.Service{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Service",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Namespace: namespaceName,
			Name:      svcName,
			Labels:    componentLabel,
		},
		Spec: v1.ServiceSpec{
			Type:     v1.ServiceTypeLoadBalancer,
			Selector: apiserverSvcSelector,
			Ports: []v1.ServicePort{
				{
					Name:       "https",
					Protocol:   "TCP",
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}

	svcWithLB := svc
	svcWithLB.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{
					IP: ip,
				},
			},
		},
	}

	credSecret := v1.Secret{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      credSecretName,
			Namespace: namespaceName,
		},
		Data: nil,
	}

	cmKubeconfigSecret := v1.Secret{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      cmKubeconfigSecretName,
			Namespace: namespaceName,
		},
		Data: nil,
	}

	pvc := v1.PersistentVolumeClaim{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "PersistentVolumeClaim",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      pvcName,
			Namespace: namespaceName,
			Labels:    componentLabel,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "yes",
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
		},
	}

	apiserver := v1beta1.Deployment{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Deployment",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      svcName,
			Namespace: namespaceName,
			Labels:    componentLabel,
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: nil,
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Name:   svcName,
					Labels: apiserverPodLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "apiserver",
							Image: image,
							Command: []string{
								"/hyperkube",
								"federation-apiserver",
								"--bind-address=0.0.0.0",
								"--etcd-servers=http://localhost:2379",
								"--service-cluster-ip-range=10.0.0.0/16",
								"--secure-port=443",
								"--client-ca-file=/etc/federation/apiserver/ca.crt",
								"--tls-cert-file=/etc/federation/apiserver/server.crt",
								"--tls-private-key-file=/etc/federation/apiserver/server.key",
								"--advertise-address=" + ip,
							},
							Ports: []v1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: 443,
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
							Image: "quay.io/coreos/etcd:v2.3.3",
							Command: []string{
								"/etcd",
								"--data-dir",
								"/var/etcd/data",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "etcddata",
									MountPath: "/var/etcd",
								},
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
						{
							Name: "etcddata",
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvcName,
								},
							},
						},
					},
				},
			},
		},
	}

	cmName := federationName + "-controller-manager"
	cm := v1beta1.Deployment{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Deployment",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      cmName,
			Namespace: namespaceName,
			Labels:    componentLabel,
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: nil,
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Name:   cmName,
					Labels: controllerManagerPodLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "controller-manager",
							Image: image,
							Command: []string{
								"/hyperkube",
								"federation-controller-manager",
								"--master=https://" + svcName,
								"--kubeconfig=/etc/federation/controller-manager/kubeconfig",
								fmt.Sprintf("--dns-provider=%s", dnsProvider),
								"--dns-provider-config=",
								fmt.Sprintf("--federation-name=%s", federationName),
								fmt.Sprintf("--zone-name=%s", dnsZoneName),
							},
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

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	extCodec := testapi.Extensions.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
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
				if !api.Semantic.DeepEqual(got, namespace) {
					return nil, fmt.Errorf("Unexpected namespace object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, namespace))
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
				if !api.Semantic.DeepEqual(got, svc) {
					return nil, fmt.Errorf("Unexpected service object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, svc))
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
				// Obtained secret contains generated data which cannot
				// be compared, so we just nullify the generated part
				// and compare the rest of the secret. The generated
				// parts are tested in other tests.
				got.Data = nil
				switch got.Name {
				case credSecretName:
					want = credSecret
				case cmKubeconfigSecretName:
					want = cmKubeconfigSecret
				}
				if !api.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("Unexpected secret object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, want))
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
				if !api.Semantic.DeepEqual(got, pvc) {
					return nil, fmt.Errorf("Unexpected PVC object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, pvc))
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
					want = apiserver
				case cmName:
					want = cm
				}
				if !api.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("Unexpected deployment object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, want))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(extCodec, &want)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	return f, nil
}

func testKubeconfigUpdate(t *testing.T, federationName, lbIP, kubeconfigGlobal, kubeconfigExplicit string) {
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
	endpoint := lbIP
	if !strings.HasSuffix(lbIP, "https://") {
		endpoint = fmt.Sprintf("https://%s", lbIP)
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
	if authInfo.Username != AdminCN {
		t.Errorf("Want username: %q, got: %q", AdminCN, authInfo.Username)
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

	c, err := tls.Dial("tcp", s.Addr().String(), cCfg)
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
