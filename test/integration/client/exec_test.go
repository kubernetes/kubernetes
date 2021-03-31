/*
Copyright 2021 The Kubernetes Authors.

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

package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/cert"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// This file tests the client-go credential plugin feature.

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type syncedHeaderValues struct {
	mu   sync.Mutex
	data [][]string
}

func (s *syncedHeaderValues) append(values []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = append(s.data, values)
}

func (s *syncedHeaderValues) get() [][]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.data
}

func TestExecPlugin(t *testing.T) {
	// These constants are used to communicate behavior to the testdata/exec-plugin.sh test fixture.
	const (
		outputEnvVar = "EXEC_PLUGIN_OUTPUT"
	)

	const (
		clientAuthorizedToken   = "authorized-token"
		clientUnauthorizedToken = "unauthorized-token"
	)

	certDir, err := ioutil.TempDir("", "kubernetes-client-exec-test-cert-dir-*")
	if err != nil {
		t.Fatal(err)
	}

	tokenFileName := writeTokenFile(t, clientAuthorizedToken)
	clientCAFileName, clientSigningCert, clientSigningKey := writeCACertFiles(t, certDir)
	clientCertFileName, clientKeyFileName := writeCerts(t, clientSigningCert, clientSigningKey, certDir, 30*time.Second)
	result := kubeapiservertesting.StartTestServerOrDie(
		t,
		nil,
		[]string{
			"--token-auth-file", tokenFileName,
			"--client-ca-file=" + clientCAFileName,
		},
		framework.SharedEtcd(),
	)
	t.Cleanup(result.TearDownFn)

	unauthorizedCert, unauthorizedKey, err := cert.GenerateSelfSignedCertKey("some-host", nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                          string
		clientConfigFunc              func(*rest.Config)
		wantAuthorizationHeaderValues [][]string
		wantCertificate               *tls.Certificate
		wantClientErrorPrefix         string
	}{
		{
			name: "unauthorized token",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: `{
										"kind": "ExecCredential",
										"apiVersion": "client.authentication.k8s.io/v1beta1",
										"status": {
											"token": "unauthorized"
										}
									}`,
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer unauthorized"}},
			wantCertificate:               &tls.Certificate{},
			wantClientErrorPrefix:         "Unauthorized",
		},
		{
			name: "unauthorized certificate",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"clientCertificateData": %q,
								"clientKeyData": %q
							}
						}`, unauthorizedCert, unauthorizedKey),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{nil},
			wantCertificate:               x509KeyPair(unauthorizedCert, unauthorizedKey, true),
			wantClientErrorPrefix:         "Unauthorized",
		},
		{
			name: "authorized token",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
						"kind": "ExecCredential",
						"apiVersion": "client.authentication.k8s.io/v1beta1",
						"status": {
							"token": "%s"
						}
					}`, clientAuthorizedToken),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientAuthorizedToken}},
			wantCertificate:               &tls.Certificate{},
		},
		{
			name: "authorized certificate",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"clientCertificateData": %s,
								"clientKeyData": %s
							}
						}`, read(t, clientCertFileName), read(t, clientKeyFileName)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{nil},
			wantCertificate:               loadX509KeyPair(clientCertFileName, clientKeyFileName),
		},
		{
			name: "authorized token and certificate",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s",
								"clientCertificateData": %s,
								"clientKeyData": %s
							}
						}`, clientAuthorizedToken, read(t, clientCertFileName), read(t, clientKeyFileName)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientAuthorizedToken}},
			wantCertificate:               loadX509KeyPair(clientCertFileName, clientKeyFileName),
		},
		{
			name: "unauthorized token and authorized certificate favors authorized certificate",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s",
								"clientCertificateData": %s,
								"clientKeyData": %s
							}
						}`, clientUnauthorizedToken, read(t, clientCertFileName), read(t, clientKeyFileName)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientUnauthorizedToken}},
			wantCertificate:               loadX509KeyPair(clientCertFileName, clientKeyFileName),
		},
		{
			name: "authorized token and unauthorized certificate favors authorized token",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s",
								"clientCertificateData": %q,
								"clientKeyData": %q
							}
						}`, clientAuthorizedToken, string(unauthorizedCert), string(unauthorizedKey)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientAuthorizedToken}},
			wantCertificate:               x509KeyPair([]byte(unauthorizedCert), []byte(unauthorizedKey), true),
		},
		{
			name: "unauthorized token and unauthorized certificate",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s",
								"clientCertificateData": %q,
								"clientKeyData": %q
							}
						}`, clientUnauthorizedToken, string(unauthorizedCert), string(unauthorizedKey)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientUnauthorizedToken}},
			wantCertificate:               x509KeyPair([]byte(unauthorizedCert), []byte(unauthorizedKey), true),
			wantClientErrorPrefix:         "Unauthorized",
		},
		{
			name: "good token with static auth basic creds favors static auth basic creds",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s"
							}
						}`, clientAuthorizedToken),
					},
				}
				c.Username = "unauthorized"
				c.Password = "unauthorized"
			},
			wantAuthorizationHeaderValues: [][]string{{"Basic " + basicAuthHeaderValue("unauthorized", "unauthorized")}},
			wantCertificate:               &tls.Certificate{},
			wantClientErrorPrefix:         "Unauthorized",
		},
		{
			name: "good token with static auth bearer token favors static auth bearer token",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s"
							}
						}`, clientAuthorizedToken),
					},
				}
				c.BearerToken = "some-unauthorized-token"
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer some-unauthorized-token"}},
			wantClientErrorPrefix:         "Unauthorized",
		},
		{
			// This is not the behavior we would expect, see
			//   https://github.com/kubernetes/kubernetes/issues/99603
			name: "good token with static auth cert and key favors exec plugin",
			clientConfigFunc: func(c *rest.Config) {
				c.ExecProvider.Env = []clientcmdapi.ExecEnvVar{
					{
						Name: outputEnvVar,
						Value: fmt.Sprintf(`{
							"kind": "ExecCredential",
							"apiVersion": "client.authentication.k8s.io/v1beta1",
							"status": {
								"token": "%s"
							}
						}`, clientAuthorizedToken),
					},
				}
				c.CertData = unauthorizedCert
				c.KeyData = unauthorizedKey
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer " + clientAuthorizedToken}},
			wantCertificate:               x509KeyPair([]byte(unauthorizedCert), []byte(unauthorizedKey), false),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpDir, err := ioutil.TempDir("", "kubernetes-client-exec-test-plugin-dir-*")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tmpDir)

			var authorizationHeaderValues syncedHeaderValues
			clientConfig := rest.AnonymousClientConfig(result.ClientConfig)
			clientConfig.ExecProvider = &clientcmdapi.ExecConfig{
				Command: "testdata/exec-plugin.sh",
				// TODO(ankeesler): move to v1 once exec plugins go GA.
				APIVersion: "client.authentication.k8s.io/v1beta1",
			}
			clientConfig.Wrap(transport.WrapperFunc(func(rt http.RoundTripper) http.RoundTripper {
				return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
					authorizationHeaderValues.append(req.Header.Values("Authorization"))
					return rt.RoundTrip(req)
				})
			}))

			if test.clientConfigFunc != nil {
				test.clientConfigFunc(clientConfig)
			}
			client := clientset.NewForConfigOrDie(clientConfig)

			ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
			defer cancel()

			// Validate that the client works as expected on its own.
			_, err = client.CoreV1().ConfigMaps("default").List(ctx, metav1.ListOptions{})
			if test.wantClientErrorPrefix != "" {
				if err == nil || !strings.HasPrefix(err.Error(), test.wantClientErrorPrefix) {
					t.Fatalf(`got %q, wanted "%s..."`, err, test.wantClientErrorPrefix)
				}
			} else if err != nil {
				t.Fatal(err)
			}

			// Validate that the right token is used.
			if diff := cmp.Diff(test.wantAuthorizationHeaderValues, authorizationHeaderValues.get()); diff != "" {
				t.Error("unexpected authorization header values; -want, +got:\n" + diff)
			}

			// Validate that the right certs are used.
			tlsConfig, err := rest.TLSConfigFor(clientConfig)
			if err != nil {
				t.Fatal(err)
			}
			if tlsConfig.GetClientCertificate == nil {
				if test.wantCertificate != nil {
					t.Error("GetClientCertificate is nil, but we expected a certificate")
				}
			} else {
				cert, err := tlsConfig.GetClientCertificate(&tls.CertificateRequestInfo{})
				if err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(test.wantCertificate, cert); diff != "" {
					t.Error("unexpected certificate; -want, +got:\n" + diff)
				}
			}
		})
	}
}

func writeTokenFile(t *testing.T, goodToken string) string {
	t.Helper()

	tokenFile, err := ioutil.TempFile("", "kubernetes-client-exec-test-token-file-*")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := tokenFile.WriteString(fmt.Sprintf(`%s,admin,uid1,"system:masters"`, goodToken)); err != nil {
		t.Fatal(err)
	}

	if err := tokenFile.Close(); err != nil {
		t.Fatal(err)
	}

	return tokenFile.Name()
}

func read(t *testing.T, fileName string) string {
	t.Helper()
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("%q", string(data))
}

func basicAuthHeaderValue(username, password string) string {
	return base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
}

func x509KeyPair(certPEMBlock, keyPEMBlock []byte, leaf bool) *tls.Certificate {
	cert, err := tls.X509KeyPair(certPEMBlock, keyPEMBlock)
	if err != nil {
		panic(err)
	}
	if leaf {
		cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			panic(err)
		}
	}
	return &cert
}

func loadX509KeyPair(certFile, keyFile string) *tls.Certificate {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		panic(err)
	}
	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		panic(err)
	}
	return &cert
}
