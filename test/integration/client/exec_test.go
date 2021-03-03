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
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// This file tests the client-go credential plugin feature.

// These constants are used to communicate behavior to the testdata/exec-plugin.sh test fixture.
const (
	outputEnvVar = "EXEC_PLUGIN_OUTPUT"
)

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

func TestExecPluginViaClient(t *testing.T) {
	result, clientAuthorizedToken, clientCertFileName, clientKeyFileName := startTestServer(t)

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
						}`, "client-unauthorized-token", read(t, clientCertFileName), read(t, clientKeyFileName)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer client-unauthorized-token"}},
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
						}`, "client-unauthorized-token", string(unauthorizedCert), string(unauthorizedKey)),
					},
				}
			},
			wantAuthorizationHeaderValues: [][]string{{"Bearer client-unauthorized-token"}},
			wantCertificate:               x509KeyPair(unauthorizedCert, unauthorizedKey, true),
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
			wantCertificate:               x509KeyPair(unauthorizedCert, unauthorizedKey, false),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var authorizationHeaderValues syncedHeaderValues
			clientConfig := rest.AnonymousClientConfig(result.ClientConfig)
			clientConfig.ExecProvider = &clientcmdapi.ExecConfig{
				Command: "testdata/exec-plugin.sh",
				// TODO(ankeesler): move to v1 once exec plugins go GA.
				APIVersion: "client.authentication.k8s.io/v1beta1",
			}
			clientConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
				return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
					authorizationHeaderValues.append(req.Header.Values("Authorization"))
					return rt.RoundTrip(req)
				})
			})

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

// objectMetaSansResourceVersionComparer compares two metav1.ObjectMeta's except for their resource
// versions. Since the underlying integration test etcd is shared, these resource versions may jump
// past the next sequential number for sequential API calls in the test.
var objectMetaSansResourceVersionComparer = cmp.Comparer(func(a, b metav1.ObjectMeta) bool {
	aa := a.DeepCopy()
	bb := b.DeepCopy()

	aa.ResourceVersion = ""
	bb.ResourceVersion = ""

	return cmp.Equal(aa, bb)
})

type oldNew struct {
	old, new interface{}
}

var oldNewComparer = cmp.Comparer(func(a, b oldNew) bool {
	return cmp.Equal(a.old, b.old, objectMetaSansResourceVersionComparer) &&
		cmp.Equal(a.new, a.new, objectMetaSansResourceVersionComparer)
})

type informerSpy struct {
	mu      sync.Mutex
	adds    []interface{}
	updates []oldNew
	deletes []interface{}
}

func (es *informerSpy) OnAdd(obj interface{}) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.adds = append(es.adds, obj)
}

func (es *informerSpy) OnUpdate(old, new interface{}) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.updates = append(es.updates, oldNew{old: old, new: new})
}

func (es *informerSpy) OnDelete(obj interface{}) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.deletes = append(es.deletes, obj)
}

// waitForEvents waits for adds, updates, and deletes to be filled with at least one event.
func (es *informerSpy) waitForEvents(t *testing.T) {
	if err := wait.PollImmediate(time.Millisecond*250, time.Second*20, func() (bool, error) {
		es.mu.Lock()
		defer es.mu.Unlock()
		return len(es.adds) > 0 && len(es.updates) > 0 && len(es.deletes) > 0, nil
	}); err != nil {
		t.Fatalf("failed to wait for events: %v", err)
	}
}

func TestExecPluginViaInformer(t *testing.T) {
	result, clientAuthorizedToken, clientCertFileName, clientKeyFileName := startTestServer(t)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*60)
	defer cancel()

	adminClient := clientset.NewForConfigOrDie(result.ClientConfig)
	ns := createNamespace(ctx, t, adminClient)

	tests := []struct {
		name                          string
		clientConfigFunc              func(*rest.Config)
		wantAuthorizationHeaderValues [][]string
		wantCertificate               *tls.Certificate
	}{
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
								"token": %q
							}
						}`, clientAuthorizedToken),
					},
				}
			},
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
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientConfig := rest.AnonymousClientConfig(result.ClientConfig)
			clientConfig.ExecProvider = &clientcmdapi.ExecConfig{
				Command: "testdata/exec-plugin.sh",
				// TODO(ankeesler): move to v1 once exec plugins go GA.
				APIVersion: "client.authentication.k8s.io/v1beta1",
			}

			if test.clientConfigFunc != nil {
				test.clientConfigFunc(clientConfig)
			}
			client := clientset.NewForConfigOrDie(clientConfig)

			informerSpy := startConfigMapInformer(ctx, t, client, ns.Name)
			createdCM, updatedCM, deletedCM := createUpdateDeleteConfigMap(ctx, t, client.CoreV1().ConfigMaps(ns.Name))
			informerSpy.waitForEvents(t)

			// Validate that the informer was called correctly.
			if diff := cmp.Diff([]interface{}{createdCM}, informerSpy.adds, objectMetaSansResourceVersionComparer); diff != "" {
				t.Errorf("unexpected add event(s), -want, +got:\n%s", diff)
			}
			if diff := cmp.Diff([]oldNew{{createdCM, updatedCM}}, informerSpy.updates, oldNewComparer); diff != "" {
				t.Errorf("unexpected update event(s), -want, +got:\n%s", diff)
			}
			if diff := cmp.Diff([]interface{}{deletedCM}, informerSpy.deletes, objectMetaSansResourceVersionComparer); diff != "" {
				t.Errorf("unexpected deleted event(s), -want, +got:\n%s", diff)
			}
		})
	}
}

func startTestServer(t *testing.T) (result *kubeapiservertesting.TestServer, clientAuthorizedToken string, clientCertFileName string, clientKeyFileName string) {
	certDir, err := ioutil.TempDir("", "kubernetes-client-exec-test-cert-dir-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(certDir); err != nil {
			t.Error(err)
		}
	})

	clientAuthorizedToken = "client-authorized-token"
	tokenFileName := writeTokenFile(t, clientAuthorizedToken)
	clientCAFileName, clientSigningCert, clientSigningKey := writeCACertFiles(t, certDir)
	clientCertFileName, clientKeyFileName = writeCerts(t, clientSigningCert, clientSigningKey, certDir, 30*time.Second)
	result = kubeapiservertesting.StartTestServerOrDie(
		t,
		nil,
		[]string{
			"--token-auth-file", tokenFileName,
			"--client-ca-file=" + clientCAFileName,
		},
		framework.SharedEtcd(),
	)
	t.Cleanup(result.TearDownFn)

	return
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

func createNamespace(ctx context.Context, t *testing.T, client clientset.Interface) *corev1.Namespace {
	t.Helper()

	ns, err := client.CoreV1().Namespaces().Create(
		ctx,
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-exec-plugin-with-informer-ns"}},
		metav1.CreateOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		// Use a new context since the one passed to this function would have timed out.
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
		defer cancel()
		if err := client.CoreV1().Namespaces().Delete(ctx, ns.Name, metav1.DeleteOptions{}); err != nil {
			t.Error(err)
		}
	})

	return ns
}

func startConfigMapInformer(ctx context.Context, t *testing.T, client clientset.Interface, namespace string) *informerSpy {
	t.Helper()

	var informerSpy informerSpy
	informerFactory := informers.NewSharedInformerFactoryWithOptions(client, 0, informers.WithNamespace(namespace))
	informerFactory.Core().V1().ConfigMaps().Informer().AddEventHandler(&informerSpy)
	informerFactory.Start(ctx.Done())
	synced := informerFactory.WaitForCacheSync(ctx.Done())
	if len(synced) != 1 {
		t.Fatalf("expected only 1 synced type, got %v", synced)
	}
	if cmSynced, ok := synced[reflect.TypeOf(&corev1.ConfigMap{})]; !(cmSynced && ok) {
		t.Fatalf("expected ConfigMaps to be synced, got %v", synced)
	}

	return &informerSpy
}

func createUpdateDeleteConfigMap(ctx context.Context, t *testing.T, cms v1.ConfigMapInterface) (created, updated, deleted *corev1.ConfigMap) {
	t.Helper()

	var err error
	created, err = cms.Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "cm"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	updated = created.DeepCopy()
	updated.Annotations = map[string]string{"tuna": "fish"}
	updated, err = cms.Update(ctx, updated, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if err := cms.Delete(ctx, updated.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	deleted = updated.DeepCopy()

	return created, updated, deleted
}
