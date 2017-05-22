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

package webhook

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd/api/v1"
)

const (
	errBadCertificate    = "Get .*: remote error: tls: bad certificate"
	errNoConfiguration   = "invalid configuration: no configuration has been provided"
	errMissingCertPath   = "invalid configuration: unable to read %s %s for %s due to open %s: .*"
	errSignedByUnknownCA = "Get .*: x509: certificate signed by unknown authority"
)

var (
	defaultCluster = v1.NamedCluster{
		Cluster: v1.Cluster{
			Server: "https://webhook.example.com",
			CertificateAuthorityData: caCert,
		},
	}
	defaultUser = v1.NamedAuthInfo{
		AuthInfo: v1.AuthInfo{
			ClientCertificateData: clientCert,
			ClientKeyData:         clientKey,
		},
	}
	namedCluster = v1.NamedCluster{
		Cluster: v1.Cluster{
			Server: "https://webhook.example.com",
			CertificateAuthorityData: caCert,
		},
		Name: "test-cluster",
	}
	groupVersions = []schema.GroupVersion{}
	retryBackoff  = time.Duration(500) * time.Millisecond
)

// TestDisabledGroupVersion ensures that requiring a group version works as expected
func TestDisabledGroupVersion(t *testing.T) {
	gv := schema.GroupVersion{Group: "webhook.util.k8s.io", Version: "v1"}
	gvs := []schema.GroupVersion{gv}
	registry := registered.NewOrDie(gv.String())
	_, err := NewGenericWebhook(registry, scheme.Codecs, "/some/path", gvs, retryBackoff)

	if err == nil {
		t.Errorf("expected an error")
	} else {
		aErrMsg := err.Error()
		eErrMsg := fmt.Sprintf("webhook plugin requires enabling extension resource: %s", gv)

		if aErrMsg != eErrMsg {
			t.Errorf("unexpected error message mismatch:\n  Expected: %s\n  Actual:   %s", eErrMsg, aErrMsg)
		}
	}
}

// TestKubeConfigFile ensures that a kube config file, regardless of validity, is handled properly
func TestKubeConfigFile(t *testing.T) {
	badCAPath := "/tmp/missing/ca.pem"
	badClientCertPath := "/tmp/missing/client.pem"
	badClientKeyPath := "/tmp/missing/client-key.pem"
	dir := bootstrapTestDir(t)

	defer os.RemoveAll(dir)

	// These tests check for all of the ways in which a Kubernetes config file could be malformed within the context of
	// configuring a webhook.  Configuration issues that arise while using the webhook are tested elsewhere.
	tests := []struct {
		test           string
		cluster        *v1.NamedCluster
		context        *v1.NamedContext
		currentContext string
		user           *v1.NamedAuthInfo
		errRegex       string
	}{
		{
			test:     "missing context (no default, none specified)",
			cluster:  &namedCluster,
			errRegex: errNoConfiguration,
		},
		{
			test:           "missing context (specified context is missing)",
			cluster:        &namedCluster,
			currentContext: "missing-context",
			errRegex:       errNoConfiguration,
		},
		{
			test: "context without cluster",
			context: &v1.NamedContext{
				Context: v1.Context{},
			},
			currentContext: "testing-context",
			errRegex:       errNoConfiguration,
		},
		{
			test:    "context without user",
			cluster: &namedCluster,
			context: &v1.NamedContext{
				Context: v1.Context{
					Cluster: namedCluster.Name,
				},
			},
			currentContext: "testing-context",
			errRegex:       "", // Not an error at parse time, only when using the webhook
		},
		{
			test:    "context with missing cluster",
			cluster: &namedCluster,
			context: &v1.NamedContext{
				Context: v1.Context{
					Cluster: "missing-cluster",
				},
			},
			errRegex: errNoConfiguration,
		},
		{
			test:    "context with missing user",
			cluster: &namedCluster,
			context: &v1.NamedContext{
				Context: v1.Context{
					Cluster:  namedCluster.Name,
					AuthInfo: "missing-user",
				},
			},
			currentContext: "testing-context",
			errRegex:       "", // Not an error at parse time, only when using the webhook
		},
		{
			test: "cluster with invalid CA certificate path",
			cluster: &v1.NamedCluster{
				Cluster: v1.Cluster{
					Server:               namedCluster.Cluster.Server,
					CertificateAuthority: badCAPath,
				},
			},
			user:     &defaultUser,
			errRegex: fmt.Sprintf(errMissingCertPath, "certificate-authority", badCAPath, "", badCAPath),
		},
		{
			test: "cluster with invalid CA certificate ",
			cluster: &v1.NamedCluster{
				Cluster: v1.Cluster{
					Server: namedCluster.Cluster.Server,
					CertificateAuthorityData: caKey,
				},
			},
			user:     &defaultUser,
			errRegex: "", // Not an error at parse time, only when using the webhook
		},
		{
			test:    "user with invalid client certificate path",
			cluster: &defaultCluster,
			user: &v1.NamedAuthInfo{
				AuthInfo: v1.AuthInfo{
					ClientCertificate: badClientCertPath,
					ClientKeyData:     defaultUser.AuthInfo.ClientKeyData,
				},
			},
			errRegex: fmt.Sprintf(errMissingCertPath, "client-cert", badClientCertPath, "", badClientCertPath),
		},
		{
			test:    "user with invalid client certificate",
			cluster: &defaultCluster,
			user: &v1.NamedAuthInfo{
				AuthInfo: v1.AuthInfo{
					ClientCertificateData: clientKey,
					ClientKeyData:         defaultUser.AuthInfo.ClientKeyData,
				},
			},
			errRegex: "tls: failed to find certificate PEM data in certificate input, but did find a private key; PEM inputs may have been switched",
		},
		{
			test:    "user with invalid client certificate path",
			cluster: &defaultCluster,
			user: &v1.NamedAuthInfo{
				AuthInfo: v1.AuthInfo{
					ClientCertificateData: defaultUser.AuthInfo.ClientCertificateData,
					ClientKey:             badClientKeyPath,
				},
			},
			errRegex: fmt.Sprintf(errMissingCertPath, "client-key", badClientKeyPath, "", badClientKeyPath),
		},
		{
			test:    "user with invalid client certificate",
			cluster: &defaultCluster,
			user: &v1.NamedAuthInfo{
				AuthInfo: v1.AuthInfo{
					ClientCertificateData: defaultUser.AuthInfo.ClientCertificateData,
					ClientKeyData:         clientCert,
				},
			},
			errRegex: "tls: found a certificate rather than a key in the PEM for the private key",
		},
		{
			test:     "valid configuration (certificate data embeded in config)",
			cluster:  &defaultCluster,
			user:     &defaultUser,
			errRegex: "",
		},
		{
			test: "valid configuration (certificate files referenced in config)",
			cluster: &v1.NamedCluster{
				Cluster: v1.Cluster{
					Server:               "https://webhook.example.com",
					CertificateAuthority: filepath.Join(dir, "ca.pem"),
				},
			},
			user: &v1.NamedAuthInfo{
				AuthInfo: v1.AuthInfo{
					ClientCertificate: filepath.Join(dir, "client.pem"),
					ClientKey:         filepath.Join(dir, "client-key.pem"),
				},
			},
			errRegex: "",
		},
	}

	for _, tt := range tests {
		// Use a closure so defer statements trigger between loop iterations.
		err := func() error {
			kubeConfig := v1.Config{}

			if tt.cluster != nil {
				kubeConfig.Clusters = []v1.NamedCluster{*tt.cluster}
			}

			if tt.context != nil {
				kubeConfig.Contexts = []v1.NamedContext{*tt.context}
			}

			if tt.user != nil {
				kubeConfig.AuthInfos = []v1.NamedAuthInfo{*tt.user}
			}

			kubeConfigFile, err := newKubeConfigFile(kubeConfig)

			if err == nil {
				defer os.Remove(kubeConfigFile)

				_, err = NewGenericWebhook(registered.NewOrDie(""), scheme.Codecs, kubeConfigFile, groupVersions, retryBackoff)
			}

			return err
		}()

		if err == nil {
			if tt.errRegex != "" {
				t.Errorf("%s: expected an error", tt.test)
			}
		} else {
			if tt.errRegex == "" {
				t.Errorf("%s: unexpected error: %v", tt.test, err)
			} else if !regexp.MustCompile(tt.errRegex).MatchString(err.Error()) {
				t.Errorf("%s: unexpected error message to match:\n  Expected: %s\n  Actual:   %s", tt.test, tt.errRegex, err.Error())
			}
		}
	}
}

// TestMissingKubeConfigFile ensures that a kube config path to a missing file is handled properly
func TestMissingKubeConfigFile(t *testing.T) {
	kubeConfigPath := "/some/missing/path"
	_, err := NewGenericWebhook(registered.NewOrDie(""), scheme.Codecs, kubeConfigPath, groupVersions, retryBackoff)

	if err == nil {
		t.Errorf("creating the webhook should had failed")
	} else if strings.Index(err.Error(), fmt.Sprintf("stat %s", kubeConfigPath)) != 0 {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestTLSConfig ensures that the TLS-based communication between client and server works as expected
func TestTLSConfig(t *testing.T) {
	invalidCert := []byte("invalid")
	tests := []struct {
		test                            string
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		errRegex                        string
	}{
		{
			test:       "invalid server CA",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: invalidCert,
			errRegex: errBadCertificate,
		},
		{
			test:       "invalid client certificate",
			clientCert: invalidCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			errRegex: "tls: failed to find any PEM data in certificate input",
		},
		{
			test:       "invalid client key",
			clientCert: clientCert, clientKey: invalidCert, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			errRegex: "tls: failed to find any PEM data in key input",
		},
		{
			test:       "client does not trust server",
			clientCert: clientCert, clientKey: clientKey,
			serverCert: serverCert, serverKey: serverKey,
			errRegex: errSignedByUnknownCA,
		},
		{
			test:       "server does not trust client",
			clientCert: clientCert, clientKey: clientKey, clientCA: badCACert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			errRegex: errSignedByUnknownCA + " .*",
		},
		{
			test:       "server requires auth, client provides it",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			errRegex: "",
		},
		{
			test:       "server does not require client auth",
			clientCA:   caCert,
			serverCert: serverCert, serverKey: serverKey,
			errRegex: "",
		},
		{
			test:       "server does not require client auth, client provides it",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey,
			errRegex: "",
		},
		{
			test:     "webhook does not support insecure servers",
			errRegex: errSignedByUnknownCA,
		},
	}

	for _, tt := range tests {
		// Use a closure so defer statements trigger between loop iterations.
		func() {
			// Create and start a simple HTTPS server
			server, err := newTestServer(tt.serverCert, tt.serverKey, tt.serverCA, nil)

			if err != nil {
				t.Errorf("%s: failed to create server: %v", tt.test, err)
				return
			}

			defer server.Close()

			// Create a Kubernetes client configuration file
			configFile, err := newKubeConfigFile(v1.Config{
				Clusters: []v1.NamedCluster{
					{
						Cluster: v1.Cluster{
							Server: server.URL,
							CertificateAuthorityData: tt.clientCA,
						},
					},
				},
				AuthInfos: []v1.NamedAuthInfo{
					{
						AuthInfo: v1.AuthInfo{
							ClientCertificateData: tt.clientCert,
							ClientKeyData:         tt.clientKey,
						},
					},
				},
			})

			if err != nil {
				t.Errorf("%s: %v", tt.test, err)
				return
			}

			defer os.Remove(configFile)

			wh, err := NewGenericWebhook(registered.NewOrDie(""), scheme.Codecs, configFile, groupVersions, retryBackoff)

			if err == nil {
				err = wh.RestClient.Get().Do().Error()
			}

			if err == nil {
				if tt.errRegex != "" {
					t.Errorf("%s: expected an error", tt.test)
				}
			} else {
				if tt.errRegex == "" {
					t.Errorf("%s: unexpected error: %v", tt.test, err)
				} else if !regexp.MustCompile(tt.errRegex).MatchString(err.Error()) {
					t.Errorf("%s: unexpected error message mismatch:\n  Expected: %s\n  Actual:   %s", tt.test, tt.errRegex, err.Error())
				}
			}
		}()
	}
}

// TestWithExponentialBackoff ensures that the webhook's exponential backoff support works as expected
func TestWithExponentialBackoff(t *testing.T) {
	count := 0 // To keep track of the requests
	gr := schema.GroupResource{
		Group:    "webhook.util.k8s.io",
		Resource: "test",
	}

	// Handler that will handle all backoff CONDITIONS
	ebHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		switch count++; count {
		case 1:
			// Timeout error with retry supplied
			w.WriteHeader(http.StatusGatewayTimeout)
			json.NewEncoder(w).Encode(apierrors.NewServerTimeout(gr, "get", 2))
		case 2:
			// Internal server error
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(apierrors.NewInternalError(fmt.Errorf("nope")))
		case 3:
			// HTTP error that is not retryable
			w.WriteHeader(http.StatusNotAcceptable)
			json.NewEncoder(w).Encode(apierrors.NewGenericServerResponse(http.StatusNotAcceptable, "get", gr, "testing", "nope", 0, false))
		case 4:
			// Successful request
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{
				"status": "OK",
			})
		}
	}

	// Create and start a simple HTTPS server
	server, err := newTestServer(clientCert, clientKey, caCert, ebHandler)

	if err != nil {
		t.Errorf("failed to create server: %v", err)
		return
	}

	defer server.Close()

	// Create a Kubernetes client configuration file
	configFile, err := newKubeConfigFile(v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{
					Server: server.URL,
					CertificateAuthorityData: caCert,
				},
			},
		},
		AuthInfos: []v1.NamedAuthInfo{
			{
				AuthInfo: v1.AuthInfo{
					ClientCertificateData: clientCert,
					ClientKeyData:         clientKey,
				},
			},
		},
	})

	if err != nil {
		t.Errorf("failed to create the client config file: %v", err)
		return
	}

	defer os.Remove(configFile)

	wh, err := NewGenericWebhook(registered.NewOrDie(""), scheme.Codecs, configFile, groupVersions, retryBackoff)

	if err != nil {
		t.Fatalf("failed to create the webhook: %v", err)
	}

	result := wh.WithExponentialBackoff(func() rest.Result {
		return wh.RestClient.Get().Do()
	})

	var statusCode int

	result.StatusCode(&statusCode)

	if statusCode != http.StatusNotAcceptable {
		t.Errorf("unexpected status code: %d", statusCode)
	}

	result = wh.WithExponentialBackoff(func() rest.Result {
		return wh.RestClient.Get().Do()
	})

	result.StatusCode(&statusCode)

	if statusCode != http.StatusOK {
		t.Errorf("unexpected status code: %d", statusCode)
	}
}

func bootstrapTestDir(t *testing.T) string {
	dir, err := ioutil.TempDir("", "")

	if err != nil {
		t.Fatal(err)
	}

	// The certificates needed on disk for the tests
	files := map[string][]byte{
		"ca.pem":         caCert,
		"client.pem":     clientCert,
		"client-key.pem": clientKey,
	}

	// Write the certificate files to disk or fail
	for fileName, fileData := range files {
		if err := ioutil.WriteFile(filepath.Join(dir, fileName), fileData, 0400); err != nil {
			t.Fatal(err)
		}
	}

	return dir
}

func newKubeConfigFile(config v1.Config) (string, error) {
	configFile, err := ioutil.TempFile("", "")

	if err != nil {
		return "", fmt.Errorf("unable to create the Kubernetes client config file: %v", err)
	}

	if err = json.NewEncoder(configFile).Encode(config); err != nil {
		return "", fmt.Errorf("unable to write the Kubernetes client configuration to disk: %v", err)
	}

	return configFile.Name(), nil
}

func newTestServer(clientCert, clientKey, caCert []byte, handler func(http.ResponseWriter, *http.Request)) (*httptest.Server, error) {
	var tlsConfig *tls.Config

	if clientCert != nil {
		cert, err := tls.X509KeyPair(clientCert, clientKey)

		if err != nil {
			return nil, err
		}

		tlsConfig = &tls.Config{
			Certificates: []tls.Certificate{cert},
		}
	}

	if caCert != nil {
		rootCAs := x509.NewCertPool()

		rootCAs.AppendCertsFromPEM(caCert)

		if tlsConfig == nil {
			tlsConfig = &tls.Config{}
		}

		tlsConfig.ClientCAs = rootCAs
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	if handler == nil {
		handler = func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte("OK"))
		}
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(handler))

	server.TLS = tlsConfig
	server.StartTLS()

	return server, nil
}
