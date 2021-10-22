// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.8

package config

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	yaml "gopkg.in/yaml.v2"
)

const (
	TLSCAChainPath        = "testdata/tls-ca-chain.pem"
	ServerCertificatePath = "testdata/server.crt"
	ServerKeyPath         = "testdata/server.key"
	ClientCertificatePath = "testdata/client.crt"
	ClientKeyNoPassPath   = "testdata/client-no-pass.key"
	InvalidCA             = "testdata/client-no-pass.key"
	WrongClientCertPath   = "testdata/self-signed-client.crt"
	WrongClientKeyPath    = "testdata/self-signed-client.key"
	EmptyFile             = "testdata/empty"
	MissingCA             = "missing/ca.crt"
	MissingCert           = "missing/cert.crt"
	MissingKey            = "missing/secret.key"

	ExpectedMessage        = "I'm here to serve you!!!"
	BearerToken            = "theanswertothegreatquestionoflifetheuniverseandeverythingisfortytwo"
	BearerTokenFile        = "testdata/bearer.token"
	MissingBearerTokenFile = "missing/bearer.token"
	ExpectedBearer         = "Bearer " + BearerToken
	ExpectedUsername       = "arthurdent"
	ExpectedPassword       = "42"
)

var invalidHTTPClientConfigs = []struct {
	httpClientConfigFile string
	errMsg               string
}{
	{
		httpClientConfigFile: "testdata/http.conf.bearer-token-and-file-set.bad.yml",
		errMsg:               "at most one of bearer_token & bearer_token_file must be configured",
	},
	{
		httpClientConfigFile: "testdata/http.conf.empty.bad.yml",
		errMsg:               "at most one of basic_auth, bearer_token & bearer_token_file must be configured",
	},
	{
		httpClientConfigFile: "testdata/http.conf.basic-auth.too-much.bad.yaml",
		errMsg:               "at most one of basic_auth password & password_file must be configured",
	},
}

func newTestServer(handler func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, error) {
	testServer := httptest.NewUnstartedServer(http.HandlerFunc(handler))

	tlsCAChain, err := ioutil.ReadFile(TLSCAChainPath)
	if err != nil {
		return nil, fmt.Errorf("Can't read %s", TLSCAChainPath)
	}
	serverCertificate, err := tls.LoadX509KeyPair(ServerCertificatePath, ServerKeyPath)
	if err != nil {
		return nil, fmt.Errorf("Can't load X509 key pair %s - %s", ServerCertificatePath, ServerKeyPath)
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(tlsCAChain)

	testServer.TLS = &tls.Config{
		Certificates: make([]tls.Certificate, 1),
		RootCAs:      rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    rootCAs}
	testServer.TLS.Certificates[0] = serverCertificate

	testServer.StartTLS()

	return testServer, nil
}

func TestNewClientFromConfig(t *testing.T) {
	var newClientValidConfig = []struct {
		clientConfig HTTPClientConfig
		handler      func(w http.ResponseWriter, r *http.Request)
	}{
		{
			clientConfig: HTTPClientConfig{
				TLSConfig: TLSConfig{
					CAFile:             "",
					CertFile:           ClientCertificatePath,
					KeyFile:            ClientKeyNoPassPath,
					ServerName:         "",
					InsecureSkipVerify: true},
			},
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, ExpectedMessage)
			},
		}, {
			clientConfig: HTTPClientConfig{
				TLSConfig: TLSConfig{
					CAFile:             TLSCAChainPath,
					CertFile:           ClientCertificatePath,
					KeyFile:            ClientKeyNoPassPath,
					ServerName:         "",
					InsecureSkipVerify: false},
			},
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, ExpectedMessage)
			},
		}, {
			clientConfig: HTTPClientConfig{
				BearerToken: BearerToken,
				TLSConfig: TLSConfig{
					CAFile:             TLSCAChainPath,
					CertFile:           ClientCertificatePath,
					KeyFile:            ClientKeyNoPassPath,
					ServerName:         "",
					InsecureSkipVerify: false},
			},
			handler: func(w http.ResponseWriter, r *http.Request) {
				bearer := r.Header.Get("Authorization")
				if bearer != ExpectedBearer {
					fmt.Fprintf(w, "The expected Bearer Authorization (%s) differs from the obtained Bearer Authorization (%s)",
						ExpectedBearer, bearer)
				} else {
					fmt.Fprint(w, ExpectedMessage)
				}
			},
		}, {
			clientConfig: HTTPClientConfig{
				BearerTokenFile: BearerTokenFile,
				TLSConfig: TLSConfig{
					CAFile:             TLSCAChainPath,
					CertFile:           ClientCertificatePath,
					KeyFile:            ClientKeyNoPassPath,
					ServerName:         "",
					InsecureSkipVerify: false},
			},
			handler: func(w http.ResponseWriter, r *http.Request) {
				bearer := r.Header.Get("Authorization")
				if bearer != ExpectedBearer {
					fmt.Fprintf(w, "The expected Bearer Authorization (%s) differs from the obtained Bearer Authorization (%s)",
						ExpectedBearer, bearer)
				} else {
					fmt.Fprint(w, ExpectedMessage)
				}
			},
		}, {
			clientConfig: HTTPClientConfig{
				BasicAuth: &BasicAuth{
					Username: ExpectedUsername,
					Password: ExpectedPassword,
				},
				TLSConfig: TLSConfig{
					CAFile:             TLSCAChainPath,
					CertFile:           ClientCertificatePath,
					KeyFile:            ClientKeyNoPassPath,
					ServerName:         "",
					InsecureSkipVerify: false},
			},
			handler: func(w http.ResponseWriter, r *http.Request) {
				username, password, ok := r.BasicAuth()
				if !ok {
					fmt.Fprintf(w, "The Authorization header wasn't set")
				} else if ExpectedUsername != username {
					fmt.Fprintf(w, "The expected username (%s) differs from the obtained username (%s).", ExpectedUsername, username)
				} else if ExpectedPassword != password {
					fmt.Fprintf(w, "The expected password (%s) differs from the obtained password (%s).", ExpectedPassword, password)
				} else {
					fmt.Fprint(w, ExpectedMessage)
				}
			},
		},
	}

	for _, validConfig := range newClientValidConfig {
		testServer, err := newTestServer(validConfig.handler)
		if err != nil {
			t.Fatal(err.Error())
		}
		defer testServer.Close()

		client, err := NewClientFromConfig(validConfig.clientConfig, "test", false)
		if err != nil {
			t.Errorf("Can't create a client from this config: %+v", validConfig.clientConfig)
			continue
		}
		response, err := client.Get(testServer.URL)
		if err != nil {
			t.Errorf("Can't connect to the test server using this config: %+v", validConfig.clientConfig)
			continue
		}

		message, err := ioutil.ReadAll(response.Body)
		response.Body.Close()
		if err != nil {
			t.Errorf("Can't read the server response body using this config: %+v", validConfig.clientConfig)
			continue
		}

		trimMessage := strings.TrimSpace(string(message))
		if ExpectedMessage != trimMessage {
			t.Errorf("The expected message (%s) differs from the obtained message (%s) using this config: %+v",
				ExpectedMessage, trimMessage, validConfig.clientConfig)
		}
	}
}

func TestNewClientFromInvalidConfig(t *testing.T) {
	var newClientInvalidConfig = []struct {
		clientConfig HTTPClientConfig
		errorMsg     string
	}{
		{
			clientConfig: HTTPClientConfig{
				TLSConfig: TLSConfig{
					CAFile:             MissingCA,
					InsecureSkipVerify: true},
			},
			errorMsg: fmt.Sprintf("unable to load specified CA cert %s:", MissingCA),
		},
		{
			clientConfig: HTTPClientConfig{
				TLSConfig: TLSConfig{
					CAFile:             InvalidCA,
					InsecureSkipVerify: true},
			},
			errorMsg: fmt.Sprintf("unable to use specified CA cert %s", InvalidCA),
		},
	}

	for _, invalidConfig := range newClientInvalidConfig {
		client, err := NewClientFromConfig(invalidConfig.clientConfig, "test", false)
		if client != nil {
			t.Errorf("A client instance was returned instead of nil using this config: %+v", invalidConfig.clientConfig)
		}
		if err == nil {
			t.Errorf("No error was returned using this config: %+v", invalidConfig.clientConfig)
		}
		if !strings.Contains(err.Error(), invalidConfig.errorMsg) {
			t.Errorf("Expected error %q does not contain %q", err.Error(), invalidConfig.errorMsg)
		}
	}
}

func TestMissingBearerAuthFile(t *testing.T) {
	cfg := HTTPClientConfig{
		BearerTokenFile: MissingBearerTokenFile,
		TLSConfig: TLSConfig{
			CAFile:             TLSCAChainPath,
			CertFile:           ClientCertificatePath,
			KeyFile:            ClientKeyNoPassPath,
			ServerName:         "",
			InsecureSkipVerify: false},
	}
	handler := func(w http.ResponseWriter, r *http.Request) {
		bearer := r.Header.Get("Authorization")
		if bearer != ExpectedBearer {
			fmt.Fprintf(w, "The expected Bearer Authorization (%s) differs from the obtained Bearer Authorization (%s)",
				ExpectedBearer, bearer)
		} else {
			fmt.Fprint(w, ExpectedMessage)
		}
	}

	testServer, err := newTestServer(handler)
	if err != nil {
		t.Fatal(err.Error())
	}
	defer testServer.Close()

	client, err := NewClientFromConfig(cfg, "test", false)
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.Get(testServer.URL)
	if err == nil {
		t.Fatal("No error is returned here")
	}

	if !strings.Contains(err.Error(), "unable to read bearer token file missing/bearer.token: open missing/bearer.token: no such file or directory") {
		t.Fatal("wrong error message being returned")
	}
}

func TestBearerAuthRoundTripper(t *testing.T) {
	const (
		newBearerToken = "goodbyeandthankyouforthefish"
	)

	fakeRoundTripper := NewRoundTripCheckRequest(func(req *http.Request) {
		bearer := req.Header.Get("Authorization")
		if bearer != ExpectedBearer {
			t.Errorf("The expected Bearer Authorization (%s) differs from the obtained Bearer Authorization (%s)",
				ExpectedBearer, bearer)
		}
	}, nil, nil)

	// Normal flow.
	bearerAuthRoundTripper := NewBearerAuthRoundTripper(BearerToken, fakeRoundTripper)
	request, _ := http.NewRequest("GET", "/hitchhiker", nil)
	request.Header.Set("User-Agent", "Douglas Adams mind")
	_, err := bearerAuthRoundTripper.RoundTrip(request)
	if err != nil {
		t.Errorf("unexpected error while executing RoundTrip: %s", err.Error())
	}

	// Should honor already Authorization header set.
	bearerAuthRoundTripperShouldNotModifyExistingAuthorization := NewBearerAuthRoundTripper(newBearerToken, fakeRoundTripper)
	request, _ = http.NewRequest("GET", "/hitchhiker", nil)
	request.Header.Set("Authorization", ExpectedBearer)
	_, err = bearerAuthRoundTripperShouldNotModifyExistingAuthorization.RoundTrip(request)
	if err != nil {
		t.Errorf("unexpected error while executing RoundTrip: %s", err.Error())
	}
}

func TestBearerAuthFileRoundTripper(t *testing.T) {
	fakeRoundTripper := NewRoundTripCheckRequest(func(req *http.Request) {
		bearer := req.Header.Get("Authorization")
		if bearer != ExpectedBearer {
			t.Errorf("The expected Bearer Authorization (%s) differs from the obtained Bearer Authorization (%s)",
				ExpectedBearer, bearer)
		}
	}, nil, nil)

	// Normal flow.
	bearerAuthRoundTripper := NewBearerAuthFileRoundTripper(BearerTokenFile, fakeRoundTripper)
	request, _ := http.NewRequest("GET", "/hitchhiker", nil)
	request.Header.Set("User-Agent", "Douglas Adams mind")
	_, err := bearerAuthRoundTripper.RoundTrip(request)
	if err != nil {
		t.Errorf("unexpected error while executing RoundTrip: %s", err.Error())
	}

	// Should honor already Authorization header set.
	bearerAuthRoundTripperShouldNotModifyExistingAuthorization := NewBearerAuthFileRoundTripper(MissingBearerTokenFile, fakeRoundTripper)
	request, _ = http.NewRequest("GET", "/hitchhiker", nil)
	request.Header.Set("Authorization", ExpectedBearer)
	_, err = bearerAuthRoundTripperShouldNotModifyExistingAuthorization.RoundTrip(request)
	if err != nil {
		t.Errorf("unexpected error while executing RoundTrip: %s", err.Error())
	}
}

func TestTLSConfig(t *testing.T) {
	configTLSConfig := TLSConfig{
		CAFile:             TLSCAChainPath,
		CertFile:           ClientCertificatePath,
		KeyFile:            ClientKeyNoPassPath,
		ServerName:         "localhost",
		InsecureSkipVerify: false}

	tlsCAChain, err := ioutil.ReadFile(TLSCAChainPath)
	if err != nil {
		t.Fatalf("Can't read the CA certificate chain (%s)",
			TLSCAChainPath)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(tlsCAChain)

	expectedTLSConfig := &tls.Config{
		RootCAs:            rootCAs,
		ServerName:         configTLSConfig.ServerName,
		InsecureSkipVerify: configTLSConfig.InsecureSkipVerify}

	tlsConfig, err := NewTLSConfig(&configTLSConfig)
	if err != nil {
		t.Fatalf("Can't create a new TLS Config from a configuration (%s).", err)
	}

	clientCertificate, err := tls.LoadX509KeyPair(ClientCertificatePath, ClientKeyNoPassPath)
	if err != nil {
		t.Fatalf("Can't load the client key pair ('%s' and '%s'). Reason: %s",
			ClientCertificatePath, ClientKeyNoPassPath, err)
	}
	cert, err := tlsConfig.GetClientCertificate(nil)
	if err != nil {
		t.Fatalf("unexpected error returned by tlsConfig.GetClientCertificate(): %s", err)
	}
	if !reflect.DeepEqual(cert, &clientCertificate) {
		t.Fatalf("Unexpected client certificate result: \n\n%+v\n expected\n\n%+v", cert, clientCertificate)
	}

	// non-nil functions are never equal.
	tlsConfig.GetClientCertificate = nil
	if !reflect.DeepEqual(tlsConfig, expectedTLSConfig) {
		t.Fatalf("Unexpected TLS Config result: \n\n%+v\n expected\n\n%+v", tlsConfig, expectedTLSConfig)
	}
}

func TestTLSConfigEmpty(t *testing.T) {
	configTLSConfig := TLSConfig{
		InsecureSkipVerify: true,
	}

	expectedTLSConfig := &tls.Config{
		InsecureSkipVerify: configTLSConfig.InsecureSkipVerify,
	}

	tlsConfig, err := NewTLSConfig(&configTLSConfig)
	if err != nil {
		t.Fatalf("Can't create a new TLS Config from a configuration (%s).", err)
	}

	if !reflect.DeepEqual(tlsConfig, expectedTLSConfig) {
		t.Fatalf("Unexpected TLS Config result: \n\n%+v\n expected\n\n%+v", tlsConfig, expectedTLSConfig)
	}
}

func TestTLSConfigInvalidCA(t *testing.T) {
	var invalidTLSConfig = []struct {
		configTLSConfig TLSConfig
		errorMessage    string
	}{
		{
			configTLSConfig: TLSConfig{
				CAFile:             MissingCA,
				CertFile:           "",
				KeyFile:            "",
				ServerName:         "",
				InsecureSkipVerify: false},
			errorMessage: fmt.Sprintf("unable to load specified CA cert %s:", MissingCA),
		}, {
			configTLSConfig: TLSConfig{
				CAFile:             "",
				CertFile:           MissingCert,
				KeyFile:            ClientKeyNoPassPath,
				ServerName:         "",
				InsecureSkipVerify: false},
			errorMessage: fmt.Sprintf("unable to use specified client cert (%s) & key (%s):", MissingCert, ClientKeyNoPassPath),
		}, {
			configTLSConfig: TLSConfig{
				CAFile:             "",
				CertFile:           ClientCertificatePath,
				KeyFile:            MissingKey,
				ServerName:         "",
				InsecureSkipVerify: false},
			errorMessage: fmt.Sprintf("unable to use specified client cert (%s) & key (%s):", ClientCertificatePath, MissingKey),
		},
	}

	for _, anInvalididTLSConfig := range invalidTLSConfig {
		tlsConfig, err := NewTLSConfig(&anInvalididTLSConfig.configTLSConfig)
		if tlsConfig != nil && err == nil {
			t.Errorf("The TLS Config could be created even with this %+v", anInvalididTLSConfig.configTLSConfig)
			continue
		}
		if !strings.Contains(err.Error(), anInvalididTLSConfig.errorMessage) {
			t.Errorf("The expected error should contain %s, but got %s", anInvalididTLSConfig.errorMessage, err)
		}
	}
}

func TestBasicAuthNoPassword(t *testing.T) {
	cfg, _, err := LoadHTTPConfigFile("testdata/http.conf.basic-auth.no-password.yaml")
	if err != nil {
		t.Fatalf("Error loading HTTP client config: %v", err)
	}
	client, err := NewClientFromConfig(*cfg, "test", false)
	if err != nil {
		t.Fatalf("Error creating HTTP Client: %v", err)
	}

	rt, ok := client.Transport.(*basicAuthRoundTripper)
	if !ok {
		t.Fatalf("Error casting to basic auth transport, %v", client.Transport)
	}

	if rt.username != "user" {
		t.Errorf("Bad HTTP client username: %s", rt.username)
	}
	if string(rt.password) != "" {
		t.Errorf("Expected empty HTTP client password: %s", rt.password)
	}
	if string(rt.passwordFile) != "" {
		t.Errorf("Expected empty HTTP client passwordFile: %s", rt.passwordFile)
	}
}

func TestBasicAuthNoUsername(t *testing.T) {
	cfg, _, err := LoadHTTPConfigFile("testdata/http.conf.basic-auth.no-username.yaml")
	if err != nil {
		t.Fatalf("Error loading HTTP client config: %v", err)
	}
	client, err := NewClientFromConfig(*cfg, "test", false)
	if err != nil {
		t.Fatalf("Error creating HTTP Client: %v", err)
	}

	rt, ok := client.Transport.(*basicAuthRoundTripper)
	if !ok {
		t.Fatalf("Error casting to basic auth transport, %v", client.Transport)
	}

	if rt.username != "" {
		t.Errorf("Got unexpected username: %s", rt.username)
	}
	if string(rt.password) != "secret" {
		t.Errorf("Unexpected HTTP client password: %s", string(rt.password))
	}
	if string(rt.passwordFile) != "" {
		t.Errorf("Expected empty HTTP client passwordFile: %s", rt.passwordFile)
	}
}

func TestBasicAuthPasswordFile(t *testing.T) {
	cfg, _, err := LoadHTTPConfigFile("testdata/http.conf.basic-auth.good.yaml")
	if err != nil {
		t.Fatalf("Error loading HTTP client config: %v", err)
	}
	client, err := NewClientFromConfig(*cfg, "test", false)
	if err != nil {
		t.Fatalf("Error creating HTTP Client: %v", err)
	}

	rt, ok := client.Transport.(*basicAuthRoundTripper)
	if !ok {
		t.Fatalf("Error casting to basic auth transport, %v", client.Transport)
	}

	if rt.username != "user" {
		t.Errorf("Bad HTTP client username: %s", rt.username)
	}
	if string(rt.password) != "" {
		t.Errorf("Bad HTTP client password: %s", rt.password)
	}
	if string(rt.passwordFile) != "testdata/basic-auth-password" {
		t.Errorf("Bad HTTP client passwordFile: %s", rt.passwordFile)
	}
}

func getCertificateBlobs(t *testing.T) map[string][]byte {
	files := []string{
		TLSCAChainPath,
		ClientCertificatePath,
		ClientKeyNoPassPath,
		ServerCertificatePath,
		ServerKeyPath,
		WrongClientCertPath,
		WrongClientKeyPath,
		EmptyFile,
	}
	bs := make(map[string][]byte, len(files)+1)
	for _, f := range files {
		b, err := ioutil.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		bs[f] = b
	}

	return bs
}

func writeCertificate(bs map[string][]byte, src string, dst string) {
	b, ok := bs[src]
	if !ok {
		panic(fmt.Sprintf("Couldn't find %q in bs", src))
	}
	if err := ioutil.WriteFile(dst, b, 0664); err != nil {
		panic(err)
	}
}

func TestTLSRoundTripper(t *testing.T) {
	bs := getCertificateBlobs(t)

	tmpDir, err := ioutil.TempDir("", "tlsroundtripper")
	if err != nil {
		t.Fatal("Failed to create tmp dir", err)
	}
	defer os.RemoveAll(tmpDir)

	ca, cert, key := filepath.Join(tmpDir, "ca"), filepath.Join(tmpDir, "cert"), filepath.Join(tmpDir, "key")

	handler := func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, ExpectedMessage)
	}
	testServer, err := newTestServer(handler)
	if err != nil {
		t.Fatal(err.Error())
	}
	defer testServer.Close()

	testCases := []struct {
		ca   string
		cert string
		key  string

		errMsg string
	}{
		{
			// Valid certs.
			ca:   TLSCAChainPath,
			cert: ClientCertificatePath,
			key:  ClientKeyNoPassPath,
		},
		{
			// CA not matching.
			ca:   ClientCertificatePath,
			cert: ClientCertificatePath,
			key:  ClientKeyNoPassPath,

			errMsg: "certificate signed by unknown authority",
		},
		{
			// Invalid client cert+key.
			ca:   TLSCAChainPath,
			cert: WrongClientCertPath,
			key:  WrongClientKeyPath,

			errMsg: "remote error: tls",
		},
		{
			// CA file empty
			ca:   EmptyFile,
			cert: ClientCertificatePath,
			key:  ClientKeyNoPassPath,

			errMsg: "unable to use specified CA cert",
		},
		{
			// cert file empty
			ca:   TLSCAChainPath,
			cert: EmptyFile,
			key:  ClientKeyNoPassPath,

			errMsg: "failed to find any PEM data in certificate input",
		},
		{
			// key file empty
			ca:   TLSCAChainPath,
			cert: ClientCertificatePath,
			key:  EmptyFile,

			errMsg: "failed to find any PEM data in key input",
		},
		{
			// Valid certs again.
			ca:   TLSCAChainPath,
			cert: ClientCertificatePath,
			key:  ClientKeyNoPassPath,
		},
	}

	cfg := HTTPClientConfig{
		TLSConfig: TLSConfig{
			CAFile:             ca,
			CertFile:           cert,
			KeyFile:            key,
			InsecureSkipVerify: false},
	}

	var c *http.Client
	for i, tc := range testCases {
		tc := tc
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			writeCertificate(bs, tc.ca, ca)
			writeCertificate(bs, tc.cert, cert)
			writeCertificate(bs, tc.key, key)
			if c == nil {
				c, err = NewClientFromConfig(cfg, "test", false)
				if err != nil {
					t.Fatalf("Error creating HTTP Client: %v", err)
				}
			}

			req, err := http.NewRequest(http.MethodGet, testServer.URL, nil)
			if err != nil {
				t.Fatalf("Error creating HTTP request: %v", err)
			}
			r, err := c.Do(req)
			if len(tc.errMsg) > 0 {
				if err == nil {
					r.Body.Close()
					t.Fatalf("Could connect to the test server.")
				}
				if !strings.Contains(err.Error(), tc.errMsg) {
					t.Fatalf("Expected error message to contain %q, got %q", tc.errMsg, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("Can't connect to the test server")
			}

			b, err := ioutil.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				t.Errorf("Can't read the server response body")
			}

			got := strings.TrimSpace(string(b))
			if ExpectedMessage != got {
				t.Errorf("The expected message %q differs from the obtained message %q", ExpectedMessage, got)
			}
		})
	}
}

func TestTLSRoundTripperRaces(t *testing.T) {
	bs := getCertificateBlobs(t)

	tmpDir, err := ioutil.TempDir("", "tlsroundtripper")
	if err != nil {
		t.Fatal("Failed to create tmp dir", err)
	}
	defer os.RemoveAll(tmpDir)

	ca, cert, key := filepath.Join(tmpDir, "ca"), filepath.Join(tmpDir, "cert"), filepath.Join(tmpDir, "key")

	handler := func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, ExpectedMessage)
	}
	testServer, err := newTestServer(handler)
	if err != nil {
		t.Fatal(err.Error())
	}
	defer testServer.Close()

	cfg := HTTPClientConfig{
		TLSConfig: TLSConfig{
			CAFile:             ca,
			CertFile:           cert,
			KeyFile:            key,
			InsecureSkipVerify: false},
	}

	var c *http.Client
	writeCertificate(bs, TLSCAChainPath, ca)
	writeCertificate(bs, ClientCertificatePath, cert)
	writeCertificate(bs, ClientKeyNoPassPath, key)
	c, err = NewClientFromConfig(cfg, "test", false)
	if err != nil {
		t.Fatalf("Error creating HTTP Client: %v", err)
	}

	var wg sync.WaitGroup
	ch := make(chan struct{})
	var total, ok int64
	// Spawn 10 Go routines polling the server concurrently.
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ch:
					return
				default:
					atomic.AddInt64(&total, 1)
					r, err := c.Get(testServer.URL)
					if err == nil {
						r.Body.Close()
						atomic.AddInt64(&ok, 1)
					}
				}
			}
		}()
	}

	// Change the CA file every 10ms for 1 second.
	wg.Add(1)
	go func() {
		defer wg.Done()
		i := 0
		for {
			tick := time.NewTicker(10 * time.Millisecond)
			<-tick.C
			if i%2 == 0 {
				writeCertificate(bs, ClientCertificatePath, ca)
			} else {
				writeCertificate(bs, TLSCAChainPath, ca)
			}
			i++
			if i > 100 {
				close(ch)
				return
			}
		}
	}()

	wg.Wait()
	if ok == total {
		t.Fatalf("Expecting some requests to fail but got %d/%d successful requests", ok, total)
	}
}

func TestHideHTTPClientConfigSecrets(t *testing.T) {
	c, _, err := LoadHTTPConfigFile("testdata/http.conf.good.yml")
	if err != nil {
		t.Errorf("Error parsing %s: %s", "testdata/http.conf.good.yml", err)
	}

	// String method must not reveal authentication credentials.
	s := c.String()
	if strings.Contains(s, "mysecret") {
		t.Fatal("http client config's String method reveals authentication credentials.")
	}
}

func TestValidateHTTPConfig(t *testing.T) {
	cfg, _, err := LoadHTTPConfigFile("testdata/http.conf.good.yml")
	if err != nil {
		t.Errorf("Error loading HTTP client config: %v", err)
	}
	err = cfg.Validate()
	if err != nil {
		t.Fatalf("Error validating %s: %s", "testdata/http.conf.good.yml", err)
	}
}

func TestInvalidHTTPConfigs(t *testing.T) {
	for _, ee := range invalidHTTPClientConfigs {
		_, _, err := LoadHTTPConfigFile(ee.httpClientConfigFile)
		if err == nil {
			t.Error("Expected error with config but got none")
			continue
		}
		if !strings.Contains(err.Error(), ee.errMsg) {
			t.Errorf("Expected error for invalid HTTP client configuration to contain %q but got: %s", ee.errMsg, err)
		}
	}
}

// LoadHTTPConfig parses the YAML input s into a HTTPClientConfig.
func LoadHTTPConfig(s string) (*HTTPClientConfig, error) {
	cfg := &HTTPClientConfig{}
	err := yaml.UnmarshalStrict([]byte(s), cfg)
	if err != nil {
		return nil, err
	}
	return cfg, nil
}

// LoadHTTPConfigFile parses the given YAML file into a HTTPClientConfig.
func LoadHTTPConfigFile(filename string) (*HTTPClientConfig, []byte, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}
	cfg, err := LoadHTTPConfig(string(content))
	if err != nil {
		return nil, nil, err
	}
	return cfg, content, nil
}

type roundTrip struct {
	theResponse *http.Response
	theError    error
}

func (rt *roundTrip) RoundTrip(r *http.Request) (*http.Response, error) {
	return rt.theResponse, rt.theError
}

type roundTripCheckRequest struct {
	checkRequest func(*http.Request)
	roundTrip
}

func (rt *roundTripCheckRequest) RoundTrip(r *http.Request) (*http.Response, error) {
	rt.checkRequest(r)
	return rt.theResponse, rt.theError
}

// NewRoundTripCheckRequest creates a new instance of a type that implements http.RoundTripper,
// which before returning theResponse and theError, executes checkRequest against a http.Request.
func NewRoundTripCheckRequest(checkRequest func(*http.Request), theResponse *http.Response, theError error) http.RoundTripper {
	return &roundTripCheckRequest{
		checkRequest: checkRequest,
		roundTrip: roundTrip{
			theResponse: theResponse,
			theError:    theError}}
}
