package session

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/awstesting"
)

var TLSBundleCertFile string
var TLSBundleKeyFile string
var TLSBundleCAFile string

func TestMain(m *testing.M) {
	var err error

	TLSBundleCertFile, TLSBundleKeyFile, TLSBundleCAFile, err = awstesting.CreateTLSBundleFiles()
	if err != nil {
		panic(err)
	}

	fmt.Println("TestMain", TLSBundleCertFile, TLSBundleKeyFile)

	code := m.Run()

	err = awstesting.CleanupTLSBundleFiles(TLSBundleCertFile, TLSBundleKeyFile, TLSBundleCAFile)
	if err != nil {
		panic(err)
	}

	os.Exit(code)
}

func TestNewSession_WithCustomCABundle_Env(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	endpoint, err := awstesting.CreateTLSServer(TLSBundleCertFile, TLSBundleKeyFile, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	os.Setenv("AWS_CA_BUNDLE", TLSBundleCAFile)

	s, err := NewSession(&aws.Config{
		HTTPClient:  &http.Client{},
		Endpoint:    aws.String(endpoint),
		Region:      aws.String("mock-region"),
		Credentials: credentials.AnonymousCredentials,
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if s == nil {
		t.Fatalf("expect session to be created, got none")
	}

	req, _ := http.NewRequest("GET", *s.Config.Endpoint, nil)
	resp, err := s.Config.HTTPClient.Do(req)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}

func TestNewSession_WithCustomCABundle_EnvNotExists(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_CA_BUNDLE", "file-not-exists")

	s, err := NewSession()
	if err == nil {
		t.Fatalf("expect error, got none")
	}
	if e, a := "LoadCustomCABundleError", err.(awserr.Error).Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}
	if s != nil {
		t.Errorf("expect nil session, got %v", s)
	}
}

func TestNewSession_WithCustomCABundle_Option(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	endpoint, err := awstesting.CreateTLSServer(TLSBundleCertFile, TLSBundleKeyFile, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	s, err := NewSessionWithOptions(Options{
		Config: aws.Config{
			HTTPClient:  &http.Client{},
			Endpoint:    aws.String(endpoint),
			Region:      aws.String("mock-region"),
			Credentials: credentials.AnonymousCredentials,
		},
		CustomCABundle: bytes.NewReader(awstesting.TLSBundleCA),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if s == nil {
		t.Fatalf("expect session to be created, got none")
	}

	req, _ := http.NewRequest("GET", *s.Config.Endpoint, nil)
	resp, err := s.Config.HTTPClient.Do(req)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}

func TestNewSession_WithCustomCABundle_HTTPProxyAvailable(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	s, err := NewSessionWithOptions(Options{
		Config: aws.Config{
			HTTPClient:  &http.Client{},
			Region:      aws.String("mock-region"),
			Credentials: credentials.AnonymousCredentials,
		},
		CustomCABundle: bytes.NewReader(awstesting.TLSBundleCA),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if s == nil {
		t.Fatalf("expect session to be created, got none")
	}

	tr := s.Config.HTTPClient.Transport.(*http.Transport)
	if tr.Proxy == nil {
		t.Fatalf("expect transport proxy, was nil")
	}
	if tr.TLSClientConfig.RootCAs == nil {
		t.Fatalf("expect TLS config to have root CAs")
	}
}

func TestNewSession_WithCustomCABundle_OptionPriority(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	endpoint, err := awstesting.CreateTLSServer(TLSBundleCertFile, TLSBundleKeyFile, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	os.Setenv("AWS_CA_BUNDLE", "file-not-exists")

	s, err := NewSessionWithOptions(Options{
		Config: aws.Config{
			HTTPClient:  &http.Client{},
			Endpoint:    aws.String(endpoint),
			Region:      aws.String("mock-region"),
			Credentials: credentials.AnonymousCredentials,
		},
		CustomCABundle: bytes.NewReader(awstesting.TLSBundleCA),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if s == nil {
		t.Fatalf("expect session to be created, got none")
	}

	req, _ := http.NewRequest("GET", *s.Config.Endpoint, nil)
	resp, err := s.Config.HTTPClient.Do(req)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}

type mockRoundTripper struct{}

func (m *mockRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	return nil, nil
}

func TestNewSession_WithCustomCABundle_UnsupportedTransport(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	s, err := NewSessionWithOptions(Options{
		Config: aws.Config{
			HTTPClient: &http.Client{
				Transport: &mockRoundTripper{},
			},
		},
		CustomCABundle: bytes.NewReader(awstesting.TLSBundleCA),
	})
	if err == nil {
		t.Fatalf("expect error, got none")
	}
	if e, a := "LoadCustomCABundleError", err.(awserr.Error).Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}
	if s != nil {
		t.Errorf("expect nil session, got %v", s)
	}
	aerrMsg := err.(awserr.Error).Message()
	if e, a := "transport unsupported type", aerrMsg; !strings.Contains(a, e) {
		t.Errorf("expect %s to be in %s", e, a)
	}
}

func TestNewSession_WithCustomCABundle_TransportSet(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	endpoint, err := awstesting.CreateTLSServer(TLSBundleCertFile, TLSBundleKeyFile, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	s, err := NewSessionWithOptions(Options{
		Config: aws.Config{
			Endpoint:    aws.String(endpoint),
			Region:      aws.String("mock-region"),
			Credentials: credentials.AnonymousCredentials,
			HTTPClient: &http.Client{
				Transport: &http.Transport{
					Proxy: http.ProxyFromEnvironment,
					Dial: (&net.Dialer{
						Timeout:   30 * time.Second,
						KeepAlive: 30 * time.Second,
						DualStack: true,
					}).Dial,
					TLSHandshakeTimeout: 2 * time.Second,
				},
			},
		},
		CustomCABundle: bytes.NewReader(awstesting.TLSBundleCA),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if s == nil {
		t.Fatalf("expect session to be created, got none")
	}

	req, _ := http.NewRequest("GET", *s.Config.Endpoint, nil)
	resp, err := s.Config.HTTPClient.Do(req)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}
